"""Re-run both benchmarks at vendor-recommended inference params (kimi removed).
Writes to v3 result files; resumable (skips tasks already ok). Atomic saves.

Usage: python3 rerun_vendor.py text   |   python3 rerun_vendor.py vl
Config locked in model-params-research.md (FINAL).
"""
import base64
import json
import mimetypes
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from endpoints import get_endpoints, _load_env

_load_env()
EPS = {e["tag"]: e for e in get_endpoints()}
ND_URL = os.environ.get("NEURALDEEP_URL", "https://api.neuraldeep.ru/v1/chat/completions")
ND_KEY = os.environ.get("NEURALDEEP_API_KEY", "")

# ---- per-model vendor params (final, user-approved) ----
TEXT_CFG = {
    "27b-v2":        {"ep": "27b-v2",  "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "thinking": True},  # precise preset (RU benchmark)
    "35b-a3b":       {"ep": "35b-a3b", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "thinking": True},  # precise preset (RU benchmark)
    "gpt-oss-120b":  {"ep": "gpt-oss-120b", "url": "https://api.groq.com/openai/v1", "temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"},
    "deepseek-flash":{"ep": "deepseek-flash", "reasoning_effort": "high"},
    "deepseek-pro":  {"ep": "deepseek-pro",  "reasoning_effort": "high"},
    "gemma-4-31b":   {"ep": "_nd_gemma", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "reasoning": {"enabled": True}},  # thinking on (neuraldeep param)
    "diffusion-gemma": {"ep": "_nd_diff", "nd_model": "diffusion-gemma", "temperature": 0.6, "top_p": 0.95, "top_k": 64, "reasoning": {"enabled": True}},  # temp 0.6 ~ mid of recommended diffusion decay 0.8->0.4 (sampler/steps are server-side, not API-settable)
    "nemotron-super": {"ep": "nemotron-super", "temperature": 1.0, "top_p": 0.95, "thinking": True},  # NVIDIA card: temp 1.0/top_p 0.95 all tasks; reasoning on
    "nemotron-omni":  {"ep": "nemotron-omni",  "temperature": 0.6, "top_p": 0.95, "thinking": True},  # NVIDIA card: temp 0.6/top_p 0.95 reasoning mode
    "kimi-k2.6":      {"ep": "_nd_kimi", "nd_model": "kimi-k2.6", "temperature": 1.0, "top_p": 0.95},  # Moonshot thinking-mode vendor params (1.0/0.95); thinking on by default
    "35b-a3b-nd":     {"ep": "_nd_qwen", "nd_model": "qwen3.6-35b-a3b", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "thinking": True},  # SAME params as the primary 35b-a3b endpoint, routed to the alternate host - isolates host/quantization
}
VL_CFG = {
    "27b-v2":      {"ep": "27b-v2",  "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "thinking": True},
    "35b-a3b":     {"ep": "35b-a3b", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "thinking": True},
    "gemma-4-31b": {"ep": "_nd_gemma", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "reasoning": {"enabled": True}},  # gemma card: 1.0 all use cases; thinking on
    "diffusion-gemma": {"ep": "_nd_diff", "nd_model": "diffusion-gemma", "temperature": 0.6, "top_p": 0.95, "top_k": 64, "reasoning": {"enabled": True}},  # temp 0.6 ~ recommended diffusion range (vision-capable)
    "llama-scout": {"ep": "llama-scout", "temperature": 0.6, "top_p": 0.9, "max_tokens": 8192, "img_max_px": 32000000},  # groq caps output at 8192 & rejects >33Mpx images
    "nemotron-omni": {"ep": "nemotron-omni", "temperature": 0.6, "top_p": 0.95, "thinking": True},  # vision-capable; NVIDIA card temp 0.6/top_p 0.95
    "kimi-k2.6":     {"ep": "_nd_kimi", "nd_model": "kimi-k2.6", "temperature": 1.0, "top_p": 0.95},  # Moonshot thinking-mode vendor params; vision-capable, thinking on by default
    "35b-a3b-nd":    {"ep": "_nd_qwen", "nd_model": "qwen3.6-35b-a3b", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "thinking": True},  # SAME params as the primary 35b-a3b endpoint, routed to the alternate host
}


def data_url(path, max_px=None):
    # groq (llama-scout) rejects images >33.18Mpx; downscale a JPEG copy when capped
    if max_px:
        try:
            import io
            import math
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(path).convert("RGB")
            if im.width * im.height > max_px:
                s = math.sqrt(max_px / (im.width * im.height))
                im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
                b = io.BytesIO(); im.save(b, "JPEG", quality=92)
                return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
        except Exception:
            pass
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


GROQ_NUDGE = ("Не вызывай инструменты/функции по-настоящему. В этой среде нет исполнения инструментов. "
              "Любые вызовы инструментов описывай ТЕКСТОМ в своём ответе (имя функции и аргументы в виде JSON), "
              "рассуждая по шагам. Отвечай обычным сообщением.")


def build_messages(prompt, image, max_px=None):
    if image:
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url(image, max_px)}}]}]
    return [{"role": "user", "content": prompt}]


def call(tag, cfg, prompt, image, max_tokens, timeout):
    """Route to the right endpoint with vendor params. Returns response dict."""
    epkey = cfg["ep"]
    msgs = build_messages(prompt, image, cfg.get("img_max_px"))
    if epkey.startswith("_nd"):
        url, key, etype = ND_URL, ND_KEY, "nd"
        model = cfg.get("nd_model", "gemma-4-31b")
    else:
        ep = EPS[epkey]
        etype = ep["type"]
        model = ep["model"]
        key = ep["key"]
        base = cfg.get("url", ep["url"])   # allow per-model URL override (e.g. gpt-oss direct groq)
        url = base + ("/v1/chat/completions" if etype == "qwen" else "/chat/completions")

    mt = min(max_tokens, cfg["max_tokens"]) if "max_tokens" in cfg else max_tokens
    payload = {"model": model, "messages": msgs}
    if etype == "groq":
        payload["max_completion_tokens"] = mt
    else:
        payload["max_tokens"] = mt
    for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "reasoning_effort", "reasoning"):
        if k in cfg:
            payload[k] = cfg[k]
    if cfg.get("thinking"):
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if etype == "groq":
        headers["User-Agent"] = "Mozilla/5.0"

    attempts = 6
    last = None
    groq_nudged = False
    for a in range(1, attempts + 1):
        t = time.monotonic()
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            dt = int((time.monotonic() - t) * 1000)
            if r.status_code == 400 and etype == "groq" and not groq_nudged and "tool_use_failed" in (r.text or ""):
                # gpt-oss tried to emit a native tool call; retry asking for text-only output
                groq_nudged = True
                payload["messages"] = [{"role": "system", "content": GROQ_NUDGE}] + msgs
                last = "groq tool_use_failed (retry text-only)"
                continue
            if r.status_code == 429:
                last = "HTTP 429"
                if a < attempts:
                    time.sleep(min(60, 5 * a)); continue
                return {"ok": False, "error": last, "latency_ms": dt}
            if r.status_code in (404, 500, 502, 503, 504):
                # 404 from neuraldeep = transient load-shedding ("page not found") under concurrency
                last = f"HTTP {r.status_code}"
                if a < attempts:
                    time.sleep(3 * a); continue
                return {"ok": False, "error": last, "latency_ms": dt}
            if r.status_code >= 400:
                return {"ok": False, "error": f"HTTP {r.status_code}", "body": r.text[:300], "latency_ms": dt}
            d = r.json()
            ch = d["choices"][0]
            msg = ch["message"]
            content = msg.get("content") or ""
            psf = msg.get("provider_specific_fields") or {}
            reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or psf.get("reasoning"))
            finish = ch.get("finish_reason")
            truncated = (finish == "length" and len(content.strip()) < 15)
            if truncated and a < attempts:
                last = "truncated; retry"; time.sleep(1); continue
            # empty content with the answer leaked into the reasoning channel — stochastic on
            # some reasoning models (nemotron-omni ~18-24%); a re-draw recovers it. Retry.
            if (not content.strip()) and a < attempts:
                last = "empty content (reasoning-leak); retry"; time.sleep(1); continue
            usage = d.get("usage", {})
            details = usage.get("completion_tokens_details") or {}
            return {"ok": True, "content": content, "reasoning": reasoning,
                    "finish_reason": finish, "truncated": truncated,
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "reasoning_tokens": details.get("reasoning_tokens"),
                    "latency_ms": dt, "attempt": a}
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
            if a < attempts:
                time.sleep(2 * a)
    return {"ok": False, "error": last or "failed", "attempt": attempts}


def save_atomic(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def run(mode):
    if mode == "text":
        SRC = "master-benchmark-results-v2.json"
        OUT = "master-benchmark-results-v3.json"
        CFG = TEXT_CFG
        MAXTOK, TIMEOUT, VL = 65536, 600, False
        meta_keys = ("id", "category", "title", "prompt")
    else:
        SRC = "vl-ru-results.json"
        OUT = "vl-ru-results-v3.json"
        CFG = VL_CFG
        MAXTOK, TIMEOUT, VL = 32768, 360, True
        meta_keys = ("id", "capability", "vertical", "title", "prompt", "notes_for_judge", "image")

    src = json.load(open(SRC))["results"]
    tags = list(CFG.keys())
    # load/resume
    out = {"timestamp": "vendor-params", "type": f"{mode}_benchmark_v3",
           "config": {t: CFG[t] for t in tags},
           "results": []}
    existing = {}
    if Path(OUT).exists():
        for r in json.load(open(OUT)).get("results", []):
            existing[r["id"]] = r
    for r in src:
        rec = existing.get(r["id"]) or {k: r[k] for k in meta_keys if k in r}
        rec.setdefault("responses", {})
        out["results"].append(rec)
    by_id = {r["id"]: r for r in out["results"]}

    jobs = []
    for r in src:
        for t in tags:
            cur = by_id[r["id"]]["responses"].get(t)
            if cur and cur.get("ok"):
                continue
            jobs.append((r["id"], t, r["prompt"], r.get("image") if VL else None))
    print(f"{mode}: {len(jobs)} jobs ({len(src)} tasks x {len(tags)} models, skipping done)", flush=True)

    lock = threading.Lock()
    done = {"n": 0}

    def do(job):
        tid, tag, prompt, image = job
        resp = call(tag, CFG[tag], prompt, image, MAXTOK, TIMEOUT)
        with lock:
            by_id[tid]["responses"][tag] = resp
            done["n"] += 1
            if done["n"] % 10 == 0:
                save_atomic(out, OUT)
                print(f"  {done['n']}/{len(jobs)} done", flush=True)

    # workers configurable (3rd arg); lower for gemma-thinking on neuraldeep (404 load-shedding)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, jobs))
    save_atomic(out, OUT)

    from collections import Counter
    print(f"=== {mode} done ===")
    for t in tags:
        c = Counter("ok" if by_id[r["id"]]["responses"].get(t, {}).get("ok") else "fail" for r in src)
        print(f"  {t:16s} {dict(c)}")


if __name__ == "__main__":
    run(sys.argv[1])
