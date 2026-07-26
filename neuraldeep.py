"""Client for neuraldeep.ru (OpenAI-compatible). Handles 429 (session/week windows),
502 retries, reasoning-token budget. Supports text and VL (image) calls.
"""
import base64
import mimetypes
import os
import time
from pathlib import Path

import requests

from endpoints import _load_env  # populates os.environ from .env

_load_env()
ND_URL = os.environ.get("NEURALDEEP_URL", "https://api.neuraldeep.ru/v1/chat/completions")
ND_KEY = os.environ.get("NEURALDEEP_API_KEY", "")

# display models available
ND_MODELS = {
    "gemma-4-31b": {"model": "gemma-4-31b", "reasoning_field": None, "vision": True},
    "kimi-k2.6": {"model": "kimi-k2.6", "reasoning_field": "reasoning_content", "vision": True},
}


def _data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def call_nd(model_tag, prompt, image_path=None, max_tokens=32768, timeout=240):
    """Returns dict {ok, content, reasoning, finish_reason, completion_tokens, latency_ms}.
    On a long rate-limit cooldown returns {ok:False, error:'rate_limited', retry_after, window}."""
    cfg = ND_MODELS[model_tag]
    if image_path:
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": _data_url(image_path)}}]
    else:
        content = prompt
    payload = {"model": cfg["model"], "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0.6}
    headers = {"Authorization": f"Bearer {ND_KEY}", "Content-Type": "application/json"}
    last = None
    for attempt in range(1, 7):
        t = time.monotonic()
        try:
            r = requests.post(ND_URL, json=payload, headers=headers, timeout=timeout)
            dt = int((time.monotonic() - t) * 1000)
            if r.status_code == 429:
                ra = int(r.headers.get("Retry-After", "0") or 0)
                win = r.headers.get("X-Window", "")
                # hard account-window limit (session ~2h / week): stop only when explicitly signaled
                if win in ("session", "week") and ra > 300:
                    return {"ok": False, "error": "rate_limited", "retry_after": ra, "window": win, "latency_ms": dt}
                # everything else (transient burst, upstream cooldown) -> back off and retry
                if attempt < 6:
                    time.sleep(min(ra, 60) if ra else 30)
                    continue
                return {"ok": False, "error": "HTTP 429 (exhausted)", "latency_ms": dt}
            if r.status_code in (500, 502, 503, 504) and attempt < 4:
                last = f"HTTP {r.status_code}"
                time.sleep(3 * attempt)   # transient upstream (incl. 500) — back off and retry
                continue
            if r.status_code >= 400:
                return {"ok": False, "error": f"HTTP {r.status_code}", "body": r.text[:300], "latency_ms": dt}
            data = r.json()
            msg = data["choices"][0]["message"]
            rf = cfg["reasoning_field"]
            psf = msg.get("provider_specific_fields") or {}
            # capture reasoning from any channel a model might use
            reasoning = ((msg.get(rf) if rf else None) or msg.get("reasoning")
                         or msg.get("reasoning_content") or psf.get("reasoning"))
            return {"ok": True, "content": msg.get("content") or "",
                    "reasoning": reasoning,
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                    "latency_ms": dt, "attempt": attempt}
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
            if attempt < 4:
                time.sleep(2 * attempt)
    return {"ok": False, "error": last or "failed", "attempt": 4}
