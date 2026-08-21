"""Run GPT-5.5 (via codex exec) as a CONTESTANT on the benchmark tasks.
reasoning effort medium or xhigh. Writes to a per-(tag,mode) file to avoid races; merge later.
VL attaches the image via -i. Resumable, parallel (subprocess pool).

Usage: python3 codex_model.py text medium [workers] | python3 codex_model.py vl xhigh [workers]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

mode = sys.argv[1]
effort = sys.argv[2]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# explicit model id (must be passed via -m; codex config default may differ). Tag derives from it.
MODEL = os.environ.get("CODEX_MODEL", "gpt-5.5")
TAG = f"{MODEL}-{effort}"

if mode == "text":
    RES = "master-benchmark-results-v3.json"
    VL = False
else:
    RES = "vl-ru-results-v3.json"
    VL = True
OUT = f"contestant-{TAG}-{mode}.json"

results = {r["id"]: r for r in json.load(open(RES))["results"]}
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
lock = threading.Lock()
todo = [tid for tid in results if not (done.get(tid, {}).get("ok"))]
prog = {"n": 0, "ok": 0, "fail": 0}
print(f"codex contestant [{TAG}/{mode}]: {len(todo)} tasks (workers={WORKERS})", flush=True)


def save():
    tmp = OUT + ".tmp"
    json.dump(done, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, OUT)


# codex runs sandbox read-only with NO tools registered, so on tool/agentic tasks the model
# aborts ("инструмент X недоступен") instead of describing the routing. Groq models get an
# equivalent nudge (GROQ_NUDGE in rerun_vendor); give codex the same so tool-shaped tasks are
# answered in text, not refused. Prepended to the prompt (codex exec has no separate system msg).
CODEX_NUDGE = (
    "В этой среде инструменты/функции НЕ исполняются — по-настоящему их вызвать нельзя, "
    "и это не ошибка окружения. Если задача подразумевает вызовы инструментов, ОПИСЫВАЙ их "
    "ТЕКСТОМ (имя функции и аргументы в виде JSON), рассуждая по шагам, и дай финальный ответ "
    "как обычным сообщением. НЕ прекращай работу из-за отсутствия инструментов и НЕ выдумывай "
    "результаты их выполнения.\n\n---\n\n"
)


def run_one(tid):
    r = results[tid]
    prompt = CODEX_NUDGE + r["prompt"]
    for attempt in (1, 2):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            outfile = tf.name
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
               "--color", "never", "-m", MODEL, "-c", f"model_reasoning_effort={effort}", "-o", outfile]
        if VL:
            cmd += ["-i", os.path.abspath(r["image"])]
        cmd += ["-"]
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, input=prompt.encode(), capture_output=True, timeout=900)
            dt = int((time.monotonic() - t0) * 1000)
            content = open(outfile).read().strip()
            os.unlink(outfile)
            if len(content) < 1:
                if attempt == 1:
                    continue
                return {"ok": False, "error": "empty", "latency_ms": dt}
            # codex prints "tokens used" then the count, which may use comma OR space/NBSP
            # thousands separators (e.g. "12,572" or "12 572"); grab the whole run and strip.
            m = re.search(r"tokens used[^\d]*([\d][\d,  ]*)", (p.stdout + p.stderr).decode(errors="ignore"))
            ctok = int(re.sub(r"[^\d]", "", m.group(1))) if m else None
            return {"ok": True, "content": content, "completion_tokens": ctok, "latency_ms": dt, "attempt": attempt}
        except Exception as e:
            try:
                os.unlink(outfile)
            except Exception:
                pass
            if attempt == 2:
                return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def work(tid):
    resp = run_one(tid)
    with lock:
        done[tid] = resp
        prog["n"] += 1
        prog["ok" if resp.get("ok") else "fail"] += 1
        if prog["n"] % 10 == 0:
            save()
            print(f"  {prog['n']}/{len(todo)} (ok={prog['ok']} fail={prog['fail']})", flush=True)


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(work, todo))
save()
print(f"=== {TAG}/{mode} done: ok={prog['ok']} fail={prog['fail']} | total {sum(1 for v in done.values() if v.get('ok'))}/{len(results)} ===")
