"""Merge codex-contestant files (GPT-5.5 / GPT-5.6) into the master/VL result files.
Globs contestant-<prefix>*-<mode>.json and derives the model tag from the filename.

Usage: python3 merge_codex_contestants.py text|vl [prefix]   (prefix default: gpt-5.5)
"""
import glob
import json
import os
import sys

mode = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.5"
RES = "master-benchmark-results-v3.json" if mode == "text" else "vl-ru-results-v3.json"
# tasks where OpenAI's platform cyber-filter blocks the request (returncode 1, empty) —
# behaviour is UNMEASURABLE via codex here; these cells are excluded from scoring.
CYBER_BLOCK_TASKS = {"m-refusal-01", "m-security-01", "m-security-05"}

data = json.load(open(RES))
by_id = {r["id"]: r for r in data["results"]}

files = sorted(glob.glob(f"contestant-{PREFIX}*-{mode}.json"))
for fn in files:
    tag = os.path.basename(fn)[len("contestant-"):-len(f"-{mode}.json")]
    cont = json.load(open(fn))
    added = 0
    for tid, v in cont.items():
        if tid not in by_id:
            continue
        resp = {
            "ok": bool(v.get("ok")),
            "content": v.get("content", ""),
            "reasoning": None,
            "finish_reason": "stop" if v.get("ok") else (v.get("error") or "error"),
            "truncated": False,
            "completion_tokens": v.get("completion_tokens"),
            "prompt_tokens": None,
            "reasoning_tokens": None,
            "latency_ms": v.get("latency_ms"),
            "attempt": v.get("attempt"),
        }
        # mark cells the OpenAI platform cyber-filter blocked (request never reached the model)
        if not resp["ok"] and tid in CYBER_BLOCK_TASKS:
            resp["cyber_blocked"] = True
        by_id[tid]["responses"][tag] = resp
        added += 1
    print(f"{tag}: merged {added} responses")

tmp = RES + ".tmp"
json.dump(data, open(tmp, "w"), ensure_ascii=False, indent=1)
os.replace(tmp, RES)
print(f"saved {RES}")
