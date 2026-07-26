"""Unified re-judge of ALL models with ONE consistent method (Opus, pinned effort=medium).
- content-only packets (no reasoning traces) -> reasoning judged from visible content for ALL
  models uniformly (reasoning is excluded from the scored Σ anyway); keeps prompts tractable.
- TEXT: original per-category focus (judge-args-v3). VL: image + generic vision criteria.
- Scores EVERY tag present in responses. cyber_blocked / ok=false / empty -> all 4 = 0.
- Skips m-refusal-01 (judge safeguard refuses it); those keep existing v3 scores (copied).
- Output -> judging-scores-v4 / vl-judging-scores-v4 (v3 preserved).

Usage: python3 build_unified_judge.py text | vl
"""
import json
import os
import sys
from pathlib import Path

mode = sys.argv[1]
CYBER = {"m-refusal-01", "m-security-01", "m-security-05"}
EXCLUDE = {"35b-a3b-nd", "gpt-5.5-xhigh"}   # host-comparison tag + dropped effort — not in dashboard
if mode == "text":
    RES = "master-benchmark-results-v3.json"
    PKT = Path("judging-packets-v4"); PKT.mkdir(exist_ok=True)
    OUT = Path("judging-scores-v4"); OUT.mkdir(exist_ok=True)
    OLD = Path("judging-scores-v3")
    focus = json.load(open("judge-args-v3.json")).get("focus", {})
    VL = False
    SKIP = {"m-refusal-01"}
else:
    RES = "vl-ru-results-v3.json"
    PKT = Path("vl-judging-packets-v4"); PKT.mkdir(exist_ok=True)
    OUT = Path("vl-judging-scores-v4"); OUT.mkdir(exist_ok=True)
    OLD = Path("vl-judging-scores-v3")
    focus = {}
    VL = True
    SKIP = set()

data = json.load(open(RES))
cat_key = "category" if not VL else "capability"

items = []
for r in data["results"]:
    tid = r["id"]
    # preserve existing scores for skipped (safeguard) tasks
    if tid in SKIP:
        src = OLD / f"{tid}.json"
        if src.exists():
            (OUT / f"{tid}.json").write_text(src.read_text())
        continue
    if (OUT / f"{tid}.json").exists():
        continue  # resumable
    # content-only packet with all tags
    pkt = {"task_id": tid, "category": r.get(cat_key, "?"),
           "prompt": r["prompt"], "notes_for_judge": r.get("notes_for_judge", ""),
           "responses": {t: {"ok": resp.get("ok"),
                             "content": resp.get("content", ""),
                             "cyber_blocked": resp.get("cyber_blocked", False)}
                         for t, resp in r["responses"].items() if t not in EXCLUDE}}
    pp = PKT / f"{tid}.json"
    pp.write_text(json.dumps(pkt, ensure_ascii=False))
    it = {"id": tid, "category": r.get(cat_key, "?"),
          "packet": str(pp.resolve()), "scoresPath": str((OUT / f"{tid}.json").resolve())}
    if VL:
        it["image"] = os.path.abspath(r["image"])
    items.append(it)

if VL:
    PROMPT_JS = r'''function buildPrompt(it) {
  return `Ты — СТРОГИЙ и беспристрастный судья ответов VL-моделей на задачу по изображению.
1) ОТКРОЙ и внимательно рассмотри изображение: ${it.image}
2) Прочитай пакет: ${it.packet}
   В нём: prompt, notes_for_judge (эталон — что реально на картинке), responses — словарь {tag:{ok,content,cyber_blocked}}. ОЦЕНИ ВСЕ модели, присутствующие в responses.
Сверяя ответы С САМОЙ КАРТИНКОЙ и notes_for_judge, оцени КАЖДУЮ модель по 4 критериям (целое 0-10):
- correctness — точность распознавания/интерпретации (главное): верно ли прочитаны данные/факты, нет ли выдумок. Галлюцинация того, чего на картинке НЕТ — correctness 0-1.
- reasoning — качество анализа, видимого в ответе (content)
- format — структурированность и ясность
- russian — качество русского
Если cyber_blocked=true ИЛИ ok=false ИЛИ content пуст — все 4 = 0.
Будь СТРОГ и последователен: одинаковое качество → одинаковый балл у всех моделей. Шкала: 10 идеально; 8-9 отлично; 6-7 хорошо; 4-5 средне; 2-3 серьёзные ошибки; 0-1 провал/выдумка.
Сохрани СТРОГО валидный JSON в ${it.scoresPath}: {"task_id":"${it.id}","category":"${it.category}","scores":{"<tag>":{"correctness":N,"reasoning":N,"format":N,"russian":N,"comment":"1-2 предложения"}, ... для КАЖДОГО tag из responses},"winner":"<tag>","notes":"1-2 предложения"}
Ничего не пиши в чат кроме "done".`
}'''
else:
    PROMPT_JS = r'''const FOCUS = __FOCUS__;
function buildPrompt(it) {
  const f = FOCUS[it.category] || { summary: it.category, correctness: 'фактическая точность по эталону', format: 'структурированность', russian: 'качество русского', extra: 'стандартная оценка' }
  return `Ты — СТРОГИЙ и беспристрастный судья ответов LLM на задачу типа «${f.summary}».
Прочитай файл ${it.packet}. В нём: prompt, notes_for_judge (эталон — ВНИМАТЕЛЬНО изучи), responses — словарь {tag:{ok,content,cyber_blocked}}. ОЦЕНИ ВСЕ модели, присутствующие в responses.
Оцени КАЖДУЮ модель по 4 критериям (целое 0-10):
- correctness — ${f.correctness}
- reasoning — глубина/качество рассуждения, видимого в ответе (content)
- format — ${f.format}
- russian — ${f.russian}
ОСОБЕННОСТЬ: ${f.extra}
Если cyber_blocked=true ИЛИ ok=false ИЛИ content пуст — все 4 = 0.
Будь СТРОГ и последователен: одинаковое качество → одинаковый балл у всех моделей. Скоринг СТРОГО по notes_for_judge, длина ≠ качество. Шкала: 10 идеально; 8-9 отлично; 6-7 хорошо; 4-5 средне; 2-3 серьёзные ошибки; 0-1 провал/выдумка/отказ.
Сохрани СТРОГО валидный JSON в ${it.scoresPath}: {"task_id":"${it.id}","category":"${it.category}","scores":{"<tag>":{"correctness":N,"reasoning":N,"format":N,"russian":N,"comment":"1-2 предложения"}, ... для КАЖДОГО tag из responses},"winner":"<tag>","notes":"1-2 предложения"}
Ничего не пиши в чат кроме "done".`
}'''

WF = '''export const meta = {
  name: 'unified-judge-%s',
  description: 'Unified strict Opus re-judge of ALL models (medium effort)',
  phases: [{ title: 'Judge' }],
}
const A = __ITEMS__;
const items = A.items || []
%s
phase('Judge')
log(`unified judge (%s): ${items.length} tasks, all models per task`)
const CHUNK = 6
const out = []
for (let i = 0; i < items.length; i += CHUNK) {
  const batch = items.slice(i, i + CHUNK)
  const r = await parallel(batch.map(it => () =>
    agent(buildPrompt(it), { label: `uj:${it.id}`, phase: 'Judge', agentType: 'general-purpose', model: 'opus', effort: 'medium' })
      .then(res => ({ id: it.id, ok: res !== null && res !== undefined }))
      .catch(e => ({ id: it.id, ok: false, error: String(e) }))
  ))
  out.push(...r)
  log(`  ${Math.min(i + CHUNK, items.length)}/${items.length} (ok ${out.filter(x => x && x.ok).length})`)
}
return { done: out.filter(r => r && r.ok).length, failed: out.filter(r => r && !r.ok).map(r => r.id) }
''' % (mode, PROMPT_JS, mode)
WF = WF.replace("__ITEMS__", json.dumps({"items": items}, ensure_ascii=False))
WF = WF.replace("__FOCUS__", json.dumps(focus, ensure_ascii=False))
open(f"unified_judge_{mode}_full.js", "w").write(WF)
print(f"{mode}: {len(items)} tasks (content-only, all models) + unified_judge_{mode}_full.js")
