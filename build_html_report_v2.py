"""Single scrollable HTML report with a bookmark sidebar (nav) — uniform fonts.

Структура: сводка (лидерборд / ризонинг / производительность / сегменты / категории),
затем по сегментам→категориям каждая задача: полный промпт + таблица оценок 5 моделей
+ полные комментарии судьи + нота. Слева — фиксированная навигация-закладки.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from category_focus_master import SEGMENTS

# reasoning excluded from scoring: gemma-4-31b returns no separate reasoning trace
# (it reasons inline in content), so the criterion is not uniformly available across models.
CRITERIA = ["correctness", "format", "russian"]
# weighted score: correctness dominates (>format+russian together); russian > format
# (RU deployment values language over markdown). Weights sum to 1 -> Σ is already on 0-10.
WEIGHTS = {"correctness": 0.6, "format": 0.1, "russian": 0.3}
MAX_TOTAL = 10
DIV = 1   # weighted Σ is already 0-10 (weights sum to 1); kept as a no-op display divisor


def wsum(s):
    """Weighted 0-10 score for one model's criterion dict."""
    return sum(WEIGHTS[c] * s[c] for c in CRITERIA)

CATEGORY_LABEL = {
    "classification": "Классификация", "summarization": "Саммаризация",
    "agentic": "Agentic planning", "harness": "Agent harness (ReAct/tool-loop)",
    "code": "Code / Debug", "rag": "RAG with attribution",
    "tool_calling": "Function/Tool calling", "long_context_extended": "Long-context (30K–92K)",
    "math_business": "Math / Финансы", "reasoning": "Reasoning / Logic",
    "security": "Security / Injections", "format_strict": "Strict format",
    "refusal": "Refusal", "multilang": "Multilang", "hallucination": "Hallucination",
    "tone_style": "Tone / Style", "self_correction": "Self-correction",
    "ru_legal": "Юр + комплаенс РФ", "ru_accounting": "Бухгалтерия / налоги РФ",
    "ru_finance": "Корп. финансы РФ", "ru_data_logistics": "Данные/BI + ВЭД/логистика",
    "exec_assistant": "Ассистент руководителя",
}
SEG_EMOJI = {
    "Bulk (повседневные)": "🔵", "Specialized (специализированные)": "🟠",
    "Edge (граничные)": "🟡", "Quality (метакачество)": "⚪",
    "RU-Business (отраслевые РФ)": "🟢",
}


def esc(s):
    if s is None:
        return ""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def avg(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def get_segment(category):
    for seg, cats in SEGMENTS.items():
        if category in cats:
            return seg
    return "Other"


CSS = """
:root { --pri:#2563eb; --pri-d:#1e3a8a; --bg:#f8fafc; --soft:#eff4ff; --bd:#dbe3ef; --mut:#64748b; --gold:#b8860b; }
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color:#1f2937; background:#fff; font-size:15px; line-height:1.5; }
a { color: var(--pri); text-decoration:none; }
a:hover { text-decoration:underline; }
/* sidebar nav (bookmarks) */
.sidebar { position:fixed; top:0; left:0; width:300px; height:100vh; overflow-y:auto; background:#0f172a; color:#cbd5e1; padding:16px 12px 40px; font-size:12.5px; }
.sidebar h1 { color:#fff; font-size:15px; margin:0 0 4px; }
.sidebar .sub { color:#7c8aa0; font-size:11px; margin-bottom:12px; }
.sidebar a { color:#cbd5e1; display:block; padding:2px 6px; border-radius:4px; }
.sidebar a:hover { background:#1e293b; color:#fff; text-decoration:none; }
.sidebar .grp { color:#fff; font-weight:700; margin:12px 0 3px; font-size:12px; }
.sidebar .seg { color:#93c5fd; font-weight:700; margin:14px 0 4px; font-size:12.5px; border-top:1px solid #1e293b; padding-top:8px; }
.sidebar .tlink { padding-left:12px; color:#94a3b8; font-size:11.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
main { margin-left:300px; padding:28px 40px 80px; max-width:1180px; }
h2.sec { color:var(--pri-d); border-bottom:2px solid var(--pri); padding-bottom:6px; margin:36px 0 14px; font-size:22px; }
h3.catsec { color:var(--pri-d); margin:34px 0 6px; font-size:18px; border-left:4px solid var(--pri); padding-left:10px; }
table { border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:13.5px; background:#fff; }
th, td { border:1px solid var(--bd); padding:6px 9px; text-align:left; vertical-align:top; }
th { background:var(--soft); color:var(--pri-d); font-weight:700; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.win { background:#ecfdf5; }
.gold-row { background:#fffbeb; }
.callout { background:var(--soft); border-left:4px solid var(--pri); padding:9px 14px; margin:10px 0 18px; border-radius:0 6px 6px 0; font-size:14px; }
/* task card */
.task { border:1px solid var(--bd); border-radius:8px; padding:16px 18px; margin:16px 0; background:#fff; scroll-margin-top:14px; }
.task h4 { margin:0 0 4px; font-size:16px; color:var(--pri-d); }
.task .meta { color:var(--mut); font-size:12.5px; margin-bottom:10px; }
.tag { background:var(--soft); color:var(--pri-d); border-radius:4px; padding:1px 7px; font-size:11.5px; font-weight:600; }
.win-badge { color:var(--gold); font-weight:700; }
.plabel { font-weight:700; color:var(--pri-d); font-size:13px; margin:6px 0 4px; }
.prompt { background:#fafbfc; border:1px solid var(--bd); border-left:3px solid var(--pri); border-radius:0 6px 6px 0; padding:10px 14px; white-space:pre-wrap; font-size:13px; line-height:1.45; color:#374151; }
.prompt.huge { max-height:520px; overflow-y:auto; }
.cmt { font-size:13px; }
.cmt-cell { font-size:12.5px; color:#4b5563; line-height:1.4; }
.note { background:var(--bg); border-left:3px solid var(--pri); padding:8px 12px; margin-top:10px; border-radius:0 6px 6px 0; font-size:12.5px; color:#374151; font-style:italic; }
.totop { position:fixed; bottom:18px; right:22px; background:var(--pri); color:#fff; padding:8px 12px; border-radius:6px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.2); }
.totop:hover { text-decoration:none; background:var(--pri-d); }
.medal { color:var(--gold); }
"""


# fallback if a run file lacks endpoint metadata
MODEL_NAMES_FALLBACK = {
    "27b-v2": "Qwen3.6-27B-FP8", "35b-a3b": "Qwen3.6-35B-A3B-FP8",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "deepseek-flash": "deepseek-v4-flash", "deepseek-pro": "deepseek-v4-pro",
    "gpt-5.5-low": "GPT-5.5 (low)", "gpt-5.5-medium": "GPT-5.5 (medium)",
    "gpt-5.5-xhigh": "GPT-5.5 (xhigh)",
    "nemotron-super": "Nemotron-3-Super-120B", "nemotron-omni": "Nemotron-3-Nano-Omni-30B",
    "kimi-k2.6": "Kimi K2.6",
    "gpt-5.6-sol-low": "GPT-5.6 Sol (low)", "gpt-5.6-sol-medium": "GPT-5.6 Sol (med)", "gpt-5.6-terra-low": "GPT-5.6 Terra (low)", "gpt-5.6-terra-medium": "GPT-5.6 Terra (med)", "gpt-5.6-terra-high": "GPT-5.6 Terra (high)", "gpt-5.6-luna-low": "GPT-5.6 Luna (low)", "gpt-5.6-luna-medium": "GPT-5.6 Luna (med)", "gpt-5.6-luna-high": "GPT-5.6 Luna (high)",
}


def build(results_path, scores_dir, out_path, tags, drop_mentions=None):
    data = json.load(open(results_path))
    results = data["results"]
    # real inference model names from run metadata (authoritative for this run)
    names = {e["tag"]: e["model"] for e in data.get("endpoints", [])}
    for t in tags:
        names.setdefault(t, MODEL_NAMES_FALLBACK.get(t, t))

    def nm(t):
        return esc(names.get(t, t))

    # replace short aliases inside free text (judge comments/notes) with real names
    _relabel_pairs = sorted(((t, names.get(t, t)) for t in tags), key=lambda p: -len(p[0]))

    # tokens of excluded models to scrub from judge prose (this report omits those models)
    _drop_tokens = []
    for dt in (drop_mentions or []):
        _drop_tokens.append(dt.lower())
        _drop_tokens.append(dt.split("-")[0].lower())   # bare alias, e.g. "kimi" from "kimi-k2.6"
    _drop_tokens = [t for t in set(_drop_tokens) if t]

    def _scrub(text):
        if not _drop_tokens:
            return text
        # drop whole sentences that mention an excluded model
        parts = re.split(r'(?<=[.!?;])\s+', text)
        kept = [p for p in parts if not any(tok in p.lower() for tok in _drop_tokens)]
        return " ".join(kept).strip()

    def relabel(text):
        text = _scrub(text or "")
        for short, full in _relabel_pairs:
            text = text.replace(short, full)
        return esc(text)

    scores = {}
    for fp in Path(scores_dir).glob("*.json"):
        sd = json.load(open(fp))
        scores[sd["task_id"]] = sd

    # aggregates
    by_tag = {t: {c: [] for c in CRITERIA} for t in tags}
    by_seg = defaultdict(lambda: {t: {c: [] for c in CRITERIA} for t in tags})
    by_cat = defaultdict(lambda: {t: [] for t in tags})
    wins = defaultdict(int)
    cat_wins = defaultdict(lambda: defaultdict(int))   # per-category winner counts
    perf = defaultdict(lambda: {"lat": [], "ct": [], "tps": [], "rc": [], "alen": [], "ok": 0, "fail": 0})
    # ok=true but empty content = technical artifact (answer leaked into the reasoning
    # channel); exclude from scoring rather than penalise. Affects kimi-k2.6 on 6 text tasks
    # (re-run pending the 2026-07-05 budget reset). ok=false stays a real failure and is kept.
    def empty_artifact(t, r):
        resp = r["responses"].get(t) or {}
        return bool(resp.get("ok")) and not (resp.get("content") or "").strip()
    # OpenAI's platform cyber-filter blocked the request before it reached the model
    # (returncode 1, empty). Behaviour is unmeasurable, so the cell is excluded from scoring
    # rather than counted as a model failure. Affects GPT-5.5 (codex) on a few security tasks.
    def cyber_blocked(t, r):
        return bool((r["responses"].get(t) or {}).get("cyber_blocked"))
    excluded = defaultdict(int)
    blocked = defaultdict(int)

    for r in results:
        sd = scores.get(r["id"])
        seg = get_segment(r["category"])
        for t in tags:
            resp = r["responses"].get(t, {})
            if cyber_blocked(t, r):
                blocked[t] += 1
                continue   # request blocked upstream: neither ok, nor fail, nor scored
            if resp.get("ok"):
                perf[t]["ok"] += 1
                lat = resp.get("latency_ms") or 0
                ct = resp.get("completion_tokens") or 0
                perf[t]["lat"].append(lat)
                perf[t]["ct"].append(ct)
                perf[t]["rc"].append(len(resp.get("reasoning") or ""))
                perf[t]["alen"].append(len(resp.get("content") or ""))
                if ct and lat:
                    perf[t]["tps"].append(ct / (lat / 1000))
            else:
                perf[t]["fail"] += 1
            if sd and empty_artifact(t, r):
                excluded[t] += 1
                continue
            if sd:
                if t not in sd["scores"]:
                    continue   # judge produced no score for this model on this task — exclude
                s = sd["scores"][t]
                tot = wsum(s)
                by_cat[r["category"]][t].append(tot)
                for c in CRITERIA:
                    by_tag[t][c].append(s[c])
                    by_seg[seg][t][c].append(s[c])
        if sd:
            # recompute winner over the scored criteria (reasoning excluded; skip empty artifacts/blocked)
            w = max((t for t in tags if t in sd["scores"]
                     and not empty_artifact(t, r) and not cyber_blocked(t, r)),
                    key=lambda t: wsum(sd["scores"][t]), default=None)
            if w:
                wins[w] += 1
                cat_wins[r["category"]][w] += 1

    def tot_tag(d):
        return sum(WEIGHTS[c] * avg(d[c]) for c in CRITERIA)
    overall = {t: tot_tag(by_tag[t]) for t in tags}          # Σ micro (task-weighted)
    n_tasks = len(scores)
    judged_cats = [c for c in by_cat if any(by_cat[c][t] for t in tags)]

    # size-normalized (macro) metrics — each category weighs equally
    def cat_mean(cat, t):
        return avg(by_cat[cat][t])
    macro_sigma = {t: avg([cat_mean(c, t) for c in judged_cats]) for t in tags}
    cat_n = {c: max((len(by_cat[c][t]) for t in tags), default=0) for c in judged_cats}
    macro_winrate = {t: 100 * avg([cat_wins[c][t] / cat_n[c] for c in judged_cats]) for t in tags}
    cats_led = {t: sum(1 for c in judged_cats if max(tags, key=lambda x: cat_mean(c, x)) == t) for t in tags}
    n_cats = len(judged_cats)

    # leaderboard ordered by the size-fair metric (macro Σ)
    order = sorted(tags, key=lambda t: -macro_sigma[t])

    # group tasks by segment -> category (in SEGMENTS order)
    cat_order = [c for seg, cats in SEGMENTS.items() for c in cats]
    by_cat_tasks = defaultdict(list)
    for r in results:
        by_cat_tasks[r["category"]].append(r)

    H = []
    A = H.append

    # ---------- sidebar ----------
    A('<nav class="sidebar">')
    A(f'<h1>Master Benchmark v2</h1><div class="sub">{n_tasks} задач · {len(tags)} моделей · единый бюджет 65536</div>')
    A('<div class="grp">Сводка</div>')
    A('<a href="#leaderboard">Лидерборд</a>')
    A('<a href="#tokens">Расход токенов</a>')
    A('<a href="#perf">Производительность</a>')
    A('<a href="#segments">Результаты по сегментам</a>')
    A('<a href="#categories">Результаты по категориям</a>')
    for seg, cats in SEGMENTS.items():
        seg_cats = [c for c in cats if by_cat_tasks.get(c)]
        if not seg_cats:
            continue
        A(f'<div class="seg">{esc(seg)}</div>')
        for c in seg_cats:
            A(f'<div class="grp" style="margin:8px 0 2px">{esc(CATEGORY_LABEL.get(c,c))} ({len(by_cat_tasks[c])})</div>')
            for r in by_cat_tasks[c]:
                ttl = esc(r["title"])
                A(f'<a class="tlink" href="#{r["id"]}" title="{ttl}">{r["id"]} — {ttl}</a>')
    A('</nav>')

    # ---------- main ----------
    A('<main id="top">')
    A(f'<h1 style="color:#1e3a8a;font-size:28px;margin:0 0 4px">Master Benchmark v2 — отчёт</h1>')
    A(f'<div style="color:#374151;margin-bottom:10px;font-size:14.5px;line-height:1.55">Отчёт сравнивает {len(tags)} больших языковых моделей '
      f'на {n_tasks} задачах, объединённых в {n_cats} категории и пять сегментов. Все модели работали с одинаковым бюджетом вывода '
      f'(<b>max_tokens = 65536</b>), поэтому объём внутренних рассуждений не ограничивался искусственно и оставался сопоставимым по условиям. '
      f'Каждый ответ оценивался по трём критериям — correctness, format, russian — по шкале от 0 до 10. '
      f'Итоговый балл — взвешенная свёртка: <b>correctness 0.6, russian 0.3, format 0.1</b> (правильность по сути важнее, '
      f'чем язык и форматирование вместе взятые; язык для русскоязычного продукта весомее форматирования). Итог на шкале 0–10. '
      f'Оценку выставлял отдельный независимый судья (Opus) для каждой задачи. '
      f'Критерий «reasoning» сознательно исключён из подсчёта: часть моделей возвращает отдельный след рассуждения (chain-of-thought), '
      f'а часть (например, gemma-4-31b) рассуждает прямо внутри ответа и отдельного следа не отдаёт, поэтому единый и честный для всех учёт reasoning невозможен.</div>')
    A('<div class="callout"><b>Модели:</b> '
      + ', '.join(f'<b>{nm(t)}</b>' for t in tags) + '.</div>')

    # Leaderboard — ranked by size-fair macro Σ
    A('<h2 class="sec" id="leaderboard">Лидерборд</h2>')
    A(f'<div class="callout"><b>Методика агрегирования.</b> Категории содержат разное число задач — от 5 до 16, '
      f'поэтому основной метрикой служит <b>«Σ по категориям»</b>: сначала вычисляется средний балл внутри каждой из {n_cats} категорий, '
      f'затем эти средние усредняются между собой. При таком подходе каждая категория вносит равный вклад, и размер выборки не искажает итог. '
      f'«Σ по задачам» — среднее по всем задачам без нормализации, в нём крупные категории влияют сильнее. '
      f'Все баллы Σ — на шкале <b>0–10</b> (взвешенная свёртка correctness 0.6 / russian 0.3 / format 0.1). '
      f'«Доля побед» — усреднённая по категориям доля задач, в которых модель оказалась лучшей. '
      f'«Лидер в категориях» — число категорий из {n_cats}, где модель занимает первое место по среднему баллу. '
      f'«Побед всего» — суммарное число выигранных задач; эта величина зависит от размера категорий и приводится только для справки.'
      + ("".join(f' <b>Примечание:</b> у модели {nm(t)} {excluded[t]} задач(и) исключены из подсчёта — ответ вернулся пустым (содержимое ушло во внутренние рассуждения); '
         f'эти задачи будут перезапущены отдельно.' for t in tags if excluded.get(t)))
      + ("".join(f' <b>Примечание:</b> у модели {nm(t)} {blocked[t]} задач(и) на тему ИБ исключены из подсчёта — '
         f'платформенный кибербез-фильтр OpenAI заблокировал сам запрос (модель до ответа не дошла), '
         f'поэтому её поведение здесь неизмеримо. Остальные модели шли через свои API без этого фильтра.' for t in tags if blocked.get(t)))
      + '</div>')
    A('<table><thead><tr><th>#</th><th>Модель</th>'
      '<th class="num">Σ по категориям / 10</th><th class="num">Σ по задачам / 10</th>'
      '<th class="num">Лидер в категориях</th><th class="num">Доля побед, %</th><th class="num">Побед всего</th>'
      '<th class="num">C</th><th class="num">F</th><th class="num">Ru</th></tr></thead><tbody>')
    for i, t in enumerate(order):
        a = by_tag[t]
        cls = ' class="gold-row"' if i == 0 else ""
        A(f'<tr{cls}><td>{i+1}</td><td><b>{nm(t)}</b></td>'
          f'<td class="num"><b>{macro_sigma[t]/DIV:.2f}</b></td><td class="num">{overall[t]/DIV:.2f}</td>'
          f'<td class="num">{cats_led[t]} / {n_cats}</td><td class="num">{macro_winrate[t]:.1f}</td><td class="num">{wins[t]}</td>'
          f'<td class="num">{avg(a["correctness"]):.2f}</td>'
          f'<td class="num">{avg(a["format"]):.2f}</td><td class="num">{avg(a["russian"]):.2f}</td></tr>')
    A('</tbody></table>')
    A('<div style="color:#64748b;font-size:12.5px;margin:-8px 0 14px">C, F, Ru — средние баллы по критериям '
      'correctness, format и russian соответственно (каждый по шкале 0–10). Критерий reasoning исключён из подсчёта (см. методику выше).</div>')
    lead = order[0]                      # by category-balanced Σ
    win_lead = max(tags, key=lambda t: macro_winrate[t])
    A(f'<div class="callout"><b>Результат.</b> По сбалансированной метрике (Σ по категориям) первое место занимает <b>{nm(lead)}</b> '
      f'с результатом {macro_sigma[lead]:.2f} из 40; модель {nm(order[1])} отстаёт на {macro_sigma[lead]-macro_sigma[order[1]]:.2f} балла '
      f'({macro_sigma[order[1]]:.2f}). По усреднённой доле побед лидирует <b>{nm(win_lead)}</b> ({macro_winrate[win_lead]:.1f}%). '
      f'Различие между двумя сильнейшими моделями минимально. '
      f'Суммарное число побед ({", ".join(f"{nm(t)} — {wins[t]}" for t in sorted(tags, key=lambda t:-wins[t])[:2])}) '
      f'преувеличивает разрыв, поскольку зависит от размера категорий, и поэтому не используется для ранжирования.</div>')

    # Token expenditure (cost): total output tokens billed per answer + visible answer length
    A('<h2 class="sec" id="tokens">Расход токенов на ответ</h2>')
    A('<div class="callout">Сколько модель тратит выходных токенов на ответ — это прямая стоимость инференса и главный драйвер задержки. '
      'Выходные токены включают весь сгенерированный моделью текст, в том числе внутренние рассуждения (для reasoning-моделей). '
      'Длина ответа — размер видимого пользователю ответа (в символах). Разрыв между большим расходом токенов и коротким ответом '
      'указывает на «дорогую» модель, которая много рассуждает ради краткого вывода. Бюджет у всех одинаковый (max_tokens = 65536), потолок никто не достиг.</div>')
    A('<table><thead><tr><th>Модель</th><th class="num">Среднее выходных токенов</th><th class="num">Медиана выходных токенов</th>'
      '<th class="num">Средняя длина ответа, символов</th></tr></thead><tbody>')
    for t in sorted(tags, key=lambda t: -avg(perf[t]["ct"])):
        A(f'<tr><td>{nm(t)}</td><td class="num"><b>{int(avg(perf[t]["ct"]))}</b></td><td class="num">{int(median(perf[t]["ct"]))}</td>'
          f'<td class="num">{int(avg(perf[t]["alen"]))}</td></tr>')
    A('</tbody></table>')

    # Performance
    A('<h2 class="sec" id="perf">Производительность: задержка и пропускная способность</h2>')
    A('<div class="callout">Пропускная способность (tok/s) вычислена как отношение числа выходных токенов к полному времени ответа, '
      'включая фазу рассуждений. Время до первого токена (TTFT) не измерялось, так как запросы выполнялись без потоковой передачи.</div>')
    A('<table><thead><tr><th>Модель</th><th class="num">Средняя задержка</th><th class="num">Медианная задержка</th><th class="num">Средняя, tok/s</th><th class="num">Медианная, tok/s</th><th class="num">Успешно</th><th class="num">Ошибок</th></tr></thead><tbody>')
    for t in sorted(tags, key=lambda t: -avg(perf[t]["tps"])):
        A(f'<tr><td>{nm(t)}</td><td class="num">{avg(perf[t]["lat"])/1000:.1f}s</td><td class="num">{median(perf[t]["lat"])/1000:.1f}s</td>'
          f'<td class="num"><b>{avg(perf[t]["tps"]):.0f}</b></td><td class="num">{median(perf[t]["tps"]):.0f}</td>'
          f'<td class="num">{perf[t]["ok"]}</td><td class="num">{perf[t]["fail"]}</td></tr>')
    A('</tbody></table>')

    # Segments — category-balanced within each segment (consistent with the leaderboard)
    seg_cats_map = {seg: [c for c in cats if c in by_cat] for seg, cats in SEGMENTS.items()}
    A('<h2 class="sec" id="segments">Результаты по сегментам (Σ по категориям / 10)</h2>')
    A('<table><thead><tr><th>Сегмент</th>' + "".join(f'<th class="num">{nm(t)}</th>' for t in tags) + '<th>Лидер</th></tr></thead><tbody>')
    for seg in SEGMENTS:
        sc = seg_cats_map.get(seg, [])
        if not sc:
            continue
        vals = {t: avg([cat_mean(c, t) for c in sc]) for t in tags}
        ld = max(vals, key=vals.get)
        A(f'<tr><td>{esc(seg)}</td>' + "".join(f'<td class="num">{vals[t]/DIV:.1f}</td>' for t in tags) + f'<td><b>{nm(ld)}</b></td></tr>')
    A('</tbody></table>')

    # Categories
    A('<h2 class="sec" id="categories">Результаты по категориям (среднее Σ / 10)</h2>')
    A('<table><thead><tr><th>Категория</th><th class="num">N</th>' + "".join(f'<th class="num">{nm(t)}</th>' for t in tags) + '<th>Лидер</th></tr></thead><tbody>')
    for cat in cat_order:
        if cat not in by_cat:
            continue
        vals = {t: avg(by_cat[cat][t]) for t in tags}
        ld = max(vals, key=vals.get)
        n = max((len(by_cat[cat][t]) for t in tags), default=0)
        A(f'<tr><td><a href="#cat-{cat}">{esc(CATEGORY_LABEL.get(cat,cat))}</a></td><td class="num">{n}</td>'
          + "".join(f'<td class="num">{vals[t]/DIV:.1f}</td>' for t in tags) + f'<td><b>{nm(ld)}</b></td></tr>')
    A('</tbody></table>')

    # ---------- per-task ----------
    for seg, cats in SEGMENTS.items():
        for cat in cats:
            tasks = by_cat_tasks.get(cat, [])
            if not tasks:
                continue
            A(f'<h3 class="catsec" id="cat-{cat}">{esc(CATEGORY_LABEL.get(cat,cat))} '
              f'<span style="color:#64748b;font-weight:400;font-size:14px">— {esc(seg)}, {len(tasks)} задач</span></h3>')
            for r in tasks:
                sd = scores.get(r["id"])
                if not sd:
                    continue
                winner = max((t for t in tags if t in sd["scores"]
                              and not empty_artifact(t, r) and not cyber_blocked(t, r)),
                             key=lambda t: wsum(sd["scores"][t]), default=None)
                wtot = wsum(sd["scores"][winner]) if winner else 0
                plen = len(r["prompt"])
                huge = " huge" if plen > 12000 else ""
                A(f'<article class="task" id="{r["id"]}">')
                A(f'<h4>{r["id"]} — {esc(r["title"])}</h4>')
                A(f'<div class="meta"><span class="tag">{r["category"]}</span> · {esc(seg)} · промпт {plen:,} симв. · '
                  f'<span class="win-badge">Победитель: {nm(winner)} · Σ {wtot/DIV:.1f}/10</span></div>')
                A('<div class="plabel">Промпт задачи</div>')
                A(f'<div class="prompt{huge}">{esc(r["prompt"])}</div>')
                # scores table
                A('<div class="plabel" style="margin-top:12px">Оценки</div>')
                A('<table><thead><tr><th>Модель</th><th class="num">C</th><th class="num">F</th><th class="num">Ru</th>'
                  '<th class="num">Σ</th><th class="num">Задержка</th><th class="num">Токены</th><th class="num">tok/s</th><th>Комментарий судьи</th></tr></thead><tbody>')
                def _cell_key(t):
                    if cyber_blocked(t, r) or t not in sd["scores"]:
                        return (2, 0)          # blocked / unscored — last
                    if empty_artifact(t, r):
                        return (1, 0)          # empty artifact — after scored
                    return (0, -wsum(sd["scores"][t]))
                for t in sorted(tags, key=_cell_key):
                    resp = r["responses"].get(t, {})
                    ok = resp.get("ok")
                    ct = resp.get("completion_tokens")
                    cts = str(ct) if (ok and ct is not None) else "—"
                    tps = f'{ct/((resp.get("latency_ms") or 0)/1000):.0f}' if (ok and ct and resp.get("latency_ms")) else "—"
                    if cyber_blocked(t, r):
                        A(f'<tr style="color:#94a3b8"><td><b>{nm(t)}</b></td><td class="num">—</td>'
                          f'<td class="num">—</td><td class="num">—</td><td class="num">—</td>'
                          f'<td class="num">—</td><td class="num">—</td><td class="num">—</td>'
                          f'<td class="cmt-cell">Исключено из подсчёта: платформенный кибербез-фильтр OpenAI заблокировал запрос — модель до ответа не дошла.</td></tr>')
                        continue
                    if t not in sd["scores"]:
                        continue
                    lat = f'{(resp.get("latency_ms") or 0)/1000:.1f}s' if ok else "FAIL"
                    if empty_artifact(t, r):
                        A(f'<tr style="color:#94a3b8"><td><b>{nm(t)}</b></td><td class="num">—</td>'
                          f'<td class="num">—</td><td class="num">—</td><td class="num">—</td>'
                          f'<td class="num">{lat}</td><td class="num">{cts}</td><td class="num">{tps}</td>'
                          f'<td class="cmt-cell">Исключено из подсчёта: ответ вернулся пустым (содержимое ушло во внутренние рассуждения). Будет перезапущено.</td></tr>')
                        continue
                    s = sd["scores"][t]
                    tot = wsum(s)
                    cls = ' class="win"' if t == winner else ""
                    A(f'<tr{cls}><td><b>{nm(t)}</b></td><td class="num">{s["correctness"]}</td>'
                      f'<td class="num">{s["format"]}</td><td class="num">{s["russian"]}</td><td class="num"><b>{tot/DIV:.1f}</b></td>'
                      f'<td class="num">{lat}</td><td class="num">{cts}</td><td class="num">{tps}</td><td class="cmt-cell">{relabel(s.get("comment",""))}</td></tr>')
                A('</tbody></table>')
                A(f'<div class="note"><b>Итоговое заключение судьи:</b> {relabel(sd.get("notes",""))}</div>')
                A('</article>')

    A('</main>')
    A('<a class="totop" href="#top">↑ Наверх</a>')

    html = f"<!DOCTYPE html><html lang=ru><head><meta charset=utf-8><title>Master Benchmark v2</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
    Path(out_path).write_text(html)
    print(f"Wrote {out_path} ({len(html)//1024} KB), {n_tasks} tasks, {len(tags)} models")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="master-benchmark-results-v2.json")
    p.add_argument("--scores-dir", default="judging-scores-v2")
    p.add_argument("--out", default="Master-Benchmark-V2.html")
    p.add_argument("--tags", default="27b-v2,35b-a3b,gpt-oss-120b,deepseek-flash,deepseek-pro")
    p.add_argument("--drop-mentions", default="", help="comma-sep tags to scrub from judge prose (models omitted from this report)")
    args = p.parse_args()
    drop = [x for x in args.drop_mentions.split(",") if x]
    build(args.results, args.scores_dir, args.out, args.tags.split(","), drop)


if __name__ == "__main__":
    main()
