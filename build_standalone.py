"""Premium standalone report generator (dark-hero dashboard) — faithful to the
'Master Benchmark v2 standalone' structure: .wrap sections, hero+rankcard(+rc-foot),
lead-finding + finding-grid, criteria-legend, chart barcharts with bc-meta chips,
collapsible detail tables, segment heatmap, per-task cards. One self-contained HTML.

Usage: python3 build_standalone.py text|vl --results ... --scores-dir ... --out ... --tags ... --title ... [--subtitle ...]
"""
import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from category_focus_master import SEGMENTS
from build_html_report_v2 import CATEGORY_LABEL, MODEL_NAMES_FALLBACK, get_segment, esc
from build_vl_report import CAP_LABEL, VERT_LABEL, thumb_data_url

THEME = Path("_standalone_theme.css").read_text()

# Fixes on top of the borrowed theme: it was authored for 12 models, so wide tables were
# clipped (.table-scroll used overflow:hidden) and long unbroken tokens in prompts spilled
# out of their container. The theme also styles images with cursor:zoom-in and ships
# .lightbox rules, but the markup/JS for it has to be emitted here.
THEME_FIX = """
/* --- wide-content fixes (20 models) --- */
.table-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;}
.heat-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;padding-bottom:8px;}
.prompt{overflow-wrap:anywhere;word-break:break-word;}
.cmt-cell{overflow-wrap:anywhere;width:auto;min-width:220px;}
article.task table{min-width:660px;}
/* VL cards have a narrow text column (page width minus the 290px image), so their score
   tables must FIT instead of scrolling: fixed layout, wrapping model names, tight numerics. */
article.task.vl .scorewrap{grid-column:1/-1;min-width:0;}
article.task.vl table{min-width:0;width:100%;table-layout:fixed;}
article.task.vl table th:first-child,article.task.vl table td:first-child{width:23%;overflow-wrap:anywhere;}
article.task.vl table th.num,article.task.vl table td.num{width:8.5%;padding-left:4px;padding-right:6px;}
article.task.vl .cmt-cell{min-width:0;width:auto;}
article.task.vl .table-scroll{overflow-x:visible;}
.bc-name{overflow-wrap:anywhere;}
main,.wrap,article.task{max-width:100%;}
@media (min-width:1500px){.wrap{max-width:1320px;}}
/* VL card: never let the text column force the page wider */
article.task.vl{grid-template-columns:290px minmax(0,1fr);}
article.task.vl .body{min-width:0;}
@media (max-width:900px){article.task.vl{grid-template-columns:1fr;}.task .imgcol{position:static;}}
/* mobile perf: with 180-300 per-task cards (tens of thousands of table cells) the full DOM
   overwhelms phone browsers (page appears not to load). content-visibility defers rendering
   of off-screen cards; contain-intrinsic-size gives the scrollbar an estimate so layout is stable. */
article.task{content-visibility:auto;contain-intrinsic-size:0 720px;}
"""
TEXT_W = {"correctness": 0.6, "format": 0.1, "russian": 0.3}
VL_W = {"correctness": 0.8, "format": 0.1, "russian": 0.1}
CRIT = ["correctness", "format", "russian"]


def wsum(s, W):
    return sum(W[c] * s[c] for c in CRIT)


def avg(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def med(xs):
    return statistics.median(xs) if xs else 0.0


# completion_tokens is the total output (answer + reasoning) for every provider EXCEPT the
# neuraldeep-streamed models below, where it counts only the visible answer and the reasoning
# tokens are reported separately. For a consistent "output tokens per task" cost we add them back.
CONTENT_ONLY_CT = set()  # qwen3.8 VL (EmpirioLabs) reports completion_tokens as total; text-qwen not in a report


def is_codex(tag):
    # GPT-5.5/5.6 run via the codex CLI: it reports only a stdout "tokens used" count (whole-turn,
    # locale-formatted, unreliable to parse) and hides the reasoning trace, and its latency includes
    # per-task subprocess overhead. So codex tokens aren't comparable and its TPS is meaningless ->
    # excluded from the token and performance metrics (still ranked on scores).
    return tag.startswith("gpt-5.5") or tag.startswith("gpt-5.6")


def out_toks(tag, resp):
    if is_codex(tag):
        return None
    # total turn tokens, comparable across HTTP hosts: completion(output)+prompt(input).
    ct = resp.get("completion_tokens") or 0
    ct += resp.get("prompt_tokens") or 0
    if tag in CONTENT_ONLY_CT:
        ct += resp.get("reasoning_tokens") or 0
    return ct


def compute(results_path, scores_dir, tags, W, vl, exclude=(), rename=None):
    data = json.load(open(results_path))
    exclude = set(exclude)
    results = [r for r in data["results"] if r["id"] not in exclude]
    names = {e["tag"]: e["model"] for e in data.get("endpoints", [])}
    for t in tags:
        names.setdefault(t, MODEL_NAMES_FALLBACK.get(t, t))
    for t, nm in (rename or {}).items():   # per-report display-name overrides (e.g. effort suffix)
        names[t] = nm
    cat_key = "capability" if vl else "category"
    scores = {}
    for fp in Path(scores_dir).glob("*.json"):
        try:
            sd = json.load(open(fp))
            if sd["task_id"] in exclude:
                continue
            scores[sd["task_id"]] = sd
        except Exception:
            pass

    def blk(t, r):
        return bool((r["responses"].get(t) or {}).get("cyber_blocked"))

    def emp(t, r):
        rr = r["responses"].get(t) or {}
        return bool(rr.get("ok")) and not (rr.get("content") or "").strip()

    by_cat = defaultdict(lambda: {t: [] for t in tags})
    by_seg = defaultdict(lambda: {t: {c: [] for c in CRIT} for t in tags})
    by_tag = {t: {c: [] for c in CRIT} for t in tags}
    cat_wins = defaultdict(lambda: defaultdict(int))
    wins = defaultdict(int)
    perf = defaultdict(lambda: {"lat": [], "ct": [], "tps": [], "alen": [], "ok": 0})
    by_cat_tok = defaultdict(lambda: defaultdict(list))   # cat -> tag -> [output tokens/task]
    blocked = defaultdict(int)

    for r in results:
        sd = scores.get(r["id"])
        seg = r.get("vertical", "?") if vl else get_segment(r.get(cat_key, "?"))
        for t in tags:
            resp = r["responses"].get(t, {})
            if blk(t, r):
                blocked[t] += 1; continue
            if resp.get("ok"):
                perf[t]["ok"] += 1
                ct = out_toks(t, resp)
                if ct is not None:   # codex excluded from token/perf metrics (out_toks -> None)
                    lat = resp.get("latency_ms") or 0
                    perf[t]["lat"].append(lat); perf[t]["ct"].append(ct)
                    perf[t]["alen"].append(len(resp.get("content") or ""))
                    by_cat_tok[r.get(cat_key, "?")][t].append(ct)
                    if ct and lat:
                        perf[t]["tps"].append(ct / (lat / 1000))
            if sd and emp(t, r):
                continue
            if sd and t in sd["scores"]:
                s = sd["scores"][t]
                by_cat[r.get(cat_key, "?")][t].append(wsum(s, W))
                for c in CRIT:
                    by_tag[t][c].append(s[c]); by_seg[seg][t][c].append(s[c])
        if sd:
            w = max((t for t in tags if t in sd["scores"] and not emp(t, r) and not blk(t, r)),
                    key=lambda t: wsum(sd["scores"][t], W), default=None)
            if w:
                wins[w] += 1; cat_wins[r.get(cat_key, "?")][w] += 1

    judged_cats = [c for c in by_cat if any(by_cat[c][t] for t in tags)]
    micro = {t: sum(W[c] * avg(by_tag[t][c]) for c in CRIT) for t in tags}
    macro = {t: avg([avg(by_cat[c][t]) for c in judged_cats if by_cat[c][t]]) for t in tags}
    cat_n = {c: max((len(by_cat[c][t]) for t in tags), default=0) for c in judged_cats}
    winrate = {t: 100 * avg([cat_wins[c][t] / cat_n[c] for c in judged_cats if cat_n[c]]) for t in tags}
    cats_led = {t: sum(1 for c in judged_cats if max(tags, key=lambda x: avg(by_cat[c][x])) == t) for t in tags}
    crit_avg = {t: {c: avg(by_tag[t][c]) for c in CRIT} for t in tags}
    return dict(results=results, scores=scores, names=names, tags=tags, W=W, vl=vl, cat_key=cat_key,
                macro=macro, micro=micro, winrate=winrate, cats_led=cats_led, wins=wins,
                by_cat=by_cat, by_seg=by_seg, judged_cats=judged_cats, crit_avg=crit_avg,
                perf=perf, blocked=blocked, n_cats=len(judged_cats), cat_wins=cat_wins,
                by_cat_tok=by_cat_tok, blk=blk, emp=emp)


def render(D, out_path, title, subtitle):
    tags, names = D["tags"], D["names"]
    macro, micro, ca = D["macro"], D["micro"], D["crit_avg"]
    order = sorted(tags, key=lambda t: -macro[t])
    n_tasks = len(D["scores"]); vl = D["vl"]; W = D["W"]; cat_key = D["cat_key"]
    H = []; A = H.append

    def nm(t):
        return esc(names.get(t, t))

    LBL = CAP_LABEL if vl else CATEGORY_LABEL

    def clbl(c):
        return LBL.get(c, c)

    def seglbl(sg):
        return (VERT_LABEL.get(sg, sg) if vl else sg.split("(")[0].strip())

    by_cat_tasks = defaultdict(list)
    for r in D["results"]:
        by_cat_tasks[r.get(cat_key, "?")].append(r)
    cat_order = ([c for seg, cats in SEGMENTS.items() for c in cats if c in by_cat_tasks]
                 if not vl else sorted(by_cat_tasks))

    # ---- sidebar: collapsible two-level nav (segment > category > tasks), as the theme expects ----
    A(f'<aside class="sidebar"><h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div>')
    for sid, lbl in [("exec", "Обзор"), ("findings", "Ключевые выводы"), ("leaderboard", "Лидерборд"),
                     ("tokens", "Расход токенов"), ("perf", "Производительность"),
                     ("segments", "Отрасли" if vl else "Сегменты"),
                     ("categories", "Способности" if vl else "Категории")]:
        A(f'<a class="nav-link" data-target="{sid}" href="#{sid}">{lbl}</a>')

    def cat_block(c):
        tasks = by_cat_tasks[c]
        A(f'<div class="nav-cat open"><button class="nav-cat-h" data-target="cat-{c}">'
          f'<span class="chev">▶</span><span class="cat-name">{esc(clbl(c))}</span>'
          f'<span class="cat-n">{len(tasks)}</span></button><div class="nav-cat-b">')
        for r in tasks:
            ttl = esc(r.get("title", ""))
            A(f'<a class="tlink" data-target="{r["id"]}" href="#{r["id"]}" title="{ttl}">'
              f'<span class="tid">{r["id"]}</span> {ttl}</a>')
        A('</div></div>')

    if vl:
        # VL has one grouping level (capability); keep it open so tasks are one click away
        A(f'<div class="nav-seg open"><button class="nav-seg-h"><span class="chev">▶</span>Задачи'
          f'<span class="seg-n">{n_tasks}</span></button><div class="nav-seg-b">')
        for c in cat_order:
            cat_block(c)
        A('</div></div>')
    else:
        for seg, cats in SEGMENTS.items():
            present = [c for c in cats if c in by_cat_tasks]
            if not present:
                continue
            cnt = sum(len(by_cat_tasks[c]) for c in present)
            A(f'<div class="nav-seg open"><button class="nav-seg-h"><span class="chev">▶</span>'
              f'{esc(seglbl(seg))}<span class="seg-n">{cnt}</span></button><div class="nav-seg-b">')
            for c in present:
                cat_block(c)
            A('</div></div>')
    A('</aside>')

    # ---- hero ----
    A('<main id="top"><header class="hero" id="exec"><div class="hero-grid"><div class="hero-main">')
    A(f'<div class="hero-eyebrow">Сравнительная оценка · {len(tags)} моделей</div>')
    words = title.split("—")[0].strip()
    A(f'<h1>{esc(words)} <span class="accent">— бенчмарк</span></h1>')
    A(f'<p class="hero-lead">{len(tags)} моделей в равных условиях: <b>{n_tasks} задач</b> и независимый судья '
      f'Opus (единый метод). Один средний балл (0–10) по трём критериям — корректность, формат, русский — решает, кто лучше.</p>')
    A('<div class="hero-kpis">')
    for val, lab in [(n_tasks, "задач"), (D["n_cats"], "категорий"), (len(tags), "моделей"),
                     (32768 if vl else 65536, "бюджет, ток.")]:
        A(f'<div class="hero-kpi"><div class="v"><span data-count="{val}">{val:,}</span></div><div class="l">{lab}</div></div>')
    A('</div></div>')
    # rankcard — all models + rc-foot
    A('<div class="hero-aside"><div class="rankcard"><div class="rc-head"><div class="rc-title">Лидерборд</div>'
      '<div class="rc-sub">средний балл · 0–10</div></div>')
    for i, t in enumerate(order, 1):
        champ = " champ" if i == 1 else ""
        w = round(100 * macro[t] / 10)
        A(f'<div class="rank-row{champ}"><span class="rk">{i}</span><div class="rm"><div class="nm">{nm(t)}</div>'
          f'<div class="track"><span class="fill" data-w="{w}" style="width:{w}%"></span></div></div>'
          f'<span class="rv">{macro[t]:.2f}</span></div>')
    A(f'<div class="rc-foot"><span class="dot"></span>Чемпион — {nm(order[0])} · '
      f'{D["cats_led"][order[0]]} / {D["n_cats"]} категорий</div>')
    A('</div></div></div></header>')

    # ---- findings ----
    A('<div class="wrap findings-sec"><h2 class="sec" id="findings">Ключевые выводы</h2>')
    ch = order[0]
    ru_lead = max(tags, key=lambda t: ca[t]["russian"])
    A('<div class="lead-finding"><div class="badge">1</div><div class="lf-body">')
    A(f'<div class="lf-k">Лидер — {nm(ch)}</div>')
    lead_txt = (f'<b>{nm(ch)}</b> возглавляет таблицу с <b>{macro[ch]:.2f} / 10</b> и берёт '
                f'<b>{D["cats_led"][ch]} / {D["n_cats"]}</b> категорий.')
    if len(order) > 2:
        lead_txt += (f' Следом — <b>{nm(order[1])}</b> ({macro[order[1]]:.2f}) и <b>{nm(order[2])}</b> '
                     f'({macro[order[2]]:.2f}).')
    A(f'<div class="lf-t">{lead_txt}</div></div></div>')
    A('<div class="finding-grid">')
    mid = order[3:6]
    if mid:
        A(f'<div class="finding"><h4><span class="ic"></span>Плотная середина</h4><p>' +
          ", ".join(f'<b>{nm(t)}</b> ({macro[t]:.2f})' for t in mid) +
          f' — разброс {macro[mid[0]]-macro[mid[-1]]:.2f} балла; выбор зависит от задачи.</p></div>')
    A(f'<div class="finding"><h4><span class="ic"></span>Аутсайдеры</h4><p>Замыкают поле '
      f'<b>{nm(order[-1])}</b> ({macro[order[-1]]:.2f}) и <b>{nm(order[-2])}</b> ({macro[order[-2]]:.2f}).</p></div>')
    A(f'<div class="finding"><h4><span class="ic"></span>Русский язык</h4><p>Лучший русский у '
      f'<b>{nm(ru_lead)}</b> ({ca[ru_lead]["russian"]:.2f}). Критерий russian весит '
      f'{W["russian"]:.0%} в итоговом Σ.</p></div>')
    blk_note = [f"{nm(t)}: {D['blocked'][t]}" for t in tags if D["blocked"].get(t)]
    if blk_note:
        A(f'<div class="finding"><h4><span class="ic"></span>Кибербез-блоки</h4><p>Фильтр OpenAI заблокировал '
          f'(исключено из подсчёта): {"; ".join(blk_note)}.</p></div>')
    A('</div></div>')

    # ---- barchart helper ----
    def barsection(sec_id, h2, callout, rows, legend=""):
        # rows: list of (label_html, value_float, bar_pct, chips_html) sorted
        A(f'<div class="wrap"><h2 class="sec" id="{sec_id}">{h2}</h2>')
        if callout:
            A(f'<div class="callout">{callout}</div>')
        if legend:
            A(legend)
        A('<div class="chart"><div class="chart-cap"><span class="ct">' + h2 +
          '</span><span class="cs">0–10 · нормировано</span></div><div class="barchart">')
        for i, (lab, val, pct, chips, disp) in enumerate(rows):
            champ = " champ" if i == 0 else ""
            A(f'<div class="bc-row{champ}"><div class="bc-name">{lab}</div>'
              f'<div class="bc-track"><span class="bc-fill" data-w="{pct}" style="width:{pct}%"></span></div>'
              f'<div class="bc-val">{disp}</div>{chips}</div>')
        A('</div></div>')

    # ---- leaderboard ----
    legend = ('<div class="criteria-legend"><span class="cl-title">Критерии судьи · 0–10</span>'
              '<span class="cl-item"><b>C</b> correctness — корректность</span>'
              '<span class="cl-item"><b>F</b> format — формат</span>'
              '<span class="cl-item"><b>Ru</b> russian — русский</span>'
              '<span class="cl-item cl-note">«Среднее» — взвешенная свёртка этих трёх</span></div>')
    rows = []
    for t in order:
        chips = (f'<div class="bc-meta"><span class="bc-chip">за задачу <b>{micro[t]:.2f}</b></span>'
                 f'<span class="bc-chip">лидер в кат. <b>{D["cats_led"][t]} / {D["n_cats"]}</b></span>'
                 f'<span class="bc-chip">доля побед <b>{D["winrate"][t]:.1f}%</b></span></div>')
        rows.append((nm(t), macro[t], round(100 * macro[t] / 10), chips, f'{macro[t]:.2f}<small>/10</small>'))
    barsection("leaderboard", "Лидерборд",
               "Средний балл по трём критериям, нормированный по категориям (крупные категории не перевешивают).",
               rows, legend)
    # collapsible full table
    A('<details class="tbl-d"><summary>Полная таблица баллов</summary><div class="table-scroll"><table><thead><tr>'
      '<th>#</th><th>Модель</th><th class="num">Средний балл</th><th class="num">За задачу</th>'
      '<th class="num">Лидер в кат.</th><th class="num">Доля побед</th><th class="num">C</th>'
      '<th class="num">F</th><th class="num">Ru</th></tr></thead><tbody>')
    for i, t in enumerate(order, 1):
        cls = ' class="gold-row"' if i == 1 else ""
        A(f'<tr{cls}><td class="num">{i}</td><td><b>{nm(t)}</b></td><td class="num">{macro[t]:.2f}</td>'
          f'<td class="num">{micro[t]:.2f}</td><td class="num">{D["cats_led"][t]} / {D["n_cats"]}</td>'
          f'<td class="num">{D["winrate"][t]:.1f}%</td><td class="num">{ca[t]["correctness"]:.1f}</td>'
          f'<td class="num">{ca[t]["format"]:.1f}</td><td class="num">{ca[t]["russian"]:.1f}</td></tr>')
    A('</tbody></table></div></details></div>')

    # ---- tokens ----
    tok_order = sorted([t for t in tags if D["perf"][t]["ct"]], key=lambda t: -med(D["perf"][t]["ct"]))
    mx = max((med(D["perf"][t]["ct"]) for t in tok_order), default=1) or 1
    rows = [(nm(t), med(D["perf"][t]["ct"]), round(100 * med(D["perf"][t]["ct"]) / mx),
             f'<div class="bc-meta"><span class="bc-chip">символов <b>{med(D["perf"][t]["alen"]):,.0f}</b></span></div>',
             f'{med(D["perf"][t]["ct"]):,.0f}') for t in tok_order]
    barsection("tokens", "Расход токенов на задачу",
               "Медиана токенов хода (вход + рассуждение + ответ). GPT-5.x через codex исключены — codex отдаёт лишь суммарный счётчик и прячет трейс, токены несопоставимы.", rows)
    A('<details class="tbl-d"><summary>Таблица токенов</summary><div class="table-scroll"><table><thead><tr>'
      '<th>Модель</th><th class="num">Медиана</th><th class="num">Среднее</th>'
      '<th class="num">символов (медиана)</th></tr></thead><tbody>')
    for t in tok_order:
        p = D["perf"][t]
        A(f'<tr><td><b>{nm(t)}</b></td><td class="num">{med(p["ct"]):,.0f}</td>'
          f'<td class="num">{avg(p["ct"]):,.0f}</td><td class="num">{med(p["alen"]):,.0f}</td></tr>')
    A('</tbody></table></div></details></div>')

    # ---- performance ----
    perf_order = sorted([t for t in tags if D["perf"][t]["tps"]], key=lambda t: -med(D["perf"][t]["tps"]))
    mx = max((med(D["perf"][t]["tps"]) for t in perf_order), default=1) or 1
    rows = [(nm(t), med(D["perf"][t]["tps"]), round(100 * med(D["perf"][t]["tps"]) / mx),
             f'<div class="bc-meta"><span class="bc-chip">задержка <b>{med(D["perf"][t]["lat"])/1000:.1f}с</b></span></div>',
             f'{med(D["perf"][t]["tps"]):.0f}<small> tok/s</small>') for t in perf_order]
    barsection("perf", "Производительность",
               "Скорость генерации (tok/s, медиана) и типовая задержка ответа.", rows)

    # ---- segment heatmap ----
    def heat_color(s):
        d = 9.2 - s
        R = min(238, max(12, 20 + 37.4 * d)); G = min(238, max(100, 112 + 24.2 * d)); B = min(238, max(92, 103 + 24.7 * d))
        return f"rgb({R:.0f},{G:.0f},{B:.0f})", ("#fff" if R < 150 else "var(--ink)")
    seg_order = [s for s in SEGMENTS if s in D["by_seg"]] + [s for s in D["by_seg"] if s not in SEGMENTS]
    A('<div class="wrap"><h2 class="sec" id="segments">Результаты по сегментам</h2>')
    A('<div class="callout">Средний балл по группам задач × модели. Теплее = выше; золотая рамка — лидер сегмента.</div>')
    A('<div class="chart"><div class="chart-cap"><span class="ct">Сегмент × модель</span><span class="cs">0–10</span></div>')
    A('<div class="heat-scroll">')
    A(f'<div class="heat" style="min-width:{150 + len(order) * 58}px;'
      f'grid-template-columns:minmax(140px,1.3fr) repeat({len(order)},minmax(52px,1fr))"><div></div>')
    for t in order:
        parts = nm(t).replace(" (", "|(").split("|", 1)
        A(f'<div class="hh">{"<br>".join(parts)}</div>')
    for seg in seg_order:
        bs = D["by_seg"][seg]
        ss = {t: (sum(W[c] * avg(bs[t][c]) for c in CRIT) if bs[t]["correctness"] else None) for t in order}
        lead = max((t for t in order if ss[t] is not None), key=lambda t: ss[t], default=None)
        A(f'<div class="hr">{esc(seglbl(seg))}</div>')
        for t in order:
            v = ss[t]
            if v is None:
                A('<div class="cell" style="background:var(--bg-soft);color:var(--faint)">—</div>'); continue
            bg, fg = heat_color(v)
            ld = '<span class="ld"></span>' if t == lead else ''
            A(f'<div class="cell{" lead" if t==lead else ""}" style="background:{bg};color:{fg}">{ld}{v:.1f}</div>')
    A('</div></div></div></div>')

    # ---- categories heatmap (category × model, same style as segments) ----
    A('<div class="wrap"><h2 class="sec" id="categories">Результаты по категориям</h2>')
    A('<div class="callout">Средний балл по каждой категории × модели. Теплее = выше; золотая рамка — лидер категории.</div>')
    A('<div class="chart"><div class="chart-cap"><span class="ct">Категория × модель</span><span class="cs">0–10</span></div>')
    A('<div class="heat-scroll">')
    A(f'<div class="heat" style="min-width:{150 + len(order) * 58}px;'
      f'grid-template-columns:minmax(140px,1.3fr) repeat({len(order)},minmax(52px,1fr))"><div></div>')
    for t in order:
        parts = nm(t).replace(" (", "|(").split("|", 1)
        A(f'<div class="hh">{"<br>".join(parts)}</div>')
    for c in cat_order:
        if c not in D["by_cat"]:
            continue
        cs = {t: (avg(D["by_cat"][c][t]) if D["by_cat"][c][t] else None) for t in order}
        lead = max((t for t in order if cs[t] is not None), key=lambda t: cs[t], default=None)
        A(f'<div class="hr">{esc(clbl(c))}</div>')
        for t in order:
            v = cs[t]
            if v is None:
                A('<div class="cell" style="background:var(--bg-soft);color:var(--faint)">—</div>'); continue
            bg, fg = heat_color(v)
            ld = '<span class="ld"></span>' if t == lead else ''
            A(f'<div class="cell{" lead" if t==lead else ""}" style="background:{bg};color:{fg}">{ld}{v:.1f}</div>')
    A('</div></div></div></div>')

    # ---- category leaders + token cost table ----
    A('<div class="wrap"><div class="chart"><div class="chart-cap"><span class="ct">Лидер и стоимость по категории</span><span class="cs">0–10</span></div>')
    A('<div class="table-scroll"><table><thead><tr><th>Категория</th><th>Лидер</th>'
      '<th class="num">Балл</th><th class="num">задач</th><th class="num">ср. ток/задача</th></tr></thead><tbody>')
    for c in cat_order:
        if c not in D["by_cat"]:
            continue
        lead = max(tags, key=lambda t: avg(D["by_cat"][c][t]))
        ctoks = [v for t in tags for v in D["by_cat_tok"][c][t]]   # cost across all models in this category
        A(f'<tr><td>{esc(clbl(c))}</td><td><b>{nm(lead)}</b></td>'
          f'<td class="num">{avg(D["by_cat"][c][lead]):.2f}</td><td class="num">{len(by_cat_tasks[c])}</td>'
          f'<td class="num">{avg(ctoks):,.0f}</td></tr>')
    A('</tbody></table></div></div></div>')

    # ---- per-task ----
    A('<div class="wrap"><h2 class="sec">Разбор по задачам</h2></div>')
    for c in cat_order:
        A(f'<div class="wrap"><h3 class="catsec" id="cat-{c}">{esc(clbl(c))}</h3>')
        for r in by_cat_tasks[c]:
            sd = D["scores"].get(r["id"])
            if not sd:
                continue
            present = [t for t in tags if t in sd["scores"] and not D["blk"](t, r) and not D["emp"](t, r)]
            winner = max(present, key=lambda t: wsum(sd["scores"][t], W), default=None)
            wtot = wsum(sd["scores"][winner], W) if winner else 0
            if vl and r.get("image"):
                # Small inline thumbnail keeps the page self-contained; the lightbox opens the
                # full-resolution file next to the report (repo/Pages) when it is available.
                thumb = thumb_data_url(r["image"], w=420)
                full = "vl-ru-images/" + os.path.basename(r["image"])
                A(f'<article class="task vl" id="{r["id"]}"><div class="imgcol">'
                  f'<img src="{thumb}" alt="{esc(r.get("title",""))}" loading="lazy" '
                  f'data-full="{esc(full)}" '
                  f'data-cap="{esc(r["id"])} — {esc(r.get("title",""))}"></div><div class="body">')
            else:
                A(f'<article class="task" id="{r["id"]}"><div class="body">')
            A(f'<h4>{esc(r["id"])} — {esc(r.get("title",""))}</h4>')
            A(f'<div class="meta"><span class="tag">{esc(r.get(cat_key,"?"))}</span> · промпт {len(r["prompt"]):,} симв. · '
              f'<span class="win-badge">Победитель: {nm(winner) if winner else "—"} · {wtot:.1f}/10</span></div>')
            A(f'<details class="prompt-d"><summary><span class="pchev">▸</span>Промпт задачи</summary>'
              f'<div class="prompt">{esc(r["prompt"][:7000])}</div></details>')
            if vl and r.get("image"):
                A('</div><div class="scorewrap">')   # table spans the whole card, not the narrow column
            A('<div class="table-scroll"><table><thead><tr><th>Модель</th><th class="num">C</th><th class="num">F</th>'
              '<th class="num">Ru</th><th class="num">Σ</th><th class="num">ток</th><th>Комментарий судьи</th></tr></thead><tbody>')

            def rk(t):
                if D["blk"](t, r):
                    return (2, 0)
                if t not in sd["scores"] or D["emp"](t, r):
                    return (3, 0)
                return (0, -wsum(sd["scores"][t], W))
            for t in sorted(tags, key=rk):
                if D["blk"](t, r):
                    A(f'<tr style="color:var(--faint)"><td>{nm(t)}</td><td class="num">—</td><td class="num">—</td>'
                      f'<td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="cmt-cell">Заблокировано кибербез-фильтром OpenAI.</td></tr>')
                    continue
                if t not in sd["scores"] or D["emp"](t, r):
                    continue
                s = sd["scores"][t]; tot = wsum(s, W)
                cls = ' class="win"' if t == winner else ""
                toks = out_toks(t, r["responses"].get(t, {}))
                toks_cell = f'{toks:,}' if toks is not None else '—'   # codex: no comparable token count
                A(f'<tr{cls}><td><b>{nm(t)}</b></td><td class="num">{s["correctness"]}</td>'
                  f'<td class="num">{s["format"]}</td><td class="num">{s["russian"]}</td>'
                  f'<td class="num"><b>{tot:.1f}</b></td><td class="num">{toks_cell}</td>'
                  f'<td class="cmt-cell">{esc(s.get("comment",""))}</td></tr>')
            A('</tbody></table></div>')
            if sd.get("notes"):
                A(f'<div class="note">{esc(sd["notes"])}</div>')
            A('</div></article>')
        A('</div>')

    A('</main><a class="totop" href="#top">↑</a>')
    if vl:
        A('<div class="lightbox" id="lb"><span class="lb-close">&times;</span>'
          '<img id="lb-img" src="" alt=""><div class="lb-cap" id="lb-cap"></div></div>')
    js = ("<script>document.querySelectorAll('[data-count]').forEach(e=>{const n=+e.dataset.count;let c=0;"
          "const st=Math.max(1,n/40);const t=setInterval(()=>{c+=st;if(c>=n){c=n;clearInterval(t)}"
          "e.textContent=Math.round(c).toLocaleString('ru')},20)});"
          # --- collapsible nav: segment/category accordions ---
          "document.querySelectorAll('.nav-seg-h').forEach(b=>b.addEventListener('click',()=>"
          "b.parentElement.classList.toggle('open')));"
          "document.querySelectorAll('.nav-cat-h').forEach(b=>b.addEventListener('click',()=>{"
          "b.parentElement.classList.toggle('open');"
          "const t=b.dataset.target,el=t&&document.getElementById(t);"
          "if(el){history.replaceState(null,'','#'+t);el.scrollIntoView({behavior:'smooth',block:'start'});}}));"
          # --- scroll-spy: highlight the section in view, open its accordions ---
          "const navBy={};document.querySelectorAll('[data-target]').forEach(el=>{"
          "(navBy[el.dataset.target]=navBy[el.dataset.target]||[]).push(el);});"
          "let cur=null;"
          "function setActive(id){if(id===cur)return;cur=id;"
          "document.querySelectorAll('.nav-link.active,.tlink.active,.nav-cat-h.cat-active')"
          ".forEach(e=>e.classList.remove('active','cat-active'));"
          "const els=navBy[id];if(!els)return;"
          "els.forEach(e=>e.classList.add(e.classList.contains('nav-cat-h')?'cat-active':'active'));"
          "const host=els[0].closest('.nav-cat');if(host){host.classList.add('open');"
          "const h=host.querySelector('.nav-cat-h');if(h)h.classList.add('cat-active');"
          "const sg=host.closest('.nav-seg');if(sg)sg.classList.add('open');}"
          "const a=els.find(e=>e.classList.contains('tlink'))||els[0];"
          # keep the active link visible by scrolling the sidebar itself; scrollIntoView here
          # would fight the page scroll and make navigation feel broken
          "const sb=document.querySelector('.sidebar');"
          "if(a&&sb){const ar=a.getBoundingClientRect(),sr=sb.getBoundingClientRect();"
          "if(ar.top<sr.top+8||ar.bottom>sr.bottom-8)sb.scrollTop+=ar.top-sr.top-sr.height/3;}}"
          "const spy=new IntersectionObserver(es=>{const vis=es.filter(e=>e.isIntersecting)"
          ".sort((x,y)=>x.boundingClientRect.top-y.boundingClientRect.top);"
          "if(vis.length)setActive(vis[0].target.id);},{rootMargin:'-72px 0px -70% 0px',threshold:0});"
          "document.querySelectorAll('#exec,h2.sec,h3.catsec,article.task').forEach(e=>{if(e.id)spy.observe(e);});"
          # open the accordion that matches the URL hash on load
          "if(location.hash){const el=document.querySelector(location.hash);if(el&&el.id)setActive(el.id);}"
          "")
    if vl:
        js += ("const lb=document.getElementById('lb'),li=document.getElementById('lb-img'),"
               "lc=document.getElementById('lb-cap');"
               "document.querySelectorAll('.imgcol img').forEach(im=>im.addEventListener('click',()=>{"
               "li.src=im.dataset.full||im.src;li.onerror=()=>{li.onerror=null;li.src=im.src;};lc.textContent=im.dataset.cap||'';lb.classList.add('open');"
               "document.body.style.overflow='hidden';}));"
               "function closeLb(){lb.classList.remove('open');document.body.style.overflow='';}"
               "lb.addEventListener('click',closeLb);"
               "document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLb();});")
    js += "</script>"
    html = (f"<!DOCTYPE html><html lang=ru><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{esc(title)}</title>"
            f'<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
            f"<style>{THEME}{THEME_FIX}</style></head><body>{''.join(H)}{js}</body></html>")
    Path(out_path).write_text(html)
    print(f"Wrote {out_path} ({len(html)//1024} KB), {n_tasks} tasks, {len(tags)} models")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--results"); p.add_argument("--scores-dir"); p.add_argument("--out")
    p.add_argument("--tags"); p.add_argument("--title"); p.add_argument("--subtitle", default="")
    p.add_argument("--exclude", default="", help="comma-separated task ids to drop from the report")
    p.add_argument("--rename", default="", help="comma-separated tag=Display Name overrides")
    a = p.parse_args()
    vl = a.mode == "vl"
    W = VL_W if vl else TEXT_W
    exclude = [t for t in a.exclude.split(",") if t]
    rename = dict(kv.split("=", 1) for kv in a.rename.split(",") if "=" in kv)
    D = compute(a.results, a.scores_dir, a.tags.split(","), W, vl, exclude, rename)
    render(D, a.out, a.title, a.subtitle)


if __name__ == "__main__":
    main()
