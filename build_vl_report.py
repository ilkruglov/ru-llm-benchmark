"""HTML report for the RU VL/CV benchmark — 3 vision models, capabilities x verticals,
with embedded image thumbnails and a bookmark sidebar. Article-style, uniform fonts.
"""
import argparse
import base64
import json
import re
import statistics
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

# reasoning excluded from scoring: gemma-4-31b returns no separate reasoning trace
# (it reasons inline in content), so the criterion is not uniformly available across models.
CRITERIA = ["correctness", "format", "russian"]
# VL weighting: correctness dominant. In grounded VL tasks russian/format are near-ceiling
# (85% of russian scores >=9) and barely discriminate, so they get equal minor weight; the
# real signal is correctness. (Text uses 0.6/0.1/0.3 — russian discriminates more there.)
WEIGHTS = {"correctness": 0.8, "format": 0.1, "russian": 0.1}
MAX_TOTAL = 10
DIV = 1   # weighted Σ is already 0-10 (weights sum to 1); kept as a no-op display divisor


def wsum(s):
    """Weighted 0-10 score for one model's criterion dict."""
    return sum(WEIGHTS[c] * s[c] for c in CRITERIA)
NAMES = {"27b-v2": "Qwen3.6-27B-FP8", "35b-a3b": "Qwen3.6-35B-A3B-FP8",
         "llama-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
         "gpt-5.5-low": "GPT-5.5 (low)", "gpt-5.5-medium": "GPT-5.5 (medium)",
         "gpt-5.5-xhigh": "GPT-5.5 (xhigh)",
         "nemotron-omni": "Nemotron-3-Nano-Omni-30B",
         "kimi-k2.6": "Kimi K2.6",
         "gpt-5.6-sol-low": "GPT-5.6 Sol (low)", "gpt-5.6-sol-medium": "GPT-5.6 Sol (med)", "gpt-5.6-terra-low": "GPT-5.6 Terra (low)", "gpt-5.6-terra-medium": "GPT-5.6 Terra (med)", "gpt-5.6-terra-high": "GPT-5.6 Terra (high)", "gpt-5.6-luna-low": "GPT-5.6 Luna (low)", "gpt-5.6-luna-medium": "GPT-5.6 Luna (med)", "gpt-5.6-luna-high": "GPT-5.6 Luna (high)"}
CAP_LABEL = {
    "ocr": "OCR — чтение текста", "kie": "Извлечение полей (KIE)", "table": "Таблицы",
    "chart": "Графики и диаграммы", "dashboard": "Дашборды (BI)", "diagram": "Схемы",
    "drawing": "Чертежи", "count": "Подсчёт объектов", "detect": "Детекция/локализация",
    "defect": "Дефектоскопия", "equipment": "Оборудование", "scene": "Сцены и объекты",
    "safety": "Охрана труда / СИЗ", "map": "Карты", "compare": "Сравнение (до/после)",
    "vqa": "Визуальное рассуждение", "ui": "Интерфейсы (UI)",
}
VERT_LABEL = {
    "стройка": "Стройка", "банки": "Банки / финтех", "промышленность": "Тяжёлая промышленность",
    "нефтегаз": "Нефтегаз", "операционка": "Операционка (бух/фин/юр/налоги)",
    "маркетинг": "Маркетинг", "дата-аналитика": "Дата-аналитика", "общее": "Общее",
}


def esc(s):
    if s is None:
        return ""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def avg(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def thumb_data_url(path, w=380):
    try:
        im = Image.open(path).convert("RGB")
        if im.width > w:
            im = im.resize((w, int(im.height * w / im.width)))
        b = BytesIO()
        im.save(b, "JPEG", quality=78)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""


CSS = """
:root{--pri:#2563eb;--pri-d:#1e3a8a;--soft:#eff4ff;--bd:#dbe3ef;--mut:#64748b;--gold:#b8860b;}
*{box-sizing:border-box;} body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#1f2937;font-size:15px;line-height:1.5;}
a{color:var(--pri);text-decoration:none;} a:hover{text-decoration:underline;}
.sidebar{position:fixed;top:0;left:0;width:300px;height:100vh;overflow-y:auto;background:#0f172a;color:#cbd5e1;padding:16px 12px 40px;font-size:12.5px;}
.sidebar h1{color:#fff;font-size:15px;margin:0 0 4px;} .sidebar .sub{color:#7c8aa0;font-size:11px;margin-bottom:12px;}
.sidebar a{color:#cbd5e1;display:block;padding:2px 6px;border-radius:4px;} .sidebar a:hover{background:#1e293b;color:#fff;text-decoration:none;}
.sidebar .grp{color:#fff;font-weight:700;margin:10px 0 3px;font-size:12px;}
.sidebar .seg{color:#93c5fd;font-weight:700;margin:14px 0 4px;font-size:12.5px;border-top:1px solid #1e293b;padding-top:8px;}
.sidebar .tlink{padding-left:12px;color:#94a3b8;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
main{margin-left:300px;padding:28px 40px 80px;max-width:1180px;}
h1.title{color:#1e3a8a;font-size:28px;margin:0 0 4px;} h2.sec{color:var(--pri-d);border-bottom:2px solid var(--pri);padding-bottom:6px;margin:34px 0 14px;font-size:22px;}
h3.catsec{color:var(--pri-d);margin:32px 0 6px;font-size:18px;border-left:4px solid var(--pri);padding-left:10px;}
table{border-collapse:collapse;width:100%;margin:10px 0 18px;font-size:13.5px;} th,td{border:1px solid var(--bd);padding:6px 9px;text-align:left;vertical-align:top;}
th{background:var(--soft);color:var(--pri-d);font-weight:700;} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
tr.win{background:#ecfdf5;} .gold-row{background:#fffbeb;}
.callout{background:var(--soft);border-left:4px solid var(--pri);padding:9px 14px;margin:10px 0 18px;border-radius:0 6px 6px 0;font-size:14px;}
.task{border:1px solid var(--bd);border-radius:8px;padding:16px 18px;margin:16px 0;display:flex;gap:18px;scroll-margin-top:14px;}
.task .imgcol{flex:0 0 390px;} .task .imgcol img{width:100%;border:1px solid var(--bd);border-radius:6px;}
.task .imgcol .cap{font-size:11.5px;color:var(--mut);margin-top:4px;}
.task .body{flex:1;min-width:0;}
.task h4{margin:0 0 4px;font-size:15px;color:var(--pri-d);} .task .meta{color:var(--mut);font-size:12.5px;margin-bottom:8px;}
.tag{background:var(--soft);color:var(--pri-d);border-radius:4px;padding:1px 7px;font-size:11.5px;font-weight:600;}
.prompt{background:#fafbfc;border-left:3px solid var(--pri);padding:7px 11px;border-radius:0 6px 6px 0;font-size:13px;margin-bottom:8px;}
.cmt-cell{font-size:12px;color:#4b5563;} .note{background:#f8fafc;border-left:3px solid var(--pri);padding:7px 11px;margin-top:8px;border-radius:0 6px 6px 0;font-size:12px;font-style:italic;color:#374151;}
.totop{position:fixed;bottom:18px;right:22px;background:var(--pri);color:#fff;padding:8px 12px;border-radius:6px;font-size:12px;}
"""


def build(results_path, scores_dir, out_path, tags, note=None, drop_mentions=None):
    # scrub judge-prose sentences mentioning models omitted from this report
    _drop_tokens = []
    for dt in (drop_mentions or []):
        for part in [dt] + dt.split("-"):
            if part:
                _drop_tokens.append(part.lower())
    _drop_tokens = list(set(_drop_tokens))

    def scrub(text):
        if not _drop_tokens or not text:
            return esc(text)
        parts = re.split(r'(?<=[.!?;])\s+', text)
        kept = [p for p in parts if not any(tok in p.lower() for tok in _drop_tokens)]
        return esc(" ".join(kept).strip())

    data = json.load(open(results_path))
    results = data["results"]
    scores = {}
    for fp in Path(scores_dir).glob("*.json"):
        try:
            sd = json.load(open(fp))
            scores[sd["task_id"]] = sd
        except Exception:
            pass

    def nm(t):
        return esc(NAMES.get(t, t))

    by_cap = defaultdict(lambda: {t: [] for t in tags})
    by_vert = defaultdict(lambda: {t: [] for t in tags})
    by_tag = {t: {c: [] for c in CRITERIA} for t in tags}
    wins = defaultdict(int)
    cap_wins = defaultdict(lambda: defaultdict(int))
    for r in results:
        sd = scores.get(r["id"])
        if not sd:
            continue
        for t in tags:
            s = sd["scores"].get(t)
            if not s:
                continue
            tot = wsum(s)
            by_cap[r["capability"]][t].append(tot)
            by_vert[r["vertical"]][t].append(tot)
            for c in CRITERIA:
                by_tag[t][c].append(s[c])
        # recompute winner over the scored criteria (reasoning excluded)
        w = max((t for t in tags if t in sd["scores"]),
                key=lambda t: wsum(sd["scores"][t]), default=None)
        if w in tags:
            wins[w] += 1
            cap_wins[r["capability"]][w] += 1

    # performance / token-expenditure metrics (latency_ms and completion_tokens are stored per response)
    perf = {t: {"lat": [], "ct": [], "tps": [], "alen": [], "ok": 0, "fail": 0} for t in tags}
    for r in results:
        for t in tags:
            resp = r["responses"].get(t)
            if not resp:
                continue
            if resp.get("ok"):
                perf[t]["ok"] += 1
                lat = resp.get("latency_ms") or 0
                ct = resp.get("completion_tokens") or 0
                if lat:
                    perf[t]["lat"].append(lat)
                if ct:
                    perf[t]["ct"].append(ct)
                if ct and lat:
                    perf[t]["tps"].append(ct / (lat / 1000))
                perf[t]["alen"].append(len(resp.get("content") or ""))
            else:
                perf[t]["fail"] += 1

    n_tasks = len([r for r in results if r["id"] in scores])
    caps = [c for c in CAP_LABEL if c in by_cap]
    # category-balanced Σ over capabilities
    def cap_mean(cap, t):
        return avg(by_cap[cap][t])
    macro = {t: avg([cap_mean(c, t) for c in caps]) for t in tags}
    micro = {t: sum(WEIGHTS[c] * avg(by_tag[t][c]) for c in CRITERIA) for t in tags}
    caps_led = {t: sum(1 for c in caps if max(tags, key=lambda x: cap_mean(c, x)) == t) for t in tags}
    # size-normalized win-rate: share of wins within each direction, averaged over directions
    cap_size = {c: max(len(by_cap[c][t]) for t in tags) for c in caps}
    macro_winrate = {t: 100 * avg([cap_wins[c][t] / cap_size[c] for c in caps]) for t in tags}
    order = sorted(tags, key=lambda t: -macro[t])

    H = []
    A = H.append
    # group by vertical -> capability
    by_vc = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["id"] in scores:
            by_vc[r["vertical"]][r["capability"]].append(r)

    # sidebar
    A('<nav class="sidebar">')
    A(f'<h1>VL/CV Benchmark — РФ</h1><div class="sub">{n_tasks} задач · {len(tags)} модели · реальные русские изображения</div>')
    A('<div class="grp">Сводка</div><a href="#lead">Лидерборд</a><a href="#caps">По направлениям VL/CV</a><a href="#verts">По отраслям</a>'
      '<a href="#tokens">Расход токенов</a><a href="#perf">Производительность</a>')
    for vert in VERT_LABEL:
        if vert not in by_vc:
            continue
        A(f'<div class="seg">{esc(VERT_LABEL[vert])}</div>')
        for cap in CAP_LABEL:
            tasks_vc = by_vc[vert].get(cap, [])
            if not tasks_vc:
                continue
            A(f'<div class="grp" style="margin:6px 0 2px">{esc(CAP_LABEL[cap])} ({len(tasks_vc)})</div>')
            for r in tasks_vc:
                A(f'<a class="tlink" href="#{esc(r["id"])}">{esc(r.get("title") or r["id"])[:40]}</a>')
    A('</nav>')

    A('<main id="top">')
    A('<h1 class="title">VL/CV-бенчмарк на реальных русских изображениях</h1>')
    A(f'<div style="color:#374151;margin-bottom:10px">Тест {len(tags)} vision-моделей на {n_tasks} задачах по реальным изображениям из российского бизнес-контекста '
      f'({len(caps)} направлений VL/CV × {len([v for v in VERT_LABEL if v in by_vc])} отраслей). Каждая модель оценивалась по 4 критериям '
      f'(correctness, format, russian) по шкале 0–10; итоговый балл — взвешенная свёртка correctness 0.8 / format 0.1 / russian 0.1 (в VL это граундед-распознавание: главное — точность чтения изображения, а format/russian у всех моделей близки к потолку). Судья — отдельный Opus, который сам видит изображение. '
      f'Критерий «reasoning» исключён из подсчёта: часть моделей возвращает отдельный след рассуждения, а часть (gemma-4-31b) рассуждает внутри ответа и отдельного следа не отдаёт — единый честный учёт невозможен.</div>')
    A('<div class="callout"><b>Модели:</b> ' + ', '.join(f'<b>{nm(t)}</b>' for t in tags) + '.</div>')
    if note:
        A(f'<div class="callout" style="border-left-color:var(--gold);background:#fffbeb"><b>Примечание.</b> {esc(note)}</div>')

    # leaderboard
    A('<h2 class="sec" id="lead">Лидерборд</h2>')
    A('<div class="callout"><b>Методика агрегирования.</b> Направления VL/CV содержат разное число задач (от 4 до 35), '
      f'поэтому основная метрика — <b>«Σ по направлениям»</b>: сначала средний балл внутри каждого из {len(caps)} направлений, '
      'затем эти средние усредняются между собой (каждое направление весит одинаково — размер выборки не искажает итог). '
      '«Σ по задачам» — среднее по всем задачам без нормализации. Все баллы Σ — на шкале <b>0–10</b> (взвешенная свёртка correctness 0.8 / format 0.1 / russian 0.1). '
      '«Доля побед» — усреднённая по направлениям доля выигранных задач. '
      '«Побед всего» — сырая сумма; зависит от размера направлений, приводится только для справки.</div>')
    A('<table><thead><tr><th>#</th><th>Модель</th><th class="num">Σ по направлениям / 10</th><th class="num">Σ по задачам / 10</th>'
      '<th class="num">Лидер в направлениях</th><th class="num">Доля побед, %</th><th class="num">Побед всего</th>'
      '<th class="num">C</th><th class="num">F</th><th class="num">Ru</th></tr></thead><tbody>')
    for i, t in enumerate(order):
        a = by_tag[t]
        cls = ' class="gold-row"' if i == 0 else ''
        A(f'<tr{cls}><td>{i+1}</td><td><b>{nm(t)}</b></td><td class="num"><b>{macro[t]/DIV:.2f}</b></td><td class="num">{micro[t]/DIV:.2f}</td>'
          f'<td class="num">{caps_led[t]} / {len(caps)}</td><td class="num">{macro_winrate[t]:.1f}</td><td class="num">{wins[t]}</td>'
          f'<td class="num">{avg(a["correctness"]):.2f}</td>'
          f'<td class="num">{avg(a["format"]):.2f}</td><td class="num">{avg(a["russian"]):.2f}</td></tr>')
    A('</tbody></table>')
    A('<div style="color:#64748b;font-size:12.5px;margin:-8px 0 14px">C/F/Ru — correctness, format, russian (0–10). Критерий reasoning исключён из подсчёта (см. методику выше).</div>')

    # by capability
    A('<h2 class="sec" id="caps">Результаты по направлениям VL/CV (Σ / 10)</h2>')
    A('<table><thead><tr><th>Направление</th><th class="num">N</th>' + ''.join(f'<th class="num">{nm(t)}</th>' for t in tags) + '<th>Лидер</th></tr></thead><tbody>')
    for cap in caps:
        vals = {t: cap_mean(cap, t) for t in tags}
        ld = max(vals, key=vals.get)
        n = max(len(by_cap[cap][t]) for t in tags)
        A(f'<tr><td><a href="#cap-{cap}">{esc(CAP_LABEL[cap])}</a></td><td class="num">{n}</td>'
          + ''.join(f'<td class="num">{vals[t]/DIV:.1f}</td>' for t in tags) + f'<td><b>{nm(ld)}</b></td></tr>')
    A('</tbody></table>')

    # by vertical
    A('<h2 class="sec" id="verts">Результаты по отраслям (Σ / 10)</h2>')
    A('<table><thead><tr><th>Отрасль</th><th class="num">N</th>' + ''.join(f'<th class="num">{nm(t)}</th>' for t in tags) + '<th>Лидер</th></tr></thead><tbody>')
    for vert in VERT_LABEL:
        if vert not in by_vert:
            continue
        vals = {t: avg(by_vert[vert][t]) for t in tags}
        ld = max(vals, key=vals.get)
        n = max(len(by_vert[vert][t]) for t in tags)
        A(f'<tr><td>{esc(VERT_LABEL[vert])}</td><td class="num">{n}</td>'
          + ''.join(f'<td class="num">{vals[t]/DIV:.1f}</td>' for t in tags) + f'<td><b>{nm(ld)}</b></td></tr>')
    A('</tbody></table>')

    # token expenditure (cost)
    A('<h2 class="sec" id="tokens">Расход токенов на ответ</h2>')
    A('<div class="callout">Сколько модель тратит выходных токенов на ответ — прямая стоимость инференса и главный драйвер задержки. '
      'Выходные токены включают весь сгенерированный текст, в том числе внутренние рассуждения (для reasoning-моделей). '
      'Длина ответа — размер видимого ответа в символах.</div>')
    A('<table><thead><tr><th>Модель</th><th class="num">Среднее выходных токенов</th><th class="num">Медиана выходных токенов</th>'
      '<th class="num">Средняя длина ответа, символов</th></tr></thead><tbody>')
    for t in sorted(tags, key=lambda t: -avg(perf[t]["ct"])):
        A(f'<tr><td>{nm(t)}</td><td class="num"><b>{int(avg(perf[t]["ct"]))}</b></td>'
          f'<td class="num">{int(statistics.median(perf[t]["ct"])) if perf[t]["ct"] else 0}</td>'
          f'<td class="num">{int(avg(perf[t]["alen"]))}</td></tr>')
    A('</tbody></table>')

    # performance
    A('<h2 class="sec" id="perf">Производительность: задержка и пропускная способность</h2>')
    A('<div class="callout">Пропускная способность (tok/s) — отношение выходных токенов к полному времени ответа (включая рассуждения и обработку изображения). '
      'Время до первого токена (TTFT) не измерялось — запросы без потоковой передачи. '
      '<b>Важно:</b> модели работают на разной инфраструктуре, поэтому задержка и tok/s отражают и серверную настройку провайдера, а не только саму модель.</div>')
    A('<table><thead><tr><th>Модель</th><th class="num">Средняя задержка</th><th class="num">Медианная задержка</th>'
      '<th class="num">Средняя, tok/s</th><th class="num">Медианная, tok/s</th><th class="num">Успешно</th><th class="num">Ошибок</th></tr></thead><tbody>')
    for t in sorted(tags, key=lambda t: -avg(perf[t]["tps"])):
        med_lat = statistics.median(perf[t]["lat"]) if perf[t]["lat"] else 0
        med_tps = statistics.median(perf[t]["tps"]) if perf[t]["tps"] else 0
        A(f'<tr><td>{nm(t)}</td><td class="num">{avg(perf[t]["lat"])/1000:.1f}s</td><td class="num">{med_lat/1000:.1f}s</td>'
          f'<td class="num"><b>{avg(perf[t]["tps"]):.0f}</b></td><td class="num">{med_tps:.0f}</td>'
          f'<td class="num">{perf[t]["ok"]}</td><td class="num">{perf[t]["fail"]}</td></tr>')
    A('</tbody></table>')

    # per-task, grouped vertical->capability
    res_by_id = {r["id"]: r for r in results}
    for vert in VERT_LABEL:
        if vert not in by_vc:
            continue
        for cap in CAP_LABEL:
            tasks_vc = by_vc[vert].get(cap, [])
            if not tasks_vc:
                continue
            A(f'<h3 class="catsec" id="cap-{cap}">{esc(CAP_LABEL[cap])} <span style="color:#64748b;font-weight:400;font-size:14px">— {esc(VERT_LABEL[vert])}, {len(tasks_vc)} задач</span></h3>')
            for r in tasks_vc:
                sd = scores.get(r["id"])
                if not sd:
                    continue
                winner = max((t for t in tags if t in sd["scores"]),
                             key=lambda t: wsum(sd["scores"][t]), default=None)
                durl = thumb_data_url(r["image"])
                A(f'<article class="task" id="{esc(r["id"])}">')
                A(f'<div class="imgcol"><img src="{durl}" alt=""><div class="cap">{esc(r.get("title") or "")}</div></div>')
                A('<div class="body">')
                A(f'<h4>{esc(r.get("title") or r["id"])}</h4>')
                A(f'<div class="meta"><span class="tag">{esc(CAP_LABEL.get(cap,cap))}</span> · {esc(VERT_LABEL.get(vert,vert))} · '
                  f'<b style="color:var(--gold)">Победитель: {nm(winner) if winner else "—"}</b></div>')
                A(f'<div class="prompt"><b>Задание:</b> {esc(r.get("prompt",""))}</div>')
                A('<table><thead><tr><th>Модель</th><th class="num">C</th><th class="num">F</th><th class="num">Ru</th><th class="num">Σ</th>'
                  '<th class="num">Задержка</th><th class="num">Токены</th><th class="num">tok/s</th><th>Комментарий судьи</th></tr></thead><tbody>')
                tt = sorted(((t, wsum(sd["scores"][t])) for t in tags if t in sd["scores"]), key=lambda x: -x[1])
                for t, tot in tt:
                    s = sd["scores"][t]
                    cls = ' class="win"' if t == winner else ''
                    resp = r["responses"].get(t, {})
                    okr = resp.get("ok")
                    latms = resp.get("latency_ms") or 0
                    ctk = resp.get("completion_tokens")
                    lat = f'{latms/1000:.1f}s' if okr else "FAIL"
                    cts = str(ctk) if (okr and ctk is not None) else "—"
                    tps = f'{ctk/(latms/1000):.0f}' if (okr and ctk and latms) else "—"
                    A(f'<tr{cls}><td><b>{nm(t)}</b></td><td class="num">{s["correctness"]}</td>'
                      f'<td class="num">{s["format"]}</td><td class="num">{s["russian"]}</td><td class="num"><b>{tot/DIV:.1f}</b></td>'
                      f'<td class="num">{lat}</td><td class="num">{cts}</td><td class="num">{tps}</td><td class="cmt-cell">{scrub(s.get("comment",""))}</td></tr>')
                A('</tbody></table>')
                A(f'<div class="note"><b>Эталон/нота судьи:</b> {scrub(sd.get("notes",""))}</div>')
                A('</div></article>')

    A('</main><a class="totop" href="#top">↑ Наверх</a>')
    html = f"<!DOCTYPE html><html lang=ru><head><meta charset=utf-8><title>VL/CV РФ бенчмарк</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
    Path(out_path).write_text(html)
    print(f"Wrote {out_path} ({len(html)//1024} KB), {n_tasks} tasks")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="vl-ru-results.json")
    p.add_argument("--scores-dir", default="vl-judging-scores")
    p.add_argument("--out", default="VL-Benchmark-RU.html")
    p.add_argument("--tags", default="27b-v2,35b-a3b,llama-scout")
    p.add_argument("--note", default=None)
    p.add_argument("--drop-mentions", default="")
    a = p.parse_args()
    build(a.results, a.scores_dir, a.out, a.tags.split(","), a.note, [x for x in a.drop_mentions.split(",") if x])


if __name__ == "__main__":
    main()
