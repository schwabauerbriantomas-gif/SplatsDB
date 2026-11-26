#!/usr/bin/env python3
"""Build SplatsDB charts from committed measurements.

HTML is generated without JavaScript. PNG rendering is optional and uses the
locally installed Chrome through Playwright. No value is typed into a chart
except labels and thresholds copied from the same versioned source document.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
PUBLIC_PNG = {"faiss": "splatsdb-faiss-comparison.png", "longmemeval": "splatsdb-longmemeval.png", "m3-rerun": "splatsdb-m3-context-rerun.png", "m3-historical": "splatsdb-m3-context.png"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def num(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def header(title: str, subtitle: str, status: str, tone: str = "neutral") -> str:
    return f"""
<header class="header">
  <div class="brand">
    <img class="brand-mark" src="../logo-splatsdb.svg" alt="SplatsDB logo"/>
    <div>
      <div class="kicker">SplatsDB / Measured evidence</div>
      <h1>{esc(title)}</h1>
      <p class="subtitle">{esc(subtitle)}</p>
    </div>
  </div>
  <div class="status {tone}">{esc(status)}</div>
</header>"""


def kpi(label: str, value: str, detail: str, tone: str = "") -> str:
    return f'<div class="kpi"><div class="kpi-label">{esc(label)}</div><div class="kpi-value {tone}">{esc(value)}</div><div class="kpi-detail">{esc(detail)}</div></div>'


def panel_head(title: str, note: str = "", tag: str = "") -> str:
    tag_html = f'<span class="tag">{esc(tag)}</span>' if tag else ""
    note_html = f'<p class="panel-note">{esc(note)}</p>' if note else ""
    return f'<div class="panel-head"><div><h2 class="panel-title">{esc(title)}</h2>{note_html}</div>{tag_html}</div>'


def bar(label: str, value: float, maximum: float, text: str, sub: str = "", fill: str = "") -> str:
    width = max(0.0, min(100.0, 100 * value / maximum))
    sub_html = f"<small>{esc(sub)}</small>" if sub else ""
    return f'<div class="bar-row"><div class="bar-label">{esc(label)}{sub_html}</div><div class="track"><div class="fill {fill}" style="width:{width:.5f}%"></div></div><div class="bar-value">{text}</div></div>'


def metric(value: str, label: str) -> str:
    return f'<div class="metric-box"><div class="metric-value">{esc(value)}</div><div class="metric-label">{esc(label)}</div></div>'


def callout(text: str, tone: str = "") -> str:
    style = ' style="margin-top:12px"' if tone else ""
    return f'<div class="callout {tone}"{style}>{text}</div>'


def footer(source: str, note: str) -> str:
    return f'<footer class="footer"><div class="source">{source}<br/>{esc(note)}</div><div class="build">HTML + CSS · measured values only</div></footer>'


def document(title: str, description: str, css: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{esc(title)}</title><meta name="description" content="{esc(description)}"/>
<style>{css}</style></head>
<body><main class="chart" role="img" aria-label="{esc(description)}">{body}</main></body></html>"""


def faiss(css: str) -> tuple[str, str, str]:
    d = load(ROOT / "bench-data/benchmark_results_hardware.json")
    gpu, faiss, ivf, native = (d[k] for k in ("splatdb_gpu_topk", "faiss_hnsw_cpu", "faiss_ivf_pq_cpu", "splatdb_hnsw_cpu"))
    assert gpu["results_match_cpu"] is True
    rows = [
        bar("SplatsDB GPU", gpu["gpu_qps_persistent"], 13000, f'{num(gpu["gpu_qps_persistent"])} <small>QPS</small>', "exact · k=10 · RTX 3090"),
        bar("Rust CPU", gpu["cpu_qps"], 13000, f'{num(gpu["cpu_qps"])} <small>QPS</small>', "exact · k=10", "deep"),
        bar("Faiss HNSW", faiss["qps"], 13000, f'{num(faiss["qps"])} <small>QPS</small>', "CPU · approximate · k=64", "secondary"),
        bar("Faiss IVF-PQ", ivf["qps"], 13000, f'{num(ivf["qps"])} <small>QPS</small>', "CPU · approximate · k=64", "secondary"),
        bar("SplatsDB HNSW", native["qps"], 13000, f'{num(native["qps"])} <small>QPS</small>', "CPU · approximate · k=64", "deep"),
    ]
    body = header(
        "Search performance, in separate regimes",
        f"{d['dataset']} · same host · Ryzen 5 3400G + RTX 3090 · Windows 11 / WSL2",
        f"HISTORICAL RUN · {d['date']}",
    )
    body += '<section class="kpis">'
    body += kpi("Exact GPU top-k", num(gpu["gpu_qps_persistent"]), "QPS · k=10 · recall 1.0 · matched CPU")
    body += kpi("CPU HNSW gap", f'{faiss["qps"] / native["qps"]:.1f}×', "Faiss faster at k=64; similar recall")
    body += kpi("HNSW recall", pct(faiss["recall_at_64"], 2), f'Faiss · SplatsDB {pct(native["recall_at_64"], 2)}')
    body += kpi("IVF-PQ trade-off", pct(ivf["recall_at_64"], 1), f'Lossy compression · {num(ivf["qps"])} QPS')
    body += '</section><section class="panel">'
    body += panel_head("Throughput — read each row in its own regime", "QPS is not compared across different k or search types.", "100K vectors · 1K queries")
    body += "".join(rows)
    body += '<div class="legend"><span class="legend-item"><i class="dot"></i>Exact GPU</span><span class="legend-item"><i class="dot secondary"></i>Faiss ANN</span><span class="legend-item"><i class="dot deep"></i>SplatsDB CPU</span></div></section>'
    body += '<section class="panels equal"><div class="panel">'
    body += panel_head("Recall", "The k changes by row; this is not a shared leaderboard.", "k varies per row")
    body += bar("SplatsDB GPU · k=10", 1, 1, "100.0%", "exact; matches CPU")
    body += bar("Faiss HNSW · k=64", faiss["recall_at_64"], 1, pct(faiss["recall_at_64"], 2), "approximate ANN", "secondary")
    body += bar("SplatsDB HNSW · k=64", native["recall_at_64"], 1, pct(native["recall_at_64"], 2), "approximate ANN", "deep")
    body += bar("Faiss IVF-PQ · k=64", ivf["recall_at_64"], 1, pct(ivf["recall_at_64"], 1), "lossy compression", "warn")
    body += '</div><div class="panel">'
    body += panel_head("Build cost & GPU upload", "Index build times use the same April run.", "not release timing")
    body += bar("Faiss IVF-PQ", ivf["build_ms"] / 1000, 100, f'{num(ivf["build_ms"] / 1000, 1)} s', "k=64", "secondary")
    body += bar("Faiss HNSW", faiss["build_ms"] / 1000, 100, f'{num(faiss["build_ms"] / 1000, 1)} s', "M=32 · efC=400", "secondary")
    body += bar("SplatsDB HNSW", native["build_ms"] / 1000, 100, f'{num(native["build_ms"] / 1000, 1)} s', "computes α and κ", "deep")
    body += callout(f'<strong>GPU persistent:</strong> {num(gpu["gpu_qps_persistent"])} QPS = {esc(gpu["speedup_persistent"].replace("x", "×"))} CPU.<br/><strong>Including one-time {gpu["upload_ms"]} ms VRAM upload:</strong> {num(gpu["gpu_qps_with_upload"])} QPS.')
    body += '</div></section>'
    source = 'Source: <a href="../../bench-data/benchmark_results_hardware.json">benchmark_results_hardware.json</a> · SplatsDB v2.5.0 run · <a href="faiss.html">standalone HTML chart</a>'
    body += footer(source, "Historical measurements, not a current release profile. GPU k=10 exact search and CPU k=64 ANN are not head-to-head.")
    title = "SplatsDB vs Faiss — measured search regimes"
    return "faiss", document(title, title, css, body), source


def longmemeval(css: str) -> tuple[str, str, str]:
    ref = load(ROOT / "bench-data/longmemeval_full_results.json")
    native = load(ROOT / "bench-data/longmemeval_splatdb_native.json")
    baseline = load(ROOT / "bench-data/longmemeval_baseline_results.json")
    recall = ref["recall"]
    body = header(
        "Two memory pipelines. Two different results.",
        f"{ref['n_questions']} questions · {num(native['n_sessions'])} sessions · MiniLM 384d · RTX 3090 embeddings",
        f"RETRIEVAL ONLY · {ref['date']}",
    )
    body += '<section class="panels equal" style="flex:0 0 460px"><div class="panel">'
    body += panel_head("NumPy brute-force reference", "Embeddings + normalized cosine; not SplatsDB's engine.", "96.6% @10")
    for key, label in (("recall@1", "Top-1"), ("recall@3", "Top-3"), ("recall@5", "Top-5"), ("recall@10", "Top-10")):
        body += bar(label, recall[key], 1, pct(recall[key], 1), "answer session in top-k")
    body += '</div><div class="panel">'
    body += panel_head("SplatsDB native pipeline", "HNSW M=32 · efS=100 · over-fetch 3× + re-rank · L2.", "30.4% @10")
    body += '<div class="split-metrics">'
    body += metric(pct(native["recall"]["hnsw_overfetch_rerank"]["@10"], 1), "Recall@10 · over-fetch + re-rank")
    body += metric(num(native["splatdb_rust"]["qps"], 2), "QPS · measured native run")
    body += metric(f'{num(native["splatdb_rust"]["p50_us"] / 1000, 3)} ms', "p50 latency · over-fetch + re-rank")
    body += metric(f'{num(native["splatdb_rust"]["build_ms"] / 1000, 2)} s', "Index build")
    body += '</div>'
    body += callout('<strong>Do not read 96.6% as native-engine recall.</strong> The native gap is real under this methodology; no end-to-end paper comparison is claimed.')
    body += '</div></section><section class="panel" style="flex:1;min-height:0">'
    body += panel_head("Recall@10 by question type", "Same 500 questions; two pipelines shown together.", "NumPy vs native")
    for name, values in ref["per_type"].items():
        n = baseline["per_type"][name]["total"]
        a = values["recall@10"]
        b = native["per_type_overfetch"][name]["@10"]
        body += '<div class="type-row">'
        body += f'<div class="type-name">{esc(name.replace("-", " "))}<small>· {n} questions</small></div>'
        body += f'<div class="type-bars"><div class="type-bar"><div class="type-bar-top"><span>NumPy</span><strong>{pct(a, 1)}</strong></div><div class="track"><div class="fill" style="width:{100*a:.4f}%"></div></div></div>'
        body += f'<div class="type-bar"><div class="type-bar-top"><span>Native</span><strong>{pct(b, 1)}</strong></div><div class="track"><div class="fill deep" style="width:{100*b:.4f}%"></div></div></div></div></div>'
    body += '<div class="legend"><span class="legend-item"><i class="dot"></i>NumPy brute force</span><span class="legend-item"><i class="dot deep"></i>SplatsDB native</span><span>Raw HNSW recall@10: 30.8% · re-ranked: 30.4%</span></div></section>'
    source = 'Sources: <a href="../../bench-data/longmemeval_full_results.json">NumPy JSON</a> · <a href="../../bench-data/longmemeval_splatdb_native.json">native JSON</a> · <a href="../../bench-data/longmemeval_baseline_results.json">question counts</a> · <a href="longmemeval.html">standalone HTML chart</a>'
    body += footer(source, "NumPy latency p50/p95/p99: 0.056 / 0.080 / 0.105 ms; no NumPy QPS was recorded. Native figures come from the native JSON, not mixed with the reference.")
    title = "LongMemEval — NumPy reference and SplatsDB native"
    return "longmemeval", document(title, title, css, body), source


def m3(css: str, historical: bool = False) -> tuple[str, str, str]:
    key = "m3-historical" if historical else "m3-rerun"
    d = load(DATA / f"{key}.json")
    a, b = "glm-5.4", "glm-5.3-flash"
    verdict = d["verdict"]
    assert verdict == ("NO_GO" if historical else "GO")
    reduction = d["context_reduction"]
    accuracy = d["accuracy"]
    judges = d["judges"]
    pairs = reduction[a]["paired_questions"]
    tone = "no-go" if historical else "go"
    title = "M3 adaptation — historical NO_GO run" if historical else "M3 adaptation — independent rerun"
    status = "NO_GO · COVERAGE" if historical else "GO · PREDECLARED PROTOCOL"
    body = header(
        title,
        f"500 questions · GLM-5.4 + GLM-5.3-Flash · crossed judges · {pairs} complete token pairs",
        status,
        tone,
    )
    body += '<section class="kpis">'
    body += kpi("Context reduction", pct((reduction[a]["reduction_percent"] + reduction[b]["reduction_percent"]) / 200, 2), "Mean of the two model reductions; each is reported below")
    body += kpi("Symmetric accuracy", pct(accuracy[a], 2), f'GLM-5.3-Flash {pct(accuracy[b], 2)} · {accuracy["scored_questions"]} scored')
    body += kpi("Complete token pairs", f'{pairs}/500', f'Minimum required: {d["required_complete_pairs"]}/500')
    if historical:
        body += kpi("M3 error calls", str(d["m3_error_calls"]), "One per model · provider-filtered question")
    else:
        body += kpi("Non-filter errors", str(d["filters"]["non_filter_errors"]), "Zero judge filters · no imputation")
    body += '</section><section class="panel" style="flex:0 0 265px">'
    body += panel_head("Prompt context — median tokens per question", "Baseline report median uses 499 calls; M3 medians use the 498 paired questions.", "≤50% threshold")
    body += bar("Full-context baseline", reduction[a]["baseline_median"], reduction[a]["baseline_median"], num(reduction[a]["baseline_median"]), "same corpus and models", "outline")
    body += bar("M3 · GLM-5.4", reduction[a]["m3_total_median"], reduction[a]["baseline_median"], num(reduction[a]["m3_total_median"], 1), f'largest-prompt median {num(reduction[a]["m3_largest_median"], 1)} · −{reduction[a]["reduction_percent"]:.2f}%')
    body += bar("M3 · GLM-5.3-Flash", reduction[b]["m3_total_median"], reduction[b]["baseline_median"], num(reduction[b]["m3_total_median"], 1), f'largest-prompt median {num(reduction[b]["m3_largest_median"], 1)} · −{reduction[b]["reduction_percent"]:.2f}%', "secondary")
    body += '</section><section class="panels equal" style="flex:1;min-height:0"><div class="panel">'
    body += panel_head("Accuracy & non-regression", "Symmetric crossed-judge accuracy; no superiority claim.", f'{accuracy["scored_questions"]} scored')
    body += bar("GLM-5.4", accuracy[a], 1, pct(accuracy[a], 2), "M3 policy", "go" if not historical else "")
    body += bar("GLM-5.3-Flash", accuracy[b], 1, pct(accuracy[b], 2), "M3 policy", "secondary")
    body += '<div class="split-metrics">'
    body += metric(f'+{accuracy["delta_pp"]:.2f} pp', "Observed paired difference")
    body += metric(f'p={accuracy["mcnemar"]["p"]:.4f}', f'{accuracy["mcnemar"]["positive"]} improved · {accuracy["mcnemar"]["negative"]} regressed · McNemar')
    body += '</div>'
    if historical:
        body += callout('<strong>Numerical non-regression gates observed.</strong> The formal coverage gate still fails: it required 500 complete pairs, not 498.', 'no-go')
    else:
        itt = d["intent_to_treat"]
        body += callout(f'<strong>Intent-to-treat sensitivity · PASS:</strong> {pct(itt["accuracy"][a], 2)} vs {pct(itt["accuracy"][b], 2)} over 500 questions; filtered/missing labels count as incorrect.', 'go')
    body += '</div><div class="panel">'
    body += panel_head("Coverage, judges & filters", "Validity and missingness stay visible next to the result.", "no imputation")
    body += '<div class="split-metrics">'
    body += metric(f'{pairs}/500', f'Token pairs · required {d["required_complete_pairs"]}/500')
    body += metric(f'{accuracy["scored_questions"]}/500', "Scored answers")
    if historical:
        body += metric(str(d["m3_error_calls"]), "M3 provider-filtered calls")
        body += metric("500/500", "Original coverage requirement")
    else:
        body += metric("0", "Non-filter transport errors")
        body += metric("2 questions", "One filtered per stage; ≤5 budget")
    body += '</div>'
    body += callout(f'<strong>Judges:</strong> GLM-5.4 {pct(judges[a]["agreement"], 2)} · κ {judges[a]["kappa"]:.4f} · invalid {judges[a]["invalid_judged"]}/1000<br/><strong>Judges:</strong> Flash {pct(judges[b]["agreement"], 2)} · κ {judges[b]["kappa"]:.4f} · invalid {judges[b]["invalid_judged"]}/1000')
    if historical:
        body += callout('<strong>NO_GO by coverage, not by observed magnitude.</strong> The later independent rerun does not reinterpret this run.', 'no-go')
    else:
        body += callout(f'<strong>Protocol {d["protocol_sha256"][:12]}…</strong> GO applies only to this protocol. The earlier 500/500 run remains NO_GO. Prompt reduction is not total-cost reduction.', 'go')
    body += '</div></section>'
    source = f'Source: <a href="data/{key}.json">{key}.json</a> (minimal measured subset) · <a href="{key}.html">standalone HTML chart</a>'
    if not historical:
        source += ' · <a href="../../docs/validation/m3-adaptation-rerun.md">methodology and limits</a>'
    else:
        source += ' · <a href="../../docs/validation/m3-adaptation.md">historical report</a>'
    body += footer(source, "Measured subset copied from the local REPORT.json; raw conversations and credentials are not published. LLM judges are not human ground truth.")
    return key, document(title, title, css, body), source


def render(html_paths: list[Path]) -> dict:
    from playwright.sync_api import sync_playwright

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    assert chrome, "Chrome/Chromium is required for --render"
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--disable-gpu", "--hide-scrollbars"])
        page = browser.new_page(viewport={"width": 1800, "height": 1400}, device_scale_factor=1)
        for html_path in html_paths:
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready.then(() => true)")
            layout = page.evaluate("""() => {
              const chart = document.querySelector('.chart');
              const bad = [...document.querySelectorAll('.chart *')].filter(el => {
                if (!el.textContent.trim() || el.clientWidth < 2 || el.clientHeight < 2) return false;
                return el.scrollWidth > el.clientWidth + 2 || el.scrollHeight > el.clientHeight + 2;
              }).map(el => ({tag: el.tagName, cls: el.className, text: el.textContent.trim().slice(0, 90),
                cw: el.clientWidth, sw: el.scrollWidth, ch: el.clientHeight, sh: el.scrollHeight}));
              return {width: chart.clientWidth, height: chart.clientHeight,
                scrollWidth: chart.scrollWidth, scrollHeight: chart.scrollHeight, overflow: bad};
            }""")
            png = ROOT / "assets" / PUBLIC_PNG[html_path.stem]
            page.locator(".chart").screenshot(path=str(png), animations="disabled")
            results[html_path.name] = {"layout": layout, "png": png.name}
            print(html_path.name, json.dumps(layout))
        browser.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render standalone HTML to PNG with local Chrome")
    args = parser.parse_args()
    css = (HERE / "theme.css").read_text(encoding="utf-8")
    builders = [faiss, longmemeval, lambda c: m3(c, historical=False), lambda c: m3(c, historical=True)]
    outputs = []
    sources = {}
    for build in builders:
        name, content, source = build(css)
        path = HERE / f"{name}.html"
        path.write_text(content, encoding="utf-8")
        outputs.append(path)
        sources[name] = source
    manifest = {"schema_version": 1, "sources": sources, "artifacts": {}}
    if args.render:
        manifest["render"] = render(outputs)
    for name in outputs + [ROOT / "assets" / value for value in PUBLIC_PNG.values()] + sorted(DATA.glob("*.json")) + [HERE / "theme.css", HERE / "README.md", HERE / "validate.py", Path(__file__).resolve()]:
        manifest["artifacts"][name.name] = {"sha256": hashlib.sha256(name.read_bytes()).hexdigest(), "bytes": name.stat().st_size}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {len(outputs)} HTML charts{' and PNGs' if args.render else ''}; manifest written.")


if __name__ == "__main__":
    main()
