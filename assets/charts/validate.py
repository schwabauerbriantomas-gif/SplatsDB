#!/usr/bin/env python3
"""Validate chart data, local links, PNG headers, visible text and artifact hashes."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import struct
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC_PNG = {
    "faiss": "splatsdb-faiss-comparison.png",
    "longmemeval": "splatsdb-longmemeval.png",
    "m3-rerun": "splatsdb-m3-context-rerun.png",
    "m3-historical": "splatsdb-m3-context.png",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.references: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text.append(data.strip())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.references.append(value)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"not PNG: {path}"
    return struct.unpack(">II", header[16:24])


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: list[str] = []
    texts: dict[str, str] = {}
    for stem, public_name in PUBLIC_PNG.items():
        html_path = HERE / f"{stem}.html"
        check(html_path.is_file(), f"missing {html_path}")
        page = Page()
        page.feed(html_path.read_text(encoding="utf-8"))
        text = " ".join(" ".join(page.text).split())
        texts[stem] = text
        check(not re.search(r"</?[A-Za-z][^>]*>", text), f"raw markup visible in {stem}")
        for reference in page.references:
            parsed = urlparse(reference)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (html_path.parent / unquote(parsed.path)).resolve()
            check(target.is_file(), f"broken link {reference} in {html_path}")
        png = ROOT / "assets" / public_name
        check(png.is_file(), f"missing {png}")
        check(png_size(png) == (1800, 1400), f"wrong dimensions: {png}")
        checks.append(f"{stem}: local links, visible text, PNG 1800x1400")

    required = {
        "faiss": ["12,195 QPS", "2,941 QPS", "9,758 QPS", "10,039 QPS", "658 QPS", "99.26%", "98.60%", "69.0%", "3.0 s", "24.5 s", "88.2 s", "k=10", "k=64"],
        "longmemeval": ["75.8%", "88.2%", "92.2%", "96.6%", "30.4%", "671.63", "1.448 ms", "106.96 s", "0.056 / 0.080 / 0.105 ms", "30.8%"],
        "m3-rerun": ["GO", "93.76%", "81.46%", "80.16%", "498/500", "499/500", "81.30%", "80.00%", "0.9403", "0.9244", "c6973edb93af", "earlier 500/500 run remains NO_GO"],
        "m3-historical": ["NO_GO", "93.77%", "81.46%", "79.56%", "498/500", "500/500", "0.1175", "0.8474", "0.9507"],
    }
    for stem, values in required.items():
        for value in values:
            check(value in texts[stem], f"{stem}: missing displayed value {value}")
        checks.append(f"{stem}: required values and verdict scope")

    hardware = load(ROOT / "bench-data/benchmark_results_hardware.json")
    check(hardware["splatdb_gpu_topk"]["results_match_cpu"] is True, "GPU/CPU mismatch")
    check(round(hardware["faiss_hnsw_cpu"]["qps"] / hardware["splatdb_hnsw_cpu"]["qps"], 1) == 14.8, "HNSW ratio")
    checks.append("faiss: source arithmetic and exact-result flag")

    ref = load(ROOT / "bench-data/longmemeval_full_results.json")
    native = load(ROOT / "bench-data/longmemeval_splatdb_native.json")
    baseline = load(ROOT / "bench-data/longmemeval_baseline_results.json")
    numpy_weighted = native_weighted = total = 0
    for name, row in baseline["per_type"].items():
        count = row["total"]
        numpy_weighted += count * ref["per_type"][name]["recall@10"]
        native_weighted += count * native["per_type_overfetch"][name]["@10"]
        total += count
    check(total == 500, "question counts do not sum to 500")
    check(round(100 * numpy_weighted / total, 1) == 96.6, "NumPy weighted recall")
    check(round(100 * native_weighted / total, 1) == 30.4, "native weighted recall")
    checks.append("longmemeval: counts and both weighted recalls")

    rerun = load(HERE / "data/m3-rerun.json")
    historical = load(HERE / "data/m3-historical.json")
    for label, data in (("rerun", rerun), ("historical", historical)):
        a, b = "glm-5.4", "glm-5.3-flash"
        check(abs(data["accuracy"]["delta_pp"] - 100 * (data["accuracy"][a] - data["accuracy"][b])) < 1e-9, f"{label}: delta")
        for model in (a, b):
            row = data["context_reduction"][model]
            expected = 100 * (1 - row["m3_total_median"] / row["baseline_median"])
            check(abs(row["reduction_percent"] - expected) < 1e-9, f"{label}: reduction {model}")
    check(rerun["verdict"] == "GO" and rerun["context_reduction"]["glm-5.4"]["paired_questions"] >= rerun["required_complete_pairs"], "rerun gate")
    check(rerun["filters"]["non_filter_errors"] == 0, "rerun non-filter errors")
    check(historical["verdict"] == "NO_GO" and historical["context_reduction"]["glm-5.4"]["paired_questions"] < historical["required_complete_pairs"], "historical gate")
    checks.append("m3: both measured subsets, reductions, deltas and verdict gates")

    chart_readme = (HERE / "README.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    historical_doc = (ROOT / "docs/validation/m3-adaptation.md").read_text(encoding="utf-8")
    micro_rows = re.findall(r"^\|\s*(\d)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)×", (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8"), re.MULTILINE)
    check(len(micro_rows) == 6, "microbenchmark must contain six runs")
    cpu_runs = [int(row[1]) for row in micro_rows]
    gpu_runs = [int(row[2]) for row in micro_rows]
    speedups = [float(row[3]) for row in micro_rows]
    check(all(abs(cpu / gpu - speed) <= 0.0050001 for cpu, gpu, speed in zip(cpu_runs, gpu_runs, speedups)), "microbenchmark per-run speedups")
    check(statistics.median(cpu_runs) == 605 and statistics.median(gpu_runs) == 19.5, "microbenchmark medians")
    check(round(statistics.median(speedups), 1) == 30.6, "microbenchmark median speedup")
    check("median CPU **605 ms**, GPU **19.5 ms**" in root_readme, "README microbenchmark medians")
    check("ratio of medians 31.03×" in (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8"), "README/BENCHMARKS ratio of medians")
    checks.append("microbenchmark: six runs, per-run speedups, medians and ratio")

    for document, content in ((HERE / "README.md", chart_readme), (ROOT / "README.md", root_readme), (ROOT / "docs/validation/m3-adaptation.md", historical_doc)):
        for target in re.findall(r'(?:href|src)="([^"]+)"', content):
            if "assets/charts" not in target and "assets/splatsdb-" not in target:
                continue
            parsed = urlparse(target)
            if parsed.scheme or not parsed.path:
                continue
            check((document.parent / unquote(parsed.path)).resolve().is_file(), f"broken documentation link: {target}")
    checks.append("documentation: chart image and source links resolve")

    manifest = load(HERE / "manifest.json")
    for name, receipt in manifest["artifacts"].items():
        if name in PUBLIC_PNG.values():
            path = ROOT / "assets" / name
        elif (HERE / "data" / name).is_file():
            path = HERE / "data" / name
        else:
            path = HERE / name
        check(path.is_file(), f"manifest target missing: {path}")
        check(hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"], f"hash mismatch: {name}")
        check(path.stat().st_size == receipt["bytes"], f"size mismatch: {name}")
    checks.append(f"manifest: {len(manifest['artifacts'])} artifact hashes and sizes")

    print(json.dumps({"status": "CHARTS_VALIDATED", "checks": checks, "count": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
