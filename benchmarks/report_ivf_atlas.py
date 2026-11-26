#!/usr/bin/env python3
"""Generate the two measured reports, or verify they exactly match their JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def ci(values: list[float]) -> str:
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def ivf_report(data: dict) -> str:
    if data.get("status") != "COMPLETE":
        return "# Matched IVFFlat and adaptive probing — measured result\n\n**Status: VALIDATING.** The receipt is not terminal; no verdict is derived.\n"
    parity = data["h3_faiss_mirror_parity"]
    lines = [
        "# Matched IVFFlat and adaptive probing — measured result",
        "",
        f"**Verdict:** `{data['verdict']}`. Families: " + ", ".join(f"{name}=`{verdict}`" for name, verdict in data["family_verdicts"].items()) + ".",
        "",
        "## Scope",
        "",
        "Same 11,069 × 384 corpus, learned `k=100` cells, 100 frozen test queries and exact L2 top-10. Policy training uses 300 new queries; configuration/comparator selection uses 100 separate queries. Recall is distance-threshold recall with a `2e-5` tolerance.",
        "",
        "This is an implementation/policy experiment on synthetic perturbed self-queries, not a human-semantic, calibrated-confidence or production-QPS benchmark.",
        "",
        "The first run stopped before producing output on a non-grid ratio-probe harness bug; see the [incident record](ivf-atlas-incidents.md). The protocol was not changed.",
        "",
        "## H3 — Faiss/mirror parity",
        "",
        f"- Comparisons: **{parity['comparisons']}**; recall mismatches: **{parity['recall_mismatches']}**; distance-multiset mismatches: **{parity['distance_mismatches']}**; scan mismatches: **{parity['scan_mismatches']}**.",
        f"- H3: **{'PASS' if parity['h3_pass'] else 'FAIL'}**.",
        "",
        "## Fixed probe grid",
        "",
        "| nprobe | Faiss recall | Faiss scan | Mirror recall | Mirror scan |",
        "|---:|---:|---:|---:|---:|",
    ]
    for p in (1, 2, 4, 8, 16, 32, 64, 100):
        f = data["faiss_grid"][str(p)]
        m = data["mirror_grid"][str(p)]
        lines.append(f"| {p} | {pct(f['recall_mean'])} | {pct(f['scan_mean'])} | {pct(m['recall_mean'])} | {pct(m['scan_mean'])} |")
    lines += ["", "## H4 — validation-selected policy outcomes", "", "A target requires test recall within 0.5 pp, at least 5% relative scan reduction, a non-inferior recall CI and a positive scan-reduction CI. The comparator is selected only on validation.", ""]
    for family, targets in data["h4_policy_targets"].items():
        lines.append(f"### {family}")
        lines.append("")
        for target, row in targets.items():
            selected = data["validation_selection"][target]
            config = row.get("selected_config")
            config_text = "fixed model" if family == "learned" else str(config)
            lines.append(f"- **Target {pct(float(target))}: `{row['status']}`**; config={config_text}; fixed comparator `nprobe={selected['fixed_nprobe']}`.")
            if "policy_recall_mean" not in row:
                lines.append(f"  - {row.get('reason', 'No measured policy outcome.')}")
                continue
            lines += [
                f"  - Recall: policy **{pct(row['policy_recall_mean'])}**, fixed **{pct(row['fixed_recall_mean'])}**; paired delta CI {ci(row['recall_delta_ci95'])}.",
                f"  - Scan: policy **{pct(row['policy_scan_mean'])}**, fixed **{pct(row['fixed_scan_mean'])}**; relative reduction **{pct(row['relative_scan_reduction'])}**, CI {ci(row['relative_scan_reduction_ci95'])}.",
                f"  - Faiss fixed p50/p95: **{row['faiss_fixed_latency']['p50_ms']:.3f}/{row['faiss_fixed_latency']['p95_ms']:.3f} ms**. Python precomputed-distance harness p50: **{row['python_precomputed_distance_harness_latency']['p50_ms']:.3f} ms**; these timings are not comparable.",
            ]
        lines.append("")
    train = data["train_regressor"]
    build = data["build"]
    lines += [
        "## Cost and reproducibility",
        "",
        f"- Faiss index addition: **{build['faiss_add_seconds']:.3f} s**; serialized index: **{build['faiss_serialized_index_bytes']:,} bytes**; mirror cell files: **{build['mirror_cell_file_bytes']:,} bytes**. KMeans training and Rust build time are not measured here.",
        f"- Learned model: {train['rows']} training rows, {train['features']} features, train RMSE **{train['rmse']:.5f}**; test rows used in training: **{train['test_rows_used']}**.",
        f"- Protocol SHA-256: `{data['receipts']['protocol_sha256']}`.",
        f"- Runner SHA-256: `{data['receipts']['runner_sha256']}`; common analyzer: `{data['receipts']['common_script_sha256']}`.",
        f"- Source HEAD: `{data['receipts']['source']['head']}`; dirty lines at execution: **{data['receipts']['source']['dirty_lines']}**.",
        "",
        "No per-query IDs or distances are published. The receipt contains aggregate/per-query recall and scan arrays, selected configurations, hashes and package/runtime scope.",
        "",
    ]
    return "\n".join(lines)


def atlas_report(data: dict) -> str:
    if data.get("status") != "COMPLETE":
        return "# Numeric embedding atlas — structural baseline\n\n**Status: VALIDATING.** The receipt is not terminal; no metrics are derived.\n"
    p = data["structural_metrics"]["pca"]
    u = data["structural_metrics"]["umap"]
    lines = [
        "# Numeric embedding atlas — structural baseline",
        "",
        f"**Viewer recommendation:** `{data['viewer_recommendation']}`. This is an isotropic splat-density prototype inspired by Topo-GS, not a Topo-GS reproduction.",
        "",
        "![PCA and UMAP scatter and splat-density panels](../../assets/atlas/embedding_atlas.png)",
        "",
        "## Structural metrics",
        "",
        "| Projection | kNN@10 | kNN@50 | Trust@10 | Continuity@10 | Spearman | Scale-invariant stress |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| PCA | {p['knn_10']:.4f} | {p['knn_50']:.4f} | {p['trustworthiness_10']:.4f} | {p['continuity_10']:.4f} | {p['spearman']:.4f} | {p['scale_invariant_stress']:.4f} |",
        f"| UMAP | {u['knn_10']:.4f} | {u['knn_50']:.4f} | {u['trustworthiness_10']:.4f} | {u['continuity_10']:.4f} | {u['spearman']:.4f} | {u['scale_invariant_stress']:.4f} |",
        "",
        "Evaluation uses 2,000 points and 100,000 non-self sampled pairs. Class separation is `not_applicable` because no labels are present; user-task accuracy was not measured.",
        "",
        "## Rendering cost and limits",
        "",
        f"- PCA fit: **{data['projection_seconds']['pca']:.3f} s**; UMAP fit: **{data['projection_seconds']['umap']:.3f} s**; figure render: **{data['render']['render_seconds']:.3f} s**.",
        f"- Each smoothed volume: **{data['volumes']['pca']['volume_bytes']:,} bytes** at `128³`; clipped fractions: PCA **{pct(data['volumes']['pca']['clipped_fraction'])}**, UMAP **{pct(data['volumes']['umap']['clipped_fraction'])}**.",
        f"- PNG: **{data['render']['png_bytes']:,} bytes**, `{data['render']['width']}×{data['render']['height']}`; SHA-256 `{data['render']['png_sha256']}`.",
        "",
        "The splat pass shares the projection coordinates. It therefore cannot improve kNN preservation, trustworthiness, continuity or stress; only PCA-versus-UMAP differences are structural. No interactive-viewer benefit is claimed without a separate user-task study.",
        "",
        f"Protocol SHA-256: `{data['receipts']['protocol_sha256']}`; runner `{data['receipts']['runner_sha256']}`; source HEAD `{data['receipts']['source']['head']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivf", type=Path)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/validation")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    jobs = []
    if args.ivf:
        jobs.append((args.output_dir / "ivf-policy.md", ivf_report(json.loads(args.ivf.read_text()))))
    if args.atlas:
        jobs.append((args.output_dir / "embedding-atlas.md", atlas_report(json.loads(args.atlas.read_text()))))
    assert jobs, "Provide --ivf and/or --atlas"
    failed = False
    for path, expected in jobs:
        if args.check:
            actual = path.read_text() if path.exists() else ""
            ok = actual == expected
            failed |= not ok
            print(("REPORT_CURRENT" if ok else "REPORT_STALE") , path)
        else:
            path.write_text(expected)
            print("REPORT_WRITTEN", path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
