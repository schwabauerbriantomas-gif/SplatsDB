#!/usr/bin/env python3
"""Render the local Gaussian-splat routing report from JSON receipts only."""
import argparse
import json
from pathlib import Path


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def points(value):
    return f"{100 * value:+.1f} pp"


def budget_table(result):
    rows = ["| Scan budget | Learned R@10 / actual scan | Random R@10 / actual scan | Paired delta | Bootstrap 95% CI | Gate | Practical reading |", "|---:|---:|---:|---:|---:|:---:|:---|"]
    for row in result["h2_learned_routing"]["budgets"]:
        learned, random = row["learned"], row["random"]
        strength = "supports utility (>5 pp)" if abs(row['delta_recall']) >= 0.05 and row['passes'] else ("exploratory (<5 pp)" if row['passes'] else "inconclusive")
        rows.append(f"| {pct(row['budget'])} | {pct(learned['recall_mean'])} / {pct(learned['scan_mean'])} | {pct(random['recall_mean'])} / {pct(random['scan_mean'])} | {points(row['delta_recall'])} | [{points(row['delta_ci95'][0])}, {points(row['delta_ci95'][1])}] | {'PASS' if row['passes'] else 'INCONCLUSIVE'} | {strength} |")
    return "\n".join(rows)


def curve_table(result):
    rows = ["| nprobe | Learned R@10 | Learned scan | Random R@10 | Random scan |", "|---:|---:|---:|---:|---:|"]
    random = {row["nprobe"]: row for row in result["curves"]["random"]}
    for learned in result["curves"]["learned"]:
        other = random[learned["nprobe"]]
        rows.append(f"| {learned['nprobe']} | {pct(learned['recall'])} | {pct(learned['scan'])} | {pct(other['recall'])} | {pct(other['scan'])} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="verify the existing report instead of writing it")
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text())
    sensitivity = json.loads(args.sensitivity.read_text())
    assert primary["verdict"] == "GO_LEARNED_ROUTING"
    h1 = primary["h1_exact_correctness"]
    receipts = primary["receipts"]
    text = f"""# Gaussian splat routing — measured technical gate

**Primary verdict: GO for learned routing utility, with a limited scope.** The learned codebook beats a random codebook at all three predeclared scan budgets. This validates candidate generation with exact raw-vector re-scoring; it does **not** validate probabilistic ranking, calibrated confidence, semantic retrieval quality or an end-to-end agent benefit.

Primary protocol: [`splat-routing-protocol.md`](splat-routing-protocol.md). Aggregate machine-readable results: [`k100`](../../bench-data/splat_routing_results.json) and [`k200`](../../bench-data/splat_routing_k200_sensitivity.json). Per-query HTTP receipts remain private; their SHA-256 hashes are recorded below with the other run hashes.

## H1 — exact field correctness

- Exact top-10 **distance parity: {100 - h1['distance_mismatches']}/100 queries**; maximum absolute score error {h1['max_score_error']:.3g}.
- {h1['id_mismatches']} queries returned a different ID at a float32-tied boundary with identical distances. The protocol explicitly allows this; distances did not mismatch.
- Exact mode visited {pct(h1['clusters_visited_fraction'], 2)} of clusters and scanned {pct(h1['scan_fraction'], 3)} of vectors. **No material pruning benefit was observed at 11,069 × 384.** H1 is a correctness pass, not a speed pass.
- Independent Python two-phase mirror matched the real HTTP route in {primary['mirror_parity']['comparisons']}/{primary['mirror_parity']['comparisons']} comparisons, with {primary['mirror_parity']['mismatches']} mismatches.

## H2 — learned routing versus random codebook

Each query uses the smallest probe count that reaches its scan budget. Actual mean scan is reported because cluster granularity makes exact budget matching impossible.

{budget_table(primary)}

All three paired bootstrap intervals exclude zero. **Small-sample caution, added after analysis without changing the frozen gate:** the house rule for 100 queries treats deltas below 5 percentage points as noise unless replicated. Only the 10%-budget effect (+8.1 pp) clears that conservative floor. The 20% and 40% effects pass the frozen CI gate but remain exploratory, not established practical benefits. A larger independent query set is required before relying on them.

The benefit is largest at low scan budgets and becomes small near 40%. There is no claim of superiority over other ANN systems: the control isolates the value of learned centroids, not a cross-engine benchmark.

## Recall/scan curve

{curve_table(primary)}

## k=200 sensitivity

The sensitivity run cannot rescue or redefine the primary gate. It confirms the low-scan effect, while the 40% comparison is inconclusive.

{budget_table(sensitivity)}

## What the splat architecture contributes here

The validated mechanism is **KMeans-learned centroid routing → probe a small set of cells → exact L2 re-scoring of raw member vectors**. The raw vectors remain available. `α` and `κ` are not used by this two-phase route; this experiment does not attribute the benefit to Gaussian energy or probabilistic semantics.

Practical interpretation: when only a limited fraction of a memory-resident corpus can be examined, learned partitions provide better candidate coverage than arbitrary partitions before exact re-scoring. The utility is **compute/coverage control**, not a claim of better semantic relevance or calibrated trust.

## Explicit non-claims and current limits

- Gaussian-energy ranking is not validated as a benefit. The prior A5 audit remains negative; see [`A5_FINDINGS.md`](../A5_FINDINGS.md).
- `topic`/`confidence` based on distance to a centroid is a membership diagnostic, not a correctness probability.
- Raw vectors are retained in this route, so this is not evidence of lossless document compression or a smaller memory footprint.
- Queries are deterministic perturbations of real embeddings. They do not measure human semantic intent, agent task success or answer correctness.
- No latency, incremental-update, persistence-scaling, distributed, GPU or 500k+ scale claim is established by this gate.
- MCP, graph, HNSW, GPU and M3 are separate product components; their presence is not evidence of a Gaussian-splat mechanism benefit.

## Next falsification work

1. Compare against an established IVF implementation at matched `nprobe` and measured scan fraction, without cross-dataset QPS claims.
2. Repeat on a second corpus with hand-written semantic-intent queries and an exact nearest-neighbor ceiling.
3. Evaluate radius-aware adaptive probing as a new mechanism only against fixed-`nprobe` and random controls.
4. Evaluate abstention separately with selective risk/coverage and end-to-end agent outcomes; do not use cluster-membership confidence as a proxy for correctness.
5. Measure incremental insertion/deletion and larger corpora before claiming operational savings.

## Reproduce analysis

From the repository root, with the private corpus, query binaries and engine data directories present under `tmp/splat_routing_20261126`:

```bash
python3 benchmarks/analyze_splat_routing.py --workdir tmp/splat_routing_20261126 --output bench-data/splat_routing_results.json
python3 benchmarks/analyze_splat_routing.py --k 200 --workdir tmp/splat_routing_20261126 --output bench-data/splat_routing_k200_sensitivity.json
python3 benchmarks/report_splat_routing.py --primary bench-data/splat_routing_results.json --sensitivity bench-data/splat_routing_k200_sensitivity.json --output docs/validation/splat-routing.md
```

Add `--check` to the report command to verify an existing report. `--raw-output` optionally writes per-query HTTP receipts; keep those files private because the publication unit is aggregate metrics and hashes. The public aggregate JSON does not contain corpus texts, embeddings or per-query result IDs.

## Provenance and harness incident

- Binary SHA-256: `{receipts['binary_sha256']}`.
- Protocol SHA-256 at run: `{receipts['protocol_sha256']}`.
- Corpus embeddings SHA-256: `{receipts['corpus_embeddings_sha256']}`.
- Raw HTTP SHA-256: `{receipts['http_sha256']}`.
- Analysis script SHA-256: `{receipts['analysis_script_sha256']}`.
- The first parser incorrectly required identical top-10 IDs and emitted `NO_GO_CORRECTNESS` for four float32 boundary ties. The protocol allowed tied IDs when distance multisets matched. The parser was corrected and the **same raw receipts** were reanalyzed; no measurement was rerun to obtain the GO.
"""
    if args.check:
        assert args.output.read_text() == text, f"stale report: {args.output}"
        print(f"REPORT_CURRENT {args.output}")
    else:
        args.output.write_text(text)
        print(args.output)


if __name__ == "__main__":
    main()
