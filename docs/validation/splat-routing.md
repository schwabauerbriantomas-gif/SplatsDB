# Gaussian splat routing — measured technical gate

**Primary verdict: GO for learned routing utility, with a limited scope.** The learned codebook beats a random codebook at all three predeclared scan budgets. This validates candidate generation with exact raw-vector re-scoring; it does **not** validate probabilistic ranking, calibrated confidence, semantic retrieval quality or an end-to-end agent benefit.

Primary protocol: [`splat-routing-protocol.md`](splat-routing-protocol.md). Aggregate machine-readable results: [`k100`](../../bench-data/splat_routing_results.json) and [`k200`](../../bench-data/splat_routing_k200_sensitivity.json). Per-query HTTP receipts remain private; their SHA-256 hashes are recorded below with the other run hashes.

## H1 — exact field correctness

- Exact top-10 **distance parity: 100/100 queries**; maximum absolute score error 3.58e-07.
- 4 queries returned a different ID at a float32-tied boundary with identical distances. The protocol explicitly allows this; distances did not mismatch.
- Exact mode visited 99.88% of clusters and scanned 99.999% of vectors. **No material pruning benefit was observed at 11,069 × 384.** H1 is a correctness pass, not a speed pass.
- Independent Python two-phase mirror matched the real HTTP route in 800/800 comparisons, with 0 mismatches.

## H2 — learned routing versus random codebook

Each query uses the smallest probe count that reaches its scan budget. Actual mean scan is reported because cluster granularity makes exact budget matching impossible.

| Scan budget | Learned R@10 / actual scan | Random R@10 / actual scan | Paired delta | Bootstrap 95% CI | Gate | Practical reading |
|---:|---:|---:|---:|---:|:---:|:---|
| 10.0% | 94.5% / 15.2% | 86.4% / 14.7% | +8.1 pp | [+4.7 pp, +11.6 pp] | PASS | supports utility (>5 pp) |
| 20.0% | 98.9% / 29.0% | 95.3% / 29.0% | +3.6 pp | [+2.0 pp, +5.4 pp] | PASS | exploratory (<5 pp) |
| 40.0% | 99.9% / 61.5% | 99.2% / 64.8% | +0.7 pp | [+0.1 pp, +1.4 pp] | PASS | exploratory (<5 pp) |

All three paired bootstrap intervals exclude zero. **Small-sample caution, added after analysis without changing the frozen gate:** the house rule for 100 queries treats deltas below 5 percentage points as noise unless replicated. Only the 10%-budget effect (+8.1 pp) clears that conservative floor. The 20% and 40% effects pass the frozen CI gate but remain exploratory, not established practical benefits. A larger independent query set is required before relying on them.

The benefit is largest at low scan budgets and becomes small near 40%. There is no claim of superiority over other ANN systems: the control isolates the value of learned centroids, not a cross-engine benchmark.

## Recall/scan curve

| nprobe | Learned R@10 | Learned scan | Random R@10 | Random scan |
|---:|---:|---:|---:|---:|
| 1 | 57.2% | 1.3% | 38.6% | 1.4% |
| 2 | 72.8% | 2.8% | 55.3% | 2.9% |
| 4 | 83.9% | 5.4% | 69.5% | 5.4% |
| 8 | 90.3% | 10.5% | 81.8% | 10.1% |
| 16 | 96.7% | 19.4% | 90.7% | 19.4% |
| 32 | 99.8% | 35.4% | 96.6% | 35.7% |
| 64 | 99.9% | 67.2% | 99.4% | 67.2% |
| 100 | 100.0% | 100.0% | 100.0% | 100.0% |

## k=200 sensitivity

The sensitivity run cannot rescue or redefine the primary gate. It confirms the low-scan effect, while the 40% comparison is inconclusive.

| Scan budget | Learned R@10 / actual scan | Random R@10 / actual scan | Paired delta | Bootstrap 95% CI | Gate | Practical reading |
|---:|---:|---:|---:|---:|:---:|:---|
| 10.0% | 96.0% / 15.2% | 89.6% / 14.5% | +6.4 pp | [+4.1 pp, +9.0 pp] | PASS | supports utility (>5 pp) |
| 20.0% | 98.9% / 30.4% | 95.1% / 28.4% | +3.8 pp | [+2.1 pp, +5.6 pp] | PASS | exploratory (<5 pp) |
| 40.0% | 100.0% / 100.0% | 99.9% / 95.3% | +0.1 pp | [+0.0 pp, +0.3 pp] | INCONCLUSIVE | inconclusive |

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

- Binary SHA-256: `2e6feb0f35b5d10b21e6f16fb387cd0889481058e96ef67599505a179554a94d`.
- Protocol SHA-256 at run: `db87dfa8fc4b472b5e53c3af8e85bbc5055a4a861f949df398d905654e047490`.
- Corpus embeddings SHA-256: `2fb754e256201728d49a60a58f7a60bffacb1a7aa43cf022e07d89a45130af7f`.
- Raw HTTP SHA-256: `03b3057973ca893ae638284956e390b4b1c886110b74c80273e2f37ee631df89`.
- Analysis script SHA-256: `d9a95e7e0077d7c2f724c78d2349f9a0c68cdc4e451bfd3c1c94d6e3c158aeaf`.
- The first parser incorrectly required identical top-10 IDs and emitted `NO_GO_CORRECTNESS` for four float32 boundary ties. The protocol allowed tied IDs when distance multisets matched. The parser was corrected and the **same raw receipts** were reanalyzed; no measurement was rerun to obtain the GO.
