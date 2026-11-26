# Matched IVFFlat and adaptive probing — measured result

**Verdict:** `NO_GO_ADAPTIVE_POLICY`. Families: ratio=`NO_GO`, doubling=`NO_GO`, learned=`NO_GO`.

## Scope

Same 11,069 × 384 corpus, learned `k=100` cells, 100 frozen test queries and exact L2 top-10. Policy training uses 300 new queries; configuration/comparator selection uses 100 separate queries. Recall is distance-threshold recall with a `2e-5` tolerance.

This is an implementation/policy experiment on synthetic perturbed self-queries, not a human-semantic, calibrated-confidence or production-QPS benchmark.

Early harness incidents are recorded in the [incident register](ivf-atlas-incidents.md), including the invalid Faiss-parser NO_GO and its preserved artifact. The protocol was not changed.

## H3 — Faiss/mirror parity

- Comparisons: **800**; recall mismatches: **0**; distance-multiset mismatches: **0**; scan mismatches: **0**.
- H3: **PASS**.

## Fixed probe grid

| nprobe | Faiss recall | Faiss scan | Mirror recall | Mirror scan |
|---:|---:|---:|---:|---:|
| 1 | 57.20% | 1.27% | 57.20% | 1.27% |
| 2 | 72.80% | 2.78% | 72.80% | 2.78% |
| 4 | 83.90% | 5.44% | 83.90% | 5.44% |
| 8 | 90.30% | 10.46% | 90.30% | 10.46% |
| 16 | 96.70% | 19.44% | 96.70% | 19.44% |
| 32 | 99.80% | 35.36% | 99.80% | 35.36% |
| 64 | 99.90% | 67.23% | 99.90% | 67.23% |
| 100 | 100.00% | 100.00% | 100.00% | 100.00% |

## H4 — validation-selected policy outcomes

A target requires test recall within 0.5 pp, at least 5% relative scan reduction, a non-inferior recall CI and a positive scan-reduction CI. The comparator is selected only on validation.

### ratio

- **Target 90.00%: `FAIL`**; config=0.2; fixed comparator `nprobe=8`.
  - Recall: policy **95.20%**, fixed **90.30%**; paired delta CI [+0.0240, +0.0760].
  - Scan: policy **11.97%**, fixed **10.46%**; relative reduction **-19.45%**, CI [-0.3813, -0.0171].
  - Faiss fixed p50/p95: **0.126/0.220 ms**. Python precomputed-distance harness p50: **0.058 ms**; these timings are not comparable.
- **Target 95.00%: `FAIL`**; config=0.4; fixed comparator `nprobe=16`.
  - Recall: policy **99.80%**, fixed **96.70%**; paired delta CI [+0.0170, +0.0470].
  - Scan: policy **43.04%**, fixed **19.44%**; relative reduction **-134.14%**, CI [-1.6277, -1.0673].
  - Faiss fixed p50/p95: **0.244/0.382 ms**. Python precomputed-distance harness p50: **0.250 ms**; these timings are not comparable.
- **Target 98.00%: `FAIL`**; config=0.4; fixed comparator `nprobe=32`.
  - Recall: policy **99.80%**, fixed **99.80%**; paired delta CI [-0.0030, +0.0030].
  - Scan: policy **43.04%**, fixed **35.36%**; relative reduction **-23.95%**, CI [-0.3759, -0.1054].
  - Faiss fixed p50/p95: **0.543/0.675 ms**. Python precomputed-distance harness p50: **0.247 ms**; these timings are not comparable.

### doubling

- **Target 90.00%: `FAIL`**; config=0.2; fixed comparator `nprobe=8`.
  - Recall: policy **99.90%**, fixed **90.30%**; paired delta CI [+0.0670, +0.1280].
  - Scan: policy **99.04%**, fixed **10.46%**; relative reduction **-949.93%**, CI [-10.1530, -8.8602].
  - Faiss fixed p50/p95: **0.126/0.220 ms**. Python precomputed-distance harness p50: **5.695 ms**; these timings are not comparable.
- **Target 95.00%: `FAIL`**; config=0.2; fixed comparator `nprobe=16`.
  - Recall: policy **99.90%**, fixed **96.70%**; paired delta CI [+0.0180, +0.0490].
  - Scan: policy **99.04%**, fixed **19.44%**; relative reduction **-432.05%**, CI [-4.5457, -4.0858].
  - Faiss fixed p50/p95: **0.244/0.382 ms**. Python precomputed-distance harness p50: **5.664 ms**; these timings are not comparable.
- **Target 98.00%: `FAIL`**; config=0.2; fixed comparator `nprobe=32`.
  - Recall: policy **99.90%**, fixed **99.80%**; paired delta CI [-0.0020, +0.0040].
  - Scan: policy **99.04%**, fixed **35.36%**; relative reduction **-184.81%**, CI [-1.9316, -1.7553].
  - Faiss fixed p50/p95: **0.543/0.675 ms**. Python precomputed-distance harness p50: **5.697 ms**; these timings are not comparable.

### learned

- **Target 90.00%: `FAIL`**; config=fixed model; fixed comparator `nprobe=8`.
  - Recall: policy **95.40%**, fixed **90.30%**; paired delta CI [+0.0270, +0.0760].
  - Scan: policy **13.41%**, fixed **10.46%**; relative reduction **-36.61%**, CI [-0.5632, -0.1790].
  - Faiss fixed p50/p95: **0.126/0.220 ms**. Python precomputed-distance harness p50: **9.721 ms**; these timings are not comparable.
- **Target 95.00%: `FAIL`**; config=fixed model; fixed comparator `nprobe=16`.
  - Recall: policy **98.40%**, fixed **96.70%**; paired delta CI [+0.0020, +0.0340].
  - Scan: policy **21.79%**, fixed **19.44%**; relative reduction **-16.35%**, CI [-0.3792, +0.0268].
  - Faiss fixed p50/p95: **0.244/0.382 ms**. Python precomputed-distance harness p50: **11.495 ms**; these timings are not comparable.
- **Target 98.00%: `FAIL`**; config=fixed model; fixed comparator `nprobe=32`.
  - Recall: policy **99.50%**, fixed **99.80%**; paired delta CI [-0.0080, +0.0010].
  - Scan: policy **33.98%**, fixed **35.36%**; relative reduction **3.89%**, CI [-0.1094, +0.1764].
  - Faiss fixed p50/p95: **0.543/0.675 ms**. Python precomputed-distance harness p50: **13.588 ms**; these timings are not comparable.

## Reading

Across **9** measured family/target outcomes, **8** increase scan relative to their validation-selected fixed comparator. No family meets the predeclared H4 rule. The learned regressor at the 98% target reduces scan by **3.89%**, below the 5% floor, with a CI that crosses zero; the other selected outcomes do not establish a useful scan/recall trade-off.

This is a NO_GO for these three policy families under this protocol, not proof that every possible adaptive policy must fail.

## Cost and reproducibility

- Faiss index addition: **0.044 s**; serialized index: **17,245,075 bytes**; mirror cell files: **198,700 bytes**. KMeans training and Rust build time are not measured here.
- Learned model: 2400 training rows, 11 features, train RMSE **0.08882**; test rows used in training: **0**.
- Protocol SHA-256: `00f339c84425bb96733ef8362a396dc242f40d0115dcb9a2064ee26f60d87667`.
- Runner SHA-256: `4935487f70defd2f175cbdd2df6855addec0180daf990674b5ed22f29162a271`; common analyzer: `9631167f05cb7790eecf4ab75f0d77fc48678af7dd2e0699b0bb9c7112025cd3`.
- Source HEAD: `6b0c10f3f4ce7c81efde3eb56a7689e7bd7d4152`; dirty lines at execution: **0**.

No per-query IDs or distances are published. The receipt contains aggregate/per-query recall and scan arrays, selected configurations, hashes and package/runtime scope.
