# IVF/probing and embedding-atlas protocol — predeclared

**Status:** frozen before the first new IVF/policy or atlas output.
**Date:** 2026-11-26.
**Scope:** CPU-only, local numeric embeddings; no text, identifiers or per-query distances are published.

## A. Matched IVFFlat and adaptive probing

### Frozen inputs and arms

- Corpus: the existing 11,069 × 384 float32 MiniLM cache, SHA-256 `cdf977ee2db99a2b3825914a7bbff39042642f35fafb647eaf1d6d17ec0c7532`; normalized inputs are assumed only after runtime verification.
- Test queries: the existing 100 perturbed queries, SHA-256 `11aa5655bcdc33a517e8734981357621f2b9a3f70efc0b603d7180a17efd2c3d`. They remain untouched until final evaluation.
- Cells: the existing learned `k=100` centroids and CSR assignments. Faiss receives those exact centroids; no Faiss retraining or different codebook is allowed.
- Primary probe grid: `nprobe ∈ {1,2,4,8,16,32,64,100}`. Exact top-10 L2 re-scoring is used by both Faiss IVFFlat and the independent NumPy mirror.
- New policy-train queries: 300, seed `20261130`; policy-validation queries: 100, seed `20261201`. Each samples corpus vectors without replacement, adds `sigma=0.05` Gaussian noise and L2-normalizes, matching the original perturbation procedure.
- Faiss and mirror run single-threaded, with two warm-ups and five measurement rounds. Faiss build time means index addition with externally supplied centroids, not KMeans training or the Rust engine build.
- Ground truth is independent NumPy exact L2. Primary recall@10 is distance-threshold recall: a returned result counts when its distance is no greater than the true tenth distance plus `2e-5`. ID equality is not required in float32 ties.

### Policy families

1. **Validation-selected fixed `nprobe`:** for each target recall `{0.90,0.95,0.98}`, select the smallest grid probe whose mean validation recall reaches the target; otherwise use `nprobe=100`. This is the comparator for every adaptive family.
2. **Ratio probing:** always probe the closest cell and every cell satisfying `d_centroid <= (1+ratio) * d_closest`. Candidate ratios are `{0.01,0.05,0.10,0.20,0.40}`. For each target, validation selects the eligible ratio with the smallest mean scan; ties use the smaller ratio. A config is eligible only if validation recall reaches the target.
3. **Productive doubling:** start at one probe. Compare the current candidate set with the set obtained by doubling the probe count. Continue only while the fraction of newly discovered candidate IDs relative to the current candidate count meets `{0.02,0.05,0.10,0.20}`, selected on validation by the same rule as ratio. Stop at the current probes otherwise; never exceed 100.
4. **Learned per-query probing:** train one `HistGradientBoostingRegressor` on the 300 train queries and all grid points, with squared error, 200 iterations, at most 15 leaves, learning rate 0.05, L2=1 and seed `20261203`. Features are probe count, log probe, probe fraction, candidate scan, log candidate scan, closest centroid distance, log closest distance, log ratios of the second/fourth/nearest centroid distances, and selected-cell centroid-distance median/max normalized by the closest. At inference, choose the smallest grid probe whose predicted recall reaches the target; otherwise use 100. The test set never participates in training, feature selection, model selection or comparator selection.

### Evaluation and verdict rules

- **H3 — implementation parity:** all 100 × 8 Faiss/mirror comparisons must have identical per-query distance-threshold recall and top-10 distance multisets within `2e-5`. ID differences in ties are allowed. Any other mismatch blocks cross-engine interpretation.
- **H4 — practical adaptive-policy value:** evaluate each family separately at all three targets. At each target, the validation-selected policy must (a) reach test recall within 0.5 pp of the comparator and the target, (b) reduce mean scan by at least 5% relative to that comparator, and (c) have bootstrap 95% CIs excluding non-inferiority violation and zero scan reduction. If the fixed comparator misses the target by more than 0.5 pp on test, that target is `INCONCLUSIVE_NO_VALID_COMPARATOR`, not a pass.
- Paired bootstrap: 10,000 resamples, seed `20261128`. Report actual recall, scan, latency, index bytes, build/add time and all validation selections. A family passes only when all three targets pass. The adaptive track is GO only if at least one predeclared family passes all targets; otherwise it is NO_GO or INCONCLUSIVE, without post-hoc target substitution.
- Python policy latency is harness timing, not a SplatsDB engine benchmark or a production QPS claim. Ratio/doubling curves are descriptive unless their predeclared H4 selection passes.

### Non-claims

Synthetic perturbed self-queries do not establish human semantic quality, calibrated uncertainty, agent-memory utility, cross-dataset superiority or a novel routing algorithm. No `α/κ` ranking or 3DGS render metric is introduced into this routing experiment.

## B. Numeric embedding atlas — Topo-GS-inspired, not a reproduction

### Frozen projections and structural metrics

- Fit PCA-to-3D (`randomized`, seed `20261203`) and UMAP-to-3D (`n_neighbors=30`, `min_dist=0.1`, `spread=1.0`, Euclidean, seed `20261203`, single job) on the same corpus.
- Structural evaluation uses 2,000 corpus points sampled without replacement with seed `20261202`. Compute exact high-dimensional and projected neighborhoods on that subset.
- Report kNN preservation at `k=10` and `k=50`, trustworthiness and continuity at the same `k`, Spearman correlation on 100,000 sampled non-self pairs (seed `20261204`), and scale-invariant stress `sqrt(sum((d_hi/mean_hi - d_lo/mean_lo)^2) / sum((d_hi/mean_hi)^2))`.
- No labels are available in the numeric corpus: class separation and user-task accuracy are `not_applicable`, not zero. No interactive-viewer benefit is claimed from projection metrics.

### Continuous splat rendering

- For each projection, map all points to a `128³` grid using per-axis 0.5th/99.5th percentile bounds and report the clipped fraction. Add point mass, apply an isotropic Gaussian filter with `sigma=1.0`, and render a maximum-intensity projection over the third axis.
- The PNG contains PCA scatter, PCA splat density, UMAP scatter and UMAP splat density at 1800×1400 pixels. It contains no text, IDs or query results.
- This is an isotropic splat-density prototype inspired by continuous-volume splatting, **not** Topo-GS's learned anisotropic representation or local-geometric optimization. The splat layer shares the projection coordinates; therefore it cannot improve kNN/trustworthiness/stress metrics. Only PCA-versus-UMAP differences are structural.
- Record monotonic projection/volume/render times, volume bytes, package versions, input/output hashes and the clipped fraction. A viewer is not justified unless a separate user-task study later shows value.

## Reproducibility

The protocol, common analyzer and both runners are committed and hashed before execution. Every public result is generated from measured JSON. Any later rule change requires a new protocol hash and an independent rerun; earlier verdicts remain visible.
