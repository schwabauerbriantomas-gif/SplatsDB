# Gaussian splat routing gate — predeclared protocol

**Status:** protocol frozen before running the new routing measurements.
**Scope:** current SplatsDB 2.11.0 binary, real cached MiniLM embeddings, CPU only.

## Hypotheses

- **H1 — exact field correctness.** Default `field-search` (triangle bounds, no `nprobe`) must return the exact L2 top-10 against independent NumPy brute-force ground truth. Float32 distance ties may reorder equivalent neighbors but must not change the distance multiset. This is a correctness gate, not a speed claim.
- **H2 — learned routing adds value.** At matched vector-scan budgets of 10%, 20% and 40%, two-phase routing with the engine's learned KMeans codebook must exceed a random codebook control in paired recall@10. The control uses randomly selected corpus centroids, the same nearest-centroid assignment, the same CSR two-phase algorithm and exact re-scoring.

## Frozen design

- Corpus: 11,069 × 384 float32, normalized `all-MiniLM-L6-v2` embeddings from the existing local A5 cache (the current cache contains 11,069 rows; older A5 prose reports 9,533). No texts or embeddings are published; only aggregate metrics and hashes.
- Primary codebook: `k=100` (approximately `sqrt(N)`); engine `ingest`, seed 42. A `k=200` run is sensitivity only and cannot rescue a failed primary gate.
- Queries: 100 corpus vectors selected without replacement with seed 20261126; add isotropic Gaussian noise `sigma=0.05`, then L2-normalize. Ground truth is exact L2 top-10 computed independently with NumPy.
- Probe grid for each codebook: `nprobe ∈ {1,2,4,8,16,32,64,k}`.
- For each query and budget, select the smallest probe count whose scanned-vector fraction reaches that budget. Report the actual scanned fraction alongside recall; never silently normalize it away.
- Random control: `k` distinct corpus vectors chosen with seed 20261127 as centroids; deterministic nearest-centroid assignment and identical two-phase candidate re-scoring.
- Paired statistic: per-query recall@10 difference, learned minus random, at each budget. Bootstrap 10,000 resamples, seed 20261128, percentile 95% CI.
- No post-hoc substitution of budgets, probe grids, seeds, codebooks or primary endpoints.

## Verdict rules

- H1 passes only with exact top-10 distance parity for every query; otherwise correctness fails and no benefit claim is allowed.
- H2 requires **all three** primary-budget deltas to be positive with bootstrap 95% CIs excluding zero. A CI touching zero is inconclusive for that budget; a non-positive effect with a CI excluding zero falsifies that budget. Mixed results are reported per budget, not collapsed into a promotional GO.
- The broad learned-routing benefit is **GO** only if H1 passes and all three H2 budgets pass. Otherwise it is NO_GO or INCONCLUSIVE according to the observed CIs.
- CLI latency is not a reported benefit: each CLI invocation reloads the store, so wall time is not an in-process search benchmark.

## Explicit non-claims

- Perturbed self-queries do not measure human semantic retrieval quality, agent task success, answer correctness or calibrated confidence.
- α/κ Gaussian energy as a ranking signal is not rehabilitated by this experiment. The earlier A5 audit remains negative.
- `topic`/`confidence` fields based on distance-to-centroid are membership diagnostics, not probabilities that a result is correct.
- MCP, graph, HNSW, GPU and M3 are integration components, not evidence of a Gaussian-splat benefit unless an experiment isolates that mechanism.
