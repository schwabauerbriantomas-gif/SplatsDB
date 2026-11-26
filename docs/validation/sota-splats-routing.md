# SOTA review — Gaussian splats, routing and agent memory

- **Date:** 2026-11-26
- **Scope:** primary-source review requested for SplatsDB's technical positioning.
- **Companion measurement:** [`splat-routing.md`](splat-routing.md).

## Executive verdict

SplatsDB has a **narrow, reproducible routing result**, not a demonstrated state-of-the-art ANN or probabilistic-retrieval system. The local gate shows that learned KMeans centroids beat a random codebook at low scan budgets before exact re-scoring; the 10%-budget effect is +8.1 pp, while the smaller effects remain exploratory under the small-sample rule. Exact top-10 distances matched brute force, but exact search scanned essentially the whole 11,069-vector corpus. [Local receipts](splat-routing.md).

The broader literature establishes six boundaries:

1. Partitioning, learned codebooks, quantization and refinement are mature ANN components[1][2]. ScaNN explicitly separates reducing the number of scored items from improving the scoring rate, and contributes anisotropic quantization for that second task[1]. SPANN uses an inverted index with balanced posting lists and query-aware pruning in a memory-disk system[2]. Faiss provides established IVF, PQ and HNSW implementations[4]. **KMeans routing plus exact re-scoring is therefore not a novel algorithmic category.**[1][2]
2. Probabilistic embeddings are a representation-learning method, not a label obtained by attaching Gaussian parameters to a cluster[5]. PCME++ addresses ambiguous image-text correspondence with learned probabilistic embeddings and a closed-form probabilistic distance[5]. SplatsDB's current `α/κ` fields do not reproduce that training problem or establish calibrated uncertainty[5].
3. Agent-memory claims require end-to-end evaluation[8][12]. LongMemEval covers five abilities, including temporal reasoning, knowledge updates and abstention[8]. MemoryAgentBench evaluates accurate retrieval, test-time learning, long-range understanding and selective forgetting, and reports that current methods do not master all four[12]. Token reduction alone is not a sufficient product claim[12].

4. Closer prior art narrows the remaining gap. GARLIC learns anisotropic Gaussian cells[13]. LIRA/DARTH/QASP cover learned or adaptive probing[14][15][16]. CalibRAG calibrates against decision correctness rather than retrieval scores[18]. The open question is narrower and empirical, not a new retrieval category.

5. 3DGS contributes useful design patterns: LOD hierarchies, anchors and scale-aware filtering[22][23][25]. MCMC and compression add further capacity-management ideas[24][27]. They are hypotheses to test with retrieval metrics, not evidence that a vector index inherits rendering quality.

6. Alternative data representations are a separate legitimate track. Topo-GS maps high-dimensional data to continuous volumes[31], GraphSplatting turns graphs into continuous fields[33], and VEG separates scalar data from appearance[34]. Their proper target is visualization and structural fidelity, not ANN recall.

**Recommended positioning:** an integrated, self-hosted retrieval runtime with learned splat-cell routing, optional exact bounds and verbatim source access. Do not claim probabilistic uncertainty, universal ANN superiority or better agent outcomes until the corresponding experiments pass.

## 1. ANN routing, quantization and exact re-scoring

ScaNN describes two distinct system tasks: reduce the number of items scored through space partitioning, and improve the rate at which items are scored through quantization[1]. This distinction matters for SplatsDB because the new gate measures only candidate coverage[1]. It does not measure compressed-domain scoring, memory footprint, QPS or index-build cost.

SPANN stores posting-list centroids in memory and posting lists on disk, uses hierarchical balanced clustering, augments lists with nearby points, and applies query-aware pruning during search[2]. That is directly relevant prior art for any future adaptive or radius-aware probing proposal[2]. ANN-Benchmarks likewise separates results by distance measure and dataset, rather than declaring one algorithm universally best[3]. Faiss remains a necessary implementation baseline because it exposes established IVF, PQ and HNSW methods[4].

**Implication for SplatsDB:** the random-codebook control is useful for isolating the value of learning, but it is not enough to establish competitiveness[1][2][3]. The next baseline should be Faiss IVFFlat on the same vectors, queries, top-k and hardware, with `nprobe`, measured scan fraction, recall, latency, memory and build time reported together. A learned centroid win over random codebooks cannot be converted into a claim of beating established IVF systems[1][2].

## 2. What probabilistic embeddings actually require

PCME++ is motivated by ambiguity in image-text matching and improves a probabilistic cross-modal representation with a closed-form probabilistic distance, pseudo-positives and mixed-sample augmentation[5]. Its uncertainty is learned for a task with many-to-many and imperfect correspondences[5]. It is not evidence that arbitrary text-embedding clusters acquire semantic uncertainty when equipped with `α` and `κ`[5].

SplatsDB currently has three different things that must not be conflated:

- **Cell geometry:** `μ` for routing and `R` for triangle bounds.
- **Experimental Gaussian energy:** `α/κ` scoring, whose local ranking audit was negative.
- **Distance diagnostics:** HIGH/MEDIUM/LOW bands and `1 − d/R`, which are not learned correctness probabilities.

**Implication:** a probabilistic-embedding claim would require a training objective, an ambiguity or correctness target, and comparisons against a deterministic `μ` representation at matched model capacity[5]. A first experiment should keep the existing ANN index fixed and test only whether learned variance adds calibration or selective-retrieval value[5]. Renaming cluster-membership confidence as “uncertainty” would not be a technical advance[5].

## 3. Confidence, abstention and risk control

The Sufficient Context paper separates whether retrieved context is sufficient to answer from whether the model answers correctly[11]. Its selective-generation method combines sufficient-context information with model confidence and improves correct answers among answered queries by 2–10% for the evaluated Gemini, GPT and Gemma configurations[11]. The result is task- and protocol-specific; it does not validate a vector-distance threshold[11].

Conformal Risk Control extends conformal prediction to control the expected value of a monotone loss[7]. It supplies a statistical framework, not an automatic guarantee for an uncalibrated similarity[7]. Applying it responsibly requires a defined downstream loss, calibration data and the procedure's assumptions to be checked; exchangeability under temporal drift cannot simply be presumed[7].

**Implication for SplatsDB:** keep verbatim sources and distance bands as diagnostics, but calibrate any abstention policy on end-to-end correctness[11]. Compare fixed thresholds, calibration-only baselines and risk-controlled acceptance using selective risk-coverage curves, with temporal splits where memory updates matter[7][11]. A centroid-distance threshold should be a baseline, not the proposed solution[11].

## 4. Agent memory is an end-to-end problem

LongMemEval evaluates information extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention, with 500 curated questions in scalable chat histories[8]. Zep combines full-text, cosine and graph traversal over a temporal knowledge graph and reports improvements over its full-context baselines on LongMemEval; those are the authors' benchmark results, not independent SplatsDB measurements[9]. Mem0 reports gains and token savings on its LOCOMO protocol, also under its own evaluation design[10].

These papers do not establish a universal winner: datasets, baselines, models, latency accounting and memory-update regimes differ[8][9][10]. MemoryAgentBench makes the missing evaluation dimensions especially clear by testing incremental multi-turn memory across four competencies rather than retrieval alone[12].

**Implication for SplatsDB:** M3's prompt-token reduction is an efficiency result, not proof of a better memory system. A product-level evaluation needs, at minimum:[8][12]

- full-context and ordinary RAG baselines with the same generator;
- Faiss IVFFlat plus the same MCP/graph/source pipeline;
- answer accuracy, selective risk and coverage, temporal-update and forgetting tests;
- p50/p95 latency, prompt and total tokens, RAM/VRAM, index size, build and update cost.

Only an ablation that changes the splat routing while holding the rest fixed can attribute an outcome to the splat architecture[12].

## 5. Novelty and evidence matrix

| Area | Established prior art | SplatsDB today | Defensible contribution | Missing evidence |
|---|---|---|---|---|
| Learned cell routing | IVF/ScaNN/SPANN/GARLIC/LIRA partitioning, probing and refinement.[1][2][13][14] | Local GO vs random codebook; strongest evidence at low scan | Integrated implementation with aggregate receipts | Matched Faiss IVFFlat comparison and larger independent query set |
| HRM2 hierarchy | Coarse/fine partitioning belongs to established ANN practice.[1][2] | Implemented two-level path; not covered by the new flat-codebook gate | Alternative index organization | Separate matched-scan recall/latency/memory comparison |
| Adaptive probing | SPANN ratio pruning, DARTH early termination and QASP recall-curve prediction.[2][15][16] | Fixed `nprobe` measured; no new policy claim | Explicit computation control only | Ratio, productive-doubling and small learned-regressor controls at matched recall/scan |
| Exact cell bounds | Triangle-bound pruning is established geometry | Exact correctness pass; essentially full scan at 11k×384 | Optional exact backend and correctness reference | Scale/latency/memory envelope across datasets |
| Probabilistic embeddings | PCME++ learns distributions for ambiguous matching[5]. | `α/κ` persisted; energy ranking negative in A5 | No current probabilistic-performance claim | Task-trained distributions and calibration experiments |
| Abstention | Sufficiency, risk control, decision calibration and downstream relevance evaluation[7][11][18][19]. | Heuristic distance bands and source text | Source-inspection affordance inside the runtime | Outcome-supervised calibration and end-to-end selective risk-coverage evaluation |
| Agent memory | LongMemEval, Zep, Mem0 and MemoryAgentBench[8][9][10].[12] | M3 token reduction; MCP/graph integration | Single self-hosted runtime for experimentation | End-to-end task outcomes and update/forgetting tests |
| GPU/HNSW/graph | Mature independent implementations and algorithms[4]. | Existing separate components | Operational integration | No splat-specific attribution without ablation |

## 6. Closer prior art and sharpened novelty boundary

The extended primary-source review narrows, rather than expands, the novelty claim.

1. **Gaussian partitioning is established territory.** GARLIC learns anisotropic Gaussian cells, selects cells with Mahalanobis distance and prunes candidates. Its covariance ablation reports lower recall-per-probe as expressiveness falls from full to diagonal to isotropic[13]. It is a Euclidean preprint, not evidence for normalized text embeddings; the authors explicitly leave von Mises–Fisher/angular counterparts for future work[13]. A low-rank or vMF experiment on angular text embeddings is therefore a testable gap, not a novelty claim.
2. **Adaptive probing is also established.** SPANN prunes posting lists using their distance relative to the closest centroid[2]. LIRA learns per-query partition probing[14], DARTH stops search using a learned recall predictor[15], and QASP predicts a per-query recall-progression curve by supervised regression[16]. “Adaptive” or “learned routing” alone is not a contribution. The narrower opening is a training-free closed-form per-splat yield/stop rule using `μ/α/κ` that beats fixed `nprobe`, the SPANN ratio, productive-cluster doubling and a small learned regressor at matched recall, scan and memory.
3. **Second-order parameters still need a task.** PCME learns distributions for ambiguous cross-modal correspondence[17]; PCME++ keeps mean-only retrieval strong and makes probabilistic scoring a separate design choice[5]. These results do not turn cluster spread into semantic uncertainty, and they are consistent with the negative local A5 energy-ranking result.
4. **The calibration target is decision correctness.** CalibRAG states that normalized reranker scores are not calibrated probabilities of correctness and trains an outcome-supervised forecaster[18]. On its HotpotQA/Llama-2-7B configuration it reports ECE 0.106 versus 0.477 for a cross-encoder and 0.248 for LLM reranking; those are protocol-specific results, not SplatsDB expectations[18]. eRAG separately finds only a small correlation between query-document relevance labels and downstream RAG performance[19]. A future confidence head therefore needs outcome labels, held-out and shift evaluation, and selective risk/coverage—not another distance band.
5. **Exactness must beat an exact scan.** PDX shows that dimension-pruning methods can lose to SIMD-optimized linear scans, while data layout changes the comparison[20]. SplatsDB's exact field route must beat an exact SIMD scan at recall 1.0 before any speed claim is defensible.

**Minimum fair-control package:** use the same vectors, held-out queries, top-k, kernels and thread counts; sweep list count; report recall versus distance computations, p50/p95 wall time, bytes and build/update cost[2][15][20]. Count covariance storage explicitly, and compare anisotropic/vMF routing against the isotropic control[13]. For confidence, compare raw cosine, the current band, a CalibRAG-style outcome-supervised head and a shuffled-signal control[18]. No new policy is justified unless it beats fixed probes and the ratio/doubling/learned controls[2][15][16].

## 7. 3D Gaussian Splatting: transferable methods and non-transfers

SplatsDB shares the Gaussian-splat representation with 3DGS, but it does not reconstruct a visual scene. The useful transfer is the design principle, never the rendering score: photometric error, camera visibility and vector retrieval quality are different objectives[21].

| 3DGS line | What the source establishes | Transferable question for SplatsDB | What does not transfer |
|---|---|---|---|
| Original 3DGS[21] | Anisotropic covariance, interleaved optimization/density control and visibility-aware rendering | Can anisotropic cells and selective computation improve candidate generation? | Camera visibility and rendered image quality are not retrieval evidence; exact raw-vector re-scoring remains necessary |
| LOD hierarchies[25][26] | Multi-resolution hierarchies and anchor levels selected for large scenes | Can HRM2 choose coarse/fine levels against a declared recall/scan target? | View distance is not query difficulty; level selection must use retrieval measurements |
| Scaffold-GS[23] | Anchor points generate local Gaussians and their attributes | Can anchored residual cells reduce index bytes without losing recall? | A view-conditioned MLP and its rendering gains do not establish compact semantic retrieval |
| Density control[24][30] | Error-driven densification and MCMC-style relocation/pruning | Should cells split, merge or relocate based on held-out retrieval errors at fixed memory? | Pixel error and opacity are not relevance; uncontrolled growth can consume the scan budget |
| Scale/anti-aliasing[22] | Scale-aware 3D smoothing and 2D Mip filters | Does bandwidth/scale normalization help across local densities or embedding-model changes? | There is no camera sampling frequency in semantic space; every new embedding model needs a rebuilt, recalibrated index |
| Compression[27][28] | Significance-based pruning/VQ and KMeans parameter codebooks | Can metadata or learned residuals be compressed while preserving recall and exact re-scoring? | Rendering compression ratios and FPS do not transfer; raw-vector storage must be counted separately |
| 2DGS geometry[29] | Oriented planar disks with depth/normal consistency for surfaces | Is a specialized representation useful only for genuine spatial/GIS/3D workloads? | A semantic vector database has no view-consistent surface to reconstruct |

### 3DGS-inspired experiment queue — proposed, not implemented

1. **Error-driven cell management at fixed memory.** Starting from the same flat `k=100/200` codebooks, split cells with high held-out boundary-miss contribution and merge low-yield overlapping cells. Compare against fixed KMeans and the matched random control at equal cell/vector bytes; report recall@10, scan fraction, build time and p50/p95. Try stochastic MCMC-style relocation only if a deterministic error-driven variant is insufficient, because stochasticity adds another source of variance[24][30].
2. **Recall-driven LOD.** Select HRM2 levels using measured or predicted recall, not a rendering-distance proxy, and compare against flat cells and adaptive `nprobe` at the same scan budget[25][26].
3. **Embedding-model shift test.** Measure whether scale/bandwidth normalization improves robustness across local densities. A new embedding model requires a new index and calibration set; never query the old index with incompatible vectors[22].
4. **Compression as a separate gate.** Measure codebook, residual, metadata and raw-vector bytes together. Do not import LightGaussian/CompGS compression ratios or claim compression from the routing result alone[27][28].

## 8. Alternative data representations and rendering-only methods

A second legitimate research track is to make data structure interpretable and interactive rather than to accelerate nearest-neighbor search[31][33]. Rendering-only methods apply directly, but the evaluation target changes from recall/QPS to structural fidelity, geometric error, visual clutter and user-task performance[32][34].

| Data or representation | Primary-source prior art | Transferable idea | Correct evaluation target |
|---|---|---|---|
| High-dimensional embeddings and manifolds | Topo-GS recasts multidimensional projection as meshless volumetric reconstruction, uses local geometric constraints instead of photometric loss and reports local-topology fidelity comparable to discrete baselines[31] | Continuous splat volumes may expose density, overlap and projection distortion hidden by scatterplots | Neighborhood/topology preservation, distance correlation and projection stress versus scatter/t-SNE/UMAP baselines; never ANN recall inferred from a picture |
| Multivariate scattered or scientific data | Rapp et al. summarize clusters with low-dimensional GMMs and splat anisotropic 3D Gaussians; VEG decouples scalar data from appearance for transfer-function-agnostic volume rendering[32][34] | Probabilistic summaries, LOD, outlier views and reusable data fields rather than baked RGB | Reconstruction and interactive-task error, transfer-function coverage, memory, frame time and outlier visibility |
| Knowledge/code/memory graphs | GraphSplatting transforms a graph into a continuous scalar density field using vertex contributions[33] | Continuous density fields may make repositories, modules and chat/memory relationships navigable without node-link clutter | Relationship-finding accuracy/time, clutter and graph-task error against node-link and matrix views |
| Spatial data with semantic features | Quantile Rendering samples dominant-influence Gaussians for high-dimensional feature maps in 3D scenes; 2DGS targets view-consistent surfaces[29][35] | Open-vocabulary/GIS queries can combine spatial splats with semantic features | Held-out spatial segmentation/retrieval plus geometric error; do not infer semantic-index quality from rendering speed |

### Proposed alternative-representation experiments — not implemented

1. **Embedding atlas, local and privacy-preserving.** Export only the existing 11,069 × 384 numeric corpus and approved aggregate labels; keep text and identifiers private. Compare scatter/PCA/t-SNE/UMAP against a Topo-GS-style continuous-volume variant[31]. Predeclare kNN preservation, trustworthiness/continuity, pairwise-distance correlation, projection stress, memory and frame time[31]. Add a small user task only after numerical baselines are complete; a visually attractive render is not evidence of better retrieval.
2. **Spatial CampoGrato/GIS track.** For genuine drone/point-cloud/surface data, compare 3DGS, 2DGS and mesh/LOD baselines with point-to-point/point-to-surface error, normal consistency where applicable, map/task error and resource use[29][35]. This is a separate spatial-rendering project, not a SplatsDB recall benchmark.
3. **Graph-density track.** Build a GraphSplatting-style density field from an approved aggregate knowledge/code/memory graph and test whether users find cross-module or cross-session relationships faster and more accurately than with node-link or matrix views[33]. Preserve the original graph for verification; a density image cannot replace provenance.

## 9. Three-category product assessment

### Validated locally

- Learned `k=100` centroids outperformed a random codebook at the predeclared low-scan endpoint; the +8.1 pp result exceeds the conservative 5 pp small-sample floor. The other frozen-CI passes are smaller and exploratory. [Receipts](splat-routing.md).
- Exact field mode matched brute-force top-10 distances on 100/100 queries; the Python mirror matched the real HTTP route in all 800 comparisons. [Receipts](splat-routing.md).
- The A5 audit falsified the claim that the tested Gaussian-energy fusion improves semantic retrieval. [A5](../A5_FINDINGS.md).

### Falsified or not supported

- **Gaussian energy as a default ranking improvement:** falsified in the local A5 setup.
- **`confidence` as probability of a correct answer:** unsupported; it is a distance diagnostic.
- **SOTA ANN throughput or recall:** unsupported; no matched Faiss/ScaNN/SPANN benchmark exists.
- **Exact-search speedup at 11k×384:** unsupported; the measured exact route scanned 99.9988% of vectors.
- **Better agent answers or hallucination prevention:** not established by token reduction, MCP or verbatim storage.
- **Compression from the routing result:** unsupported because raw member vectors remain available for re-scoring.

### Potential, not demonstrated

| Direction | Utility if it works | Cost | Cheap falsification path |
|---|---|---|---|
| Adaptive query-aware probing | Better recall at lower scan than fixed `nprobe` | Medium: policy and tuning complexity | Compare against fixed `nprobe`, SPANN ratio, productive-cluster doubling and DARTH/QASP-style learned controls at matched recall/scan |
| Calibrated selective retrieval | Lower selective error at controlled coverage | Medium: labels, calibration and drift handling | Start with downstream calibration; no new index until signal exists |
| Learned probabilistic codebooks | Task-specific ambiguity modelling | High: training data, objectives and capacity-matched controls | Test learned variance against deterministic `μ` on a task with real ambiguity |
| 3DGS-inspired cell management | Better recall per byte or scan at fixed capacity | Medium-high: held-out error attribution and stochastic controls | Compare error-driven split/merge/relocation against fixed KMeans at matched memory before adding MCMC or anchors |
| Alternative data visualization | Better inspection of embeddings, graphs, spatial/scientific data | Medium: projection, rendering and user-task evaluation | Compare against scatter/UMAP or node-link baselines using topology, geometry and task metrics before investing in a viewer |
| Integrated agent-memory runtime | Operational simplicity and reproducible ablations | Medium-high: systems and evaluation work | Hold generator/pipeline fixed and compare full-context, vanilla RAG and Faiss-based variants |
| Exact-bound pruning at scale | Exactness with less compute on favorable geometry | Medium: corpus/hardware matrix | Beat an exact SIMD scan at recall 1.0 before claiming speed; PDX shows layout can invalidate pruning gains[20] |

## 10. Recommended sequence

1. **Run the matched IVF and probe-policy baselines before any new algorithm.** Use the same k-means cells and SIMD kernels, then add the SPANN ratio, productive-cluster doubling and a small learned regressor. This is the highest-information, lowest-cost next step and can falsify the remaining differentiator.
2. **Expand independent semantic queries.** Use human-intent queries rather than lexical paraphrases, increase the query count and report paired deltas with intervals and the noise floor.
3. **Calibrate downstream, not at the centroid.** Compare raw cosine and the current band with an outcome-supervised correctness head on held-out and shifted queries; report selective risk/coverage before calling any field uncertainty.
4. **Evaluate adaptive probing and angular cells as controlled experiments.** Do not add complexity unless a policy beats fixed-`nprobe`, ratio/doubling and learned controls, or an angular variant beats isotropic routing at matched scan and memory.
5. **Test error-driven cell management before adding hierarchy.** Use held-out retrieval errors at fixed memory and compare against fixed KMeans; only then evaluate recall-driven LOD or MCMC-style relocation.
6. **Run an end-to-end memory ablation.** Only then make an agent-memory utility claim.
7. **Treat alternative data visualization as a separate track.** Start with a numeric-embedding atlas or a real spatial dataset and predeclared topology/geometry/task metrics; do not fold its results into the routing claim.

No new routing algorithm is justified by this review alone. The necessary new code for the current step was the reproducible analyzer, independent HTTP mirror, raw receipts and report generator—not a novelty feature introduced before the missing baselines exist.

## 11. Source quality and research limits

| Source | Status | Reading depth used here |
|---|---|---|
| ScaNN [1] | ICML 2020 | Full text; architecture and loss sections |
| SPANN [2] | NeurIPS 2021 | Full text; index and query-aware pruning design |
| PCME++ [5] | ICLR 2024 | Full text; motivation and probabilistic-distance formulation |
| Sufficient Context [6][11] | ICLR 2025 | Official proceedings plus full text; sufficiency definition and selective-generation results |
| Conformal Risk Control [7] | arXiv v4, 2025 | Abstract; used only for the framework claim |
| LongMemEval [8] | ICLR 2025 | Abstract; benchmark abilities and size |
| Zep [9] | arXiv preprint, 2025 | Full text; architecture and reported benchmark tables |
| Mem0 [10] | arXiv preprint, 2025 | Abstract; attributed claims only |
| MemoryAgentBench [12] | ICLR 2026 | Full text; competencies and incremental evaluation design |
| GARLIC [13] | arXiv preprint, 2025 | Abstract, method, covariance ablation and limitations; Euclidean only |
| LIRA [14] | WWW 2025 | Abstract and probing-model design; distance-computation regime |
| DARTH [15] | SIGMOD 2026 | Abstract and recall-predictor design; requires search traces |
| QASP [16] | arXiv v3, 2026 | Abstract and feature model; recent supervised policy, not independently reproduced |
| PCME [17] | CVPR 2021 | Abstract and probabilistic representation motivation; cross-modal regime |
| CalibRAG [18] | NeurIPS 2025 | Abstract, calibration method and reranker comparison; values tied to its protocol |
| eRAG [19] | arXiv, 2024 | Abstract; downstream-relevance correlation claim |
| PDX [20] | SIGMOD 2025 | Abstract; exact/pruning layout comparison, not a SplatsDB benchmark |
| Original 3DGS [21] | SIGGRAPH/ACM TOG 2023 | Abstract; representation and density-control foundation |
| Mip-Splatting [22] | CVPR 2024 | Abstract; scale-aware filtering, not a retrieval experiment |
| Scaffold-GS [23] | CVPR 2024 | Abstract; anchor-generated local Gaussians |
| 3DGS-MCMC [24] | NeurIPS 2024 | Abstract; SGLD and deterministic densification/pruning transition |
| Hierarchical 3DGS [25] / Octree-GS [26] | CVPR 2024 / arXiv v2 | Abstracts; LOD design and multi-resolution anchors |
| LightGaussian [27] / CompGS [28] | NeurIPS 2024 / ECCV 2024 | Abstracts; significance pruning and VQ for rendering parameters |
| 2DGS [29] / Revising Densification [30] | SIGGRAPH 2024 / ECCV 2024 | Abstracts; surface representation and error-driven density control |
| Topo-GS [31] | arXiv preprint, 2026 | Abstract/introduction; high-dimensional projection as continuous volumes, not independently reproduced |
| Rapp et al. [32] | IEEE TVCG 2021; author version 2020 | Abstract; probabilistic multivariate summaries and splatting |
| GraphSplatting [33] | Eurographics/IEEE VGTC 2001 | Primary paper; graph-to-continuous-field visualization |
| VEG [34] | arXiv v2, 2026 | Abstract; scalar-data/transfer-function separation for scientific volumes |
| Quantile Rendering [35] | arXiv preprint, 2025 | Abstract; high-dimensional feature rendering in 3D scenes |
| ANN-Benchmarks [3] / Faiss [4] | Official benchmark / official documentation | Live result structure and implementation scope |

- This is a targeted technical review, not an exhaustive systematic literature review or an independent reproduction of cited benchmark scores.
- Recent preprints and vendor-authored evaluations require extra caution; no cited result is treated as a SplatsDB measurement. GARLIC and QASP are especially recent, and their regimes are kept explicit.
- The attempted LongMemEval-V2 page was rate-limited (HTTP 429) and is excluded from the findings.
- No source found supports the claim that a KMeans-derived `α/κ` or `1 − d/R` is calibrated correctness uncertainty.
- 3DGS sources are used as design-pattern evidence only. Alternative-representation work is evaluated separately by topology, geometry, visualization and user-task metrics; no PSNR/SSIM/FPS result is claimed as a database result.
- Local conclusions remain limited to one 11,069×384 corpus, deterministic perturbation queries and 100 queries per gate run.

## Sources

[1] https://arxiv.org/html/1908.10396v4
    > "One task is to reduce the number of items that are scored to identify the top result. This is typically done with a space partitioning method. The other task is improving the rate at which items are scored. This is typically done with quantization, and is where the main contribution of our work lies."
[2] https://proceedings.neurips.cc/paper_files/paper/2021/file/299dc35e747eb77177d9cea10a802da2-Paper.pdf
    > "In the search stage, we use a query-aware scheme to dynamically prune the access of unnecessary posting lists."
    > "Instead of searching closest K posting lists for all queries, we dynamically decide a posting list to be searched only if the distance between its centroid and query is almost the same as the distance between query and the closest centroid."
[3] https://ann-benchmarks.com
    > "Results are split by distance measure and dataset."
[4] https://github.com/facebookresearch/faiss/wiki
    > "Faiss is a library for efficient similarity search and clustering of dense vectors."
[5] https://arxiv.org/html/2305.18171
    > "Deterministic functions are not sufficiently powerful to capture ambiguity, prompting the exploration of probabilistic embeddings to tackle the challenge."
[6] https://proceedings.iclr.cc/paper_files/paper/2025/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf
    > "Our method improves the fraction of correct answers among times where the model responds by 2–10% for Gemini, GPT, and Gemma."
[7] https://arxiv.org/abs/2208.02814
    > "We extend conformal prediction to control the expected value of any monotone loss function."
[8] https://arxiv.org/abs/2410.10813
    > "With 500 meticulously curated questions embedded within freely scalable user-assistant chat histories, LongMemEval presents a significant challenge to existing memory systems, with commercial chat assistants and long-context LLMs showing a 30% accuracy drop on memorizing information across sustained interactions."
[9] https://arxiv.org/html/2501.13956
    > "Using gpt-4o-mini, Zep achieved a 15.2% accuracy improvement over the baseline, while gpt-4o showed an 18.5% improvement."
[10] https://arxiv.org/abs/2504.19413
    > "Notably, Mem0 achieves 26% relative improvements in the LLM-as-a-Judge metric over OpenAI, while Mem0 with graph memory achieves around 2% higher overall score than the base configuration."
[11] https://ar5iv.labs.arxiv.org/html/2411.06037
    > "Our method improves the fraction of correct answers among times where the model responds by 2–10% for Gemini, GPT, and Gemma."
[12] https://proceedings.iclr.cc/paper_files/paper/2026/file/fd1eff9dd295df50a41f2521942fa31d-Paper-Conference.pdf
    > "Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents."
[13] https://arxiv.org/html/2505.24608
    > "GARLIC employs Mahalanobis distance and Gaussian primitives, which assume a Euclidean metric and are therefore not directly compatible with angular similarity. This may be addressed by adopting angular counterparts, such as von Mises–Fisher distributions, which we leave for future work."
    > "As shown in Table 2, reducing the Gaussian expressiveness from full to diagonal and then to isotropic leads to a notable decline in average Recall@1 per probe."
[14] https://arxiv.org/html/2503.23409
    > "This can reduce probing waste and allow for query-aware probing with nprobe individually."
[15] https://arxiv.org/html/2505.19001
    > "DARTH uses a novel method for providing target declarative recall on top of an ANNS index by employing an adaptive early termination strategy integrated into the search algorithm."
[16] https://arxiv.org/html/2607.29606v3
    > "predicts the complete recall progression curve per query via a single upfront supervised regression, from which a search policy is derived for any recall target"
[17] https://arxiv.org/abs/2101.05068
    > "Instead, we propose to use Probabilistic Cross-Modal Embedding (PCME), where samples from the different modalities are represented as probabilistic distributions in the common embedding space."
[18] https://arxiv.org/html/2411.08891
    > "Thus, the resulting scores, although often normalized between 0 and 1, are not calibrated probabilities of correctness and cannot be directly used for decision calibration."
    > "HotpotQA Cross-encoder 60.74 34.98 0.477 0.477 LLM-rerank 60.57 38.52 0.248 0.297 CalibRAG 72.47 42.37 0.106 0.206"
[19] https://arxiv.org/abs/2404.13781
    > "evaluation of the retrieval model's performance based on query-document relevance labels shows a small correlation with the RAG system's downstream performance."
[20] https://arxiv.org/abs/2503.04422
    > "We found that these algorithms on the horizontal vector layout can lose to SIMD-optimized linear scans, even if they are SIMD-optimized."
[21] https://arxiv.org/abs/2308.04079
    > "we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene"
[22] https://arxiv.org/abs/2311.16493
    > "we introduce a 3D smoothing filter which constrains the size of the 3D Gaussian primitives based on the maximal sampling frequency induced by the input views"
[23] https://arxiv.org/abs/2312.00109
    > "We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum."
[24] https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html
    > "We then rewrite the densification and pruning strategies in 3D Gaussian Splatting as simply a deterministic state transition of MCMC samples, removing these heuristics from the framework."
[25] https://arxiv.org/abs/2406.12080
    > "We introduce a hierarchy of 3D Gaussians that preserves visual quality for very large scenes, while offering an efficient Level-of-Detail (LOD) solution for efficient rendering of distant content with effective level selection and smooth transitions between levels."
[26] https://arxiv.org/abs/2403.17898
    > "Our model dynamically selects the appropriate level from the set of multi-resolution anchor points, ensuring consistent rendering performance with adaptive LOD adjustments while maintaining high-fidelity rendering results."
[27] https://arxiv.org/abs/2311.17245
    > "Gaussian Vector Quantization, based on each Gaussian's global significance, further lowers bitwidth with minimal accuracy loss."
[28] https://arxiv.org/abs/2311.18159
    > "we introduce a simple vector quantization method based on K-means algorithm to quantize the Gaussian parameters"
[29] https://arxiv.org/abs/2403.17888
    > "Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks."
[30] https://arxiv.org/abs/2404.06109
    > "Our main contribution is a more principled, pixel-error driven formulation for density control in 3DGS, leveraging an auxiliary, per-pixel error function as the criterion for densification."
[31] https://arxiv.org/html/2605.17011
    > "we introduce Topo-GS, a framework that repurposes 3D Gaussian Splatting (3DGS) to cast multidimensional projection as a meshless volumetric reconstruction process."
    > "Instead of standard photometric losses, Topo-GS is driven by local geometric constraints."
    > "Quantitative and visual evaluations demonstrate that Topo-GS successfully transforms discrete scatter plots into continuous volumetric representations, where inherent projection distortions explicitly manifest as observable geometric variations, while preserving local topological fidelity comparable to discrete baselines."
[32] https://arxiv.org/abs/2008.09544
    > "we represent high-dimensional distributions by combinations of low-dimensional Gaussian mixture models."
    > "Furthermore, we propose a spatial visualization by splatting anisotropic 3D Gaussians for which we derive a closed-form solution."
[33] https://homepages.cwi.nl/~robertl/papers/2001/splat/paper.pdf
    > "GraphSplatting is a technique that transforms a graph into a continuous field."
[34] https://arxiv.org/abs/2504.13339
    > "VEG decouple the visual appearance from the data representation by encoding only scalar values, enabling transfer-function-agnostic rendering of 3DGS models for interactive scientific visualization."
[35] https://arxiv.org/abs/2512.20927
    > "Unlike conventional volume rendering, which densely samples all 3D Gaussians intersecting each ray, Q-Render sparsely samples only those with dominant influence along the ray."
