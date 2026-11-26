# Changelog

## [Unreleased]

### Added
- **validation**: frozen H1/H2 splat-routing protocol, independent HTTP mirror, aggregate JSON results, raw-receipt hashes, report generator and extended 35-source primary-source SOTA, 3DGS and alternative-representation review. The learned `k=100` codebook beats a random control at the frozen endpoints; only the +8.1 pp low-scan effect clears the conservative 5-point small-sample floor. Exact top-10 distance parity is 100/100, with no material pruning at 11k×384.
- **validation**: LongMemEval-S scope ladder on frozen MiniLM embeddings — haystack-restricted vs one global memory — plus engine parity receipts. The current engine's haystack-restricted path returns 484/500 (reference 483/500) and its unfiltered exact path equals NumPy (167/500); chunk-max global 43.0% loses to keyword search 54.6% at hit@10. The April native 0.304 is shown to match an unfiltered search, not an engine gap. The report is generated from the receipt with a sync check.
- **validation**: frozen matched Faiss IVF policy gate and embedding-atlas harness. Faiss and the NumPy mirror agree on 800/800 recall, distance and scan comparisons; ratio, doubling and learned probing are all **NO_GO** under the predeclared rule. The PCA/UMAP atlas publishes structural metrics and density-render costs without claiming user-task or ANN quality. Harness incidents and invalid artifacts are preserved in the incident register.
- **docs**: README repositioned around learned splat-cell routing, exact raw-vector re-scoring and explicit non-claims; MCP/graph/GPU integration is separated from splat-specific evidence.

### Changed
- **MCP/verbatim**: HIGH/MEDIUM/LOW outputs are described as heuristic vector-distance bands, not calibrated correctness probabilities or hallucination prevention. Additive `close_or_moderate` and `diagnostic_only` fields clarify the summary; legacy `reliable` remains only for compatibility.
- **bench**: `bench-long-mem-eval` gains `--scope wing|global` and `--dump`. Default behaviour and legacy JSON keys are unchanged; the wing scope is documented as the benchmark shortcut it is.
- **docs**: README and BENCHMARKS label the 96.6% LongMemEval figure as haystack-restricted and record the 2026-11-26 re-measurement next to the unchanged 2026-04-09 numbers.

### Fixed
- README documented a non-existent `bench-longmemeval` subcommand (the CLI name is `bench-long-mem-eval`), and `benchmark_reproducible.py::run_longmemeval_benchmark` returned hardcoded numbers with `code_reproducible: True` while carrying throughput figures withdrawn on 2026-10-25. It is now an explicit placeholder that reads the stored result files and measures nothing; the CLI no longer prints its integrity banner over a run in which nothing was measured.

## [v2.11.0] - 2026-11-22

### Added
- **api/serve**: persistencia por chunks del store cuando hay `--data-dir`. Cada `/store` agrega una entrada a un WAL append-only propio (`{data_dir}/wal/store.wal`, formato length-prefixed JSON con `id/auto_id/idx/vector`), escrito **bajo el lock del store** para que el orden del WAL replique exactamente el orden de índices. El WAL hace flush por store (un `kill -9` no pierde nada) y checkpoint con fsync cada 32 stores (protege contra corte de luz). Cada 1000 stores se compacta: shard `"default"` completo a SQLite (`data/vectors/default.bin`, mismo formato que el CLI vía `save_vectors`) + truncate crash-safe del WAL. Boot: shard + replay de las entradas post-compact (dedup por `idx`, seeds `next_id` con el máximo `auto_id`). Shutdown graceful en SIGINT/SIGTERM (`with_graceful_shutdown`): persiste shard y trunca WAL antes de soltar el listener — un restart re-levanta el dataset entero; un `kill -9` pierde a lo sumo lo no flusheado (con flush por store: nada), no el dataset entero (antes: TODO, y `ingested.json` externo quedaba envenenado). Drop de `Durability` flushea (panic/exit sin TERM).
- **AppState**: campo `durability: Option<Arc<Durability>>` (WAL + persistence + contadores). None en tests sin data-dir → comportamiento in-memory clásico intacto.
- **tests**: `store_survives_restart_via_wal` (crash sin compact → replay idéntico, orden de índices y continuidad de `next_id`) y `compact_persists_shard_and_wal_replay_completes_it` (shard+post-compact → reaplicación exacta de índices y distancias). Helper `test_app_with_durability` espeja el boot de `run_server`.

### Fixed
- `api_server.rs`: doc-comment de `field_cmds::persist_field` referenciaba un call site inexistente — la ruta `/store` del server HTTP jamás persistió a disco (el MCP server sí persiste docs; el HTTP no). Con v2.11.0 el hueco queda cerrado.



### Added
- **api**: `POST /recap` — recap SAC-style de handoff, LLM-free (ver módulo `src/recap.rs`). Recibe `{objective, actions[], max_words}` inline (stateless: sin store, sin mutex) y devuelve un recap determinista por extracción: clasificación por heurística (changes/verification/dead-ends/facts) + ranking TF-IDF contra objetivo y acciones recientes + `semantic_compress`, con trim post-render que garantiza el contrato de palabras (max 397/400 sobre 30 recaps reales). 8 tests nuevos (6 unit + 2 router). Latencia: 9.1ms server mediana, p90 13.3ms sobre 30 recaps de trayectorias reales.
- **semantic-hub**: `sac_recap.py` — cliente para compactar el estado de una sesión Hermes: lee state.db en RO (copia a /tmp), arma el action log y pide el recap al hub. Es la variante agéntica validada en sac_falsif.

### Measured
- A/B en el harness sac_falsif (10 bugs reales de `requests`, GLM-5.3-flash, 40 runs): brazo SACDB (recap por /recap en cada acción) 10/10 con wall mediana **162s** vs SAC-LLM 689s (4.45×) y FHA 323s (3.06×); contexto pico 2.7K tokens; overhead de recap 4.7s TOTAL en ~450 compactaciones. Detalles: `~/sac_falsif/STATE.md`.

### Fixed
- `emeans.rs`: 3 `needless_range_loop` en el fixture de tests (clippy nuevo); mismos valores, mismo orden de sampling del RNG.

## [v2.8.0] - 2026-11-13

### Added
- **clustering**: trainer e-means (port del faiss contrib `e_means`, Meta FAIR arXiv:2611.09501) como alternativa a Lloyd para el fit jerárquico HRM2. Módulo nuevo `src/emeans.rs` (~390 LOC CPU): init por filas aleatorias, batch law Ec.7 por dimensión (quality d≤16 / throughput d>16), step law Ec.6 con clamp [1e-4,1], annealing coseno a 0.1·α₀, EMA por masa, reactivación de clusters muertos cada 5 updates (split ∝ masa, Apéndice J) y finalize con medias no ponderadas. Unit tests con los valores de referencia verificados 1:1 contra `base.py` de FAISS real (batch=32, α₀=0.270245 para n=248, k=31, T=5, d=768). KMeans::fit_with(TrainerKind) despacha lloyd (histórico, default) o emeans; `--trainer lloyd|emeans` global en el CLI; campo `cluster_trainer` en config con serde default (configs viejas siguen parseando). HRM2 coarse y fine respetan el flag. Determinista por seed.

### Measured
- Standalone (port, CPU single-thread, hub real 5.871×768, coarse k=16, fine k≤64, 5 réplicas): Lloyd-100/50 1.269s / loss 0.1713 vs e-means-5/5 0.864s (−31.9% t, gate ≤0.75×) / loss 0.1703 (−0.56%, mejor en mediana y mínimo). Contra-caso conocido: blobs/GMM bien separados pierde hasta +51.7% de inercia (documentado en tests).
- E2E in-repo (serve --gpu sobre copia del hub, 5 arranques fresh por trainer, tiempo a /health): Lloyd mediana 1.083s vs e-means 0.939s (−13.3% startup). Gate /search: top-10 idéntico con ambos trainers (HNSW+GPU exact intocables). El fit corre en CADA arranque de serve (splats.rs build_index_if_needed lo ejecuta aunque HNSW esté cacheada); default `lloyd` mantiene el comportamiento de producción hasta que se flipa el flag.

### Security
- **deps**: rusqlite 0.31 → 0.41.1 / libsqlite3-sys 0.28 → 0.39.1 (SQLite bundled 3.54.1) — RUSTSEC-2026-0459 / CVE-2026-119375: SQLite < 3.54.0 puede devolver memoria no inicializada del page-cache al abrir una DB con un WAL hermano de page size distinto (vector local, CVSS:4.0 AV:L/VC:H). La advisory entró al RUSTSEC DB el 13-nov-2026 13:33 UTC; fue la primera CI que la vio. Único cambio de API: rusqlite ≥0.41 quitó FromSql para usize (SQLite es i64) — un cast en sqlite_store.rs.

### Removed
- `SearchBackend` (config/types.rs) + campo `search_backend` + default `SearchBackend::Hrm2`: config muerta sin un solo consumidor fuera de la definición (verificado por alcanzabilidad grep — enum con variantes Hrm2/Hnsw/Lsh/Quantized que nadie leía; el backend real lo deciden los comandos, no la config)

## [v2.7.2] - 2026-11-06

### Added
- **cli**: `append` ahora hace el insert incremental HNSW real que el help prometía — el CLI corría con `enable_hnsw=false`, así que `candidates` era `None`, `hnsw_indexed` quedaba en 0 y el índice se ignoraba. Al detectar `{data-dir}/hnsw_index.bin` (creado por `hnsw-search`, preset advanced) replica sus parámetros (m=32, ef_c=400, ef_s=100), carga el índice persistido, inserta los nuevos vectores sin rebuild y persiste el índice extendido. Medido sobre copia del hub (5.638+20): 476ms total, `hnsw_indexed = n_after`; sin índice persistido el comportamiento es idéntico al anterior

### Fixed
- **cli**: `--query` de `quant/field/fused/hnsw/lsh-search` y `--embedding` de `graph-add-{doc,entity}` abortaban en clap (`unexpected argument '-0'`) cuando el primer float era negativo — única salida la forma `--query="..."`; `allow_hyphen_values` en los 7 args (el `search` base ya la tenía)

### Security
- **deps**: rusqlite 0.31 → 0.41.1 / libsqlite3-sys 0.28 → 0.39.1 (SQLite bundled 3.54.1) — RUSTSEC-2026-0459 / CVE-2026-119375: SQLite < 3.54.0 puede devolver memoria no inicializada del page-cache al abrir una DB con un WAL hermano de page size distinto (vector local, CVSS:4.0 AV:L/VC:H). La advisory entró al RUSTSEC DB el 13-nov-2026 13:33 UTC; fue la primera CI que la vio. Único cambio de API: rusqlite ≥0.41 quitó FromSql para usize (SQLite es i64) — un cast en sqlite_store.rs.

### Removed
- Poda v2.7.2 (commits `bf47b88`, `662d9bb`, `2696042`): 9.338 LOC (27%, src/ 34.007 → 24.800), cadenas muertas verificadas por alcanzabilidad. Wave 1 (5.011 LOC, cero camino de ejecución desde cualquier entry point): `semantic_memory.rs`, `bm25_index.rs`, `embedding_model.rs` (SemanticMemory nunca se construía; el léxico vivo es el FTS5 del hook), `query_optimizer.rs`, `optimized_api.rs`, `query_router.rs`, `search_supervisor.rs`, `quality_reflector.rs`, `mapreduce_indexer.rs`, `gpu_vector_index.rs`, `gpu_hierarchical_search.rs`, `cuda_search.rs`, `gpu_auto_tune.rs` (el GPU vivo es el dir `gpu/` con kernels PTX), flags `enable_semantic_memory`/`gpu_auto_tune`. Wave 2 (superficie CLI sin uso, stack single-machine): `src/cluster/` (sharding distribuido: 5 módulos + 7 comandos `cluster-*`), `src/ebm/` (`soc-check`/`soc-avalanche`/`soc-relax` — el ranking por energía ya estaba condenado por bench-gaussian, agreement vs L2 ≈ 4.6%), `entity_extractor.rs`, `evaluate_embeddings.rs`, `ml_cmds.rs`, preset `distributed`. `git show bf47b88~1:<ruta>` recupera cualquier archivo cortado. README honest-assets pass: architecture diagram, feature tables, presets table, benchmark provenance notes. Kept deliberately: `hrm2_engine` (el store lo envuelve — ES el store), `quantization`, `verbatim` + `text_compression` (MCP), `lsh_index`, `candidate_backend`, `gpu/`, `loaders/ quant/ storage/`

### Docs
- A5: audit de recall de TurboQuant sobre datos reales — `quant-search` 4-bit (config `for_search`) da **R@10 mediana 0.10 vs exacto** (self-match fuera del top-10 en ~90% de queries): no usar para ranking ni gating; bug negative-CSV marcado FIXED

## [v2.7.1] - 2026-10-25

### Fixed
- **storage**: atomic tmp+rename writes for metadata JSON, vector shards, energy state and `field/` members.idx/radii.bin — a crash or disk-full mid-write no longer truncates shards or silently discards metadata; a corrupt metadata file is quarantined to `*.corrupt-<timestamp>` with a loud stderr note instead of being dropped
- **storage**: `WriteAheadLog::truncate` rewritten crash-safe — the old error path pointed the file handle at `/dev/null`, silently losing every later `log_operation` (and always failing on Windows); it now builds a fsynced sibling temp and swaps the handle only after every fallible step succeeds
- **cli**: `bench-hnsw` panicked (index out of bounds) with fewer than 100 query samples — percentiles clamp to `n-1` and zero samples exits with an error
- **cli**: `field-search` panicked on query-dim mismatch (ndarray shape panic); clean error naming query/splats/raw dims now (also validated in `search`/`search-file`)
- **cli**: `cluster-reset` removed the wrong file (`cluster_state.bin` vs the real `cluster_state.json`) and panicked on IO errors
- **cli**: `quant-index --bits` validated against the library contract (1..=16; TurboQuant 2..=16) instead of panicking; `doc-add` warns instead of silently discarding invalid metadata JSON
- **cli**: `verbatim store` never inserted the embedded vector — searches attributed neighbor vectors to documents whose metadata pointed at indices no vector occupied, defeating the anti-hallucination purpose; vectors are now really inserted and persisted with their real index, and `decompress` no longer panics on non-ASCII hex (invalid hex is an error, not silently dropped digits)
- **cli**: `graph-*` commands were in-memory only — `graph-add-relation` always failed with `NodeNotFound`, `graph-stats` always printed zeros, and `graph-search` injected a fake demo document; `GaussianGraphStore` now persists to `graph_state.json` (atomic write, duplicate-id/dangling-edge rejection on load) and every subcommand loads and saves; the fake injection is gone
- **ebm**: `soc-avalanche` panicked on NaN energies (`partial_cmp().unwrap()`); Ordering fallback now
- **text_compression**: `semantic_compress` panicked or silently corrupted output on Unicode whose lowercase changes byte length (`İ` U+0130 → 3 bytes); single-string char-wise matching now, with regression tests
- **hnsw**: `HNSWIndex::new` enforces the same `m` range as `load` ([2, 4096]) — an index saved by `new` can always be reloaded
- **cli**: `.bin` loaders cap header-driven allocations (1e9 elements) and validate real file size before allocating — a hostile header can no longer OOM the ingest/search/bench paths

### Security
- **deps**: rusqlite 0.31 → 0.41.1 / libsqlite3-sys 0.28 → 0.39.1 (SQLite bundled 3.54.1) — RUSTSEC-2026-0459 / CVE-2026-119375: SQLite < 3.54.0 puede devolver memoria no inicializada del page-cache al abrir una DB con un WAL hermano de page size distinto (vector local, CVSS:4.0 AV:L/VC:H). La advisory entró al RUSTSEC DB el 13-nov-2026 13:33 UTC; fue la primera CI que la vio. Único cambio de API: rusqlite ≥0.41 quitó FromSql para usize (SQLite es i64) — un cast en sqlite_store.rs.

### Removed
- Dead code (zero callers verified by search): `src/energy.rs` (239 LOC), `src/ebm/exploration.rs` (447 LOC)

### Docs
- Consistency pass: measured test counts, every benchmark table cites its persisted artifact in `bench-data/`, BEIR v2 labeled NOT VALIDATED, broken anchors fixed, TOC completed, MCP tool tables completed (18/18), stale PDFs with superseded claims removed

## [v2.7.0] - 2026-10-25

### Added
- Process-level VRAM dataset cache (`gpu::cached_dataset_index`): the dataset uploads to GPU once per dataset version instead of per query/batch; `SplatStore::data_version()` (global epoch-seeded) keys the single-entry cache; alpha/kappa ride along; sub-threshold datasets (≤10K floats) bypass it
- `HNSWIndex::len()`/`dim()`/`is_empty()` accessors; `SyncAction::seq` monotonic identity
- Regression test `test_stale_persisted_hnsw_rejected_and_rebuilt`: a stale persisted HNSW (fewer vectors than the store) is rejected on load and rebuilt — vectors added after the snapshot stay findable via the fused path (covers the `1e6b4ea` fix)

### Fixed
- `semantic_memory`: hybrid search fabricated `vec_{n}` ids that never matched real doc ids — fusion never joined vector+BM25 results (CRITICAL); honest capacity/dim errors on store; `with_encoder()` documented + warns (drops stored vectors); `stats()` counted deletes twice
- `splats`: panics when the HNSW backend returns <k candidates; stale or dim-mismatched persisted HNSW now rejected and rebuilt (was silently accepted → permanent recall loss); `hnsw_indexed_count` no longer advances past failed backend inserts (vectors were lost forever); per-vector fallback stops at first failure (positional shift = silent corruption)
- `hnsw_index`: `m`/`m_max0` from file headers bounded (DoS via ~100-byte hostile index); `m ≥ 2` keeps `level_mult` finite
- HTTP/MCP: embeddings with NaN/inf rejected at every entry point (poisoned the whole index); hex decompress no longer panics the MCP process on odd/multibyte input; oversized-line reader preserves bytes after the newline (stream no longer desyncs); `/field-search` runs its O(n·d) scan on `spawn_blocking` (was freezing the whole runtime); `load_field` validates file size before header-driven allocations
- cluster: rendezvous placement unified between CLI ingest and library (split-brain mapping); `mark_flushed` no longer drops re-queued failed actions; sharding stable under edge reordering
- GPU: no more global panic-hook swap (fallback causes now logged once); extended-kernel PTX cached per process (was JIT-loaded from disk per call, even inside the k-means loop); l2/cosine top-k honors >48KB shared-memory opt-in (silent failures for K≥24); sentinel −1 no longer surfaces as vector 0; CPU fallbacks log once instead of silently degrading
- CI: `cargo-audit` job; `Cargo.lock` versioned; release binaries smoke-tested before publish; dead Python tests (m2m-memory inheritance) removed; pentest setups fail loudly instead of passing empty
- MCP stdio dispatcher runs behind `catch_unwind`: a panic in a tool handler returns JSON-RPC `-32603` instead of killing the server process; GPU clamp test asserts exact result length and sentinel/NaN-free outputs under CUDA
- docs: badge/tests count measured (393), GPU microbenchmark re-measured with per-run values transcribed in `BENCHMARKS.md` (no JSON artifact present): 6 runs, median of run-level speedups 30.6×, GPU 20 ms @ 10K×640D×100q, longmemeval NumPy-vs-native results separated (native: 0.304 recall / 671.63 QPS), unsourced HRM2/TurboQuant claims marked NOT VALIDATED, 15 missing CLI commands documented, verbatim/compression status corrected, inert config keys (`semantic_fusion`, `semantic_vector_weight`, `semantic_bm25_*`, `semantic_decay_halflife`, `autoscale_*`) documented as unused in code (doc-comments in `src/config/mod.rs`), `build_single` input preconditions documented in code (doc-comment in `src/encoding.rs`), LOC badge recount (34,461)

### Security
- **deps**: rusqlite 0.31 → 0.41.1 / libsqlite3-sys 0.28 → 0.39.1 (SQLite bundled 3.54.1) — RUSTSEC-2026-0459 / CVE-2026-119375: SQLite < 3.54.0 puede devolver memoria no inicializada del page-cache al abrir una DB con un WAL hermano de page size distinto (vector local, CVSS:4.0 AV:L/VC:H). La advisory entró al RUSTSEC DB el 13-nov-2026 13:33 UTC; fue la primera CI que la vio. Único cambio de API: rusqlite ≥0.41 quitó FromSql para usize (SQLite es i64) — un cast en sqlite_store.rs.

### Removed
- Dead code, wave-4 audit (zero callers verified by search before each deletion): `error::SplatsDBError`, `engine::SplatsDBEngine` + `compute_expert_distances*`, geometry helpers (`normalize_sphere_1d`, `geodesic_distance_batch`, `exp_map`, `log_map`, `project_to_tangent`, `knn_geodesic`), `BM25Index::search_ids`, `SplatStore::decay_frequencies`, cluster `protocol`/`edge_node`/`sync` modules + `HealthChecker` (`EdgeNodeInfo`/`LoadMetrics`/`GeoLocation` kept — router/sharding use them), `backend_comm`, `auto_scaling`, `api::coordinator_api` + `api::edge_api` — ~2.8K lines

### Verified
- 393 tests green (317 lib + 76 integration), `clippy -D warnings` clean on cuda + cpu-only, GPU kernel benchmark median 30.6× across 6 runs (range 25.45×–34.32×, GPU median 20ms @ 10K×640D×100q, same-run CPU control)

## [v2.6.1] - 2026-10-25

### Added
- Two-phase IVF over the splat field: `field-search --nprobe N` (CLI) and `/field-search {"mode":"two-phase","nprobe":N}` (HTTP) — phase 1 selects top-N clusters by d(q,μ), phase 2 re-ranks their members exactly; `--gpu` runs phase 1 on the PTX kernel (`15de959`)
- `/search` results now carry `topic` + `confidence` per hit (1 − d(x,μ_topic)/R_topic) when a `field/` is persisted; dead-weight splats pruned during ingest (`4d144be`)
- `semantic-hub/faststart.sh`: one-command stack — embeddinggemma-2 embedder (:8860) + exact GPU hub (:8871) + end-to-end smoke (`4d144be`)

### Fixed
- `serve --gpu`: the `/search` GPU index now searches the full chunk corpus (`field.raw`) instead of the cluster centroids μ, so `topic`/`confidence` always refer to real chunks (`2ed5e6c`)
- CI: `gpu_info()` no longer panics when `cudarc` finds no driver (`7afb768`); cudarc-without-driver panic guarded in the rest of the GPU path and clippy 1.99 lints fixed (`867b9da`)

### Security
- **deps**: rusqlite 0.31 → 0.41.1 / libsqlite3-sys 0.28 → 0.39.1 (SQLite bundled 3.54.1) — RUSTSEC-2026-0459 / CVE-2026-119375: SQLite < 3.54.0 puede devolver memoria no inicializada del page-cache al abrir una DB con un WAL hermano de page size distinto (vector local, CVSS:4.0 AV:L/VC:H). La advisory entró al RUSTSEC DB el 13-nov-2026 13:33 UTC; fue la primera CI que la vio. Único cambio de API: rusqlite ≥0.41 quitó FromSql para usize (SQLite es i64) — un cast en sqlite_store.rs.

### Removed
- `training` preset and its subsystems (data lake, `train_embeddings`, training config fields) as unvalidated dead weight — ~1,250 lines deleted (`4d144be`)

## [v2.6.0] - 2026-10-20

### Added
- `field-search` CLI: búsqueda exacta por cotas sobre el campo splat (celdas con radio + desigualdad triangular); radios y membresías CSR persistidos por cada ingest (`field/`, shard `raw`) — recall 0.99 vs GT exacto
- `serve --gpu`: índice exacto en VRAM (kernel PTX `l2_topk_kernel` fused, herencia m2m); `/search` con embedding corre GPU y marca `"backend":"gpu"` — p50 11.4 ms E2E en 26k×768, recall 1.0
- Endpoint `/field-search`: sweep exacto server-side con métricas de poda reales
- Feature `cuda` ON por defecto (build.rs degrada sin nvcc; runtime sin driver cae a CPU)
- `bench-latent/`: GT exacto 30 queries + comparativa de modos — HNSW 0.997, campo exacto 0.99, scoring de energía gaussiana descartado (recall 0.00-0.05, alpha domina = popularidad)

### Fixed
- `ingest-hierarchical` no persistía splats: calculaba, imprimía éxito y descartaba todo
- `simple_kmeans`: asignación por GEMM (init LCG + update f64 intactos) — flat 30-60+ min → 63 s, hierarchical 38,5 → 9,2 s
- Poda por cotas: `kth` solo es cota válida con el top-k completo (recall 0.837 → 0.99)

### Changed
- CI: matriz de features (cpu-only + default-cuda), runtime smoke sin GPU, clippy en ambas configs

## [v2.5.0] - 2026-07-04

### Added
- GitHub Actions CI workflow (test + lint)
- PyPI Trusted Publisher auto-release workflow
- 256 tests passing (CPU mode)

### Fixed
- Various CI/CD improvements
