# LongMemEval-S: haystack-restricted vs one global memory — measured result

**Verdicts:** engine parity `PASS` · wing scope `FILTER_RESTRICTS_TO_HAYSTACK` · global retrieval `DENSE_BELOW_KEYWORD`.

## Scope

Same dataset, same embeddings, one question at a time: is one of the question's answer sessions in the top-k? 500 questions, 23,867 haystack session slots, 19,195 unique sessions (3,942 are shared between questions). Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, 256-token model truncation, torch.float32. The unit of retrieval is the unique session.

Two scopes, and the difference between them is the whole point:

- **within** — candidates are the question's own haystack sessions (median 48). The historical 96.6% headline lives here. The scope pre-selects the answer's neighbourhood; an agent with one memory cannot apply it.
- **global** — candidates are all 19,195 unique sessions, no filter. This is what a single memory looks like.

The engine part uses the `head` vectors only (one vector per session). This is retrieval-only and MiniLM-only; it says nothing about end-to-end answer accuracy.

## Ladder (NumPy, no engine)

| configuration | @1 | @3 | @5 | @10 | CI95 hit@10 |
|---|---:|---:|---:|---:|---|
| head, within | 373/500 = 74.6% | 446/500 = 89.2% | 462/500 = 92.4% | 484/500 = 96.8% | [95.2%, 98.2%] |
| head, global | 52/500 = 10.4% | 104/500 = 20.8% | 126/500 = 25.2% | 167/500 = 33.4% | [29.2%, 37.6%] |
| chunk-max, within | 425/500 = 85.0% | 472/500 = 94.4% | 489/500 = 97.8% | 495/500 = 99.0% | [98.0%, 99.8%] |
| chunk-max, global | 81/500 = 16.2% | 146/500 = 29.2% | 188/500 = 37.6% | 215/500 = 43.0% | [38.6%, 47.2%] |
| keyword, within | 443/500 = 88.6% | 479/500 = 95.8% | 486/500 = 97.2% | 496/500 = 99.2% | [98.4%, 99.8%] |
| keyword, global | 107/500 = 21.4% | 197/500 = 39.4% | 221/500 = 44.2% | 273/500 = 54.6% | [50.2%, 58.8%] |

Paired differences on hit@10 (same 500 questions, bootstrap):

- `chunkmax_global_minus_keyword_global@10`: -11.6 pp, CI95 [-15.8, -7.4]
- `head_global_minus_keyword_global@10`: -21.2 pp, CI95 [-25.8, -16.6]
- `chunkmax_within_minus_head_within@10`: +2.2 pp, CI95 [+0.8, +3.6]
- `chunkmax_global_minus_chunkmax_within@10`: -56.0 pp, CI95 [-60.4, -51.8]

### Per-type hit@10

| question type | n | head within | head global | chunk-max within | chunk-max global | keyword within | keyword global | 2026-04 native | 2026-04 reference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 70 | 88.6% | 8.6% | 98.6% | 34.3% | 100.0% | 42.9% | 5.7% | 88.6% |
| multi-session | 133 | 99.2% | 30.8% | 100.0% | 32.3% | 98.5% | 42.9% | 30.1% | 99.2% |
| single-session-preference | 30 | 96.7% | 16.7% | 96.7% | 16.7% | 96.7% | 10.0% | 16.7% | 96.7% |
| temporal-reasoning | 133 | 96.2% | 33.8% | 97.7% | 28.6% | 99.2% | 51.1% | 28.6% | 95.5% |
| knowledge-update | 78 | 100.0% | 26.9% | 100.0% | 62.8% | 100.0% | 79.5% | 20.5% | 100.0% |
| single-session-assistant | 56 | 98.2% | 87.5% | 100.0% | 100.0% | 100.0% | 94.6% | 87.5% | 98.2% |
| **all** | 500 | 96.8% | 33.4% | 99.0% | 43.0% | 99.2% | 54.6% | 30.4% | 96.6% |

## Engine (SplatsDB binary on the same head vectors)

Release binary at `afc0d49f1672` (SHA-256 `4792a97d0f46906a…`), HNSW M=32, ef_construction=400, ef_search=100, over_fetch=3, metric `l2`, single job, single thread. Latency is as measured on this machine and is not a normalised claim.

| scope | store rows | path | @1 | @3 | @5 | @10 | p50 ms | p95 ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| wing | 23,867 | filtered candidates, exact re-rank | 373/500 = 74.6% | 446/500 = 89.2% | 462/500 = 92.4% | 484/500 = 96.8% | 0.825 | 1.011 |
| global | 19,195 | HNSW candidates, exact re-rank | 52/500 = 10.4% | 104/500 = 20.8% | 126/500 = 25.2% | 168/500 = 33.6% | 1.719 | 2.380 |
| global | 19,195 | linear scan | 52/500 = 10.4% | 104/500 = 20.8% | 126/500 = 25.2% | 167/500 = 33.4% | 7.885 | — |

Index build: wing 94.4 s, global 71.4 s. Both search paths return the first k distinct session ids.

## Predeclared readings and outcomes

Rules were frozen before the corresponding run (hashes in the receipt). Outcomes are computed from the receipt.

- **R1 — April diagnosis.** The haystack-restricted head search reproduces the April *reference*: 484/500 against 483/500, with 5 of 6 per-type rates equal at the precision stored in April's JSON. The April *native* per-type pattern is reproduced by the **unfiltered** head search: 2 of 6 per-type rates equal at that precision (`single-session-preference`, `single-session-assistant`), 167/500 against 152/500; the other types are `single-session-user` +2, `multi-session` +1, `temporal-reasoning` +7, `knowledge-update` +5 questions (all above the April figure). April's script and inputs are not in the repository, so the mechanism inside that run cannot be shown; what is measured is that its result matches an unfiltered search and not a haystack-restricted one.
- **R2 — memory ceiling.** chunk-max global minus keyword global: -11.6 pp, CI95 [-15.8, -7.4] → `DENSE_BELOW_KEYWORD`: chunk-max loses to keyword search on a single global memory.
- **R3 — engine parity.** `PASS`: exact path 167 vs NumPy 167 (difference 0), mean top-10 overlap 1.00000; ANN path 168 (difference 1), ANN-vs-exact overlap 0.99480; independent rescoring of the engine dump equals the engine counts: True.
- **R4 — wing scope.** `FILTER_RESTRICTS_TO_HAYSTACK`: 484/500 against the haystack-restricted reference 484/500 (difference 0); the global reference is 167/500 (difference 317). The pre-filter does restrict to the question's haystack.

## What this changes

- The 96.6% headline is a haystack-restricted number: it ranks within the question's own ~48 sessions. Its unfiltered counterpart with the same vectors and model is 33.4% hit@10.
- The April native 30.4% is not a like-for-like comparison against it. With the current binary the restricted path returns exactly the reference, and the unfiltered path returns the April-like pattern.
- `bench-long-mem-eval` now has `--scope wing|global` and `--dump`. The default is unchanged; the wing scope is labelled as the shortcut it is.
- `benchmarks/benchmark_reproducible.py::run_longmemeval_benchmark` is labelled as the placeholder it is: it runs nothing and points to the reproducible runners.

## What this does not show

- Not an end-to-end result: no answer generation, no judge, no abstention. A session in the top-k is not a correct answer.
- MiniLM only. The Hermes memory hook uses a different embedder (`embeddinggemma-2`, 768D); its own measurement is reported in `docs/hermes-memory.md` without stating its scope, so it is not comparable with this table until that scope is reconciled.
- One dataset, one chunk budget (200 word-pieces, no overlap, untuned), one keyword implementation (SQLite FTS5). No claim transfers to other corpora or settings.
- Speed is measured at 19,195 vectors only, where the linear scan costs 7.9 ms p50 against 1.72 ms for the ANN path.
- Hybrid RRF is exploratory (written after the frozen results, untuned): keyword+chunk-max 270/500, keyword+chunk-max+head 250/500, keyword+head 254/500, against keyword alone 273/500 at hit@10.

## Incidents

1. First Phase 1 launch held 4.38 GB RSS (whole-corpus token-id lists during chunking) with MemAvailable 3.10 GiB, below the 4 GiB floor: killed by exact PID before producing any result; rerun with block-wise chunking (RSS 2.52 GiB at the last memory checkpoint, per run log).
2. The April inputs and script are absent from the repository, so April's pipeline itself is not reproducible; dataset identity is established by identical corpus statistics.
3. ANN and exact paths differ by one question (168 vs 167) with 18 queries showing one flipped neighbour: inside the predeclared tolerance, reported unadjusted.
4. benchmarks/benchmark_reproducible.py::run_longmemeval_benchmark returned hardcoded numbers and pointed to a non-existing benches/ directory.
5. README documented the subcommand as bench-longmemeval; the CLI name is bench-long-mem-eval.

## Reproducibility

```bash
# 1. Ladder: embeddings (GPU if present) + NumPy + SQLite FTS5 — aggregate output only
python benchmarks/longmemeval_scope.py --dataset longmemeval_s_cleaned.json --cache-dir /tmp/lm_cache \
    --reuse --hybrid-exploratory

# 2. Engine: whole store, no filter; per-query rows are dumped for independent rescoring
splatsdb bench-long-mem-eval --scope global --sessions sessions.bin --queries queries.bin --meta meta.json --dump dump.jsonl

# 3. This report is generated from the receipt; the check fails if they diverge
python benchmarks/report_longmemeval_scope.py --check
```

Receipt: [`bench-data/longmemeval_scope_results.json`](../../bench-data/longmemeval_scope_results.json) (dataset SHA-256 `d6f21ea9d60a0d56…`; frozen protocol SHA-256 `44f53cdd6a765ab9`, engine protocol `7a894cc0f031dd8c`; published script reproduces the frozen counts: True).
