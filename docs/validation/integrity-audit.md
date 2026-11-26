# Integrity audit: benchmark claims, result files and benchmark code

**Date:** 2026-11-26 · **Branch:** `fix/integrity-audit` · **Status:** corrections applied on the branch; the items still open are listed at the end.

## Why this audit

On 2026-11-26 the LongMemEval figure of April 2026 (96.6% recall@10) turned out to rank every
question inside its own sessions, not over one memory ([longmemeval-scope.md](longmemeval-scope.md)).
This audit looks for more of the same kind of problem in the rest of the repository: numbers presented
as measured that are not, claims without a source behind them, and comparisons that do not compare
like with like.

## How it was done

- **Schema provenance.** For every stored result file: are its keys written by a function of this repository?
- **Table reconciliation.** For every number that a `README.md` or `BENCHMARKS.md` table prints as a bare cell:
  does a result file cited next to the table contain it, at the printed precision, directly or as a difference
  or ratio of two of its values? The reconciler's own false positives were fixed before any finding was taken from it.
- **Reading before running.** Every script was read in full before it was executed. Every correction was exercised:
  16 unit tests, an end-to-end probe on synthetic data, one real run on official data.
- The one-off audit scripts were session artifacts and are not committed. The check anyone can repeat is
  `python benchmarks/reconcile_claims.py`; the corrections are in this branch.

## Findings

### 1. The keyword baseline the README cited contradicts itself — HIGH

- **Published:** "Temporal-reasoning (95.5%) and preference questions (96.7%) — where the original keyword
  baseline showed 30% and 0% — are dramatically improved by semantic search."
- **Evidence:** `bench-data/longmemeval_baseline_results.json` (April 2026). `per_type[*].kw_top10` is `0`
  for all six question types, while the `baseline` block of the same file reports 91.0% keyword recall.
  The file also reports the answer as present for 52.0% of the questions; in the dataset it is present for 500 of 500.
- **Measured replacement:** `bench-data/longmemeval_scope_results.json`. Keyword search (SQLite FTS5)
  reaches 99.2% (496/500) inside each question's own sessions — temporal 99.2% (132/133), preference
  96.7% (29/30) — and 54.6% (273/500) over one global memory. The sentence embeddings reach 96.6% in the
  first scope and 33.4% (167/500) in the second.
- **Conclusion:** the sentence rested on a broken comparator. Inside the haystack, semantic search is not
  better than keywords; over one memory it is worse.
- **Corrected:** README, LongMemEval section, and the "benchmark script" / "previous keyword baseline" lines under it.

### 2. The ANN suite reported a NumPy brute force as SplatsDB — HIGH

- **Published:** the suite printed `[SplatsDB CPU Brute Force]` and stored its result under `splatsdb_cpu`.
- **Evidence (the code):** the block called no SplatsDB binary or library. It was a NumPy matrix product.
  It could not run as written: it allocated a `test × train × dim` tensor (about 5 TB for SIFT, 11.6 GB for NYTimes).
  The Faiss IVF and HNSW indexes were built with L2 for datasets whose ground truth is angular, so their
  recall meant nothing. The statistic stored as `p50_ms` was the mean per query.
- **Why it matters:** fixing only the download URLs (finding 3) would have made this output reachable for the first time.
- **Corrected:** `run_ann_benchmark` rewritten. The exact search is a block-wise NumPy reference labelled as NumPy
  (`"engine": "NumPy"`, `splatsdb_measured: false`). Faiss uses the metric of each dataset (inner product on
  unit vectors for angular ones). The latency statistic is named `mean_ms_per_query`. The suite produces no
  SplatsDB number and says so.

### 3. "Reproducible results" that were three download errors — HIGH

- **Published:** `benchmarks/results_reproducible.json`, named as reproducible results. The README summary table also listed "ANN-Benchmarks (SIFT, GloVe, NYTimes) — ✅ Published".
- **Evidence for the table row:** there was no result for NYTimes at all; the SIFT and GloVe files are Faiss numbers with no producing script in this
  repository. Corrected to "baselines only, none of SplatsDB".
- **Evidence:** the file held three entries `HTTP Error 401: Unauthorized` and nothing else. The three dataset
  URLs pointed to `huggingface.co/datasets/ann-benchmarks/…`, a repository with no datasets, so `--suite ann`
  could never download its data. `ann-benchmarks.com` serves the files.
- **Verified on the official file** (`nytimes-256-angular`): downloaded from ann-benchmarks.com, loaded
  (290,000 × 256 train, 10,000 × 100 neighbors), and an exact cosine search over 200 queries overlaps the
  stored ground truth in 9.9 of 10 neighbors on average.
- **Found while running it:** when a dependency was missing the suite returned `None`, the run exited with
  code 0 and the output file carried no trace of the suite. A run that measured nothing looked like a success.
- **Corrected:** URLs replaced by the official host; the file removed. A suite that returns nothing now leaves
  a `<suite>_not_run` entry, and a run that measures nothing exits with code 2.

### 4. Verification flags that verified nothing — MEDIUM

- **Published:** every suite wrote an `integrity_checklist` (21 constant booleans in the five suites
  that run; the LongMemEval placeholder carried 8 more, among them `code_reproducible: true`, while running nothing),
  a hardware block typed by hand, and the banner "INTEGRITY: run with NO hardcoding…".
- **Evidence:** constants in the source. The run verified none of them. The LongMemEval hero chart repeated the hardware that the April script had typed
  by hand ("RTX 3090 embeddings"); that phrase was removed from the chart.
- **Corrected:** each checklist now sits next to a note saying the flags are self-declared in the script's source
  and that the run verifies none of them. The placeholder prints no checklist. Hardware is read at run time
  from `/proc/cpuinfo`, `os.cpu_count()` and memory. Every suite result records whether it was measured in that
  run, its elapsed time and its completion time. The banner says what is true.
- **Still open:** the 21 constants remain in the five suites. They are declarations about how those
  suites are written; turning them into checks is a separate change.

### 5. "+19% showed BM25 helps" — an attribution the artifact cannot support — MEDIUM

- **Published:** "The +19% improvement on temporal queries showed BM25 helps when queries contain specific keywords."
- **Evidence:** `bench-data/locomo_v2_results.json` lists three changes against v1 (`v2_changes`: model,
  session-level retrieval, BM25 fusion). The temporal gain (31.8% to 50.8%) belongs to the three together.
- **Corrected:** rewritten as a difference between two pipelines, with the three changes named.

### 6. "This confirms the bottleneck is the semantic gap" — LOW

- **Evidence:** the rows it rests on are marked "source not preserved" two paragraphs above them.
- **Corrected:** rewritten as a directional reading that confirms nothing until re-run.

### 7. "✅ 100% Recall@10" for MemoBench — LOW

- **Evidence:** 15 synthetic questions generated by the script itself.
- **Corrected:** sample size and "illustrative, not a validation" added in both places; the summary table now says
  what its check marks mean (that a result file exists).

### 8. HNSW table: no artifact, and two notes that contradicted each other — MEDIUM

- **Published:** one note said the README table was measured on 2026-07-05, the other that "the figures below come
  from April 2026"; the first also said the validated JSON "lives on `main`".
- **Evidence:** the table's figures are the July synthetic ones (789 QPS / 0.861 at 10K). Commit `ab81716` exists;
  `bench-data/hnsw_validated_results.json` is in no commit of any branch. No raw artifact backs that table or the
  April table in BENCHMARKS.md.
- **Corrected:** one note, saying what is and is not preserved.
- **Still open:** a re-run with persisted results (needs the SIFT/GloVe binaries, which are not on this machine).

### 9. BEIR "QPS" excludes query encoding — LOW

- **Evidence:** the script times only the `index.search` call.
- **Corrected:** column headed "QPS (search only)" with a note under the table.

### 10. Graph traversal: an explanation the artifact does not show — LOW

- **Published:** "Dense already finds relevant docs; graph expansion adds topologically-connected but irrelevant docs."
- **Evidence:** `ndcg_at_10` is 0.72 in the baseline arm and 0.72 in the graph arm of `bench-data/ab_test_results.json`:
  at the stored precision, the metric did not change. The
  harness is a Python mirror of the Rust feature and applies a boost of at most 0.02 on cosine scores, far below any
  dense similarity in the top-100. It cannot tell "the graph adds nothing" from "the boost is too small to matter".
  The result file covers scifact only; the README row named scifact and arguana.
- **Corrected:** the row now says what was measured and what the experiment cannot distinguish.

### 11. HTTP ingest stores a hash as if it were an embedding, and says `stored` — MEDIUM (product)

- **Evidence:** in `src/api_server.rs`, a `/ingest` request without an `embedding` vector stores the text under
  `simple_hash_embedding`: one hash of the text seeds a PRNG that fills the dimensions. It has no semantic content.
  The response is `status: "stored"` with no flag. The README did warn about it, but described the method of the
  CLI's verbatim commands (`simcos_embed`, trigram hashing) instead of the one the HTTP server uses.
- **Corrected:** README now describes the method that the server applies.
- **Still open (needs the maintainer's decision, it changes the API's response):** flag the response, or refuse to
  store text without an embedding.

### 12. References to files that are not in the repository — LOW

- `bench-data/hnsw_validated_results.json` (README.md:382)
- `bench-data/hnsw_validated_results.json` (README.md:384)
- `bench-data/hnsw_validated_results.json` (BENCHMARKS.md:81)
- `bench-data/longmemeval_splatdb_native.py` (BENCHMARKS.md:170)

Each is now labelled as not being in the repository. References to gitignored data files are intentional and were
left alone. Two result files have no writer in the repository (`benchmarks/results_sift.json`,
`benchmarks/results_glove.json`: Faiss HNSW numbers for SIFT and GloVe whose keys are not written by any script here), and
`bench-data/faiss_vs_splatdb_results.json` is cited by no document.

## What checked out

- Tables BEIR, LoCoMo, LoCoMo v2, MemoBench and HotpotQA: every bare numeric cell is contained in the cited result
  file. Nothing was found without backing.
- The schema of those files is compatible with having been written by the functions in this repository. This does
  not prove that the runs happened; it says that nothing contradicts it.
- No withdrawn claim (3,125 QPS, "beats all", SOTA) is published again outside the correction log.
- In the Rust sources, the `#[ignore]` tests are evidence probes that print; the product's filtered-search path was
  measured directly ([longmemeval-scope.md](longmemeval-scope.md)).

## ANN-Benchmarks: first run of the repaired suite

- **Input:** the official `nytimes-256-angular` file from ann-benchmarks.com. **Receipt:**
  `bench-data/ann_nytimes_receipt.json` (command, thread policy, detected hardware, SHA-256 of script, data and output).
- **What it holds:** the NumPy exact reference and Faiss IVF / HNSW baselines. No SplatsDB number.
- **One thread, one machine:** the throughput figures are not comparable with the rows of the public leaderboard.
- **Why the exact reference scores 0.9854 and not 1.0** (measured): the corpus has 41,031 duplicate rows among 290,000
  and 239 zero rows. Of the 10,000 queries, 9 are zero vectors (similarity undefined, recall 0) and 880 further
  queries miss part of the top-10; in the 400 of them that were sampled, the gap between the 10th and the 11th
  similarity was exactly 0 (tied duplicates compete for the last places, and the stored ground truth breaks the tie
  differently). Among 400 queries with recall 1.0 the median gap is 3.0e-3. Any exact method is capped near 0.986
  on this file unless it breaks ties the way the stored ground truth does.

## New guards

- `benchmarks/reconcile_claims.py`, run by CI: every bare number in a README.md or BENCHMARKS.md table must be
  contained in a result file cited next to the table, or be on the reviewed list
  `bench-data/claims_untraceable.json`. A new number without a source fails the build. Every entry on that list today sits
  in a table whose rows the README marks as having no preserved source, or in the graph row that explains its own 0.0000.
  What the check does not do: tables that cite no result file are not checked, and a number that coincides with a
  difference or ratio of two stored values passes. It proves the absence of a source, never its presence.
- `tests/test_benchmark_integrity.py`, run by CI: 16 unit tests on the provenance helpers, the exact-search
  reference and the labels of the ANN suite, plus 3 on the reconciler.
- `run_provenance` on every suite result: measured in this run or read from stored files, elapsed time,
  completion time, and the hardware detected on the machine that produced the file.

## Not verified, and open

- Whether the April runs of BEIR / LoCoMo / MemoBench happened as their files say. They are compatible with it; the
  download host of BEIR did not answer from this machine, so nothing was re-downloaded or re-run.
- The HNSW tables (finding 8) were not re-measured.
- The scope of the 63.1% context recall reported for the Hermes hook (`docs/hermes-memory.md` does not say whether
  it is per-question or global).
- Whether `/ingest` should flag or refuse text without an embedding (finding 11).
- Whether keyword search (54.6%) or chunk-max (43.0%) lead to better answers than head embeddings (33.4%) over one
  global memory: only retrieval was measured, not answers.

## Corrections to statements made during the audit

- An interim count of "27 hardcoded booleans" came from a grep over an earlier state of the script. The count by
  syntax tree is 29: 21 in the five suites and 8 in the placeholder (removed).
