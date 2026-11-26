# IVF/atlas validation incidents

## 2026-11-26 — non-grid ratio probes stopped the first IVF run

- **Observed:** `KeyError: 3` in `policy_result` while evaluating validation ratio policies. The frozen ratio rule can select 3, 5 or other probe counts outside the fixed grid; the candidate plan originally contained only grid probes.
- **Effect:** the process exited nonzero before writing `ivf_policy_results.json`. No result, target, threshold or verdict was produced or changed.
- **Fix:** preserve the centroid order and construct candidate lists for arbitrary probe counts through `get_candidates`; add a regression test for a non-grid ratio probe.
- **Protocol change:** none. The same corpus, queries, seeds, ratios, targets and verdict rules remain frozen. The corrected runner receives a new commit/hash before rerunning.

## 2026-11-26 — missing learned selection key stopped the second IVF run

- **Observed:** `KeyError: 'learned'` when the target loop read the selected configuration. Ratio and doubling selections were present, but the fixed learned model had no explicit selection entry.
- **Effect:** the process exited nonzero before writing `ivf_policy_results.json`; again, no measurements or verdicts were published.
- **Fix:** record `learned: null` explicitly and recheck the complete target/selection flow before rerunning.
- **Protocol change:** none.

## 2026-11-26 — Faiss return-order parser produced an invalid NO_GO

- **Observed:** the third run reported 800/800 distance mismatches and 795/800 recall mismatches with zero scan mismatches. A direct probe showed identical selected cells, IDs and scores to the mirror within float32 error.
- **Root cause:** `faiss_recall_scan` assigned the second return value from `index.search` (IDs) to `scores` instead of the first return value (distances).
- **Invalid artifact preserved:** `ivf_policy_results.INVALID_parity_parser.json`, SHA-256 `1410dec5e21993deb0bb0abf5b880fa92c157e4bd85476ee3d4459af195d7eed`. Its `NO_GO_IMPLEMENTATION_PARITY` is not a valid product verdict.
- **Fix:** swap the return assignment and add a synthetic end-to-end Faiss/mirror score-parity regression test. Re-run the identical frozen protocol; do not reinterpret the invalid JSON as a NO_GO.

## 2026-11-26 — atlas runner stopped on a wrong import

- **Observed:** `ImportError` before any projection, metric or image was produced: `pairwise_distances` was imported from SciPy instead of scikit-learn.
- **Fix:** import it from `sklearn.metrics`, compute continuity as reversed-role trustworthiness because scikit-learn does not export a `continuity` function, and add a runner-import test so CI catches missing symbols before measurement.
- **Protocol change:** none; the valid IVF receipt from commit `6b0c10f` is unaffected.


