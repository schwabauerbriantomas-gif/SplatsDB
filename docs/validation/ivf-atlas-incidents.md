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

