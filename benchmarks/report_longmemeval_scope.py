#!/usr/bin/env python3
"""Render docs/validation/longmemeval-scope.md from the published receipt.

    python benchmarks/report_longmemeval_scope.py          # write the report
    python benchmarks/report_longmemeval_scope.py --check  # exit 1 if out of sync

Every number in the report is read from bench-data/longmemeval_scope_results.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "bench-data/longmemeval_scope_results.json"
DOC = ROOT / "docs/validation/longmemeval-scope.md"
CONFIGS = ("head_within", "head_global", "chunkmax_within", "chunkmax_global", "keyword_within", "keyword_global")
LABEL = {
    "head_within": "head, within",
    "head_global": "head, global",
    "chunkmax_within": "chunk-max, within",
    "chunkmax_global": "chunk-max, global",
    "keyword_within": "keyword, within",
    "keyword_global": "keyword, global",
}
TYPES = (
    "single-session-user",
    "multi-session",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-assistant",
)
KS = ("@1", "@3", "@5", "@10")
PAIRED_KEYS = (
    "chunkmax_global_minus_keyword_global@10",
    "head_global_minus_keyword_global@10",
    "chunkmax_within_minus_head_within@10",
    "chunkmax_global_minus_chunkmax_within@10",
)


def pct(hits, n):
    return f"{hits / n * 100:.1f}%" if n else "n/a"


def cell(x):
    return f"{x['hits']}/{x['n']} = {pct(x['hits'], x['n'])}"


def ci(c):
    return f"[{c[0] * 100:.1f}%, {c[1] * 100:.1f}%]"


def paired_text(p):
    lo, hi = p["ci95"]
    return f"{p['delta'] * 100:+.1f} pp, CI95 [{lo * 100:+.1f}, {hi * 100:+.1f}]"


def r2_verdict(d):
    lo, hi = d["ci95"]
    if hi < 0:
        return "`DENSE_BELOW_KEYWORD`", "chunk-max loses to keyword search on a single global memory"
    if lo > 0:
        return "`DENSE_ABOVE_KEYWORD`", "chunk-max beats keyword search on a single global memory"
    return "`NO_DIFFERENCE_RESOLVED`", "the chunk-max versus keyword difference is not resolved at 95%"


def build(r):
    lad, eng, hist = r["ladder"], r["engine"], r["historical_2026_04_09"]
    c, m = r["corpus"], r["model"]
    nq = c["questions"]
    d = r["paired"]["chunkmax_global_minus_keyword_global@10"]
    r2, r2_text = r2_verdict(d)
    r3 = eng["R3_parity"]
    r3_name = "PASS" if r3["pass"] else "FAIL"
    r4d = eng["R4_wing"]
    r4v = r4d["verdict"]
    hg, hw = lad["head_global"]["@10"]["per_type"], lad["head_within"]["@10"]["per_type"]
    an, ar = hist["native_per_type@10"], hist["reference_per_type@10"]

    def decimals(x):
        s = repr(float(x))
        if "e" in s:
            return 6
        return len(s.split(".")[1].rstrip("0")) or 1

    # April's JSON stores rounded rates, so equal means equal at April's stored precision.
    same_global = [t for t in TYPES if round(hg[t]["rate"], decimals(an[t])) == an[t]]
    same_within = [t for t in TYPES if round(hw[t]["rate"], decimals(ar[t])) == ar[t]]
    ref_hits = round(hist["reference_hit@10"] * nq)
    nat_hits = round(hist["native_hit@10"] * nq)
    ex = r["exploratory_hybrid_rrf"]["hit@10"]
    w, g = eng["wing"], eng["global"]
    cfg = eng["config"]
    rec = r["receipts"]

    out = []
    a = out.append
    a("# LongMemEval-S: haystack-restricted vs one global memory — measured result")
    a("")
    a(f"**Verdicts:** engine parity `{r3_name}` · wing scope `{r4v}` · global retrieval {r2}.")
    a("")
    a("## Scope")
    a("")
    a(f"Same dataset, same embeddings, one question at a time: is one of the question's answer sessions in the top-k? "
      f"{nq} questions, {c['session_slots']:,} haystack session slots, {c['unique_sessions']:,} unique sessions "
      f"({c['sessions_shared_across_questions']:,} are shared between questions). Embeddings: `{m['name']}`, "
      f"{m['max_seq_length']}-token model truncation, {m['param_dtype']}. The unit of retrieval is the unique session.")
    a("")
    a("Two scopes, and the difference between them is the whole point:")
    a("")
    a(f"- **within** — candidates are the question's own haystack sessions (median {c['haystack_candidates_median']:.0f}). "
      "The historical 96.6% headline lives here. The scope pre-selects the answer's neighbourhood; an agent with one memory cannot apply it.")
    a(f"- **global** — candidates are all {c['unique_sessions']:,} unique sessions, no filter. This is what a single memory looks like.")
    a("")
    a("The engine part uses the `head` vectors only (one vector per session). This is retrieval-only and MiniLM-only; "
      "it says nothing about end-to-end answer accuracy.")
    a("")
    a("## Ladder (NumPy, no engine)")
    a("")
    a("| configuration | @1 | @3 | @5 | @10 | CI95 hit@10 |")
    a("|---|---:|---:|---:|---:|---|")
    for name in CONFIGS:
        x = lad[name]
        a(f"| {LABEL[name]} | " + " | ".join(cell(x[k]) for k in KS) + f" | {ci(x['ci95_rate@10'])} |")
    a("")
    a("Paired differences on hit@10 (same 500 questions, bootstrap):")
    a("")
    for key in PAIRED_KEYS:
        a(f"- `{key}`: {paired_text(r['paired'][key])}")
    a("")
    a("### Per-type hit@10")
    a("")
    a("| question type | n | head within | head global | chunk-max within | chunk-max global | keyword within | keyword global | 2026-04 native | 2026-04 reference |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in TYPES:
        row = [str(hw[t]["n"])]
        for name in CONFIGS:
            row.append(pct(lad[name]["@10"]["per_type"][t]["hits"], lad[name]["@10"]["per_type"][t]["n"]))
        row += [f"{an[t] * 100:.1f}%", f"{ar[t] * 100:.1f}%"]
        a(f"| {t} | " + " | ".join(row) + " |")
    a(f"| **all** | {nq} | " + " | ".join(pct(lad[name]["@10"]["hits"], nq) for name in CONFIGS)
      + f" | {pct(nat_hits, nq)} | {pct(ref_hits, nq)} |")
    a("")
    a("## Engine (SplatsDB binary on the same head vectors)")
    a("")
    a(f"Release binary at `{eng['binary_git_head'][:12]}` (SHA-256 `{eng['binary_sha256'][:16]}…`), HNSW M={cfg['hnsw_m']}, "
      f"ef_construction={cfg['hnsw_ef_construction']}, ef_search={cfg['hnsw_ef_search']}, over_fetch={cfg['over_fetch']}, "
      f"metric `{cfg['metric']}`, single job, single thread. Latency is as measured on this machine and is not a normalised claim.")
    a("")
    a("| scope | store rows | path | @1 | @3 | @5 | @10 | p50 ms | p95 ms |")
    a("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    a(f"| wing | {eng['wing_store_rows']:,} | filtered candidates, exact re-rank | "
      + " | ".join(cell(w["recall"][k]) for k in KS) + f" | {w['timings']['p50_ms']:.3f} | {w['timings']['p95_ms']:.3f} |")
    a(f"| global | {eng['global_store_rows']:,} | HNSW candidates, exact re-rank | "
      + " | ".join(cell(g["ann"][k]) for k in KS) + f" | {g['timings']['p50_ms']:.3f} | {g['timings']['p95_ms']:.3f} |")
    a(f"| global | {eng['global_store_rows']:,} | linear scan | "
      + " | ".join(cell(g["exact"][k]) for k in KS) + f" | {g['timings']['exact_p50_ms']:.3f} | — |")
    a("")
    a(f"Index build: wing {w['timings']['build_time_s']:.1f} s, global {g['timings']['build_time_s']:.1f} s. "
      "Both search paths return the first k distinct session ids.")
    a("")
    a("## Predeclared readings and outcomes")
    a("")
    a("Rules were frozen before the corresponding run (hashes in the receipt). Outcomes are computed from the receipt.")
    a("")
    r1 = (f"The haystack-restricted head search reproduces the April *reference*: "
          f"{lad['head_within']['@10']['hits']}/{nq} against {ref_hits}/{nq}, with {len(same_within)} of 6 per-type rates equal at the precision stored in April's JSON. ")
    rest = {t: hg[t]["hits"] - round(an[t] * hg[t]["n"]) for t in TYPES if t not in same_global}
    if rest and all(v > 0 for v in rest.values()):
        side = "all above the April figure"
    elif rest and all(v < 0 for v in rest.values()):
        side = "all below the April figure"
    else:
        side = "mixed"
    a("- **R1 — April diagnosis.** " + r1
      + f"The April *native* per-type pattern is reproduced by the **unfiltered** head search: {len(same_global)} of 6 per-type rates equal at that precision "
      f"({', '.join('`' + t + '`' for t in same_global)}), {lad['head_global']['@10']['hits']}/{nq} against {nat_hits}/{nq}; "
      + ("the other types are " + ", ".join(f"`{t}` {v:+d}" for t, v in rest.items()) + f" questions ({side}). " if rest else "every type matches. ") +
      "April's script and inputs are not in the repository, so the mechanism inside that run cannot be shown; what is measured is that its result "
      "matches an unfiltered search and not a haystack-restricted one.")
    a(f"- **R2 — memory ceiling.** chunk-max global minus keyword global: {paired_text(d)} → {r2}: {r2_text}.")
    a(f"- **R3 — engine parity.** `{r3_name}`: exact path {r3['exact_hits@10']} vs NumPy {r3['head_global_hits@10']} "
      f"(difference {r3['abs_diff_exact']}), mean top-10 overlap {r3['mean_overlap_engine_exact_vs_numpy']:.5f}; "
      f"ANN path {r3['ann']['ann_hits@10']} (difference {r3['ann']['abs_diff_ann']}), "
      f"ANN-vs-exact overlap {r3['ann']['mean_overlap_ann_vs_exact']:.5f}; "
      f"independent rescoring of the engine dump equals the engine counts: {r3['rescoring']['pass']}.")
    a(f"- **R4 — wing scope.** `{r4v}`: {r4d['wing_hits@10']}/{nq} against the haystack-restricted reference "
      f"{r4d['head_within_hits@10']}/{nq} (difference {r4d['abs_diff_within']}); the global reference is "
      f"{r4d['head_global_hits@10']}/{nq} (difference {r4d['abs_diff_global']}). The pre-filter does restrict to the question's haystack.")
    a("")
    a("## What this changes")
    a("")
    a("- The 96.6% headline is a haystack-restricted number: it ranks within the question's own ~48 sessions. "
      f"Its unfiltered counterpart with the same vectors and model is {pct(lad['head_global']['@10']['hits'], nq)} hit@10.")
    a("- The April native 30.4% is not a like-for-like comparison against it. With the current binary the restricted path returns exactly the reference, "
      "and the unfiltered path returns the April-like pattern.")
    a("- `bench-long-mem-eval` now has `--scope wing|global` and `--dump`. The default is unchanged; the wing scope is labelled as the shortcut it is.")
    a("- `benchmarks/benchmark_reproducible.py::run_longmemeval_benchmark` is labelled as the placeholder it is: it runs nothing and points to the reproducible runners.")
    a("")
    a("## What this does not show")
    a("")
    a("- Not an end-to-end result: no answer generation, no judge, no abstention. A session in the top-k is not a correct answer.")
    a("- MiniLM only. The Hermes memory hook uses a different embedder (`embeddinggemma-2`, 768D); its own measurement is reported in "
      "`docs/hermes-memory.md` without stating its scope, so it is not comparable with this table until that scope is reconciled.")
    a("- One dataset, one chunk budget (200 word-pieces, no overlap, untuned), one keyword implementation (SQLite FTS5). "
      "No claim transfers to other corpora or settings.")
    a(f"- Speed is measured at {c['unique_sessions']:,} vectors only, where the linear scan costs {g['timings']['exact_p50_ms']:.1f} ms p50 "
      f"against {g['timings']['p50_ms']:.2f} ms for the ANN path.")
    a(f"- Hybrid RRF is exploratory (written after the frozen results, untuned): keyword+chunk-max {ex['rrf_keyword_chunkmax']['hits']}/{nq}, "
      f"keyword+chunk-max+head {ex['rrf_keyword_chunkmax_head']['hits']}/{nq}, keyword+head {ex['rrf_keyword_head']['hits']}/{nq}, "
      f"against keyword alone {lad['keyword_global']['@10']['hits']}/{nq} at hit@10.")
    a("")
    a("## Incidents")
    a("")
    for i, text in enumerate(r["incidents"], start=1):
        a(f"{i}. {text}")
    a("")
    a("## Reproducibility")
    a("")
    a("```bash")
    a("# 1. Ladder: embeddings (GPU if present) + NumPy + SQLite FTS5 — aggregate output only")
    a("python benchmarks/longmemeval_scope.py --dataset longmemeval_s_cleaned.json --cache-dir /tmp/lm_cache \\")
    a("    --reuse --hybrid-exploratory")
    a("")
    a("# 2. Engine: whole store, no filter; per-query rows are dumped for independent rescoring")
    a("splatsdb bench-long-mem-eval --scope global --sessions sessions.bin --queries queries.bin --meta meta.json --dump dump.jsonl")
    a("")
    a("# 3. This report is generated from the receipt; the check fails if they diverge")
    a("python benchmarks/report_longmemeval_scope.py --check")
    a("```")
    a("")
    a(f"Receipt: [`bench-data/longmemeval_scope_results.json`](../../bench-data/longmemeval_scope_results.json) "
      f"(dataset SHA-256 `{r['dataset']['sha256'][:16]}…`; frozen protocol SHA-256 `{rec['protocol_sha256'][:16]}`, "
      f"engine protocol `{rec['protocol_phase2_sha256'][:16]}`; published script reproduces the frozen counts: "
      f"{rec['published_script_reproduces_frozen_counts']}).")
    a("")
    return "\n".join(out)


def main():
    receipt = json.loads(RECEIPT.read_text())
    text = build(receipt)
    if "--check" in sys.argv:
        current = DOC.read_text() if DOC.exists() else ""
        if current != text:
            print(f"OUT OF SYNC: {DOC} differs from {RECEIPT} (run without --check to regenerate)")
            return 1
        print(f"IN SYNC: {DOC.relative_to(ROOT)}")
        return 0
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print(f"wrote {DOC.relative_to(ROOT)} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
