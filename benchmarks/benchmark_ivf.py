#!/usr/bin/env python3
"""Run the frozen matched-IVFFlat and adaptive-probing experiment."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

import ivf_atlas_common as common

ROOT = Path(__file__).resolve().parents[1]


def norms_error(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1.0)))


def prepare_queries(queries: np.ndarray, corpus: np.ndarray, mu: np.ndarray, groups: list[np.ndarray]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    plans = [common.candidate_plan(q, mu, groups) for q in queries]
    all_d2 = common.brute_d2(corpus, queries)
    gt = [common.top_k(row) for row in all_d2]
    gt_kth = np.asarray([scores[-1] for _, scores in gt], dtype=np.float32)
    return plans, all_d2, gt_kth


def faiss_recall_scan(index, quantizer, queries: np.ndarray, all_d2: np.ndarray, gt_kth: np.ndarray, p: int, n: int) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    index.nprobe = p
    recalls, scans, score_rows = [], [], []
    for i, query in enumerate(queries):
        _, scores = index.search(query[None, :], 10)
        scores = scores[0].astype(np.float32)
        score_rows.append(scores)
        recalls.append(common.threshold_recall(scores[None, :], gt_kth[i : i + 1])[0])
        cell_ids = quantizer.search(query[None, :], p)[1][0]
        scans.append(sum(index.get_list_size(int(c)) for c in cell_ids) / n)
    return {"recall": np.asarray(recalls), "scan": np.asarray(scans)}, score_rows


def timed(operation, queries: np.ndarray, warmups: int = 2, rounds: int = 5) -> np.ndarray:
    for _ in range(warmups):
        for query in queries:
            operation(query)
    samples = []
    for _ in range(rounds):
        for query in queries:
            start = time.perf_counter_ns()
            operation(query)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return np.asarray(samples)


def python_operation(family: str, config: float | None, target: float, model, plan: dict, i: int, all_d2: np.ndarray, corpus_size: int):
    def run(_: np.ndarray):
        if family == "ratio":
            assert config is not None
            p = common.ratio_probe(plan, float(config))
        elif family == "doubling":
            assert config is not None
            p = common.productive_probe(plan, float(config))
        else:
            p = common.predicted_probe(model, plan, corpus_size, target)
        common.score_candidates(common.get_candidates(plan, p), all_d2[i])
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "bench-data/ivf_policy_results.json")
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    workdir = args.workdir.resolve()
    protocol = ROOT / "docs/validation/ivf-atlas-protocol.md"
    manifest = json.loads((workdir / "manifest.json").read_text())
    corpus = common.read_bin(workdir / "corpus.bin")
    test_queries = common.read_bin(workdir / "queries.bin")
    assert corpus.shape == (11069, 384) and test_queries.shape == (100, 384)
    n = len(corpus)
    assert norms_error(corpus) < 2e-5 and norms_error(test_queries) < 2e-5
    mu, groups = common.load_cells(workdir)
    assert common.sha256_file(workdir / "corpus.bin") == manifest["corpus_bin_sha256"]
    assert common.sha256_file(workdir / "queries.bin") == manifest["queries_bin_sha256"]

    test_plans, test_d2, test_gt_kth = prepare_queries(test_queries, corpus, mu, groups)
    mirror_grid = common.grid_recall_scan(test_plans, test_d2, test_gt_kth, n)

    quantizer = faiss.IndexFlatL2(corpus.shape[1])
    quantizer.add(mu)
    index = faiss.IndexIVFFlat(quantizer, corpus.shape[1], len(mu), faiss.METRIC_L2)
    start = time.perf_counter()
    index.add(corpus)
    faiss_add_s = time.perf_counter() - start
    faiss_grid, faiss_scores = {}, {}
    for p in common.PROBES:
        faiss_grid[p], faiss_scores[p] = faiss_recall_scan(index, quantizer, test_queries, test_d2, test_gt_kth, p, n)

    parity = {"comparisons": len(test_queries) * len(common.PROBES), "recall_mismatches": 0, "distance_mismatches": 0, "scan_mismatches": 0}
    for i, plan in enumerate(test_plans):
        for p in common.PROBES:
            _, mirror_scores = common.score_candidates(plan["candidates"][p], test_d2[i])
            parity["recall_mismatches"] += int(mirror_grid[p]["recall"][i] != faiss_grid[p]["recall"][i])
            parity["distance_mismatches"] += int(not np.allclose(np.sort(mirror_scores), np.sort(faiss_scores[p][i]), rtol=2e-5, atol=2e-5))
            parity["scan_mismatches"] += int(abs(mirror_grid[p]["scan"][i] - faiss_grid[p]["scan"][i]) > 1e-12)
    parity["h3_pass"] = parity["recall_mismatches"] == parity["distance_mismatches"] == 0

    train_queries, _ = common.make_queries(corpus, 300, 20261130)
    validation_queries, _ = common.make_queries(corpus, 100, 20261201)
    train_plans, train_d2, train_gt_kth = prepare_queries(train_queries, corpus, mu, groups)
    validation_plans, validation_d2, validation_gt_kth = prepare_queries(validation_queries, corpus, mu, groups)
    train_grid = common.grid_recall_scan(train_plans, train_d2, train_gt_kth, n)
    validation_grid = common.grid_recall_scan(validation_plans, validation_d2, validation_gt_kth, n)

    features, targets = [], []
    for i, plan in enumerate(train_plans):
        for p in common.PROBES:
            features.append(common.plan_features(plan, p, n))
            targets.append(train_grid[p]["recall"][i])
    model = common.fit_probe_regressor(np.asarray(features), np.asarray(targets))
    train_predictions = model.predict(np.asarray(features))
    train_rmse = float(np.sqrt(np.mean((train_predictions - np.asarray(targets)) ** 2)))

    ratio_validation = {r: common.policy_result(validation_plans, validation_d2, validation_gt_kth, [common.ratio_probe(p, r) for p in validation_plans], n) for r in common.RATIOS}
    doubling_validation = {t: common.policy_result(validation_plans, validation_d2, validation_gt_kth, [common.productive_probe(p, t) for p in validation_plans], n) for t in common.DOUBLING_THRESHOLDS}
    boot = common.bootstrap_indices(len(test_queries))
    policies = {"ratio": {}, "doubling": {}, "learned": {}}
    selections = {}
    faiss_latency = {}

    for target in common.TARGETS:
        fixed_p = common.select_fixed(validation_grid, target)
        fixed = mirror_grid[fixed_p]
        selections[str(target)] = {
            "fixed_nprobe": fixed_p,
            "fixed_validation_recall": float(validation_grid[fixed_p]["recall"].mean()),
            "fixed_validation_scan": float(validation_grid[fixed_p]["scan"].mean()),
            "ratio": common.select_config(common.RATIOS, ratio_validation, target),
            "doubling": common.select_config(common.DOUBLING_THRESHOLDS, doubling_validation, target),
            "learned": None,
        }
        if fixed_p not in faiss_latency:
            index.nprobe = fixed_p
            faiss_latency[fixed_p] = common.latency_summary(timed(lambda q: index.search(q[None, :], 10), test_queries))
        for family in ("ratio", "doubling", "learned"):
            config = selections[str(target)][family]
            if family in {"ratio", "doubling"} and config is None:
                policies[family][str(target)] = {
                    "status": "INCONCLUSIVE_NO_ELIGIBLE_CONFIG",
                    "selected_config": None,
                    "reason": "No predeclared configuration reached the target on validation.",
                }
                continue
            if family == "ratio":
                probes = [common.ratio_probe(plan, config) for plan in test_plans]
            elif family == "doubling":
                probes = [common.productive_probe(plan, config) for plan in test_plans]
            else:
                config = None
                probes = [common.predicted_probe(model, plan, n, target) for plan in test_plans]
            result = common.policy_result(test_plans, test_d2, test_gt_kth, probes, n)
            evaluation = common.evaluate_target(result, fixed, target, boot)
            samples = []
            for i, plan in enumerate(test_plans):
                samples.extend(timed(python_operation(family, config, target, model, plan, i, test_d2, n), test_queries[i : i + 1]))
            harness_latency = common.latency_summary(np.asarray(samples))
            harness_latency.pop("qps", None)
            evaluation.update({
                "selected_config": config,
                "probe_per_query": probes,
                "python_precomputed_distance_harness_latency": harness_latency,
                "faiss_fixed_latency": faiss_latency[fixed_p],
                "latency_scope": "Faiss is end-to-end in-process IVFFlat search. Python timing includes policy selection and candidate scoring but uses precomputed full-corpus distances; it is not comparable and is not a QPS benchmark.",
            })
            policies[family][str(target)] = evaluation

    verdicts = {family: common.family_verdict(targets) for family, targets in policies.items()}
    overall = "GO_ADAPTIVE_POLICY" if any(v == "GO" for v in verdicts.values()) else ("INCONCLUSIVE_ADAPTIVE_POLICY" if any(v == "INCONCLUSIVE" for v in verdicts.values()) else "NO_GO_ADAPTIVE_POLICY")
    if not parity["h3_pass"]:
        overall = "NO_GO_IMPLEMENTATION_PARITY"
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": overall,
        "family_verdicts": verdicts,
        "h3_faiss_mirror_parity": parity,
        "h4_policy_targets": policies,
        "validation_selection": selections,
        "train_regressor": {"rmse": train_rmse, "features": 11, "rows": len(features), "test_rows_used": 0},
        "faiss_grid": {str(p): {"recall_mean": float(faiss_grid[p]["recall"].mean()), "scan_mean": float(faiss_grid[p]["scan"].mean())} for p in common.PROBES},
        "mirror_grid": {str(p): {"recall_mean": float(mirror_grid[p]["recall"].mean()), "scan_mean": float(mirror_grid[p]["scan"].mean())} for p in common.PROBES},
        "build": {
            "faiss_add_seconds": faiss_add_s,
            "faiss_serialized_index_bytes": int(faiss.serialize_index(index).nbytes),
            "mirror_cell_file_bytes": int((workdir / "k100/field/members.idx").stat().st_size + mu.nbytes),
            "faiss_version": faiss.__version__,
            "threads": 1,
            "warmups": 2,
            "rounds": 5,
        },
        "scope": {
            "corpus_shape": list(corpus.shape),
            "test_queries": len(test_queries),
            "train_queries": len(train_queries),
            "validation_queries": len(validation_queries),
            "semantic_quality_measured": False,
            "production_latency_measured": False,
            "alpha_kappa_used": False,
            "new_algorithm_claimed": False,
            "distance_recall_tolerance": common.DISTANCE_TOL,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20261128,
        },
        "receipts": {
            "protocol_sha256": common.sha256_file(protocol),
            "common_script_sha256": common.sha256_file(ROOT / "benchmarks/ivf_atlas_common.py"),
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "corpus_bin_sha256": common.sha256_file(workdir / "corpus.bin"),
            "queries_bin_sha256": common.sha256_file(workdir / "queries.bin"),
            "centroids_sha256": common.sha256_file(workdir / "k100/data/vectors/default.bin"),
            "members_sha256": common.sha256_file(workdir / "k100/field/members.idx"),
            "source": common.source_receipt(ROOT, [protocol, ROOT / "benchmarks/ivf_atlas_common.py", Path(__file__).resolve()]),
        },
    }
    common.write_json(args.output, result)
    print(json.dumps({"verdict": overall, "parity": parity, "family_verdicts": verdicts, "train_rmse": train_rmse}, indent=2))


if __name__ == "__main__":
    main()
