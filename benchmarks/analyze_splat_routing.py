#!/usr/bin/env python3
"""Analyze the predeclared SplatsDB learned-routing gate from real receipts.

Requires: corpus.bin, queries.bin, manifest.json, k100_http.json and the engine's
data/vectors/default.bin + field/members.idx under --workdir. Produces aggregate
JSON only; corpus text and embeddings are never copied into the repository.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BUDGETS = (0.10, 0.20, 0.40)


def read_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    rows, cols = struct.unpack_from("<QQ", raw)
    values = np.frombuffer(raw, dtype="<f4", offset=16)
    assert values.size == rows * cols, (path, rows, cols, values.size)
    return values.reshape(rows, cols).copy()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top10(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.argpartition(distances, 10)[:10]
    ids = ids[np.argsort(distances[ids], kind="stable")]
    return ids, distances[ids]


def field_layout(workdir: Path, k: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mu = read_bin(workdir / f"k{k}/data/vectors/default.bin")
    raw = (workdir / f"k{k}/field/members.idx").read_bytes()
    n, k = struct.unpack_from("<QQ", raw)
    offsets = np.frombuffer(raw, dtype="<u8", count=k + 1, offset=16)
    ids = np.frombuffer(raw, dtype="<u4", count=n, offset=16 + 8 * (k + 1))
    groups = [ids[offsets[c] : offsets[c + 1]].copy() for c in range(k)]
    assert sum(map(len, groups)) == n
    return mu, ids, groups


def make_groups(corpus: np.ndarray, centroids: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    x2 = np.einsum("ij,ij->i", corpus, corpus)
    m2 = np.einsum("ij,ij->i", centroids, centroids)
    d2 = x2[:, None] - 2 * (corpus @ centroids.T) + m2[None, :]
    labels = np.argmin(d2, axis=1)
    return [np.flatnonzero(labels == c).astype(np.uint32) for c in range(len(centroids))], labels


def two_phase(groups: list[np.ndarray], query_d2: np.ndarray, mu: np.ndarray, query: np.ndarray, nprobe: int, n: int):
    delta = mu - query
    dmu = np.einsum("ij,ij->i", delta, delta)
    winners = np.argpartition(dmu, nprobe)[:nprobe] if nprobe < len(dmu) else np.arange(len(dmu), dtype=np.int64)
    candidates = np.concatenate([groups[c] for c in winners])
    scores = query_d2[candidates]
    order = np.argsort(scores, kind="stable")[:10]
    return candidates[order].astype(np.int64), scores[order], len(candidates) / n


def bootstrap_ci(delta: np.ndarray, indices: np.ndarray) -> list[float]:
    means = delta[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, help="optional private per-query HTTP receipt; publish only aggregate metrics and hashes")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    out = args.output.resolve()
    k = args.k
    probes = (1, 2, 4, 8, 16, 32, 64, k)
    manifest = json.loads((workdir / "manifest.json").read_text())
    http = json.loads((workdir / f"k{k}_http.json").read_text())
    corpus = read_bin(workdir / "corpus.bin")
    queries = read_bin(workdir / "queries.bin")
    mu, _, learned_groups = field_layout(workdir, k)
    n, nq = len(corpus), len(queries)
    assert corpus.shape == (11069, 384) and nq == 100 and len(mu) == k

    x2 = np.einsum("ij,ij->i", corpus, corpus)
    gt_ids, gt_scores, all_d2 = [], [], []
    for query in queries:
        d2 = (x2 - 2 * (corpus @ query) + float(query @ query)).astype(np.float32)
        ids, scores = top10(d2)
        gt_ids.append(ids)
        gt_scores.append(scores)
        all_d2.append(d2)

    exact_score_errors = []
    exact_id_mismatches = exact_distance_mismatches = 0
    for i, response in enumerate(http["exact"]):
        ids = np.array([r["index"] for r in response["results"]], dtype=np.int64)
        scores = np.array([r["score"] for r in response["results"]], dtype=np.float32)
        exact_score_errors.append(float(np.max(np.abs(scores - gt_scores[i]))))
        exact_id_mismatches += int(not np.array_equal(ids, gt_ids[i]))
        exact_distance_mismatches += int(not np.allclose(scores, gt_scores[i], rtol=2e-5, atol=2e-5))
    h1 = exact_distance_mismatches == 0 and max(exact_score_errors) < 2e-5

    mirror_mismatches = 0
    learned = {}
    for nprobe in probes:
        recalls, scans = [], []
        for i, response in enumerate(http["two_phase"][str(nprobe)]):
            ids = np.array([r["index"] for r in response["results"]], dtype=np.int64)
            scores = np.array([r["score"] for r in response["results"]], dtype=np.float32)
            expected_ids, expected_scores, scan = two_phase(learned_groups, all_d2[i], mu, queries[i], nprobe, n)
            mirror_mismatches += int(not np.array_equal(ids, expected_ids) or not np.allclose(scores, expected_scores, rtol=2e-5, atol=2e-5))
            recalls.append(len(set(ids) & set(gt_ids[i])) / 10)
            scans.append(response["vectors_scanned"] / n)
        learned[nprobe] = {"recall": np.array(recalls), "scan": np.array(scans)}

    rng = np.random.default_rng(manifest["random_seed"])
    random_mu = corpus[rng.choice(n, size=k, replace=False)].copy()
    random_groups, _ = make_groups(corpus, random_mu)
    random = {}
    for nprobe in probes:
        recalls, scans = [], []
        for i in range(nq):
            ids, _, scan = two_phase(random_groups, all_d2[i], random_mu, queries[i], nprobe, n)
            recalls.append(len(set(ids) & set(gt_ids[i])) / 10)
            scans.append(scan)
        random[nprobe] = {"recall": np.array(recalls), "scan": np.array(scans)}

    boot_rng = np.random.default_rng(manifest["bootstrap_seed"])
    boot_indices = boot_rng.integers(0, nq, size=(10000, nq))
    budgets = []
    for budget in BUDGETS:
        row = {"budget": budget}
        deltas = []
        for name, arm in (("learned", learned), ("random", random)):
            recall = np.empty(nq)
            scan = np.empty(nq)
            for i in range(nq):
                chosen = next(p for p in probes if arm[p]["scan"][i] >= budget)
                recall[i] = arm[chosen]["recall"][i]
                scan[i] = arm[chosen]["scan"][i]
            row[name] = {"recall_mean": float(recall.mean()), "scan_mean": float(scan.mean()), "recall": recall, "scan": scan}
            if name == "learned":
                learned_recall = recall
            else:
                random_recall = recall
        delta = learned_recall - random_recall
        row["delta_recall"] = float(delta.mean())
        row["delta_ci95"] = bootstrap_ci(delta, boot_indices)
        row["passes"] = bool(delta.mean() > 0 and row["delta_ci95"][0] > 0)
        budgets.append(row)

    h2 = all(row["passes"] for row in budgets)
    if not h1:
        verdict = "NO_GO_CORRECTNESS"
    elif h2:
        verdict = "GO_LEARNED_ROUTING"
    elif any(row["delta_recall"] < 0 and row["delta_ci95"][1] < 0 for row in budgets):
        verdict = "NO_GO_LEARNED_ROUTING"
    else:
        verdict = "INCONCLUSIVE_LEARNED_ROUTING"

    public_budgets = []
    for row in budgets:
        public = {k: v for k, v in row.items() if k not in {"learned", "random"}}
        for name in ("learned", "random"):
            public[name] = {k: v for k, v in row[name].items() if k not in {"recall", "scan"}}
            public[name]["recall_per_query"] = row[name]["recall"].tolist()
            public[name]["scan_per_query"] = row[name]["scan"].tolist()
        public_budgets.append(public)
    curves = {
        name: [
            {"nprobe": p, "recall": float(arm[p]["recall"].mean()), "scan": float(arm[p]["scan"].mean())}
            for p in probes
        ]
        for name, arm in (("learned", learned), ("random", random))
    }
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "h1_exact_correctness": {"pass": h1, "id_mismatches": exact_id_mismatches, "distance_mismatches": exact_distance_mismatches, "max_score_error": max(exact_score_errors), "clusters_visited_fraction": float(np.mean([r["cluster_frac"] for r in http["exact"]])), "scan_fraction": float(np.mean([r["scan_frac"] for r in http["exact"]]))},
        "h2_learned_routing": {"pass": h2, "budgets": public_budgets},
        "mirror_parity": {"mismatches": mirror_mismatches, "comparisons": nq * len(probes)},
        "curves": curves,
        "receipts": {**manifest, "http_sha256": sha256(workdir / f"k{k}_http.json"), "analysis_script_sha256": sha256(Path(__file__))},
        "scope": {"queries": nq, "corpus_vectors": n, "dimensions": corpus.shape[1], "k": k, "sensitivity_only": k != 100, "semantic_quality_measured": False, "latency_benchmark": False, "alpha_kappa_used_by_two_phase": False, "mechanism": "learned centroid routing plus exact raw-vector L2 rescoring"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.raw_output:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_bytes(gzip.compress((workdir / f"k{k}_http.json").read_bytes(), mtime=0))
    print(json.dumps({"verdict": verdict, "h1": result["h1_exact_correctness"], "h2": [{k: v for k, v in row.items() if k not in ("learned", "random")} for row in public_budgets], "mirror": result["mirror_parity"]}, indent=2))


if __name__ == "__main__":
    main()
