#!/usr/bin/env python3
"""Shared, deterministic primitives for the frozen IVF/atlas validation.

The module never reads or writes corpus text. Public receipts contain aggregates,
configuration, versions and hashes; candidate IDs and distances stay in memory.
"""
from __future__ import annotations

import hashlib
import json
import platform
import struct
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PROBES = (1, 2, 4, 8, 16, 32, 64, 100)
TARGETS = (0.90, 0.95, 0.98)
RATIOS = (0.01, 0.05, 0.10, 0.20, 0.40)
DOUBLING_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)
DISTANCE_TOL = 2e-5


def read_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    rows, cols = struct.unpack_from("<QQ", raw)
    values = np.frombuffer(raw, dtype="<f4", offset=16)
    assert values.size == rows * cols, (path, rows, cols, values.size)
    return values.reshape(rows, cols).copy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_cells(workdir: Path, k: int = 100) -> tuple[np.ndarray, list[np.ndarray]]:
    mu = read_bin(workdir / f"k{k}/data/vectors/default.bin")
    raw = (workdir / f"k{k}/field/members.idx").read_bytes()
    n, stored_k = struct.unpack_from("<QQ", raw)
    assert stored_k == k
    offsets = np.frombuffer(raw, dtype="<u8", count=k + 1, offset=16)
    ids = np.frombuffer(raw, dtype="<u4", count=n, offset=16 + 8 * (k + 1))
    groups = [ids[offsets[c] : offsets[c + 1]].copy() for c in range(k)]
    assert sum(map(len, groups)) == n and np.array_equal(offsets[-1], n)
    return mu, groups


def make_queries(corpus: np.ndarray, n: int, seed: int, sigma: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(corpus), size=n, replace=False)
    queries = corpus[indices] + rng.normal(0.0, sigma, size=(n, corpus.shape[1]))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return queries.astype(np.float32), indices


def brute_d2(corpus: np.ndarray, queries: np.ndarray) -> np.ndarray:
    x2 = np.einsum("ij,ij->i", corpus, corpus)
    rows = []
    for query in queries:
        rows.append((x2 - 2 * (corpus @ query) + float(query @ query)).astype(np.float32))
    return np.stack(rows)


def top_k(d2: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    ids = np.argpartition(d2, k)[:k]
    ids = ids[np.argsort(d2[ids], kind="stable")]
    return ids, d2[ids]


def threshold_recall(scores: np.ndarray, gt_kth: np.ndarray, tol: float = DISTANCE_TOL) -> np.ndarray:
    return np.mean(scores <= gt_kth[:, None] + tol, axis=1)


def candidate_plan(query: np.ndarray, mu: np.ndarray, groups: Sequence[np.ndarray], probes: Iterable[int] = PROBES) -> dict:
    delta = mu - query
    dmu = np.einsum("ij,ij->i", delta, delta).astype(np.float32)
    order = np.argsort(dmu, kind="stable")
    ordered_groups = [groups[c] for c in order]
    candidates = {int(p): np.concatenate(ordered_groups[:p]) for p in probes}
    return {"dmu": dmu, "order": order, "ordered_groups": ordered_groups, "candidates": candidates}


def get_candidates(plan: dict, p: int) -> np.ndarray:
    p = int(p)
    if p not in plan["candidates"]:
        plan["candidates"][p] = np.concatenate(plan["ordered_groups"][:p])
    return plan["candidates"][p]


def score_candidates(candidates: np.ndarray, query_d2: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    scores = query_d2[candidates]
    order = np.argsort(scores, kind="stable")[:k]
    return candidates[order].astype(np.int64), scores[order]


def grid_recall_scan(plans: Sequence[dict], all_d2: np.ndarray, gt_kth: np.ndarray, n: int) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    for p in PROBES:
        recalls, scans = [], []
        for i, plan in enumerate(plans):
            _, scores = score_candidates(get_candidates(plan, p), all_d2[i])
            recalls.append(threshold_recall(scores[None, :], gt_kth[i : i + 1])[0])
            scans.append(len(plan["candidates"][p]) / n)
        result[p] = {"recall": np.asarray(recalls), "scan": np.asarray(scans)}
    return result


def ratio_probe(plan: dict, ratio: float) -> int:
    dmu = plan["dmu"]
    closest = dmu[plan["order"][0]]
    return max(1, int(np.count_nonzero(dmu <= (1.0 + ratio) * closest + 1e-12)))


def productive_probe(plan: dict, threshold: float) -> int:
    current = 1
    while current < PROBES[-1]:
        following = min(PROBES[-1], current * 2)
        previous = plan["candidates"][current]
        new_ids = np.setdiff1d(plan["candidates"][following], previous, assume_unique=False)
        productive = len(new_ids) / max(1, len(previous))
        if productive < threshold:
            break
        current = following
    return current


def plan_features(plan: dict, p: int, n: int, k: int = 100) -> list[float]:
    dmu = plan["dmu"]
    order = plan["order"]
    closest = max(float(dmu[order[0]]), 1e-9)
    selected = dmu[order[:p]]
    scan = len(plan["candidates"][p]) / n
    return [
        float(p),
        float(np.log1p(p)),
        float(p / k),
        float(scan),
        float(np.log1p(scan)),
        closest,
        float(np.log(closest)),
        float(np.log((float(dmu[order[1]]) + 1e-9) / closest)),
        float(np.log((float(dmu[order[3]]) + 1e-9) / closest)),
        float(np.median(selected) / closest),
        float(np.max(selected) / closest),
    ]


def fit_probe_regressor(features: np.ndarray, targets: np.ndarray, seed: int = 20261203) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=seed,
        early_stopping=False,
    )
    model.fit(features, targets)
    return model


def predicted_probe(model: HistGradientBoostingRegressor, plan: dict, n: int, target: float) -> int:
    for p in PROBES:
        prediction = float(model.predict(np.asarray([plan_features(plan, p, n)]))[0])
        if prediction >= target:
            return p
    return PROBES[-1]


def policy_result(plans: Sequence[dict], all_d2: np.ndarray, gt_kth: np.ndarray, probe_for_query: Sequence[int], n: int) -> dict[str, np.ndarray]:
    recalls, scans = [], []
    for i, (plan, p) in enumerate(zip(plans, probe_for_query)):
        _, scores = score_candidates(get_candidates(plan, int(p)), all_d2[i])
        recalls.append(threshold_recall(scores[None, :], gt_kth[i : i + 1])[0])
        scans.append(len(plan["candidates"][int(p)]) / n)
    return {"recall": np.asarray(recalls), "scan": np.asarray(scans)}


def bootstrap_indices(n: int, seed: int = 20261128, resamples: int = 10000) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_ci(delta: np.ndarray, indices: np.ndarray) -> list[float]:
    means = delta[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def select_fixed(grid: dict[int, dict[str, np.ndarray]], target: float) -> int:
    for p in PROBES:
        if float(grid[p]["recall"].mean()) >= target:
            return p
    return PROBES[-1]


def select_config(configs: Sequence[float], outcomes: dict[float, dict[str, np.ndarray]], target: float) -> float | None:
    eligible = [c for c in configs if float(outcomes[c]["recall"].mean()) >= target]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (float(outcomes[c]["scan"].mean()), c))


def evaluate_target(policy: dict[str, np.ndarray], fixed: dict[str, np.ndarray], target: float, boot: np.ndarray) -> dict:
    recall_delta = policy["recall"] - fixed["recall"]
    scan_reduction = (fixed["scan"] - policy["scan"]) / fixed["scan"]
    recall_ci = bootstrap_ci(recall_delta, boot)
    reduction_ci = bootstrap_ci(scan_reduction, boot)
    comparator_valid = bool(fixed["recall"].mean() >= target - 0.005)
    checks = {
        "comparator_valid": comparator_valid,
        "policy_reaches_target": bool(policy["recall"].mean() >= target - 0.005),
        "recall_noninferior": bool(recall_ci[0] > -0.005),
        "scan_reduction_at_least_5pct": bool(scan_reduction.mean() >= 0.05),
        "scan_reduction_ci_positive": bool(reduction_ci[0] > 0.0),
    }
    if not comparator_valid:
        status = "INCONCLUSIVE_NO_VALID_COMPARATOR"
    elif all(checks.values()):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "checks": checks,
        "policy_recall_mean": float(policy["recall"].mean()),
        "fixed_recall_mean": float(fixed["recall"].mean()),
        "policy_scan_mean": float(policy["scan"].mean()),
        "fixed_scan_mean": float(fixed["scan"].mean()),
        "recall_delta": float(recall_delta.mean()),
        "recall_delta_ci95": recall_ci,
        "relative_scan_reduction": float(scan_reduction.mean()),
        "relative_scan_reduction_ci95": reduction_ci,
        "policy_recall_per_query": policy["recall"].tolist(),
        "policy_scan_per_query": policy["scan"].tolist(),
        "fixed_recall_per_query": fixed["recall"].tolist(),
        "fixed_scan_per_query": fixed["scan"].tolist(),
    }


def family_verdict(targets: dict[str, dict]) -> str:
    states = [row["status"] for row in targets.values()]
    if all(s == "PASS" for s in states):
        return "GO"
    if any(s.startswith("INCONCLUSIVE") for s in states):
        return "INCONCLUSIVE"
    return "NO_GO"


def latency_summary(milliseconds: np.ndarray) -> dict:
    values = np.asarray(milliseconds, dtype=np.float64)
    mean = float(values.mean())
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": mean,
        "qps": float(1000.0 / mean) if mean > 0 else None,
        "samples": int(values.size),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def source_receipt(root: Path, files: Sequence[Path]) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    return {
        "head": head,
        "dirty_lines": len(dirty),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "files": {str(path.relative_to(root)): sha256_file(path) for path in files},
    }


def knn_overlap(x: np.ndarray, z: np.ndarray, k: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    x_ids = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    z_ids = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1).fit(z).kneighbors(z, return_distance=False)[:, 1:]
    overlaps = [len(set(a) & set(b)) / k for a, b in zip(x_ids, z_ids)]
    return float(np.mean(overlaps))


def pair_sample(n: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=count)
    right = rng.integers(0, n, size=count)
    mask = left != right
    while not np.all(mask):
        bad = np.flatnonzero(~mask)
        left[bad] = rng.integers(0, n, size=len(bad))
        right[bad] = rng.integers(0, n, size=len(bad))
        mask = left != right
    return left, right


def pair_metrics(d_hi: np.ndarray, d_lo: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    hi = np.asarray(d_hi, dtype=np.float64)
    lo = np.asarray(d_lo, dtype=np.float64)
    hi_scaled = hi / hi.mean()
    lo_scaled = lo / lo.mean()
    stress = float(np.sqrt(np.sum((hi_scaled - lo_scaled) ** 2) / np.sum(hi_scaled**2)))
    return {"spearman": float(spearmanr(hi, lo).statistic), "scale_invariant_stress": stress}


def splat_volume(coordinates: np.ndarray, grid: int = 128, sigma: float = 1.0) -> tuple[np.ndarray, dict]:
    from scipy.ndimage import gaussian_filter

    values = np.asarray(coordinates, dtype=np.float32)
    lower = np.percentile(values, 0.5, axis=0)
    upper = np.percentile(values, 99.5, axis=0)
    span = np.maximum(upper - lower, 1e-9)
    scaled = (values - lower) / span * (grid - 1)
    clipped = float(np.mean(np.any((scaled < 0) | (scaled > grid - 1), axis=1)))
    cells = np.clip(np.rint(scaled).astype(np.int32), 0, grid - 1)
    flat = (cells[:, 0] * grid + cells[:, 1]) * grid + cells[:, 2]
    volume = np.bincount(flat, minlength=grid**3).reshape(grid, grid, grid).astype(np.float32)
    smoothed = gaussian_filter(volume, sigma=sigma, mode="constant")
    mip = smoothed.max(axis=2)
    return mip, {
        "grid": grid,
        "sigma": sigma,
        "clipped_fraction": clipped,
        "volume_bytes": int(smoothed.nbytes),
        "volume_sha256": sha256_array(smoothed),
        "mip_sha256": sha256_array(mip),
    }
