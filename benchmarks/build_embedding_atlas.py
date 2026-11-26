#!/usr/bin/env python3
"""Build the frozen numeric embedding atlas and structural-metrics receipt."""
from __future__ import annotations

import argparse
import importlib.metadata
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness, continuity

import ivf_atlas_common as common

ROOT = Path(__file__).resolve().parents[1]
GRID = 128
SIGMA = 1.0
EVAL_N = 2000
PAIR_N = 100_000


def projection_metrics(corpus: np.ndarray, coordinates: np.ndarray, eval_indices: np.ndarray) -> dict:
    x = corpus[eval_indices]
    z = coordinates[eval_indices]
    d_hi = pairwise_distances(x, metric="euclidean", n_jobs=1)
    d_lo = pairwise_distances(z, metric="euclidean", n_jobs=1)
    left, right = common.pair_sample(len(eval_indices), PAIR_N, 20261204)
    pairs = common.pair_metrics(d_hi[left, right], d_lo[left, right])
    result = dict(pairs)
    for k in (10, 50):
        result[f"knn_{k}"] = common.knn_overlap(x, z, k)
        result[f"trustworthiness_{k}"] = float(trustworthiness(x, z, n_neighbors=k, metric="euclidean"))
        result[f"continuity_{k}"] = float(continuity(x, z, n_neighbors=k, metric="euclidean"))
    return result


def render_atlas(pca: np.ndarray, umap: np.ndarray, mips: dict, metrics: dict, output: Path) -> dict:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.333333), dpi=150)
    panels = [(axes[0, 0], "pca", pca, "PCA — scatter", metrics["pca"]), (axes[0, 1], "pca", pca, "PCA — isotropic splat density", metrics["pca"]), (axes[1, 0], "umap", umap, "UMAP — scatter", metrics["umap"]), (axes[1, 1], "umap", umap, "UMAP — isotropic splat density", metrics["umap"])]
    start = time.perf_counter()
    for ax, name, coordinates, title, row in panels:
        bounds = np.percentile(coordinates, [0.5, 99.5], axis=0)
        if "density" in title:
            mip = mips[name]
            ax.imshow(mip, origin="lower", cmap="magma", extent=(0, GRID - 1, 0, GRID - 1), vmax=float(np.percentile(mip, 99.8)), interpolation="nearest")
            ax.set_xlabel("projection 1")
            ax.set_ylabel("projection 2")
        else:
            ax.scatter(coordinates[:, 0], coordinates[:, 1], s=0.55, alpha=0.20, linewidths=0, rasterized=True)
            ax.set_xlim(bounds[:, 0])
            ax.set_ylim(bounds[:, 1])
            ax.set_xlabel("projection 1")
            ax.set_ylabel("projection 2")
        ax.set_title(f"{title}\nkNN@10={row['knn_10']:.3f} | trust@10={row['trustworthiness_10']:.3f} | stress={row['scale_invariant_stress']:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)
    figure.suptitle("Embedding atlas — projection metrics, not retrieval quality", fontsize=15, fontweight="bold")
    figure.text(0.5, 0.035, f"Structural eval: {EVAL_N:,} points + {PAIR_N:,} pairs • no labels or user task • MIP over z • grid={GRID}³ • sigma={SIGMA:g}", ha="center", fontsize=9)
    figure.text(0.5, 0.012, "The splat layer shares coordinates with its scatter panel: it changes density inspection, not kNN/trust/stress.", ha="center", fontsize=9)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.13, hspace=0.38, wspace=0.23)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    return {"render_seconds": time.perf_counter() - start, "png_bytes": output.stat().st_size, "png_sha256": common.sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "bench-data/embedding_atlas_results.json")
    parser.add_argument("--png", type=Path, default=ROOT / "assets/atlas/embedding_atlas.png")
    args = parser.parse_args()
    protocol = ROOT / "docs/validation/ivf-atlas-protocol.md"
    corpus_path = args.workdir.resolve() / "corpus.bin"
    corpus = common.read_bin(corpus_path)
    assert corpus.shape == (11069, 384)
    eval_indices = np.random.default_rng(20261202).choice(len(corpus), EVAL_N, replace=False)

    start = time.perf_counter()
    pca = PCA(n_components=3, svd_solver="randomized", random_state=20261203).fit_transform(corpus).astype(np.float32)
    pca_seconds = time.perf_counter() - start
    import umap

    start = time.perf_counter()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, spread=1.0, metric="euclidean", n_components=3, random_state=20261203, n_jobs=1)
    umap_coordinates = reducer.fit_transform(corpus).astype(np.float32)
    umap_seconds = time.perf_counter() - start

    metrics = {"pca": projection_metrics(corpus, pca, eval_indices), "umap": projection_metrics(corpus, umap_coordinates, eval_indices)}
    volumes = {}
    mips = {}
    for name, coordinates in (("pca", pca), ("umap", umap_coordinates)):
        mips[name], volumes[name] = common.splat_volume(coordinates, GRID, SIGMA)
    render = render_atlas(pca, umap_coordinates, mips, metrics, args.png)
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "umap-learn", "matplotlib")}
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "representation": "Topo-GS-inspired isotropic splat-density prototype; not a Topo-GS reproduction",
        "viewer_recommendation": "DEFER_USER_TASK",
        "structural_metrics": metrics,
        "projection_seconds": {"pca": pca_seconds, "umap": umap_seconds},
        "volumes": volumes,
        "render": {**render, "width": 1800, "height": 1400},
        "scope": {
            "corpus_shape": list(corpus.shape),
            "eval_points": EVAL_N,
            "sampled_pairs": PAIR_N,
            "class_separation": "not_applicable_no_labels",
            "user_task_accuracy": "not_measured",
            "splat_changes_structural_metrics": False,
            "ann_quality_measured": False,
            "interactive_viewer_benefit_claimed": False,
        },
        "packages": packages,
        "receipts": {
            "protocol_sha256": common.sha256_file(protocol),
            "common_script_sha256": common.sha256_file(ROOT / "benchmarks/ivf_atlas_common.py"),
            "runner_sha256": common.sha256_file(Path(__file__).resolve()),
            "corpus_bin_sha256": common.sha256_file(corpus_path),
            "source": common.source_receipt(ROOT, [protocol, ROOT / "benchmarks/ivf_atlas_common.py", Path(__file__).resolve()]),
        },
    }
    common.write_json(args.output, result)
    print({name: {key: row[key] for key in ("knn_10", "knn_50", "trustworthiness_10", "continuity_10", "spearman", "scale_invariant_stress")} for name, row in metrics.items()})


if __name__ == "__main__":
    main()
