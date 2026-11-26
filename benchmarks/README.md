# SplatsDB Benchmark Suite

Automated benchmarks for the SplatsDB vector search engine.

## Quick Start

```bash
# Build the binary first
cargo build --release --features cuda

# Built-in HNSW benchmark (CPU, no CUDA needed)
./target/release/splatsdb bench-hnsw \
  --train <train.bin> --queries <query.bin> --gt <gt.bin> \
  -d 128 -k 10 --metric l2 \
  --ef-search 100 --ef-construction 400 --over-fetch 2

# GPU benchmark (requires --features cuda)
./target/release/splatsdb bench-gpu --n-vectors 100000 --dim 640 --n-queries 100 --top-k 10
```

### Binary data format

`bench-hnsw` expects raw binary files (not HDF5):

| File | Format |
|------|--------|
| `--train` | `[u64 rows][u64 cols][f32 data row-major]` |
| `--queries` | `[u64 rows][u64 cols][f32 data row-major]` |
| `--gt` | `[u64 n_queries][u64 k][i64 indices]` |

To convert from ANN-Benchmarks HDF5 to this format, use `h5py` + `struct`.

## What It Runs

### bench-hnsw

HNSW graph search with persistence. Measures build time, p50/p95/p99 latency, QPS, and recall@k against ground truth.

| Config | Default |
|--------|---------|
| `--metric` | `l2` (or `cosine`) |
| `--ef-construction` | 400 |
| `--ef-search` | 100 |
| `--over-fetch` | 2 |
| `-k` | 10 |

### bench-gpu (requires `--features cuda`)

GPU vs CPU search performance with varied dataset sizes:

| Config | Vectors | Dim | Queries | Top-K | Metric |
|--------|---------|-----|---------|-------|--------|
| Small  | 1K      | 640 | 100     | 10    | L2     |
| Medium | 10K     | 640 | 100     | 10    | L2     |
| Large  | 100K    | 640 | 100     | 10    | L2     |

Measures: CPU QPS, GPU upload QPS, GPU persistent QPS.

## Results

Results from the latest runs are stored in [`bench-data/`](../bench-data/). Each JSON is self-documenting with hardware metadata, timestamps, and an integrity checklist.

## Reproducibility

### Matched IVF policies and embedding atlas (2026-11-26)

Use Python 3.12 with `numpy`, `scipy`, `scikit-learn`, `faiss-cpu`, `matplotlib` and `umap-learn`; run one heavy process at a time.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
python benchmarks/benchmark_ivf.py --workdir "$SPLATSDB_WORKDIR" --output bench-data/ivf_policy_results.json
python benchmarks/build_embedding_atlas.py --workdir "$SPLATSDB_WORKDIR" --output bench-data/embedding_atlas_results.json --png assets/atlas/embedding_atlas.png
python benchmarks/report_ivf_atlas.py --ivf bench-data/ivf_policy_results.json --atlas bench-data/embedding_atlas_results.json --output-dir docs/validation --check
```

The workdir must reproduce the frozen corpus, queries, centroids and cell files listed in [the protocol](../docs/validation/ivf-atlas-protocol.md). See [IVF results](../docs/validation/ivf-policy.md) and [atlas results](../docs/validation/embedding-atlas.md). These are CPU structural/policy experiments, not production-QPS benchmarks.

All benchmark numbers published in the [main README](../README.md) trace to a JSON in `bench-data/`. If a number cannot be traced to a results file, it should not be published. See the README's **Integrity pledge** section.
