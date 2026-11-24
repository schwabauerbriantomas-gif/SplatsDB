//! A/B release probe for F04c (bit-packed codes). Prints only.
//!
//! Uses the API common to 9f319b2 (before: unpacked codes) and the candidate (after:
//! packed codes), so the same source measures both: N=10000, d=640, m=256, k=10,
//! TurboQuant, `bits` from `AB_BITS` (default 4), 10 queries per pass, 2 warm-up passes
//! + `AB_RUNS` (at least 5) timed passes, rayon threads from the environment. Reports
//!   per-query latency (median / p95, nearest rank) over every sample of every pass,
//!   throughput per pass (median), encode time, encoded bytes, reported memory,
//!   `is_bit_packed`, compression ratio, the heap bytes retained by the store after the
//!   encode and the peak live delta during the searches (counting global allocator),
//!   VmRSS / VmHWM, and a hash of (id, f32 score bits) over the FULL ranking of every
//!   query in both modes — identical across commits iff ids, order and scores are.
//!   Run ALONE, release profile:
//!
//! `cargo test --release --no-default-features --test quant_contracts probe_ab_release
//!  -- --ignored --nocapture --exact --test-threads=1`
//!
//! `probe_serde_golden` prints the serde JSON of the unpacked code types for a fixed
//! vector: the byte-identical golden of `tests/quant_contracts/packing.rs`, produced by
//! whichever commit it runs on.

use super::ab_bench::{median, quantile, rss_kb, LIVE, PEAK, TOTAL};
use super::d18_probe::{gaussian_matrix, mix, HASH_INIT};
use serde_json::json;
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore, ScoreMode};
use std::hint::black_box;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

const DIM: usize = 640;
const PROJECTIONS: usize = 256;
const K: usize = 10;
const WARMUP_PASSES: usize = 2;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// FNV-1a over (id, f32 score bits) of the full ranking of every query under `mode`.
fn ranking_hash(
    store: &QuantizedStore,
    queries: &ndarray::Array2<f32>,
    mode: ScoreMode,
) -> (u64, usize) {
    let mut hash = HASH_INIT;
    let mut nonfinite = 0;
    for qi in 0..queries.nrows() {
        let hits = store
            .search_scored(&queries.row(qi), usize::MAX, mode)
            .expect("every query is finite");
        for hit in hits {
            hash = mix(hash, hit.id);
            hash = mix(hash, hit.score.to_bits() as u64);
            nonfinite += usize::from(!hit.score.is_finite());
        }
    }
    (hash, nonfinite)
}

#[test]
#[ignore = "evidence probe (F04c A/B, release): prints, run alone with --ignored --nocapture --exact --test-threads=1"]
fn probe_ab_release() {
    let n = env_usize("AB_N", 10_000);
    let bits = env_usize("AB_BITS", 4) as u8;
    let runs = env_usize("AB_RUNS", 5).max(5);
    let n_queries = env_usize("AB_QUERIES", 10);
    let data = gaussian_matrix(n, DIM, 11);
    let queries = gaussian_matrix(n_queries, DIM, 12);
    let config = QuantConfig {
        bits,
        projections: PROJECTIONS,
        seed: 42,
        algorithm: QuantAlgorithm::TurboQuant,
    };

    // ---- encode: one fresh store; wall time and heap delta of add_batch ----
    let mut store = QuantizedStore::new(DIM, config).unwrap();
    let live0 = LIVE.load(Relaxed);
    let (rss_empty_kb, _) = rss_kb();
    let t0 = Instant::now();
    store.add_batch(&data, 0).unwrap();
    let encode_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(store.len(), n);
    let heap_retained = LIVE.load(Relaxed) - live0;
    let (rss_after_encode_kb, hwm_after_encode_kb) = rss_kb();

    // ---- parity: ids + f32 score bits of the full ranking, both modes ----
    let (hash_l2, nonfinite_l2) = ranking_hash(&store, &queries, ScoreMode::L2Heuristic);
    let (hash_ip, nonfinite_ip) = ranking_hash(&store, &queries, ScoreMode::InnerProduct);
    let top10_first: Vec<u64> = store
        .search(&queries.row(0), K)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    // ---- search: warm-up, then timed passes over the queries ----
    for pass in 0..WARMUP_PASSES {
        for qi in 0..n_queries {
            black_box(store.search(&queries.row((qi + pass) % n_queries), K));
        }
    }
    let live1 = LIVE.load(Relaxed);
    TOTAL.store(0, Relaxed);
    PEAK.store(live1, Relaxed);
    let mut latency_ms = Vec::with_capacity(runs * n_queries);
    let mut run_qps = Vec::with_capacity(runs);
    let mut run_median_ms = Vec::with_capacity(runs);
    for run in 0..runs {
        let t_run = Instant::now();
        let mut run_latency = Vec::with_capacity(n_queries);
        for qi in 0..n_queries {
            let query = &queries.row((qi + run) % n_queries);
            let t = Instant::now();
            let results = black_box(store.search(query, K));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            latency_ms.push(ms);
            run_latency.push(ms);
            assert_eq!(results.len(), K);
        }
        run_qps.push(n_queries as f64 / t_run.elapsed().as_secs_f64());
        run_median_ms.push(median(&run_latency));
    }
    let alloc_total = TOTAL.load(Relaxed);
    let peak_delta = PEAK.load(Relaxed) - live1;
    let (rss_after_search_kb, hwm_after_search_kb) = rss_kb();
    latency_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        json!({
            "probe": "ab_release",
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "n": n,
            "dim": DIM,
            "bits": bits,
            "projections": PROJECTIONS,
            "k": K,
            "n_queries_per_pass": n_queries,
            "warmup_passes": WARMUP_PASSES,
            "timed_passes": runs,
            "latency_samples": latency_ms.len(),
            "threads_rayon": std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
            "encode_ms": encode_ms,
            "heap_retained_after_encode_bytes": heap_retained,
            "encoded_bytes": store.encoded_bytes(),
            "code_layout_bytes": store.code_layout_bytes(),
            "memory_usage": store.memory_usage(),
            "is_bit_packed": store.is_bit_packed(),
            "compression_ratio": store.compression_ratio(),
            "rss_empty_kb": rss_empty_kb,
            "rss_after_encode_kb": rss_after_encode_kb,
            "rss_after_search_kb": rss_after_search_kb,
            "hwm_after_encode_kb": hwm_after_encode_kb,
            "hwm_after_search_kb": hwm_after_search_kb,
            "search_latency_ms_median": quantile(&latency_ms, 0.5),
            "search_latency_ms_p95": quantile(&latency_ms, 0.95),
            "search_latency_ms_min": latency_ms.first().copied().unwrap_or(f64::NAN),
            "search_latency_ms_max": latency_ms.last().copied().unwrap_or(f64::NAN),
            "search_latency_ms_median_per_pass": run_median_ms,
            "search_throughput_qps_median": median(&run_qps),
            "search_throughput_qps_min": run_qps.iter().cloned().fold(f64::INFINITY, f64::min),
            "search_alloc_bytes_per_query": alloc_total as f64 / (runs * n_queries) as f64,
            "search_peak_live_delta_bytes": peak_delta,
            "ranking_hash_l2_heuristic": format!("{hash_l2:016x}"),
            "ranking_hash_ip": format!("{hash_ip:016x}"),
            "nonfinite_scores_l2_heuristic": nonfinite_l2,
            "nonfinite_scores_ip": nonfinite_ip,
            "top10_first_query": top10_first,
        })
    );
}

#[test]
#[ignore = "evidence probe (F04c serde golden): prints the wire JSON of the unpacked code types"]
fn probe_serde_golden() {
    use splatsdb::quant::{PolarQuantizer, QjlQuantizer, TurboQuantizer};
    let x: Vec<f32> = (0..16)
        .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
        .collect();
    let turbo = TurboQuantizer::new(16, 4, 8, 42).unwrap();
    let polar = PolarQuantizer::new(16, 3, 7).unwrap();
    let qjl = QjlQuantizer::new(16, 8, 9).unwrap();
    let code = turbo.encode(&x).unwrap();
    println!(
        "{}",
        json!({
            "probe": "serde_golden",
            "vector": x,
            "turbo_code": serde_json::to_string(&code).unwrap(),
            "polar_code": serde_json::to_string(&polar.encode(&x).unwrap()).unwrap(),
            "qjl_sketch_v1": serde_json::to_string(&qjl.sketch(&x).unwrap()).unwrap(),
        })
    );
}
