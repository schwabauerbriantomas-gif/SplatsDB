//! A/B bench probe for A02-F02 (projection cache + per-query context).
//!
//! Prints only. Uses the API common to 330d1a7 (before: projection matrix regenerated
//! and query rotated PER CODE) and HEAD (after), so the same source measures both with
//! the same harness: N=1024, d=640, m=256, 4-bit TurboQuant, 4 queries per pass,
//! 2 warm-up passes + 9 timed passes (>= 7), k=10. Per-query latency (median / p95,
//! nearest rank), throughput per pass (median), encode time (median of 3 builds),
//! RSS (VmRSS / VmHWM), encoded bytes, reported memory and — through a counting global
//! allocator — bytes allocated per query and the peak live delta during the searches.
//!
//! The ranking objective of `search` differs between the two commits (IP at 330d1a7,
//! l2_heuristic at HEAD); the work per code is the same for both, so the timings are
//! comparable. Run ALONE (the allocator counters are process-wide):
//!
//! `cargo test --no-default-features --test quant_contracts probe_ab_search -- --ignored
//!  --nocapture --exact --test-threads=1`

use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const N: usize = 1024;
const DIM: usize = 640;
const PROJECTIONS: usize = 256;
const BITS: u8 = 4;
const N_QUERIES: usize = 4;
const K: usize = 10;
const WARMUP_PASSES: usize = 2;
const TIMED_PASSES: usize = 9;
const ENCODE_BUILDS: usize = 3;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL: AtomicUsize = AtomicUsize::new(0);

struct Counting;

fn account(size: usize) {
    TOTAL.fetch_add(size, Ordering::Relaxed);
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            account(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            account(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |(_, _)| StandardNormal.sample(&mut rng))
}

/// Nearest-rank quantile of a sorted slice.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    quantile(&sorted, 0.5)
}

/// `VmRSS` and `VmHWM` of this process in kB (Linux procfs).
fn rss_kb() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0;
    let mut hwm = 0;
    for line in status.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            rss = v
                .trim()
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("VmHWM:") {
            hwm = v
                .trim()
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        }
    }
    (rss, hwm)
}

#[test]
#[ignore = "evidence probe (A/B bench): prints, run alone with --ignored --nocapture --exact --test-threads=1"]
fn probe_ab_search() {
    let data = gaussian_matrix(N, DIM, 11);
    let queries = gaussian_matrix(N_QUERIES, DIM, 12);
    let config = QuantConfig {
        bits: BITS,
        projections: PROJECTIONS,
        seed: 42,
        algorithm: QuantAlgorithm::TurboQuant,
    };

    // ---- encode: ENCODE_BUILDS fresh stores, wall time of add_batch ----
    let mut encode_ms = Vec::new();
    let mut store = QuantizedStore::new(DIM, config.clone()).unwrap();
    for _ in 0..ENCODE_BUILDS {
        store = QuantizedStore::new(DIM, config.clone()).unwrap();
        let t0 = Instant::now();
        store.add_batch(&data, 0).unwrap();
        encode_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    assert_eq!(store.len(), N);
    let (rss_after_encode_kb, hwm_after_encode_kb) = rss_kb();

    // ---- search: warm-up, then timed passes over the N_QUERIES queries ----
    let top10_first: Vec<u64> = store
        .search(&queries.row(0), K)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    for pass in 0..WARMUP_PASSES {
        for qi in 0..N_QUERIES {
            std::hint::black_box(store.search(&queries.row((qi + pass) % N_QUERIES), K));
        }
    }

    let live0 = LIVE.load(Ordering::Relaxed);
    TOTAL.store(0, Ordering::Relaxed);
    PEAK.store(live0, Ordering::Relaxed);
    let mut latency_ms = Vec::with_capacity(TIMED_PASSES * N_QUERIES);
    let mut pass_qps = Vec::with_capacity(TIMED_PASSES);
    for pass in 0..TIMED_PASSES {
        let t_pass = Instant::now();
        for qi in 0..N_QUERIES {
            let q = &queries.row((qi + pass) % N_QUERIES);
            let t0 = Instant::now();
            let results = std::hint::black_box(store.search(q, K));
            latency_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(results.len(), K);
        }
        pass_qps.push(N_QUERIES as f64 / t_pass.elapsed().as_secs_f64());
    }
    let alloc_total = TOTAL.load(Ordering::Relaxed);
    let peak_delta = PEAK.load(Ordering::Relaxed) - live0;
    let (rss_after_search_kb, hwm_after_search_kb) = rss_kb();
    latency_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "{}",
        json!({
            "probe": "ab_search",
            "n": N,
            "dim": DIM,
            "bits": BITS,
            "projections": PROJECTIONS,
            "k": K,
            "n_queries_per_pass": N_QUERIES,
            "warmup_passes": WARMUP_PASSES,
            "timed_passes": TIMED_PASSES,
            "latency_samples": latency_ms.len(),
            "threads_rayon": std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
            "encode_ms": encode_ms,
            "encode_ms_median": median(&encode_ms),
            "search_latency_ms_median": quantile(&latency_ms, 0.5),
            "search_latency_ms_p95": quantile(&latency_ms, 0.95),
            "search_latency_ms_min": latency_ms.first().copied().unwrap_or(f64::NAN),
            "search_latency_ms_max": latency_ms.last().copied().unwrap_or(f64::NAN),
            "search_throughput_qps_median": median(&pass_qps),
            "search_throughput_qps_min": pass_qps.iter().cloned().fold(f64::INFINITY, f64::min),
            "search_alloc_bytes_per_query": alloc_total as f64 / (TIMED_PASSES * N_QUERIES) as f64,
            "search_peak_live_delta_bytes": peak_delta,
            "encoded_bytes": store.encoded_bytes(),
            "memory_usage": store.memory_usage(),
            "rss_after_encode_kb": rss_after_encode_kb,
            "rss_after_search_kb": rss_after_search_kb,
            "hwm_after_encode_kb": hwm_after_encode_kb,
            "hwm_after_search_kb": hwm_after_search_kb,
            "top10_first_query": top10_first,
        })
    );
}
