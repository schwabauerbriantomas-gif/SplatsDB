//! Cross-commit evidence probe (D18): the DEFAULT order of `QuantizedStore::search` —
//! recall@10 against exact-L2 and exact-IP ground truths and a hash of the full ranking
//! (ids + score bits). It uses only the API that exists at 617822d (legacy default
//! `estimate/||x||`), at 330d1a7 (F04a: unbiased inner product as default) and at HEAD
//! (`l2_heuristic` default), so the very same source yields comparable numbers for each
//! commit: identical `ranking_hash` => identical ids AND score bits.
//!
//! Same 4 datasets/configs as `properties::probe_recall_vs_exact_l2` (uniform [0,1) and
//! centred gaussian, d=640, n=400, 16 queries, k=10, start_id 0), GT in f64.
//!
//! `cargo test --no-default-features --test quant_contracts d18_probe -- --ignored
//!  --nocapture --test-threads=2`

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use splatsdb::quantization::{QuantConfig, QuantizedStore};
use std::collections::HashSet;

const DIM: usize = 640;
const N: usize = 400;
const N_QUERIES: usize = 16;
const K: usize = 10;

fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |(_, _)| StandardNormal.sample(&mut rng))
}

fn uniform_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |(_, _)| rng.gen::<f32>())
}

/// Exact top-k by an f64 metric (ties by index): ascending for distances
/// (`descending = false`), descending for similarities.
fn exact_top_k(
    query: &[f32],
    data: &Array2<f32>,
    k: usize,
    metric: fn(&[f32], &[f32]) -> f64,
    descending: bool,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = data
        .outer_iter()
        .enumerate()
        .map(|(i, row)| (i, metric(&row.to_vec(), query)))
        .collect();
    scored.sort_by(|a, b| {
        let ord = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
        let by_metric = if descending { ord.reverse() } else { ord };
        by_metric.then(a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

fn l2_sq(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum()
}

fn ip_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn exact_l2_top_k(query: &[f32], data: &Array2<f32>, k: usize) -> Vec<usize> {
    exact_top_k(query, data, k, l2_sq, false)
}

fn exact_ip_top_k(query: &[f32], data: &Array2<f32>, k: usize) -> Vec<usize> {
    exact_top_k(query, data, k, ip_f64, true)
}

/// FNV-1a over the little-endian bytes of `value`, seeded with `hash`.
fn mix(hash: u64, value: u64) -> u64 {
    let mut h = hash;
    for byte in value.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

const HASH_INIT: u64 = 0xcbf2_9ce4_8422_2325;

fn hash_ids(ids: &[u64]) -> u64 {
    ids.iter().fold(HASH_INIT, |h, id| mix(h, *id))
}

fn overlap(a: &[u64], b: &[usize]) -> f64 {
    let set: HashSet<usize> = a.iter().map(|id| *id as usize).collect();
    b.iter().filter(|i| set.contains(i)).count() as f64
}

#[test]
#[ignore = "evidence probe: prints, run with --ignored --nocapture"]
fn probe_default_order() {
    let configs = [
        ("search_4bit", QuantConfig::for_search(DIM)),
        ("default_8bit", QuantConfig::default()),
    ];
    let families = [
        (
            "uniform01",
            uniform_matrix(N, DIM, 1001),
            uniform_matrix(N_QUERIES, DIM, 2001),
        ),
        (
            "gaussian",
            gaussian_matrix(N, DIM, 1001),
            gaussian_matrix(N_QUERIES, DIM, 2001),
        ),
    ];
    for (family, data, queries) in families {
        for (name, config) in &configs {
            let mut store = QuantizedStore::new(DIM, config.clone()).unwrap();
            let _ = store.add_batch(&data, 0);
            let mut recall_l2 = 0.0f64;
            let mut recall_ip = 0.0f64;
            let mut ranking_hash = HASH_INIT;
            let mut top10_hash = HASH_INIT;
            let mut gt_l2_hash = HASH_INIT;
            let mut gt_ip_hash = HASH_INIT;
            let mut nonfinite = 0usize;
            for qi in 0..N_QUERIES {
                let query = queries.row(qi).to_vec();
                let gt_l2 = exact_l2_top_k(&query, &data, K);
                let gt_ip = exact_ip_top_k(&query, &data, K);
                let full = store.search(&queries.row(qi), usize::MAX);
                nonfinite += full.iter().filter(|(_, s)| !s.is_finite()).count();
                for (id, score) in &full {
                    ranking_hash = mix(ranking_hash, *id);
                    ranking_hash = mix(ranking_hash, score.to_bits() as u64);
                }
                let top10: Vec<u64> = full.iter().take(K).map(|(id, _)| *id).collect();
                top10_hash = hash_ids(&top10);
                gt_l2_hash = gt_l2.iter().fold(gt_l2_hash, |h, i| mix(h, *i as u64));
                gt_ip_hash = gt_ip.iter().fold(gt_ip_hash, |h, i| mix(h, *i as u64));
                recall_l2 += overlap(&top10, &gt_l2) / K as f64;
                recall_ip += overlap(&top10, &gt_ip) / K as f64;
            }
            println!(
                "{}",
                json!({
                    "probe": "d18_default_order",
                    "family": family,
                    "config": name,
                    "bits": config.bits,
                    "projections": config.projections,
                    "dim": DIM,
                    "n": N,
                    "n_queries": N_QUERIES,
                    "k": K,
                    "start_id": 0,
                    "len": store.len(),
                    "recall_vs_exact_l2": recall_l2 / N_QUERIES as f64,
                    "recall_vs_exact_ip": recall_ip / N_QUERIES as f64,
                    "ranking_hash": format!("{ranking_hash:016x}"),
                    "top10_hash": format!("{top10_hash:016x}"),
                    "gt_l2_hash": format!("{gt_l2_hash:016x}"),
                    "gt_ip_hash": format!("{gt_ip_hash:016x}"),
                    "nonfinite_scores": nonfinite,
                    "encoded_bytes": store.encoded_bytes(),
                    "memory_usage": store.memory_usage(),
                })
            );
        }
    }
}
