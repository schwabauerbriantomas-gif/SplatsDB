//! D18: the ranking objective and the unbiased estimator are separate contracts.
//!
//! * The default, `l2_heuristic`, is the legacy order `estimate / ||x||` — reproduced bit
//!   for bit at the store level (scores AND order) against an independent reference
//!   implementation of the <= 2.11.0 estimator, for non-unit and zero norms.
//! * `ip` ranks by the unbiased inner product and keeps the F04a unbiasedness property;
//!   the heuristic is shown NOT to be an unbiased estimator (control).
//! * PolarQuant has no QJL stage: both modes are the same (legacy) order.
//! * A NaN/±Inf query is rejected before anything is ranked.
//! * `probe_modes` (ignored): recall per mode against exact-L2 and exact-IP ground truths
//!   on the same 4 datasets as `d18_probe`, plus a hash of the full ranking per mode.

use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use splatsdb::quant::{PolarQuantizer, QjlQuantizer, QuantError, TurboQuantizer};
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore, ScoreMode};
use std::f32::consts::PI;

fn gaussian(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| StandardNormal.sample(rng)).collect()
}

/// Gaussian direction scaled to norm `norm` (f64; `0.0` gives the zero vector).
fn scaled(rng: &mut ChaCha8Rng, dim: usize, norm: f32) -> Vec<f32> {
    let v = gaussian(rng, dim);
    let n = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    v.iter()
        .map(|x| ((*x as f64) * norm as f64 / n) as f32)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>() as f32
}

fn mixed(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ b.wrapping_add(0x632B_E599_66D0_24A5)
        ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// --- Independent reference of what <= 2.11.0 computed ---------------------------
// The projection matrix exactly as the old code regenerated it per call (same RNG
// recipe, same sample order), and the two QJL terms as plain loops.

fn legacy_projection(dim: usize, projections: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF_1234_5678));
    (0..projections)
        .map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect()
}

/// `sum_i sign_i * (g_i . query)` with the old accumulation order.
fn qjl_sum(g: &[Vec<f32>], signs: &[i8], query: &[f32]) -> f32 {
    g.iter()
        .zip(signs.iter())
        .map(|(row, &sign)| {
            let g_dot_query: f32 = row.iter().zip(query.iter()).map(|(g, q)| g * q).sum();
            sign as f32 * g_dot_query
        })
        .sum()
}

/// Legacy (<= 2.11.0) QJL term: `pi / (2m) * sum_i sign_i (g_i.q)`.
fn legacy_qjl(g: &[Vec<f32>], signs: &[i8], query: &[f32]) -> f32 {
    let m = signs.len() as f32;
    let scale = PI / (2.0 * m);
    scale * qjl_sum(g, signs, query)
}

/// F04a unbiased QJL term: `sqrt(pi/2) * ||x|| / m * sum_i sign_i (g_i.q)`.
fn unbiased_qjl(g: &[Vec<f32>], signs: &[i8], norm: f32, query: &[f32]) -> f32 {
    let m = signs.len() as f32;
    let scale = (PI / 2.0).sqrt() * norm / m;
    scale * qjl_sum(g, signs, query)
}

/// Reference score of every row of `data` against `query` under `mode`, built only from
/// the public encoders and the formulas above (never from the store): for TurboQuant
/// `polar_est + QJL term of the residual sketch`, with the seeds `TurboQuantizer::new`
/// derives; for PolarQuant the polar estimate (no QJL stage, same in both modes).
fn reference_store_scores(
    config: &QuantConfig,
    dim: usize,
    data: &Array2<f32>,
    query: &[f32],
    mode: ScoreMode,
) -> Vec<f32> {
    match config.algorithm {
        QuantAlgorithm::TurboQuant => {
            let polar = PolarQuantizer::new(dim, config.bits - 1, config.seed).unwrap();
            let qjl_seed = config.seed.wrapping_add(0xCAFE_BABE_0000_0001);
            let qjl = QjlQuantizer::new(dim, config.projections, qjl_seed).unwrap();
            let g = legacy_projection(dim, config.projections, qjl_seed);
            (0..data.nrows())
                .map(|i| {
                    let x: Vec<f32> = data.row(i).to_vec();
                    let code = polar.encode(&x).unwrap();
                    let reconstruction = polar.decode(&code).unwrap();
                    let residual: Vec<f32> = x
                        .iter()
                        .zip(reconstruction.iter())
                        .map(|(o, r)| o - r)
                        .collect();
                    let sketch = qjl.sketch(&residual).unwrap();
                    let qjl_term = match mode {
                        ScoreMode::L2Heuristic => legacy_qjl(&g, &sketch.signs, query),
                        ScoreMode::InnerProduct => {
                            unbiased_qjl(&g, &sketch.signs, sketch.norm, query)
                        }
                    };
                    polar.inner_product_estimate(&code, query).unwrap() + qjl_term
                })
                .collect()
        }
        QuantAlgorithm::PolarQuant => {
            let polar = PolarQuantizer::new(dim, config.bits, config.seed).unwrap();
            (0..data.nrows())
                .map(|i| {
                    let code = polar.encode(&data.row(i).to_vec()).unwrap();
                    polar.inner_product_estimate(&code, query).unwrap()
                })
                .collect()
        }
    }
}

/// The order the legacy store produced: stable sort, higher score first.
fn legacy_order(scores: &[f32]) -> Vec<(usize, f32)> {
    let mut order: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    order
}

fn three_configs(dim: usize) -> Vec<QuantConfig> {
    vec![
        QuantConfig::for_search(dim),
        QuantConfig::default(),
        QuantConfig {
            bits: 4,
            projections: 0,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        },
    ]
}

// --- Strict contract tests ------------------------------------------------------

#[test]
fn score_mode_wire_names() {
    assert_eq!(ScoreMode::default(), ScoreMode::L2Heuristic);
    assert_eq!(QuantizedStore::default_score_mode(), ScoreMode::L2Heuristic);
    for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
        assert_eq!(ScoreMode::parse(mode.as_str()), Some(mode));
        assert_eq!(mode.to_string(), mode.as_str());
    }
    assert_eq!(ScoreMode::as_str(ScoreMode::L2Heuristic), "l2_heuristic");
    assert_eq!(ScoreMode::as_str(ScoreMode::InnerProduct), "ip");
    assert_eq!(ScoreMode::parse("IP"), None);
    assert_eq!(ScoreMode::parse("cosine"), None);
}

/// QJL level: the default mode IS the legacy estimator (same f32 bits), the `ip` mode
/// IS the F04a unbiased estimator, and they differ by exactly the legacy factor
/// `sqrt(pi/2) / ||x||` — which is why the order is the same for ||x|| > 0 and why the
/// zero vector needs no division.
#[test]
fn qjl_default_mode_reproduces_the_legacy_estimator_bit_for_bit() {
    const DIM: usize = 64;
    const M: usize = 128;
    const SEED: u64 = 4242;
    let q = QjlQuantizer::new(DIM, M, SEED).unwrap();
    let g = legacy_projection(DIM, M, SEED);
    let m = M as f32;
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    // None of these is sqrt(pi/2) ~= 1.2533 (where the two modes coincide) and two are
    // far from unit norm; 0.0 is the zero vector.
    for norm in [0.0f32, 0.05, 0.25, 1.0, 4.0, 16.0, 1000.0] {
        for rep in 0..200 {
            let x = scaled(&mut rng, DIM, norm);
            let query = gaussian(&mut rng, DIM);
            let sketch = q.sketch(&x).unwrap();
            let s = qjl_sum(&g, &sketch.signs, &query);
            let legacy = PI / (2.0 * m) * s;
            let unbiased = (PI / 2.0).sqrt() * sketch.norm / m * s;

            let default = q.score(&sketch, &query, ScoreMode::default()).unwrap();
            let l2 = q.score(&sketch, &query, ScoreMode::L2Heuristic).unwrap();
            let ip = q.score(&sketch, &query, ScoreMode::InnerProduct).unwrap();
            let via_api = q.inner_product_estimate(&sketch, &query).unwrap();

            assert_eq!(l2.to_bits(), legacy.to_bits(), "norm={norm} rep={rep}");
            assert_eq!(default.to_bits(), legacy.to_bits(), "norm={norm} rep={rep}");
            assert_eq!(ip.to_bits(), unbiased.to_bits(), "norm={norm} rep={rep}");
            assert_eq!(
                via_api.to_bits(),
                ip.to_bits(),
                "inner_product_estimate is the ip mode"
            );

            if norm > 0.0 {
                let expected_ratio = (PI / 2.0).sqrt() / sketch.norm;
                if ip.abs() > 1e-3 {
                    assert!(
                        ((l2 / ip) - expected_ratio).abs() <= 1e-4 * expected_ratio,
                        "norm={norm}: l2/ip={} expected={expected_ratio}",
                        l2 / ip
                    );
                }
            } else {
                // Zero norm: deterministic, finite, never NaN; the unbiased estimate is
                // exactly 0 and the heuristic keeps the legacy value (all signs are +1).
                assert!(sketch.signs.iter().all(|&s| s == 1));
                assert_eq!(ip, 0.0);
                assert!(l2.is_finite() && l2 != 0.0, "legacy zero-vector score {l2}");
            }
        }
    }
}

/// Store level, both algorithms: default mode = legacy scores AND legacy order for
/// non-unit and zero norms (ties by insertion order); the `ip` mode = unbiased
/// reference; `search` is the default mode; a bounded k is a prefix of the ranking.
#[test]
fn store_default_order_is_the_legacy_order_for_non_unit_and_zero_norms() {
    const DIM: usize = 64;
    let norms = [0.0f32, 0.0, 0.01, 0.25, 1.0, 1.0, 4.0, 16.0, 1e4, 0.5];
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let rows: Vec<Vec<f32>> = (0..60)
        .map(|i| scaled(&mut rng, DIM, norms[i % norms.len()]))
        .collect();
    let data = Array2::from_shape_fn((rows.len(), DIM), |(r, c)| rows[r][c]);
    let mut query_rows: Vec<Vec<f32>> = (0..5).map(|_| gaussian(&mut rng, DIM)).collect();
    for i in [2usize, 7, 11] {
        query_rows.push(data.row(i).iter().map(|v| -v).collect());
    }
    query_rows.push(vec![0.0; DIM]);
    let queries = Array2::from_shape_fn((query_rows.len(), DIM), |(r, c)| query_rows[r][c]);
    let n = data.nrows();

    for config in three_configs(DIM) {
        let mut store = QuantizedStore::new(DIM, config.clone()).unwrap();
        store.add_batch(&data, 0).unwrap();
        assert_eq!(store.len(), n);
        for qi in 0..queries.nrows() {
            let query = queries.row(qi).to_vec();
            let mut per_mode = Vec::new();
            for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
                let expected =
                    legacy_order(&reference_store_scores(&config, DIM, &data, &query, mode));
                let hits = store
                    .search_scored(&queries.row(qi), usize::MAX, mode)
                    .unwrap();
                assert_eq!(hits.len(), n);
                assert!(hits.iter().all(|h| h.mode == mode));
                for (hit, (idx, score)) in hits.iter().zip(expected.iter()) {
                    assert_eq!(hit.id as usize, *idx, "{:?} q{qi}", config.algorithm);
                    assert_eq!(
                        hit.score.to_bits(),
                        score.to_bits(),
                        "{:?} q{qi} id={} mode={mode:?}",
                        config.algorithm,
                        hit.id
                    );
                }
                let key: Vec<(u64, u32)> = hits.iter().map(|h| (h.id, h.score.to_bits())).collect();
                // A bounded k is a prefix of the full ranking.
                let top5 = store.search_scored(&queries.row(qi), 5, mode).unwrap();
                assert_eq!(
                    top5.iter()
                        .map(|h| (h.id, h.score.to_bits()))
                        .collect::<Vec<_>>(),
                    key[..5]
                );
                per_mode.push(key);
            }
            if config.algorithm == QuantAlgorithm::PolarQuant {
                // No QJL stage: nothing was ever divided by a norm; both modes coincide.
                assert_eq!(per_mode[0], per_mode[1]);
            }
            // `search` IS the default mode.
            let compat = store.search(&queries.row(qi), usize::MAX);
            let compat: Vec<(u64, u32)> =
                compat.into_iter().map(|(i, s)| (i, s.to_bits())).collect();
            assert_eq!(compat, per_mode[0], "q{qi}");
        }
    }
}

/// The `ip` mode keeps the F04a property (mean estimate within 2% of the exact inner
/// product for every norm group, same harness and tolerance as `properties.rs`), while
/// the default heuristic is off by exactly the legacy factor sqrt(pi/2)/||x|| — the
/// same control that flags a biased estimator flags it in every group. Two different
/// contracts, both stated.
#[test]
fn ip_mode_keeps_the_unbiased_property_and_the_default_is_flagged_as_heuristic() {
    const DIM: usize = 32;
    const M: usize = 256;
    const SEEDS: u64 = 8;
    const PAIRS: usize = 250;
    for norm in [0.25f32, 1.0, 4.0, 16.0] {
        let mut sum_exact = 0.0f64;
        let mut sum_ip = 0.0f64;
        let mut sum_l2 = 0.0f64;
        for seed in 0..SEEDS {
            let q = QjlQuantizer::new(DIM, M, seed).unwrap();
            for i in 0..PAIRS {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(mixed(seed, i as u64, norm.to_bits() as u64));
                let x = scaled(&mut rng, DIM, norm);
                let noise = gaussian(&mut rng, DIM);
                let beta = norm * 0.5 / (DIM as f32).sqrt();
                let query: Vec<f32> = x
                    .iter()
                    .zip(noise.iter())
                    .map(|(xi, n)| xi + n * beta)
                    .collect();
                let sketch = q.sketch(&x).unwrap();
                sum_exact += f64::from(dot(&x, &query));
                sum_ip += f64::from(q.score(&sketch, &query, ScoreMode::InnerProduct).unwrap());
                sum_l2 += f64::from(q.score(&sketch, &query, ScoreMode::L2Heuristic).unwrap());
            }
        }
        let pairs = (SEEDS as usize * PAIRS) as f64;
        let mean_exact = (sum_exact / pairs) as f32;
        let mean_ip = (sum_ip / pairs) as f32;
        let mean_l2 = (sum_l2 / pairs) as f32;
        let tolerance = 0.02 * mean_exact.abs() + 1e-4;
        assert!(
            (mean_ip - mean_exact).abs() <= tolerance,
            "||x||={norm}: ip mode biased: {mean_ip:.5} vs {mean_exact:.5} (tol {tolerance:.5})"
        );
        assert!(
            (mean_l2 - mean_exact).abs() > tolerance,
            "||x||={norm}: the l2_heuristic must not pass the unbiased-IP control"
        );
        let expected_ratio = (PI / 2.0).sqrt() / norm;
        assert!(
            ((mean_l2 / mean_exact) - expected_ratio).abs() <= 0.05 * expected_ratio,
            "||x||={norm}: l2/ip ratio {} vs legacy factor {expected_ratio}",
            mean_l2 / mean_exact
        );
    }
    // Same check for the TurboQuant combined estimator through the mode API: the sum
    // over projection seeds of the ip mode is unbiased, the l2_heuristic is not.
    let mut rng = ChaCha8Rng::seed_from_u64(mixed(77, 32, 2));
    let x = scaled(&mut rng, 32, 4.0);
    let noise = gaussian(&mut rng, 32);
    let query: Vec<f32> = x
        .iter()
        .zip(noise.iter())
        .map(|(xi, n)| 2.0 * xi + 0.25 * n)
        .collect();
    let exact = dot(&x, &query);
    let mut sum_ip = 0.0f64;
    let mut sum_l2 = 0.0f64;
    for seed in 0..1000u64 {
        let quantizer = TurboQuantizer::new(32, 2, 256, seed).unwrap();
        let code = quantizer.encode(&x).unwrap();
        sum_ip += f64::from(
            quantizer
                .score(&code, &query, ScoreMode::InnerProduct)
                .unwrap(),
        );
        sum_l2 += f64::from(
            quantizer
                .score(&code, &query, ScoreMode::L2Heuristic)
                .unwrap(),
        );
    }
    let mean_ip = (sum_ip / 1000.0) as f32;
    let mean_l2 = (sum_l2 / 1000.0) as f32;
    let tolerance = 0.02 * exact.abs() + 1e-4;
    assert!(
        (mean_ip - exact).abs() <= tolerance,
        "turbo ip: {mean_ip:.5} vs {exact:.5}"
    );
    assert!(
        (mean_l2 - exact).abs() > tolerance,
        "turbo l2_heuristic: {mean_l2:.5} vs {exact:.5}"
    );
}

/// A NaN/±Inf query is rejected BEFORE anything is ranked: `search_scored` returns the
/// error naming the index, the compatibility `search` returns nothing (never NaN
/// scores), and a finite query still ranks every code.
#[test]
fn non_finite_query_is_rejected_before_ranking() {
    let dim = 16;
    for config in three_configs(dim) {
        let mut store = QuantizedStore::new(dim, config.clone()).unwrap();
        let good = Array2::from_shape_fn((6, dim), |(r, c)| {
            (r as f32 + 1.0) * 0.1 + (c as f32) * 0.01
        });
        store.add_batch(&good, 0).unwrap();
        for (value, name) in [
            (f32::NAN, "NaN"),
            (f32::INFINITY, "+Inf"),
            (f32::NEG_INFINITY, "-Inf"),
        ] {
            for pos in [0usize, dim / 2, dim - 1] {
                let mut query: Vec<f32> = good.row(0).to_vec();
                query[pos] = value;
                let view = ndarray::ArrayView1::from(&query);
                for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
                    match store.search_scored(&view, 5, mode) {
                        Err(QuantError::NonFiniteValue { index }) => assert_eq!(index, pos),
                        other => {
                            panic!("{name} at {pos} {mode:?}: must be rejected, got {other:?}")
                        }
                    }
                }
                assert!(
                    store.search(&view, 5).is_empty(),
                    "{name} at {pos}: `search` must not rank a non-finite query"
                );
            }
        }
        // The other rejection: a query of the wrong dimension is an error, not a ranking.
        let short_vec = vec![0.5f32; dim - 1];
        let short = ndarray::ArrayView1::from(&short_vec);
        assert!(matches!(
            store.search_scored(&short, 5, ScoreMode::L2Heuristic),
            Err(QuantError::DimensionMismatch {
                expected,
                got
            }) if expected == dim && got == dim - 1
        ));
        assert!(store.search(&short, 5).is_empty());
        // Finite queries are unaffected; a non-finite query row in measure_recall
        // contributes zero instead of panicking.
        let hits = store
            .search_scored(&good.row(0), usize::MAX, ScoreMode::L2Heuristic)
            .unwrap();
        assert_eq!(hits.len(), 6);
        assert!(hits.iter().all(|h| h.score.is_finite()));
        let mut bad = good.row(1).to_owned();
        bad[0] = f32::NAN;
        let bad_queries =
            Array2::from_shape_fn(
                (2, dim),
                |(r, c)| {
                    if r == 0 {
                        bad[c]
                    } else {
                        good[(2, c)]
                    }
                },
            );
        let recall = store.measure_recall(&bad_queries, &good, 2);
        assert!((0.0..=0.5).contains(&recall), "recall={recall}");
    }
}

// --- Evidence probe -------------------------------------------------------------

#[test]
#[ignore = "evidence probe: prints, run with --ignored --nocapture"]
fn probe_modes() {
    use super::d18_probe::{
        exact_ip_top_k, exact_l2_top_k, gaussian_matrix, mix, overlap, uniform_matrix, HASH_INIT,
    };
    const DIM: usize = 640;
    const N: usize = 400;
    const N_QUERIES: usize = 16;
    const K: usize = 10;
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
            store.add_batch(&data, 0).unwrap();
            let mut top10_per_mode = Vec::new();
            for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
                let mut recall_l2 = 0.0f64;
                let mut recall_ip = 0.0f64;
                let mut ranking_hash = HASH_INIT;
                let mut top10_hash = HASH_INIT;
                let mut top10_all = Vec::new();
                let mut nonfinite = 0usize;
                for qi in 0..N_QUERIES {
                    let query = queries.row(qi).to_vec();
                    let gt_l2 = exact_l2_top_k(&query, &data, K);
                    let gt_ip = exact_ip_top_k(&query, &data, K);
                    let full = store
                        .search_scored(&queries.row(qi), usize::MAX, mode)
                        .unwrap();
                    nonfinite += full.iter().filter(|h| !h.score.is_finite()).count();
                    for hit in &full {
                        ranking_hash = mix(ranking_hash, hit.id);
                        ranking_hash = mix(ranking_hash, hit.score.to_bits() as u64);
                    }
                    let top10: Vec<u64> = full.iter().take(K).map(|h| h.id).collect();
                    top10_hash = top10.iter().fold(top10_hash, |h, id| mix(h, *id));
                    recall_l2 += overlap(&top10, &gt_l2) / K as f64;
                    recall_ip += overlap(&top10, &gt_ip) / K as f64;
                    top10_all.extend(top10);
                }
                top10_per_mode.push(top10_all);
                let measured = store.measure_recall_with_mode(&queries, &data, K, mode);
                let measured_against_l2 =
                    store.measure_recall_against_l2_with_mode(&queries, &data, K, mode);
                println!(
                    "{}",
                    json!({
                        "probe": "d18_modes",
                        "family": family,
                        "config": name,
                        "mode": mode.as_str(),
                        "bits": config.bits,
                        "projections": config.projections,
                        "n": N,
                        "n_queries": N_QUERIES,
                        "k": K,
                        "gt": "exact f64 L2 (recall_vs_exact_l2) | exact f64 inner product (recall_vs_exact_ip)",
                        "recall_vs_exact_l2": recall_l2 / N_QUERIES as f64,
                        "recall_vs_exact_ip": recall_ip / N_QUERIES as f64,
                        "measure_recall_with_mode": measured,
                        "measure_recall_against_l2_with_mode": measured_against_l2,
                        "ranking_hash": format!("{ranking_hash:016x}"),
                        "top10_hash": format!("{top10_hash:016x}"),
                        "nonfinite_scores": nonfinite,
                    })
                );
            }
            let l2: Vec<u64> = top10_per_mode[0].clone();
            let ip: Vec<u64> = top10_per_mode[1].clone();
            let same = l2
                .iter()
                .zip(ip.iter())
                .take(N_QUERIES * K)
                .filter(|(a, b)| a == b)
                .count();
            println!(
                "{}",
                json!({
                    "probe": "d18_modes_agreement",
                    "family": family,
                    "config": name,
                    "top10_positions": N_QUERIES * K,
                    "top10_identical_positions_l2_vs_ip": same,
                })
            );
        }
    }
}
