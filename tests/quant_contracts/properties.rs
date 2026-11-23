//! Evidence harness for the quantization math contracts (A02-F03 / A02-F04 / A02-F07).
//!
//! Two kinds of tests live here:
//!
//! * Strict property tests (run by default):
//!   `qjl_estimator_unbiased_for_non_unit_norms` and
//!   `turbo_combined_estimator_unbiased_across_projection_seeds`. They fail on a biased
//!   estimator.
//! * `probe_*` evidence probes (`#[ignore]`d): they print what the code does — bias per
//!   norm group, recall@k against exact-L2 ground truth, `add_batch` atomicity, memory
//!   metrics — without asserting, so the very same file compiles against the baseline API
//!   (617822d) and against the fixed one and yields comparable numbers:
//!
//!   `cargo test --no-default-features --test quant_contracts -- --ignored --nocapture`
//!
//! Everything is deterministic: fixed seeds, sequential f32 accumulation, no wall clock.

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use splatsdb::quant::{QjlQuantizer, TurboQuantizer};
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore};
use std::collections::HashSet;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Pre-declared parameters. Fixed before the first run; never tuned to an observed error.
// ---------------------------------------------------------------------------
/// Per norm group, the mean estimate must be within 2% of the mean exact inner product.
const REL_TOL: f32 = 0.02;
/// Absolute floor of the tolerance (only guards against a degenerate zero mean).
const ABS_TOL: f32 = 1e-4;
/// None of these equals sqrt(pi/2) ~= 1.2533, the single norm at which the biased
/// estimator was accidentally right.
const NORM_GROUPS: [f32; 4] = [0.25, 1.0, 4.0, 16.0];
const BIAS_DIM: usize = 32;
const BIAS_PROJECTIONS: usize = 256;
const BIAS_SEEDS: u64 = 8;
const PAIRS_PER_SEED: usize = 250;
/// 8 projection matrices x 250 pairs = 2000 pairs per norm group.
const PAIRS_PER_GROUP: usize = BIAS_SEEDS as usize * PAIRS_PER_SEED;

fn mixed(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ b.wrapping_add(0x632B_E599_66D0_24A5)
        ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gaussian(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| StandardNormal.sample(rng)).collect()
}

fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |(_, _)| StandardNormal.sample(&mut rng))
}

fn uniform_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |(_, _)| rng.gen::<f32>())
}

/// Gaussian direction scaled to norm `scale` (norm computed in f64, stored as f32).
fn scaled_unit_vector(rng: &mut ChaCha8Rng, dim: usize, scale: f32) -> Vec<f32> {
    let v = gaussian(rng, dim);
    let n = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    v.iter()
        .map(|x| ((*x as f64) * scale as f64 / n) as f32)
        .collect()
}

/// Reference inner product, accumulated in f64.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>() as f32
}

struct GroupStats {
    norm: f32,
    mean_exact: f32,
    mean_estimate: f32,
    per_seed_ratio: Vec<f32>,
}

/// Mean of the QJL estimate vs the exact inner product, per norm group of the sketched
/// vector `x`. For every pair `query = x + beta * noise` with beta = s / (2 sqrt(d)), so
/// `<x, query>` stays clearly positive (s^2 +- ~9%) and the ratio of means is well
/// conditioned while ||x|| = s is far from sqrt(pi/2).
fn compute_qjl_bias_groups() -> Vec<GroupStats> {
    NORM_GROUPS
        .iter()
        .map(|&s| {
            let mut sum_exact = 0.0f64;
            let mut sum_estimate = 0.0f64;
            let mut per_seed_ratio = Vec::new();
            for seed in 0..BIAS_SEEDS {
                let quantizer = QjlQuantizer::new(BIAS_DIM, BIAS_PROJECTIONS, seed).unwrap();
                let mut seed_exact = 0.0f64;
                let mut seed_estimate = 0.0f64;
                for i in 0..PAIRS_PER_SEED {
                    let mut rng =
                        ChaCha8Rng::seed_from_u64(mixed(seed, i as u64, s.to_bits() as u64));
                    let x = scaled_unit_vector(&mut rng, BIAS_DIM, s);
                    let noise = gaussian(&mut rng, BIAS_DIM);
                    let beta = s * 0.5 / (BIAS_DIM as f32).sqrt();
                    let query: Vec<f32> = x
                        .iter()
                        .zip(noise.iter())
                        .map(|(xi, n)| xi + n * beta)
                        .collect();
                    let exact = dot(&x, &query);
                    let sketch = quantizer.sketch(&x).unwrap();
                    let estimate = quantizer.inner_product_estimate(&sketch, &query).unwrap();
                    sum_exact += f64::from(exact);
                    sum_estimate += f64::from(estimate);
                    seed_exact += f64::from(exact);
                    seed_estimate += f64::from(estimate);
                }
                per_seed_ratio.push((seed_estimate / seed_exact) as f32);
            }
            GroupStats {
                norm: s,
                mean_exact: (sum_exact / PAIRS_PER_GROUP as f64) as f32,
                mean_estimate: (sum_estimate / PAIRS_PER_GROUP as f64) as f32,
                per_seed_ratio,
            }
        })
        .collect()
}

/// Computed once per process and shared by the property test and the evidence probe.
fn qjl_bias_groups() -> &'static [GroupStats] {
    static GROUPS: std::sync::OnceLock<Vec<GroupStats>> = std::sync::OnceLock::new();
    GROUPS.get_or_init(compute_qjl_bias_groups)
}

/// Property (A02-F03): for g ~ N(0, I), E[sign(g.x)(g.q)] = sqrt(2/pi) <x,q>/||x||, so the
/// estimator must be `<x,q> ~ sqrt(pi/2) ||x|| / m * sum_i sign(g_i.x)(g_i.q)` for ANY
/// ||x||. Checked on the mean over 2000 deterministic pairs per norm group.
#[test]
fn qjl_estimator_unbiased_for_non_unit_norms() {
    let mut failures = Vec::new();
    for g in qjl_bias_groups() {
        let tolerance = REL_TOL * g.mean_exact.abs() + ABS_TOL;
        let deviation = (g.mean_estimate - g.mean_exact).abs();
        let line = format!(
            "||x||={:<5} pairs={} mean_estimate={:.5} mean_exact={:.5} ratio={:.5} tolerance={:.5}",
            g.norm,
            PAIRS_PER_GROUP,
            g.mean_estimate,
            g.mean_exact,
            g.mean_estimate / g.mean_exact,
            tolerance
        );
        if !matches!(
            deviation.partial_cmp(&tolerance),
            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
        ) {
            failures.push(format!("biased: {line}"));
        }
        // Control: the pre-fix estimator equals today's estimate times sqrt(pi/2)/||x||.
        // The same harness and the same tolerance must flag it in every group —
        // otherwise the tolerance would be wider than the effect it has to detect.
        let legacy_mean = g.mean_estimate * (PI / 2.0).sqrt() / g.norm;
        if (legacy_mean - g.mean_exact).abs() <= tolerance {
            failures.push(format!("control not flagged: {line}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Property (A02-F03, TurboQuant): averaged over projection matrices, polar estimate +
/// QJL residual correction must equal <x,q> even when the residual norm is neither 1 nor
/// sqrt(pi/2).
#[test]
fn turbo_combined_estimator_unbiased_across_projection_seeds() {
    const DIM: usize = 32;
    const BITS: u8 = 2;
    const PROJECTIONS: usize = 256;
    const SEEDS: u64 = 1000;
    let mut rng = ChaCha8Rng::seed_from_u64(mixed(77, DIM as u64, BITS as u64));
    // ||x|| = 4; query = 2x + noise keeps <x, query> large and positive.
    let x = scaled_unit_vector(&mut rng, DIM, 4.0);
    let noise = gaussian(&mut rng, DIM);
    let query: Vec<f32> = x
        .iter()
        .zip(noise.iter())
        .map(|(xi, n)| 2.0 * xi + 0.25 * n)
        .collect();
    let exact = dot(&x, &query);
    let mut sum = 0.0f64;
    let mut residual_norm = 0.0f32;
    for seed in 0..SEEDS {
        let quantizer = TurboQuantizer::new(DIM, BITS, PROJECTIONS, seed).unwrap();
        let code = quantizer.encode(&x).unwrap();
        sum += f64::from(quantizer.inner_product_estimate(&code, &query).unwrap());
        if seed == 0 {
            let reconstruction = quantizer.decode_approximate(&code).unwrap();
            let residual: Vec<f32> = x
                .iter()
                .zip(reconstruction.iter())
                .map(|(o, r)| o - r)
                .collect();
            residual_norm = dot(&residual, &residual).sqrt();
        }
    }
    // Guard: the residual norm must not sit at an accidental fix point of the biased
    // estimator (unbiased there only for ||residual|| = sqrt(pi/2)).
    assert!(
        (residual_norm - 1.0).abs() > 0.5 && (residual_norm - (PI / 2.0).sqrt()).abs() > 0.5,
        "residual_norm={residual_norm}"
    );
    let mean = (sum / SEEDS as f64) as f32;
    let tolerance = REL_TOL * exact.abs() + ABS_TOL;
    println!(
        "turbo: mean={mean:.5} exact={exact:.5} ratio={:.5} residual_norm={residual_norm:.4} tolerance={tolerance:.5}",
        mean / exact
    );
    assert!(
        (mean - exact).abs() <= tolerance,
        "mean over {SEEDS} projection seeds = {mean:.5}, exact = {exact:.5}, tolerance = {tolerance:.5}"
    );
}

// ---------------------------------------------------------------------------
// Evidence probes: print only.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "evidence probe: prints, run with --ignored --nocapture"]
fn probe_qjl_bias() {
    for g in qjl_bias_groups() {
        println!(
            "{}",
            json!({
                "probe": "qjl_bias",
                "dim": BIAS_DIM,
                "projections": BIAS_PROJECTIONS,
                "pairs": PAIRS_PER_GROUP,
                "group_norm": g.norm,
                "mean_exact": g.mean_exact,
                "mean_estimate": g.mean_estimate,
                "observed_ratio": g.mean_estimate / g.mean_exact,
                "per_seed_ratio": g.per_seed_ratio,
                "predicted_legacy_ratio_sqrt_pi_over_2_over_norm": (PI / 2.0).sqrt() / g.norm,
                "tolerance_rel": REL_TOL,
                "tolerance_abs": ABS_TOL,
            })
        );
    }
}

/// Exact-L2 ground truth in f64: indices of the k nearest rows of `data` (ties by index).
fn exact_l2_top_k(query: &[f32], data: &Array2<f32>, k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f64)> = data
        .outer_iter()
        .enumerate()
        .map(|(i, row)| {
            let d: f64 = row
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let diff = (*a as f64) - (*b as f64);
                    diff * diff
                })
                .sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    dists.truncate(k);
    dists.into_iter().map(|(i, _)| i).collect()
}

fn harness_recall(
    store: &QuantizedStore,
    queries: &Array2<f32>,
    data: &Array2<f32>,
    k: usize,
) -> f32 {
    let mut total = 0.0f64;
    for qi in 0..queries.nrows() {
        let query = queries.row(qi);
        let gt: HashSet<usize> = exact_l2_top_k(&query.to_vec(), data, k)
            .into_iter()
            .collect();
        let got: HashSet<usize> = store
            .search(&query, k)
            .into_iter()
            .map(|(id, _)| id as usize)
            .collect();
        total += gt.intersection(&got).count() as f64 / k as f64;
    }
    (total / queries.nrows() as f64) as f32
}

#[test]
#[ignore = "evidence probe: prints, run with --ignored --nocapture"]
fn probe_recall_vs_exact_l2() {
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
            let _ = store.add_batch(&data, 0);
            let store_recall = store.measure_recall(&queries, &data, K);
            let harness = harness_recall(&store, &queries, &data, K);
            println!(
                "{}",
                json!({
                    "probe": "recall",
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
                    "recall_measure_recall": store_recall,
                    "recall_harness_f64_gt": harness,
                })
            );
        }
    }
}

fn snapshot(
    store: &QuantizedStore,
    query: &ndarray::ArrayView1<f32>,
    step: &str,
) -> serde_json::Value {
    let results = store.search(query, usize::MAX);
    json!({
        "probe": "add_batch",
        "step": step,
        "len": store.len(),
        "searchable_codes": results.len(),
        "nonfinite_scores": results.iter().filter(|(_, s)| !s.is_finite()).count(),
    })
}

#[test]
#[ignore = "evidence probe: prints, run with --ignored --nocapture"]
fn probe_add_batch_atomicity() {
    let dim = 16;
    let mut store = QuantizedStore::new(dim, QuantConfig::for_search(dim)).unwrap();
    let good = uniform_matrix(8, dim, 7);
    let _ = store.add_batch(&good, 0);
    let query = good.row(0);
    println!("{}", snapshot(&store, &query, "after 8 valid rows"));

    let wrong_shape = uniform_matrix(4, dim + 1, 8);
    let _ = store.add_batch(&wrong_shape, 8);
    println!(
        "{}",
        snapshot(&store, &query, "after rejected? batch ncols=dim+1 x4 rows")
    );

    let mut with_nan = uniform_matrix(4, dim, 9);
    with_nan[[1, 3]] = f32::NAN;
    let _ = store.add_batch(&with_nan, 12);
    println!(
        "{}",
        snapshot(&store, &query, "after batch with one NaN x4 rows")
    );

    let mut with_inf = uniform_matrix(4, dim, 10);
    with_inf[[2, 5]] = f32::INFINITY;
    let _ = store.add_batch(&with_inf, 16);
    println!(
        "{}",
        snapshot(&store, &query, "after batch with one +Inf x4 rows")
    );
}

#[test]
#[ignore = "evidence probe: prints, run with --ignored --nocapture"]
fn probe_memory_metrics() {
    const DIM: usize = 640;
    const N: usize = 1000;
    let vectors = uniform_matrix(N, DIM, 11);
    for bits in [2u8, 4, 8] {
        for algorithm in [QuantAlgorithm::TurboQuant, QuantAlgorithm::PolarQuant] {
            let turbo = matches!(algorithm, QuantAlgorithm::TurboQuant);
            let config = QuantConfig {
                bits,
                projections: if turbo { DIM / 8 } else { 0 },
                seed: 42,
                algorithm,
            };
            let mut store = QuantizedStore::new(DIM, config).unwrap();
            let _ = store.add_batch(&vectors, 0);
            let ratio = store.compression_ratio();
            println!(
                "{}",
                json!({
                    "probe": "memory",
                    "bits": bits,
                    "algorithm": format!("{algorithm:?}"),
                    "n": N,
                    "dim": DIM,
                    "len": store.len(),
                    "memory_usage": store.memory_usage(),
                    "raw_bytes": N * DIM * 4,
                    "compression_ratio": ratio,
                    "bytes_per_vector_implied_by_ratio": ratio.recip() * DIM as f32 * 4.0,
                    "nominal_ratio_32_over_bits": 32.0 / bits as f32,
                    "unpacked_payload_layout_bytes": N * (3 * DIM + if turbo { DIM / 8 } else { 0 }),
                })
            );
        }
    }
}
