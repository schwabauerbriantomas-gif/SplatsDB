//! F04d — exact cos/sin table per bit width: bit-exact parity of the score.
//!
//! The score evaluates `cos`/`sin` of the dequantized angle once per (code, pair, query),
//! although that angle depends only on `(angle index, bits)`. A table per bit width removes
//! the repeated trigonometry; these tests hold the candidate to the pre-table behaviour with
//! a reference written from the formulas only (raw dequantization + `f32::cos`/`f32::sin`,
//! never through the table): every f32 score bit, every id and every order position must be
//! identical for PolarQuant and TurboQuant, both `ScoreMode`s, even and odd bit widths and
//! norms 0 / 1e-4 / 1 / 1e4, and the rejected inputs must keep their behaviour (non-finite
//! query, wrong dimension, foreign or malformed code).
//!
//! `probe_lut_bytes` (ignored, run alone) measures the heap bytes the first search leaves
//! behind — the table — with the counting global allocator of `ab_bench.rs`.
//!
//! `cargo test --no-default-features --test quant_contracts trig_lut`

use super::ab_bench::LIVE;
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use splatsdb::quant::{
    PolarCode, PolarQuantizer, QjlQuantizer, QuantError, TurboCode, TurboQuantizer,
};
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore, ScoreMode};
use std::f32::consts::PI;
use std::sync::atomic::Ordering::Relaxed;

const MODES: [ScoreMode; 2] = [ScoreMode::L2Heuristic, ScoreMode::InnerProduct];
const NORMS: [f32; 4] = [0.0, 1e-4, 1.0, 1e4];
/// PolarQuant bit widths: even, odd, both edges and the widest table.
const POLAR_BITS: [u8; 11] = [1, 2, 3, 4, 5, 7, 8, 9, 11, 15, 16];
/// TurboQuant bit widths (the polar stage scores with `bits - 1`).
const TURBO_BITS: [u8; 6] = [2, 3, 4, 5, 8, 16];

// ---------------------------------------------------------------------------
// Reference: the formulas as the hot loop evaluated them before the table.
// ---------------------------------------------------------------------------

/// The dequantized angle, written out from the definition (f32, same constants).
fn raw_dequantize(index: u16, bits: u8) -> f32 {
    let levels = 1u32 << bits;
    (index as f32 / levels as f32) * (2.0 * PI) - PI
}

/// The polar estimate exactly as the hot loop evaluated it before the table: direct
/// `f32::cos` / `f32::sin` per pair, same accumulation order and types.
fn reference_polar_estimate(code: &PolarCode, rotated_query: &[f32]) -> f32 {
    let mut estimate = 0.0f32;
    for i in 0..code.dim / 2 {
        let theta = raw_dequantize(code.angle_indices[i], code.bits);
        let r = code.radii[i];
        let q_a = rotated_query[2 * i];
        let q_b = rotated_query[2 * i + 1];
        estimate += r * (q_a * theta.cos() + q_b * theta.sin());
    }
    estimate
}

/// QJL projection rows with the seed `QjlQuantizer` derives (as `ranking.rs`).
fn reference_projection(dim: usize, projections: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF_1234_5678));
    (0..projections)
        .map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect()
}

fn reference_qjl_sum(g: &[Vec<f32>], signs: &[i8], query: &[f32]) -> f32 {
    g.iter()
        .zip(signs.iter())
        .map(|(row, &sign)| {
            let g_dot_query: f32 = row.iter().zip(query.iter()).map(|(g, q)| g * q).sum();
            sign as f32 * g_dot_query
        })
        .sum()
}

/// Reference QJL term of the residual sketch of `x` (legacy and unbiased scales).
fn reference_qjl_term(
    x: &[f32],
    query: &[f32],
    polar: &PolarQuantizer,
    qjl: &QjlQuantizer,
    g: &[Vec<f32>],
    mode: ScoreMode,
) -> f32 {
    let code = polar.encode(x).unwrap();
    let reconstruction = polar.decode(&code).unwrap();
    let residual: Vec<f32> = x
        .iter()
        .zip(reconstruction.iter())
        .map(|(o, r)| o - r)
        .collect();
    let sketch = qjl.sketch(&residual).unwrap();
    let m = sketch.signs.len() as f32;
    let scale = match mode {
        ScoreMode::L2Heuristic => PI / (2.0 * m),
        ScoreMode::InnerProduct => (PI / 2.0).sqrt() * sketch.norm / m,
    };
    scale * reference_qjl_sum(g, &sketch.signs, query)
}

/// Full reference score of one TurboQuant row: reference polar estimate + QJL term.
fn reference_turbo_score(
    x: &[f32],
    query: &[f32],
    rotated_query: &[f32],
    polar: &PolarQuantizer,
    qjl: &QjlQuantizer,
    g: &[Vec<f32>],
    mode: ScoreMode,
) -> f32 {
    let code = polar.encode(x).unwrap();
    let reconstruction = polar.decode(&code).unwrap();
    let residual: Vec<f32> = x
        .iter()
        .zip(reconstruction.iter())
        .map(|(o, r)| o - r)
        .collect();
    let sketch = qjl.sketch(&residual).unwrap();
    let m = sketch.signs.len() as f32;
    let scale = match mode {
        ScoreMode::L2Heuristic => PI / (2.0 * m),
        ScoreMode::InnerProduct => (PI / 2.0).sqrt() * sketch.norm / m,
    };
    reference_polar_estimate(&code, rotated_query)
        + scale * reference_qjl_sum(g, &sketch.signs, query)
}

/// The order of the store: stable sort, higher score first (ties keep insertion order).
fn reference_ranking(scores: &[f32]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

fn gaussian(mut rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
}

/// A gaussian vector rescaled to `norm` (the zero vector for `norm == 0`).
fn with_norm(mut x: Vec<f32>, norm: f32) -> Vec<f32> {
    if norm == 0.0 {
        return vec![0.0; x.len()];
    }
    let raw: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    for value in &mut x {
        *value = *value / raw * norm;
    }
    x
}

/// `QuantizedStore::search` takes an `ArrayView1`.
fn view(vector: &[f32]) -> ArrayView1<'_, f32> {
    ArrayView1::from(vector)
}

fn assert_same_bits(got: f32, expected: f32, what: &str) {
    assert_eq!(
        got.to_bits(),
        expected.to_bits(),
        "{}: got {:e} ({:#010x}), reference {:e} ({:#010x})",
        what,
        got,
        got.to_bits(),
        expected,
        expected.to_bits()
    );
}

// ---------------------------------------------------------------------------
// Parity
// ---------------------------------------------------------------------------

#[test]
fn polar_score_is_bit_identical_to_the_direct_trig_loop() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xF04D);
    let mut compared = 0usize;
    // Every width at small and mid dimensions, then d=640 (the A/B dimension) at three
    // widths: building one 640x640 rotation is O(d^3), slow in a debug test.
    let cases: Vec<(u8, usize)> = POLAR_BITS
        .iter()
        .flat_map(|&bits| [2usize, 6, 16, 130].map(move |dim| (bits, dim)))
        .chain([1u8, 4, 16].map(|bits| (bits, 640usize)))
        .collect();
    for (bits, dim) in cases {
        let polar = PolarQuantizer::new(dim, bits, 7).unwrap();
        let query = gaussian(&mut rng, dim);
        let rotated = polar.rotated_query(&query).unwrap();
        for norm in NORMS {
            for _ in 0..2 {
                let x = with_norm(gaussian(&mut rng, dim), norm);
                let code = polar.encode(&x).unwrap();
                let expected = reference_polar_estimate(&code, &rotated);
                assert_same_bits(
                    polar
                        .inner_product_estimate_rotated(&code, &rotated)
                        .unwrap(),
                    expected,
                    &format!("rotated bits={bits} dim={dim} norm={norm}"),
                );
                // The one-shot path is the rotated path (same bits).
                assert_same_bits(
                    polar.inner_product_estimate(&code, &query).unwrap(),
                    expected,
                    &format!("one-shot bits={bits} dim={dim} norm={norm}"),
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, (POLAR_BITS.len() * 4 + 3) * NORMS.len() * 2);
}

#[test]
fn turbo_score_is_bit_identical_to_the_direct_trig_loop_in_both_modes() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xF04D + 1);
    for &bits in &TURBO_BITS {
        for dim in [16usize, 130, 640] {
            for projections in [1usize, 7, 256] {
                // d=640 (the A/B dimension) only for the A/B configuration and the widest
                // table: building one 640x640 rotation is O(d^3), slow in a debug test.
                if dim == 640
                    && !((bits == 4 && projections == 256) || (bits == 16 && projections == 7))
                {
                    continue;
                }
                let turbo = TurboQuantizer::new(dim, bits, projections, 42).unwrap();
                // The seeds `TurboQuantizer::new` derives (checked on the polar code below).
                let polar = PolarQuantizer::new(dim, bits - 1, 42).unwrap();
                let qjl_seed = 42u64.wrapping_add(0xCAFE_BABE_0000_0001);
                let qjl = QjlQuantizer::new(dim, projections, qjl_seed).unwrap();
                let g = reference_projection(dim, projections, qjl_seed);
                let query = gaussian(&mut rng, dim);
                let rotated = polar.rotated_query(&query).unwrap();
                let context = turbo.prepare_query(&query).unwrap();
                for norm in NORMS {
                    for _ in 0..if dim == 640 { 1 } else { 2 } {
                        let x = with_norm(gaussian(&mut rng, dim), norm);
                        let code = turbo.encode(&x).unwrap();
                        assert_eq!(code.polar_code, polar.encode(&x).unwrap());
                        for mode in MODES {
                            let what = format!(
                                "{mode:?} bits={bits} dim={dim} m={projections} norm={norm}"
                            );
                            let expected =
                                reference_turbo_score(&x, &query, &rotated, &polar, &qjl, &g, mode);
                            assert_same_bits(
                                turbo.score(&code, &query, mode).unwrap(),
                                expected,
                                &what,
                            );
                            assert_same_bits(
                                turbo.score_projected(&code, &context, mode).unwrap(),
                                expected,
                                &what,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn store_ranking_is_bit_identical_to_the_reference_in_both_modes() {
    let dim = 64;
    let mut rng = ChaCha8Rng::seed_from_u64(0xF04D + 2);
    // 120 rows: gaussians, a zero row, a 1e-4-norm row and a 1e4-norm row.
    let rows: Vec<Vec<f32>> = (0..120)
        .map(|i| {
            let x = gaussian(&mut rng, dim);
            match i {
                10 => vec![0.0; dim],
                11 => with_norm(x, 1e-4),
                12 => with_norm(x, 1e4),
                _ => x,
            }
        })
        .collect();
    let data = Array2::from_shape_fn((rows.len(), dim), |(r, c)| rows[r][c]);
    let mut configs = Vec::new();
    for bits in [2u8, 3, 4, 8] {
        configs.push(QuantConfig {
            bits,
            projections: 16,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        });
    }
    for bits in [1u8, 3, 4, 16] {
        configs.push(QuantConfig {
            bits,
            projections: 0,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        });
    }
    let mut compared = 0usize;
    for config in &configs {
        let mut store = QuantizedStore::new(dim, config.clone()).unwrap();
        store.add_batch(&data, 0).unwrap();
        // The reference pipeline of this config, built from the public encoders only.
        let (polar, qjl, g) = match config.algorithm {
            QuantAlgorithm::TurboQuant => {
                let qjl_seed = config.seed.wrapping_add(0xCAFE_BABE_0000_0001);
                (
                    PolarQuantizer::new(dim, config.bits - 1, config.seed).unwrap(),
                    Some(QjlQuantizer::new(dim, config.projections, qjl_seed).unwrap()),
                    Some(reference_projection(dim, config.projections, qjl_seed)),
                )
            }
            QuantAlgorithm::PolarQuant => (
                PolarQuantizer::new(dim, config.bits, config.seed).unwrap(),
                None,
                None,
            ),
        };
        for qi in 0..4 {
            let query = with_norm(gaussian(&mut rng, dim), NORMS[qi]);
            let rotated = polar.rotated_query(&query).unwrap();
            for mode in MODES {
                let reference: Vec<f32> = rows
                    .iter()
                    .map(|x| match (&qjl, &g) {
                        (Some(qjl), Some(g)) => {
                            reference_turbo_score(x, &query, &rotated, &polar, qjl, g, mode)
                        }
                        _ => reference_polar_estimate(&polar.encode(x).unwrap(), &rotated),
                    })
                    .collect();
                let expected = reference_ranking(&reference);
                let hits = store
                    .search_scored(&view(&query), usize::MAX, mode)
                    .unwrap();
                assert_eq!(hits.len(), rows.len());
                for (position, (hit, (row, score))) in hits.iter().zip(&expected).enumerate() {
                    let what = format!(
                        "{mode:?} {:?} bits={} position {position}",
                        config.algorithm, config.bits
                    );
                    assert_eq!(hit.id, *row as u64, "{what}");
                    assert_eq!(hit.mode, mode, "{what}");
                    assert_same_bits(hit.score, *score, &what);
                }
                compared += hits.len();
            }
            // The default `search` is the `L2Heuristic` ranking, bits included.
            let default = store.search(&view(&query), usize::MAX);
            let explicit = store
                .search_scored(&view(&query), usize::MAX, ScoreMode::L2Heuristic)
                .unwrap();
            for ((id, score), hit) in default.iter().zip(&explicit) {
                assert_eq!(*id, hit.id);
                assert_same_bits(*score, hit.score, "search vs search_scored");
            }
        }
    }
    assert_eq!(compared, configs.len() * 4 * 2 * rows.len());
}

#[test]
fn foreign_and_malformed_codes_keep_their_behaviour() {
    let dim = 16;
    let mut rng = ChaCha8Rng::seed_from_u64(0xF04D + 3);
    let x = gaussian(&mut rng, dim);
    let query = gaussian(&mut rng, dim);
    let polar = PolarQuantizer::new(dim, 3, 5).unwrap();
    let code = polar.encode(&x).unwrap();
    let rotated = polar.rotated_query(&query).unwrap();
    let turbo = TurboQuantizer::new(dim, 4, 8, 5).unwrap();
    let turbo_code = turbo.encode(&x).unwrap();
    let qjl_seed = 5u64.wrapping_add(0xCAFE_BABE_0000_0001);
    let qjl = QjlQuantizer::new(dim, 8, qjl_seed).unwrap();
    let g = reference_projection(dim, 8, qjl_seed);

    // A code of another bit width / dimension is rejected before anything is scored.
    let foreign_bits = PolarCode {
        bits: 4,
        ..code.clone()
    };
    assert!(matches!(
        polar.inner_product_estimate(&foreign_bits, &query),
        Err(QuantError::MalformedCode(_))
    ));
    let foreign_dim = PolarCode {
        dim: 14,
        ..code.clone()
    };
    assert!(matches!(
        polar.inner_product_estimate_rotated(&foreign_dim, &rotated),
        Err(QuantError::DimensionMismatch {
            expected: 16,
            got: 14
        })
    ));
    let foreign_turbo = TurboCode {
        polar_code: foreign_bits,
        ..turbo_code.clone()
    };
    assert!(matches!(
        turbo.score(&foreign_turbo, &query, ScoreMode::InnerProduct),
        Err(QuantError::MalformedCode(_))
    ));
    // A rotated query of another length is rejected.
    assert!(matches!(
        polar.inner_product_estimate_rotated(&code, &rotated[..15]),
        Err(QuantError::DimensionMismatch {
            expected: 16,
            got: 15
        })
    ));

    // An angle index outside [0, 2^bits) is not rejected before scoring: it must yield
    // the direct value of that code (no new panic from the table).
    for bad in [8u16, 100, 4096, u16::MAX] {
        for position in [0usize, 3, 7] {
            let mut malformed = code.clone();
            malformed.angle_indices[position] = bad;
            let expected = reference_polar_estimate(&malformed, &rotated);
            assert_same_bits(
                polar
                    .inner_product_estimate_rotated(&malformed, &rotated)
                    .unwrap(),
                expected,
                &format!("polar index {bad} at {position}"),
            );
            let mut malformed_turbo = turbo_code.clone();
            malformed_turbo.polar_code = malformed.clone();
            for mode in MODES {
                // The sketch is untouched: reference polar estimate of the mutated code
                // + the reference QJL term of x.
                let expected = expected + reference_qjl_term(&x, &query, &polar, &qjl, &g, mode);
                assert_same_bits(
                    turbo.score(&malformed_turbo, &query, mode).unwrap(),
                    expected,
                    &format!("turbo {mode:?} index {bad} at {position}"),
                );
            }
        }
    }
}

#[test]
fn non_finite_and_wrong_dimension_queries_keep_their_behaviour() {
    let dim = 16;
    let mut rng = ChaCha8Rng::seed_from_u64(0xF04D + 4);
    let x = gaussian(&mut rng, dim);
    let polar = PolarQuantizer::new(dim, 4, 9).unwrap();
    let code = polar.encode(&x).unwrap();
    let turbo = TurboQuantizer::new(dim, 5, 8, 9).unwrap();
    let turbo_code = turbo.encode(&x).unwrap();
    let batch: Vec<Vec<f32>> = (0..4).map(|_| gaussian(&mut rng, dim)).collect();
    let mut store = QuantizedStore::new(
        dim,
        QuantConfig {
            bits: 5,
            projections: 8,
            seed: 9,
            algorithm: QuantAlgorithm::TurboQuant,
        },
    )
    .unwrap();
    store
        .add_batch(&Array2::from_shape_fn((4, dim), |(r, c)| batch[r][c]), 0)
        .unwrap();

    for (value, name) in [
        (f32::NAN, "NaN"),
        (f32::INFINITY, "+Inf"),
        (f32::NEG_INFINITY, "-Inf"),
    ] {
        let mut query = x.clone();
        query[3] = value;
        for result in [
            polar.inner_product_estimate(&code, &query).map(|_| ()),
            polar.rotated_query(&query).map(|_| ()),
            turbo
                .score(&turbo_code, &query, ScoreMode::InnerProduct)
                .map(|_| ()),
            turbo.prepare_query(&query).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(QuantError::NonFiniteValue { index: 3 })),
                "{name}"
            );
        }
        assert!(matches!(
            store.search_scored(&view(&query), 4, ScoreMode::L2Heuristic),
            Err(QuantError::NonFiniteValue { index: 3 })
        ));
        assert!(store.search(&view(&query), 4).is_empty(), "{name}");
    }
    // Wrong dimension.
    let short = x[..dim - 1].to_vec();
    for result in [
        polar.inner_product_estimate(&code, &short).map(|_| ()),
        polar.rotated_query(&short).map(|_| ()),
        turbo
            .score(&turbo_code, &short, ScoreMode::L2Heuristic)
            .map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(QuantError::DimensionMismatch {
                expected: 16,
                got: 15
            })
        ));
    }
    assert!(matches!(
        store.search_scored(&view(&short), 4, ScoreMode::L2Heuristic),
        Err(QuantError::DimensionMismatch {
            expected: 16,
            got: 15
        })
    ));
    assert!(store.search(&view(&short), 4).is_empty());
}

#[test]
#[ignore = "evidence probe (F04d): prints the heap bytes left by the first search (the table), run alone"]
fn probe_lut_bytes() {
    let env =
        |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_string());
    let bits: u8 = env("AB_BITS", "4").parse().unwrap();
    let polar_only = env("AB_ALGO", "turbo") == "polar";
    let dim = 64;
    let mut rng = ChaCha8Rng::seed_from_u64(0xF04D);
    let batch: Vec<Vec<f32>> = (0..64).map(|_| gaussian(&mut rng, dim)).collect();
    let query: Vec<f32> = gaussian(&mut rng, dim);
    let config = QuantConfig {
        bits,
        projections: 16,
        seed: 42,
        algorithm: if polar_only {
            QuantAlgorithm::PolarQuant
        } else {
            QuantAlgorithm::TurboQuant
        },
    };
    let mut store = QuantizedStore::new(dim, config).unwrap();
    store
        .add_batch(&Array2::from_shape_fn((64, dim), |(r, c)| batch[r][c]), 0)
        .unwrap();

    let before = LIVE.load(Relaxed);
    let hits = std::hint::black_box(store.search(&view(&query), 10));
    let retained = LIVE.load(Relaxed).saturating_sub(before);
    let entries = 1usize << (if polar_only { bits } else { bits - 1 });
    println!(
        "{}",
        json!({
            "probe": "lut_bytes",
            "bits": bits,
            "algorithm": if polar_only { "polar" } else { "turbo" },
            "table_entries": entries,
            "table_bytes_nominal": entries * std::mem::size_of::<[f32; 2]>(),
            "heap_retained_by_first_search_bytes": retained,
            "hits": hits.len(),
            "memory_usage": store.memory_usage(),
        })
    );
}
