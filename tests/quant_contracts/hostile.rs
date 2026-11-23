//! F04e — fail-closed quantization: hostile inputs, hostile codes, hostile sketches.
//!
//! First written in RED form against the API of c2c6206 only, so that every assertion
//! was observed failing on that commit (11 of 12 tests; log and file hash in
//! reports/F04e.json): a finite input whose radius overflows, hostile codes (lengths,
//! radii, angle indices), QJL signs other than +/-1, the bytes of the cos/sin table at
//! widths without a table, and the recall of the `ip` mode.
//!
//! This GREEN form keeps those assertions, tightens them to the explicit error variants
//! and adds: the justification of the store's fast path (`encode` only emits valid
//! codes), the "no ranking with a hole" contract of a search, hostile quantizer payloads
//! and the cross-metric recall against L2 ground truth.
use super::d18_probe::{exact_ip_top_k, exact_l2_top_k, gaussian_matrix, overlap};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use splatsdb::quant::turbo::TurboQueryContext;
use splatsdb::quant::{
    PolarCode, PolarQuantizer, QjlQuantizer, QjlSketch, QuantError, TurboCode, TurboQuantizer,
};
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore, ScoreMode};

const MODES: [ScoreMode; 2] = [ScoreMode::L2Heuristic, ScoreMode::InnerProduct];
const DIM: usize = 16;

fn view(v: &[f32]) -> ArrayView1<'_, f32> {
    ArrayView1::from(v)
}

fn x_vector() -> Vec<f32> {
    (0..DIM).map(|i| (i as f32) * 0.1 - 0.8).collect()
}

fn query_vector() -> Vec<f32> {
    (0..DIM).map(|i| 0.05 * i as f32 - 0.4).collect()
}

/// 8 finite rows of d=16 in `[-mag/2, mag/2)` (the rows of the validator's probe).
fn scaled_batch(mag: f32) -> Array2<f32> {
    let mut batch = super::uniform_matrix(8, DIM, 5);
    for r in 0..8 {
        for c in 0..DIM {
            batch[[r, c]] = (batch[[r, c]] - 0.5) * mag;
        }
    }
    batch
}

fn polar_config(bits: u8) -> QuantConfig {
    QuantConfig {
        bits,
        projections: 0,
        seed: 42,
        algorithm: QuantAlgorithm::PolarQuant,
    }
}

fn turbo_config(bits: u8) -> QuantConfig {
    QuantConfig {
        bits,
        projections: 8,
        seed: 42,
        algorithm: QuantAlgorithm::TurboQuant,
    }
}

// ---------------------------------------------------------------------------
// 2. PolarQuant: a finite input whose rotated radius is Inf is an Err, never a code.
// ---------------------------------------------------------------------------

#[test]
fn polar_encode_rejects_a_finite_input_whose_radius_overflows() {
    let polar = PolarQuantizer::new(DIM, 4, 42).unwrap();
    let turbo = TurboQuantizer::new(DIM, 4, 8, 42).unwrap();
    // Below the overflow every radius is finite and the row is a code.
    for mag in [1e10f32, 1e18] {
        for row in 0..8 {
            let x: Vec<f32> = scaled_batch(mag).row(row).to_vec();
            assert!(x.iter().all(|v| v.is_finite()));
            let code = polar.encode(&x).expect("finite radii are codes");
            assert!(code.radii.iter().all(|r| r.is_finite()));
            assert!(turbo.encode(&x).is_ok());
        }
    }
    for mag in [1e20f32, 1e30, 3e38] {
        for row in 0..8 {
            let x: Vec<f32> = scaled_batch(mag).row(row).to_vec();
            assert!(x.iter().all(|v| v.is_finite()), "every input is finite");
            // The first pair whose rotated radius overflows, from the public rotated query.
            let rotated = polar.rotated_query(&x).unwrap();
            let expected = (0..DIM / 2)
                .find(|&i| {
                    let a = rotated[2 * i];
                    let b = rotated[2 * i + 1];
                    !(a * a + b * b).sqrt().is_finite()
                })
                .expect("some pair overflows");
            let err = polar
                .encode(&x)
                .expect_err("an Inf radius must not become a code");
            assert!(
                matches!(err, QuantError::NonFiniteRadius { pair } if pair == expected),
                "mag={mag} row={row}: expected pair {expected}, got {err:?}"
            );
            // The polar stage of TurboQuant rotates with the same matrix: same first pair.
            let err = turbo
                .encode(&x)
                .expect_err("turbo stores no Inf radius either");
            assert!(
                matches!(err, QuantError::NonFiniteRadius { pair } if pair == expected),
                "turbo mag={mag} row={row}: expected pair {expected}, got {err:?}"
            );
        }
    }
}

#[test]
fn add_batch_is_atomic_when_a_finite_row_overflows_the_radius() {
    for config in [polar_config(4), turbo_config(4)] {
        for mag in [1e20f32, 1e30, 3e38] {
            let mut store = QuantizedStore::new(DIM, config.clone()).unwrap();
            let good = super::uniform_matrix(3, DIM, 9);
            store.add_batch(&good, 0).unwrap();
            let before = (
                store.len(),
                store.memory_usage(),
                store.search(&good.row(0), usize::MAX),
            );
            let err = store
                .add_batch(&scaled_batch(mag), 3)
                .expect_err("an overflowing finite row rejects the whole batch");
            assert!(
                err.contains("row ") && err.contains("non-finite radius"),
                "{err}"
            );
            let after = (
                store.len(),
                store.memory_usage(),
                store.search(&good.row(0), usize::MAX),
            );
            assert_eq!(before, after, "len, memory and ranking untouched");
        }
    }
    // A fresh store: Err and 0 new codes.
    for config in [polar_config(4), turbo_config(4)] {
        let mut store = QuantizedStore::new(DIM, config).unwrap();
        assert!(store.add_batch(&scaled_batch(1e20), 0).is_err());
        assert_eq!(store.len(), 0);
        assert!(store.search(&view(&[0.5; DIM]), usize::MAX).is_empty());
    }
}

// ---------------------------------------------------------------------------
// 3. Hostile codes: Err on every path (manual, serde, decode, direct score, store).
// ---------------------------------------------------------------------------

struct Fixture {
    polar: PolarQuantizer,
    turbo: TurboQuantizer,
    x: Vec<f32>,
    query: Vec<f32>,
    rotated: Vec<f32>,
    context: TurboQueryContext,
    code: PolarCode,
    turbo_code: TurboCode,
}

impl Fixture {
    /// The polar stage of `turbo` has exactly `bits` angle bits.
    fn new(bits: u8) -> Self {
        let polar = PolarQuantizer::new(DIM, bits, 42).unwrap();
        let turbo = TurboQuantizer::new(DIM, bits + 1, 8, 42).unwrap();
        let x = x_vector();
        let query = query_vector();
        let rotated = polar.rotated_query(&query).unwrap();
        let context = turbo.prepare_query(&query).unwrap();
        let code = polar.encode(&x).unwrap();
        let turbo_code = turbo.encode(&x).unwrap();
        Self {
            polar,
            turbo,
            x,
            query,
            rotated,
            context,
            code,
            turbo_code,
        }
    }

    /// `mutate` applied to a polar code and to the polar stage of a turbo code: every
    /// consumer answers `Err(MalformedCode)`, nothing panics, nothing returns a number.
    fn assert_rejected_everywhere(&self, mutate: impl Fn(&mut PolarCode), what: &str) {
        let mut bad = self.code.clone();
        mutate(&mut bad);
        let mut bad_turbo = self.turbo_code.clone();
        mutate(&mut bad_turbo.polar_code);
        let results = [
            self.polar.decode(&bad).map(|_| ()),
            self.polar
                .inner_product_estimate(&bad, &self.query)
                .map(|_| ()),
            self.polar
                .inner_product_estimate_rotated(&bad, &self.rotated)
                .map(|_| ()),
            self.turbo
                .score(&bad_turbo, &self.query, ScoreMode::L2Heuristic)
                .map(|_| ()),
            self.turbo
                .score(&bad_turbo, &self.query, ScoreMode::InnerProduct)
                .map(|_| ()),
            self.turbo
                .score_projected(&bad_turbo, &self.context, ScoreMode::InnerProduct)
                .map(|_| ()),
            self.turbo
                .inner_product_estimate(&bad_turbo, &self.query)
                .map(|_| ()),
            self.turbo
                .l2_distance_estimate(&bad_turbo, &self.query)
                .map(|_| ()),
            self.turbo.decode_approximate(&bad_turbo).map(|_| ()),
        ];
        for (path, result) in results.iter().enumerate() {
            assert!(
                matches!(result, Err(QuantError::MalformedCode(_))),
                "{what}: path {path} returned {result:?}"
            );
        }
        // Serde (a hostile payload is rejected when it is read, not when it is used).
        let wire = serde_json::to_string(&bad).unwrap();
        assert!(
            serde_json::from_str::<PolarCode>(&wire).is_err(),
            "{what}: serde accepts a hostile PolarCode"
        );
        let wire = serde_json::to_string(&bad_turbo).unwrap();
        assert!(
            serde_json::from_str::<TurboCode>(&wire).is_err(),
            "{what}: serde accepts a hostile TurboCode"
        );
    }
}

#[test]
fn hostile_code_lengths_are_rejected_before_any_indexing() {
    let f = Fixture::new(4);
    for len in [0usize, 1, DIM / 2 - 1, DIM / 2 + 1, DIM, 1000] {
        f.assert_rejected_everywhere(
            |code| code.radii = vec![1.0; len],
            &format!("radii.len()={len}"),
        );
        f.assert_rejected_everywhere(
            |code| code.angle_indices = vec![1; len],
            &format!("angle_indices.len()={len}"),
        );
    }
}

#[test]
fn hostile_radii_are_rejected_on_every_path() {
    let f = Fixture::new(4);
    for (value, name) in [
        (f32::NAN, "NaN"),
        (f32::INFINITY, "+Inf"),
        (f32::NEG_INFINITY, "-Inf"),
        (-1.0, "-1"),
        (f32::MIN, "f32::MIN"),
    ] {
        for pair in [0usize, 3, DIM / 2 - 1] {
            f.assert_rejected_everywhere(
                |code| code.radii[pair] = value,
                &format!("radius {name} at pair {pair}"),
            );
        }
    }
}

#[test]
fn hostile_angle_indices_are_rejected_before_indexing_the_table() {
    for bits in [1u8, 3, 4, 8] {
        let f = Fixture::new(bits);
        let levels = 1u16 << bits;
        for index in [levels, levels + 1, 4096, u16::MAX] {
            for pair in [0usize, 3, DIM / 2 - 1] {
                f.assert_rejected_everywhere(
                    |code| code.angle_indices[pair] = index,
                    &format!("bits={bits} index {index} at pair {pair}"),
                );
            }
        }
        // The last valid index is still a code.
        for pair in [0usize, 3, DIM / 2 - 1] {
            let mut ok = f.code.clone();
            ok.angle_indices[pair] = levels - 1;
            assert!(f
                .polar
                .inner_product_estimate(&ok, &f.query)
                .unwrap()
                .is_finite());
        }
    }
}

// ---------------------------------------------------------------------------
// A valid-shape code with huge radii, or a finite huge query: Err, never +/-Inf/NaN.
// ---------------------------------------------------------------------------

#[test]
fn huge_radii_or_a_huge_query_never_score_non_finite() {
    let f = Fixture::new(4);
    // 1. Huge finite radii (a valid code) against an ordinary query.
    let mut huge = f.code.clone();
    huge.radii = vec![f32::MAX; DIM / 2];
    let mut huge_turbo = f.turbo_code.clone();
    huge_turbo.polar_code = huge.clone();
    let results = [
        f.polar.inner_product_estimate(&huge, &f.query),
        f.polar.inner_product_estimate_rotated(&huge, &f.rotated),
        f.turbo.score(&huge_turbo, &f.query, ScoreMode::L2Heuristic),
        f.turbo
            .score(&huge_turbo, &f.query, ScoreMode::InnerProduct),
        f.turbo.l2_distance_estimate(&huge_turbo, &f.query),
    ];
    for (path, result) in results.iter().enumerate() {
        // Valid code, valid query context: only the score can overflow.
        assert!(
            matches!(result, Err(QuantError::NonFiniteScore)),
            "huge radii: path {path} returned {result:?}"
        );
    }
    // 2. An ordinary code against a finite huge query.
    let huge_query = vec![3e38f32; DIM];
    assert!(huge_query.iter().all(|q| q.is_finite()));
    let qjl = QjlQuantizer::new(DIM, 8, 1).unwrap();
    let sketch = qjl.sketch(&f.x).unwrap();
    let results = [
        f.polar.inner_product_estimate(&f.code, &huge_query),
        f.turbo
            .score(&f.turbo_code, &huge_query, ScoreMode::L2Heuristic),
        f.turbo
            .score(&f.turbo_code, &huge_query, ScoreMode::InnerProduct),
        f.turbo.l2_distance_estimate(&f.turbo_code, &huge_query),
        qjl.score(&sketch, &huge_query, ScoreMode::L2Heuristic),
        qjl.score(&sketch, &huge_query, ScoreMode::InnerProduct),
        qjl.inner_product_estimate(&sketch, &huge_query),
    ];
    for (path, result) in results.iter().enumerate() {
        assert!(
            matches!(
                result,
                Err(QuantError::NonFiniteQueryContext { .. }) | Err(QuantError::NonFiniteScore)
            ),
            "huge query: path {path} returned {result:?}"
        );
    }
    // 3. Store search: an Err for the query, never a ranking with holes or NaN scores.
    for config in [polar_config(4), turbo_config(4)] {
        let mut store = QuantizedStore::new(DIM, config).unwrap();
        store
            .add_batch(&super::uniform_matrix(4, DIM, 11), 0)
            .unwrap();
        for mode in MODES {
            let hits = store.search_scored(&view(&huge_query), 4, mode);
            let err = hits.expect_err("a huge query is an Err for the store");
            assert!(
                matches!(
                    err,
                    QuantError::NonFiniteQueryContext { .. } | QuantError::NonFiniteScore
                ),
                "store {mode}: {err:?}"
            );
        }
        assert!(store.search(&view(&huge_query), 4).is_empty());
        let hits = store
            .search_scored(&view(&f.query), 4, ScoreMode::L2Heuristic)
            .unwrap();
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|h| h.score.is_finite()));
    }
}

// ---------------------------------------------------------------------------
// 4. QJL: signs other than +/-1, sizes that overflow.
// ---------------------------------------------------------------------------

#[test]
fn qjl_sketch_rejects_signs_other_than_plus_minus_one_everywhere() {
    let quantizer = QjlQuantizer::new(8, 16, 7).unwrap();
    let query = [0.25f32; 8];
    let projected = quantizer.projected_query(&query).unwrap();
    // Valid sketches round-trip bit-exactly (zero vector: norm +0.0, every sign +1).
    for x in [
        vec![0.5f32; 8],
        vec![0.0f32; 8],
        (1..=8).map(|i| i as f32 * 0.5 - 2.0).collect(),
    ] {
        let sketch = quantizer.sketch(&x).unwrap();
        assert!(sketch.signs.iter().all(|&s| s == 1 || s == -1));
        let back: QjlSketch =
            serde_json::from_str(&serde_json::to_string(&sketch).unwrap()).unwrap();
        assert_eq!(back, sketch);
        for mode in MODES {
            assert_eq!(
                quantizer.score(&back, &query, mode).unwrap().to_bits(),
                quantizer.score(&sketch, &query, mode).unwrap().to_bits()
            );
        }
    }
    let sketch = quantizer.sketch(&[0.5f32; 8]).unwrap();
    let polar = PolarQuantizer::new(8, 3, 7).unwrap();
    let polar_code = polar.encode(&[0.5f32; 8]).unwrap();
    for bad in [0i8, 7, -128, 2] {
        for at in [0usize, 5, 15] {
            let mut signs = sketch.signs.clone();
            signs[at] = bad;
            // Serde: alone and nested in a TurboCode.
            let payload = serde_json::json!({
                "dim": 8, "projections": 16, "signs": signs, "norm": sketch.norm
            });
            assert!(
                serde_json::from_value::<QjlSketch>(payload.clone()).is_err(),
                "serde accepts sign {bad} at {at}"
            );
            let turbo_payload = serde_json::to_string(&TurboCode {
                polar_code: polar_code.clone(),
                residual_sketch: QjlSketch {
                    dim: 8,
                    projections: 16,
                    signs: signs.clone(),
                    norm: sketch.norm,
                },
            })
            .unwrap();
            assert!(
                serde_json::from_str::<TurboCode>(&turbo_payload).is_err(),
                "serde accepts a TurboCode with sign {bad} at {at}"
            );
            // Manual construction: never scored.
            let manual = QjlSketch {
                dim: 8,
                projections: 16,
                signs,
                norm: sketch.norm,
            };
            let results = [
                quantizer
                    .score(&manual, &query, ScoreMode::L2Heuristic)
                    .map(|_| ()),
                quantizer
                    .score(&manual, &query, ScoreMode::InnerProduct)
                    .map(|_| ()),
                quantizer
                    .score_projected(&manual, &projected, ScoreMode::InnerProduct)
                    .map(|_| ()),
                quantizer
                    .inner_product_estimate(&manual, &query)
                    .map(|_| ()),
            ];
            for (path, result) in results.iter().enumerate() {
                assert!(
                    matches!(result, Err(QuantError::MalformedCode(_))),
                    "sign {bad} at {at}: path {path} returned {result:?}"
                );
            }
        }
    }
    // v0 stays rejected; norm -0.0 is not a norm `sketch` can produce.
    let v0 = serde_json::json!({"dim": 8, "projections": 16, "signs": sketch.signs});
    assert!(serde_json::from_value::<QjlSketch>(v0).is_err());
    let negative_zero = serde_json::json!({
        "dim": 8, "projections": 16, "signs": sketch.signs, "norm": -0.0
    });
    assert!(serde_json::from_value::<QjlSketch>(negative_zero).is_err());
}

#[test]
fn qjl_quantizer_rejects_sizes_that_overflow_before_allocating() {
    // projections * dim overflows usize.
    assert!(QjlQuantizer::new(2, usize::MAX, 0).is_err());
    assert!(QjlQuantizer::new(usize::MAX, 2, 0).is_err());
    // The element count fits, its byte size does not (2^62 * 4 B).
    assert!(QjlQuantizer::new(1 << 31, 1 << 31, 0).is_err());
    // Deserialization of a hostile quantizer is an Err, not an overflow.
    let hostile = format!(r#"{{"dim":2,"projections":{},"seed":0}}"#, usize::MAX);
    assert!(serde_json::from_str::<QjlQuantizer>(&hostile).is_err());
    // No arbitrary limit: a large matrix that fits is still built.
    let quantizer = QjlQuantizer::new(2048, 2048, 0).unwrap();
    assert_eq!(quantizer.projection_bytes(), 2048 * 2048 * 4);
}

// ---------------------------------------------------------------------------
// 6. Metrics: the cos/sin table is reported once, per bit width that exists.
// ---------------------------------------------------------------------------

#[test]
fn trig_table_bytes_is_zero_for_widths_without_a_table() {
    for bits in 1..=16u8 {
        assert_eq!(PolarCode::trig_table_bytes(bits), (1usize << bits) * 8);
    }
    for bits in [0u8, 17, 18, 31, 32, 33, 63, 64, 65, 127, 128, 200, 255] {
        assert_eq!(PolarCode::trig_table_bytes(bits), 0, "bits={bits}");
    }
    // The accessors of valid quantizers are unchanged.
    assert_eq!(
        PolarQuantizer::new(DIM, 3, 1).unwrap().trig_table_bytes(),
        64
    );
    assert_eq!(
        TurboQuantizer::new(DIM, 4, 8, 1)
            .unwrap()
            .trig_table_bytes(),
        64
    );
}

#[test]
fn dequantize_index_never_panics_on_any_bit_width() {
    for bits in 0..=255u8 {
        for index in [0u16, 1, 7, u16::MAX] {
            let _ = PolarCode::dequantize_index(index, bits);
        }
    }
}

#[test]
fn memory_usage_excludes_the_shared_cos_sin_table() {
    let config = polar_config(5);
    let mut used = QuantizedStore::new(DIM, config.clone()).unwrap();
    used.add_batch(&super::uniform_matrix(4, DIM, 1), 0)
        .unwrap();
    let fresh = QuantizedStore::new(DIM, config).unwrap();
    let (mem_used, mem_fresh) = (used.memory_usage(), fresh.memory_usage());
    // The first score of a bit width builds its table; the store does not grow by it.
    assert_eq!(used.search(&view(&[0.1; DIM]), 2).len(), 2);
    assert_eq!(used.memory_usage(), mem_used);
    assert_eq!(fresh.memory_usage(), mem_fresh);
    // The table is reported once, per bit width, by its own accessor.
    assert_eq!(PolarCode::trig_table_bytes(5), 256);
}

// ---------------------------------------------------------------------------
// 5. Recall contract: the `ip` mode has the recall of the ip objective.
// ---------------------------------------------------------------------------

#[test]
fn ip_mode_recall_is_measured_against_the_ip_ground_truth() {
    const DIM: usize = 640;
    const N: usize = 400;
    const NQ: usize = 16;
    const K: usize = 10;
    let data = gaussian_matrix(N, DIM, 1001);
    let queries = gaussian_matrix(NQ, DIM, 2001);
    let mut store = QuantizedStore::new(DIM, QuantConfig::default()).unwrap();
    store.add_batch(&data, 0).unwrap();
    for mode in MODES {
        let mut recall_l2 = 0.0f64;
        let mut recall_ip = 0.0f64;
        for qi in 0..NQ {
            let q = queries.row(qi).to_vec();
            let top10: Vec<u64> = store
                .search_scored(&view(&q), K, mode)
                .unwrap()
                .iter()
                .map(|hit| hit.id)
                .collect();
            recall_l2 += overlap(&top10, &exact_l2_top_k(&q, &data, K)) / K as f64;
            recall_ip += overlap(&top10, &exact_ip_top_k(&q, &data, K)) / K as f64;
        }
        let (recall_l2, recall_ip) = (recall_l2 / NQ as f64, recall_ip / NQ as f64);
        let api = f64::from(store.measure_recall_with_mode(&queries, &data, K, mode));
        let cross_l2 =
            f64::from(store.measure_recall_against_l2_with_mode(&queries, &data, K, mode));
        println!("RECALL mode={} cross_against_l2={cross_l2}", mode.as_str());
        println!(
            "RECALL mode={} api={api} vs_exact_l2={recall_l2} vs_exact_ip={recall_ip}",
            mode.as_str()
        );
        // The API returns f32: 0.4 comes back as 0.40000003576278687. The tolerance is
        // 1e-6 (f32 resolution), not 1e-9.
        match mode {
            ScoreMode::L2Heuristic => {
                assert!((api - recall_l2).abs() < 1e-6, "{api} vs {recall_l2}");
                assert!(
                    (cross_l2 - api).abs() < 1e-6,
                    "against L2 ground truth the default mode is the same number"
                );
            }
            ScoreMode::InnerProduct => {
                assert!(
                    (api - recall_ip).abs() < 1e-6,
                    "the ip recall of the ip mode is {recall_ip}; the API reports {api}, the overlap with the L2 neighbours ({recall_l2})"
                );
                // The validator's number (0.4125) is this cross metric, not the IP
                // recall (0.925): it has its own, explicit name now.
                assert!(
                    (cross_l2 - recall_l2).abs() < 1e-6,
                    "{cross_l2} vs {recall_l2}"
                );
                assert!(
                    api - cross_l2 > 0.3,
                    "the cross metric ({cross_l2}) and the IP recall ({api}) must be told apart"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GREEN additions: the store's fast path, a search without holes, hostile payloads.
// ---------------------------------------------------------------------------

/// The only errors a search of valid codes can return for a finite query: the query
/// context or a score overflowed f32.
fn assert_overflow(err: &QuantError, what: &str) {
    assert!(
        matches!(
            err,
            QuantError::NonFiniteQueryContext { .. } | QuantError::NonFiniteScore
        ),
        "{what}: {err:?}"
    );
}

#[test]
fn every_code_an_encoder_emits_is_accepted_by_the_validating_paths() {
    // The justification of the store's fast path (it does not re-validate its codes per
    // query): whatever `encode` returns is a valid code. Magnitudes from the denormal
    // range to past the overflow; past it, the only accepted error is the radius one.
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for bits in [1u8, 2, 3, 4, 8, 15, 16] {
        let polar = PolarQuantizer::new(DIM, bits, 3).unwrap();
        let turbo = TurboQuantizer::new(DIM, bits.max(2), 8, 3).unwrap();
        for mag in [1e-30f32, 1e-6, 1.0, 1e6, 1e12, 1e17, 5e18, 1e19, 1e20, 1e30] {
            for seed in 0..4u64 {
                let mut rng = ChaCha8Rng::seed_from_u64(seed + 100u64 * u64::from(bits));
                let x: Vec<f32> = (0..DIM)
                    .map(|_| {
                        let normal: f32 = StandardNormal.sample(&mut rng);
                        mag * normal
                    })
                    .collect();
                // A unit-norm query of the same direction: a finite score if the code is.
                let unit: Vec<f32> = x.iter().map(|v| v / mag).collect();
                match polar.encode(&x) {
                    Ok(code) => {
                        assert_eq!(
                            (code.radii.len(), code.angle_indices.len()),
                            (DIM / 2, DIM / 2)
                        );
                        assert!(code.radii.iter().all(|r| r.is_finite() && *r >= 0.0));
                        assert!(code
                            .angle_indices
                            .iter()
                            .all(|&index| u32::from(index) < 1u32 << bits));
                        assert!(polar.decode(&code).unwrap().iter().all(|v| v.is_finite()));
                        assert!(polar.inner_product_estimate(&code, &unit).is_ok());
                        let wire = serde_json::to_string(&code).unwrap();
                        assert_eq!(serde_json::from_str::<PolarCode>(&wire).unwrap(), code);
                        accepted += 1;
                    }
                    Err(QuantError::NonFiniteRadius { .. }) => rejected += 1,
                    Err(other) => panic!("bits={bits} mag={mag}: {other:?}"),
                }
                if let Ok(code) = turbo.encode(&x) {
                    assert_eq!(code.polar_code.bits, bits.max(2) - 1);
                    assert!(code.residual_sketch.validate().is_ok());
                    assert!(turbo.decode_approximate(&code).is_ok());
                    assert!(turbo.score(&code, &unit, ScoreMode::InnerProduct).is_ok());
                    let wire = serde_json::to_string(&code).unwrap();
                    assert_eq!(serde_json::from_str::<TurboCode>(&wire).unwrap(), code);
                }
            }
        }
    }
    assert!(
        accepted > 100 && rejected > 0,
        "accepted={accepted} rejected={rejected}"
    );
}

#[test]
fn a_search_never_returns_a_ranking_with_a_hole() {
    let mut finite = 0usize;
    let mut overflowed = 0usize;
    for config in [polar_config(4), turbo_config(4)] {
        for code_mag in [4e18f32, 8e18, 1.2e19] {
            let mut store = QuantizedStore::new(DIM, config.clone()).unwrap();
            // Codes close to the overflow limit are still codes: every radius is finite.
            if store.add_batch(&scaled_batch(code_mag), 0).is_err() {
                assert_eq!(store.len(), 0, "a rejected batch stores nothing");
                continue;
            }
            assert_eq!(store.len(), 8);
            for query_mag in [1.0f32, 1e10, 1e18, 1e19, 1e20, 3e38] {
                let query: Vec<f32> = query_vector().iter().map(|v| v * query_mag).collect();
                assert!(query.iter().all(|v| v.is_finite()));
                for mode in MODES {
                    let what = format!("{config:?} code={code_mag} query={query_mag} {mode}");
                    match store.search_scored(&view(&query), 8, mode) {
                        Ok(hits) => {
                            // Every stored code is ranked, with a finite score: no hole.
                            assert_eq!(hits.len(), 8, "{what}");
                            assert!(hits.iter().all(|h| h.score.is_finite()), "{what}");
                            finite += 1;
                        }
                        Err(err) => {
                            assert_overflow(&err, &what);
                            overflowed += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        finite >= 2 && overflowed >= 2,
        "finite={finite} overflowed={overflowed}"
    );
}

#[test]
fn l2_distance_estimate_never_returns_a_non_finite_distance() {
    let turbo = TurboQuantizer::new(DIM, 4, 8, 42).unwrap();
    let code = turbo.encode(&x_vector()).unwrap();
    assert!(turbo
        .l2_distance_estimate(&code, &query_vector())
        .unwrap()
        .is_finite());
    // `ip` fails first here (the query context overflows): never a number.
    let err = turbo
        .l2_distance_estimate(&code, &[3e38f32; DIM])
        .expect_err("a huge query has no distance");
    assert_overflow(&err, "huge query");
}

#[test]
fn hostile_payloads_of_codes_and_quantizers_are_rejected_when_read() {
    let x = x_vector();
    // TurboCode: halves that describe vectors of different dimensions, or a polar half
    // whose vectors have the wrong length.
    let turbo = TurboQuantizer::new(DIM, 4, 8, 5).unwrap();
    let code = turbo.encode(&x).unwrap();
    let wire = serde_json::to_value(&code).unwrap();
    assert_eq!(
        serde_json::from_value::<TurboCode>(wire.clone()).unwrap(),
        code
    );
    let mut hostile = wire.clone();
    hostile["residual_sketch"]["dim"] = serde_json::json!(8);
    assert!(serde_json::from_value::<TurboCode>(hostile).is_err());
    let mut hostile = wire;
    hostile["polar_code"]["radii"] = serde_json::json!([]);
    assert!(serde_json::from_value::<TurboCode>(hostile).is_err());

    // PolarCode: a width or a dimension no quantizer accepts.
    let polar = PolarQuantizer::new(DIM, 3, 5).unwrap();
    let code = polar.encode(&x).unwrap();
    let wire = serde_json::to_value(&code).unwrap();
    assert_eq!(
        serde_json::from_value::<PolarCode>(wire.clone()).unwrap(),
        code
    );
    for (field, value) in [
        ("bits", serde_json::json!(17)),
        ("bits", serde_json::json!(0)),
        ("dim", serde_json::json!(15)),
    ] {
        let mut hostile = wire.clone();
        hostile[field] = value.clone();
        assert!(
            serde_json::from_value::<PolarCode>(hostile).is_err(),
            "{field}={value}"
        );
    }

    // Quantizers: a configuration `new` never builds (a width that would overflow the
    // encoder's `1 << bits`, a rotation or a stage of another size). A valid payload
    // round-trips into the same codes.
    let wire = serde_json::to_value(&polar).unwrap();
    let back: PolarQuantizer = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!((back.dim(), back.bits()), (DIM, 3));
    assert_eq!(back.encode(&x).unwrap(), code);
    for (field, value) in [
        ("bits", serde_json::json!(200)),
        ("bits", serde_json::json!(17)),
        ("dim", serde_json::json!(14)),
    ] {
        let mut hostile = wire.clone();
        hostile[field] = value.clone();
        assert!(
            serde_json::from_value::<PolarQuantizer>(hostile).is_err(),
            "{field}={value}"
        );
    }
    let wire = serde_json::to_value(&turbo).unwrap();
    let back: TurboQuantizer = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(back.encode(&x).unwrap(), turbo.encode(&x).unwrap());
    for (field, value) in [
        ("bits", serde_json::json!(200)),
        ("bits", serde_json::json!(5)),
        ("projections", serde_json::json!(7)),
        ("dim", serde_json::json!(14)),
    ] {
        let mut hostile = wire.clone();
        hostile[field] = value.clone();
        assert!(
            serde_json::from_value::<TurboQuantizer>(hostile).is_err(),
            "{field}={value}"
        );
    }
}
