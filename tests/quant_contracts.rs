//! Contract tests for the quantization layer (fixer F04a).
//!
//! A02-F07: `QuantizedStore::add_batch` is atomic — a batch with a wrong column count,
//! a NaN/±Inf, or a row that fails to encode returns `Err` and leaves ids, codes and
//! `len()` untouched; non-finite input never becomes a code.
//!
//! A02-F03: the estimator contract (property tests + evidence probes) lives in
//! `quant_contracts/properties.rs`.
//!
//! D18: ranking objective vs unbiased estimator (default `l2_heuristic` = legacy order,
//! `ip` separate; parity, zero norm) lives in `quant_contracts/ranking.rs`.
//!
//! A02-F02: the projection cache and the per-query context (bit-identity, memory counted
//! once, A/B bench) live in `quant_contracts/ab_bench.rs`; `quant_contracts/d18_probe.rs`
//! is the cross-commit evidence probe (default order: recall vs exact-L2 GT + hash).
//!
//! F04c (A02-F04): lossless bit-packing — bit-stream round trips, packed == unpacked
//! ids/order/score bits, store vs unpacked reference on the F04b datasets, atomicity,
//! serde goldens — in `quant_contracts/packing.rs`; `quant_contracts/ab_release.rs` is
//! the release A/B probe (N=10000) and the source of the serde goldens.
//!
//! Run: `cargo test --no-default-features --test quant_contracts`

#[path = "quant_contracts/ab_bench.rs"]
mod ab_bench;
#[path = "quant_contracts/ab_release.rs"]
mod ab_release;
#[path = "quant_contracts/d18_probe.rs"]
mod d18_probe;
#[path = "quant_contracts/packing.rs"]
mod packing;
#[path = "quant_contracts/properties.rs"]
mod properties;
#[path = "quant_contracts/ranking.rs"]
mod ranking;

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use splatsdb::quant::{PolarQuantizer, QjlQuantizer, QuantError, TurboQuantizer};
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore};

fn uniform_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |(_, _)| rng.gen::<f32>())
}

/// Every id the store can return for `query` with an unbounded k, sorted.
fn sorted_ids(store: &QuantizedStore, query: &ndarray::ArrayView1<f32>) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(query, usize::MAX)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort_unstable();
    ids
}

fn search_and_polar_configs(dim: usize) -> [QuantConfig; 2] {
    [
        QuantConfig::for_search(dim),
        QuantConfig {
            bits: 4,
            projections: 0,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        },
    ]
}

#[test]
fn add_batch_wrong_column_count_is_rejected_and_atomic() {
    let dim = 16;
    for config in search_and_polar_configs(dim) {
        let mut store = QuantizedStore::new(dim, config).unwrap();
        let good = uniform_matrix(10, dim, 1);
        assert_eq!(store.add_batch(&good, 0).unwrap(), 10);

        // (rows, cols) of the invalid batch: too wide, too narrow, empty-but-wide.
        for (rows, cols) in [(1usize, dim + 1), (4, dim + 1), (4, dim - 1), (0, dim + 1)] {
            let bad = uniform_matrix(rows, cols, 2);
            let err = store
                .add_batch(&bad, 10)
                .expect_err("wrong-shaped batch must be rejected");
            assert!(
                err.contains(&format!("{cols} columns")) && err.contains("store dimension is 16"),
                "{err}"
            );
            assert_eq!(
                store.len(),
                10,
                "len must be untouched after a rejected batch"
            );
            assert_eq!(
                sorted_ids(&store, &good.row(0)),
                (0..10).collect::<Vec<u64>>()
            );
        }
    }
}

#[test]
fn add_batch_non_finite_is_rejected_and_atomic() {
    let dim = 16;
    for config in search_and_polar_configs(dim) {
        let mut store = QuantizedStore::new(dim, config).unwrap();
        let good = uniform_matrix(6, dim, 3);
        assert_eq!(store.add_batch(&good, 0).unwrap(), 6);
        for (value, name) in [
            (f32::NAN, "NaN"),
            (f32::INFINITY, "+Inf"),
            (f32::NEG_INFINITY, "-Inf"),
        ] {
            for (row, col) in [(0usize, 0usize), (3, 7), (5, dim - 1)] {
                let mut batch = uniform_matrix(6, dim, 4);
                batch[[row, col]] = value;
                let err = store.add_batch(&batch, 100).expect_err(name);
                assert!(
                    err.contains("non-finite value (NaN/Inf)")
                        && err.contains(&format!("row {row}, col {col}")),
                    "{name}: {err}"
                );
                assert_eq!(
                    store.len(),
                    6,
                    "{name}: a rejected batch must not change len"
                );
            }
        }
        let results = store.search(&good.row(0), usize::MAX);
        assert_eq!(results.len(), 6);
        assert!(results.iter().all(|(_, score)| score.is_finite()));
        assert_eq!(
            sorted_ids(&store, &good.row(0)),
            (0..6).collect::<Vec<u64>>()
        );
    }
}

#[test]
fn add_batch_ids_only_for_encoded_rows() {
    let dim = 16;
    for config in search_and_polar_configs(dim) {
        let mut store = QuantizedStore::new(dim, config).unwrap();
        let first = uniform_matrix(10, dim, 5);
        let second = uniform_matrix(5, dim, 6);
        assert_eq!(store.add_batch(&first, 0).unwrap(), 10);
        assert_eq!(store.add_batch(&second, 10).unwrap(), 5);
        assert_eq!(store.len(), 15);
        // Every id resolves to exactly one code and every code to one id.
        assert_eq!(store.search(&first.row(0), usize::MAX).len(), 15);
        assert_eq!(
            sorted_ids(&store, &first.row(0)),
            (0..15).collect::<Vec<u64>>()
        );
        // An empty batch is a no-op.
        assert_eq!(
            store
                .add_batch(&Array2::<f32>::zeros((0, dim)), 15)
                .unwrap(),
            0
        );
        assert_eq!(store.len(), 15);
    }
}

#[test]
fn add_single_rejects_non_finite_without_touching_state() {
    let dim = 16;
    for config in search_and_polar_configs(dim) {
        let mut store = QuantizedStore::new(dim, config).unwrap();
        let good = uniform_matrix(3, dim, 7);
        assert!(store.add_single(&good.row(0), 0));
        let mut bad = good.row(1).to_owned();
        bad[4] = f32::NAN;
        assert!(!store.add_single(&bad.view(), 1));
        assert_eq!(store.len(), 1);
        assert!(store.add_single(&good.row(2), 2));
        assert_eq!(sorted_ids(&store, &good.row(0)), vec![0, 2]);
    }
}

#[test]
fn encoders_reject_non_finite_inputs() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = |v: f32| vec![1.0f32, -2.0, v, 0.5, 3.0, 0.25, -1.0, 2.0];
        assert!(matches!(
            TurboQuantizer::new(8, 4, 4, 0).unwrap().encode(&x(value)),
            Err(QuantError::NonFiniteValue { index: 2 })
        ));
        assert!(matches!(
            PolarQuantizer::new(8, 3, 0).unwrap().encode(&x(value)),
            Err(QuantError::NonFiniteValue { index: 2 })
        ));
        assert!(matches!(
            QjlQuantizer::new(8, 4, 0).unwrap().sketch(&x(value)),
            Err(QuantError::NonFiniteValue { index: 2 })
        ));
    }
    // Finite inputs still encode.
    let x = vec![1.0f32; 8];
    assert!(TurboQuantizer::new(8, 4, 4, 0).unwrap().encode(&x).is_ok());
    assert!(PolarQuantizer::new(8, 3, 0).unwrap().encode(&x).is_ok());
    assert!(QjlQuantizer::new(8, 4, 0).unwrap().sketch(&x).is_ok());
}

// ---------------------------------------------------------------------------
// A02-F03 (schema): QjlSketch v1 carries ||x||; v0 payloads are rejected, never
// reinterpreted.
// ---------------------------------------------------------------------------

#[test]
fn qjl_sketch_schema_v0_without_norm_is_rejected_not_reinterpreted() {
    use splatsdb::quant::qjl::QJL_SKETCH_SCHEMA_VERSION;
    use splatsdb::quant::{QjlSketch, TurboCode};
    assert_eq!(QJL_SKETCH_SCHEMA_VERSION, 1);
    // Wire payload of schema v0 (<= 2.11.0): {dim, projections, signs}.
    let v0 = serde_json::json!({"dim": 8, "projections": 4, "signs": [1, -1, 1, -1]});
    let err = serde_json::from_value::<QjlSketch>(v0.clone())
        .expect_err("a v0 sketch must not deserialize")
        .to_string();
    assert!(
        err.contains("schema v0") && err.contains("re-encode"),
        "{err}"
    );
    // The same holds nested in a TurboCode.
    let v0_code = serde_json::json!({
        "polar_code": {"dim": 8, "bits": 3, "radii": [1.0, 1.0, 1.0, 1.0], "angle_indices": [0, 1, 2, 3]},
        "residual_sketch": v0,
    });
    assert!(serde_json::from_value::<TurboCode>(v0_code).is_err());
}

#[test]
fn qjl_sketch_schema_v1_roundtrip_and_byte_cost() {
    use splatsdb::quant::{QjlQuantizer, QjlSketch, TurboCode, TurboQuantizer};
    let quantizer = QjlQuantizer::new(8, 16, 7).unwrap();
    let x: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
    let sketch = quantizer.sketch(&x).unwrap();
    let exact_norm = (x.iter().map(|v| v * v).sum::<f32>()).sqrt();
    assert!((sketch.norm - exact_norm).abs() < 1e-6 * exact_norm);

    let wire = serde_json::to_string(&sketch).unwrap();
    assert_eq!(wire.matches("\"norm\"").count(), 1);
    let back: QjlSketch = serde_json::from_str(&wire).unwrap();
    assert_eq!(back, sketch);

    // Byte cost of the schema change: +4 B per sketch, signs stay 1 B each.
    assert_eq!(sketch.encoded_bytes(), 16 + std::mem::size_of::<f32>());

    // A whole TurboCode round-trips through serde with the v1 sketch inside.
    let turbo = TurboQuantizer::new(8, 4, 16, 7).unwrap();
    let code = turbo.encode(&x).unwrap();
    let code_wire = serde_json::to_string(&code).unwrap();
    let code_back: TurboCode = serde_json::from_str(&code_wire).unwrap();
    assert_eq!(code_back, code);

    // Malformed v1 payloads are rejected too.
    for bad in [
        serde_json::json!({"dim": 8, "projections": 3, "signs": [1, -1], "norm": 1.0}),
        serde_json::json!({"dim": 8, "projections": 2, "signs": [1, -1], "norm": -1.0}),
        serde_json::json!({"dim": 8, "projections": 2, "signs": [1, -1], "norm": 1.0e39}),
    ] {
        assert!(serde_json::from_value::<QjlSketch>(bad).is_err());
    }
}

// ---------------------------------------------------------------------------
// A02-F04 / F04c: memory metrics reflect the real (bit-packed) layout, not dim*bits/8.
// ---------------------------------------------------------------------------

#[test]
fn memory_metrics_report_the_real_packed_layout_for_bits_2_4_8() {
    use splatsdb::quant::polar::PackedPolarCode;
    use splatsdb::quant::turbo::PackedTurboCode;
    use splatsdb::quant::{PolarCode, QjlSketch};
    const DIM: usize = 640;
    const N: usize = 1000;
    let vectors = uniform_matrix(N, DIM, 21);
    let mut report = Vec::new();
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
            store.add_batch(&vectors, 0).unwrap();
            let n = store.len();
            assert_eq!(n, N);

            // Independent count of the packed layout: one exact f32 radius per pair, the
            // angle index in `bits` bits (`bits - 1` for TurboQuant) in one stream, one
            // sign bit per projection and an f32 residual norm (TurboQuant only).
            let angle_bits = if turbo { bits - 1 } else { bits } as usize;
            let angle_bytes = (DIM / 2 * angle_bits).div_ceil(8);
            let per_code =
                DIM / 2 * 4 + angle_bytes + if turbo { (DIM / 8).div_ceil(8) + 4 } else { 0 };
            // The unpacked layout this replaced (<= 9f319b2): 6 B per pair + i8 signs.
            let unpacked = DIM / 2 * 6 + if turbo { DIM / 8 + 4 } else { 0 };
            let nominal = 32.0 / bits as f32;
            let raw = n * DIM * 4;
            let real = n * per_code;

            assert_eq!(
                PackedPolarCode::layout_bytes(DIM, angle_bits as u8),
                DIM / 2 * 4 + angle_bytes
            );
            if turbo {
                assert_eq!(PackedTurboCode::layout_bytes(DIM, bits, DIM / 8), per_code);
            } else {
                assert_eq!(
                    store.code_layout_bytes(),
                    PackedPolarCode::layout_bytes(DIM, bits)
                );
            }
            assert_eq!(PolarCode::layout_bytes(DIM), DIM / 2 * 6);
            assert_eq!(QjlSketch::layout_bytes(DIM / 8), DIM / 8 + 4);
            assert_eq!(store.code_layout_bytes(), per_code);
            assert_eq!(store.encoded_bytes(), real, "count must match the layout");
            assert!(store.is_bit_packed());
            let ratio = store.compression_ratio();
            assert!(
                (ratio - raw as f32 / real as f32).abs() < 1e-5,
                "ratio={ratio}"
            );
            assert!(
                ratio < nominal,
                "bits={bits} {algorithm:?}: nominal {nominal}x is not what the layout delivers ({ratio}x)"
            );
            let inline = if turbo {
                std::mem::size_of::<PackedTurboCode>()
            } else {
                std::mem::size_of::<PackedPolarCode>()
            };
            // The f32 norm is inline in PackedTurboCode (already inside `inline`), so it
            // is subtracted from the logical payload: memory_usage() = heap + struct + id.
            let inline_norm = if turbo { std::mem::size_of::<f32>() } else { 0 };
            // Quantizer matrices held once per store: the d×d rotation and, for
            // TurboQuant, the m×d QJL projection cache (A02-F02) — never per code.
            let quantizer = DIM * DIM * 4 + if turbo { DIM / 8 * DIM * 4 } else { 0 };
            assert_eq!(store.quantizer_memory_usage(), quantizer);
            assert_eq!(
                store.projection_cache_bytes(),
                if turbo { DIM / 8 * DIM * 4 } else { 0 }
            );
            assert_eq!(
                store.memory_usage(),
                real - n * inline_norm + n * (inline + std::mem::size_of::<u64>()) + quantizer
            );
            report.push(format!(
                "bits={bits} {algorithm:?} m={}: raw={raw} B, encoded={real} B ({per_code} B/code, was {unpacked} B unpacked: {:.3}x less payload), ratio={ratio:.4} (nominal {nominal}), memory_usage={} B",
                if turbo { DIM / 8 } else { 0 },
                unpacked as f64 / per_code as f64,
                store.memory_usage()
            ));
        }
    }
    println!("\n{}", report.join("\n"));
}

#[test]
fn memory_usage_on_a_10k_code_store_matches_the_declared_overhead() {
    use splatsdb::quant::turbo::PackedTurboCode;
    const DIM: usize = 640;
    const N: usize = 10_000;
    let mut store = QuantizedStore::new(DIM, QuantConfig::for_search(DIM)).unwrap();
    store.add_batch(&uniform_matrix(N, DIM, 22), 0).unwrap();
    assert_eq!(store.len(), N);
    // for_search(640) = 4 bits, 80 projections: 1280 B radii + 120 B angles (3 bits *
    // 320 pairs) + 10 B signs + 4 B norm.
    let per_code = store.code_layout_bytes();
    assert_eq!(
        per_code,
        DIM / 2 * 4 + (DIM / 2 * 3).div_ceil(8) + (DIM / 8).div_ceil(8) + 4
    );
    assert_eq!(per_code, 1414);
    assert!(store.is_bit_packed());
    assert_eq!(store.encoded_bytes(), N * per_code);
    // The f32 norm lives inline in PackedTurboCode (counted through its size_of), so
    // memory_usage() = encoded_bytes() - 4 B/code + struct + 8 B/id: no double count.
    // Plus the quantizer matrices counted once: rotation d×d + QJL projection m×d.
    let quantizer = DIM * DIM * 4 + DIM / 8 * DIM * 4;
    assert_eq!(store.quantizer_memory_usage(), quantizer);
    let declared = store.encoded_bytes() - N * std::mem::size_of::<f32>()
        + N * (std::mem::size_of::<PackedTurboCode>() + std::mem::size_of::<u64>())
        + quantizer;
    assert_eq!(store.memory_usage(), declared);
    let raw = N * DIM * 4;
    println!(
        "10k x 640-D 4-bit TurboQuant (m=80), packed: raw={raw} B, encoded={} B, memory_usage={declared} B, ratio={} (nominal 8.0; unpacked layout was 2004 B/code, 2120 B/code with headers)",
        store.encoded_bytes(),
        raw as f32 / store.encoded_bytes() as f32
    );
    // 1414 B payload - 4 B inline norm + struct + 8 B id per vector < 2560 B raw.
    assert!(store.memory_usage() < raw);
    assert!(store.memory_usage() > raw / 2);
}
