//! Contract tests for the quantization layer (fixer F04a).
//!
//! A02-F07: `QuantizedStore::add_batch` is atomic — a batch with a wrong column count,
//! a NaN/±Inf, or a row that fails to encode returns `Err` and leaves ids, codes and
//! `len()` untouched; non-finite input never becomes a code.
//!
//! Run: `cargo test --no-default-features --test quant_contracts`

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
