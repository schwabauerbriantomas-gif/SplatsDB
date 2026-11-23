//! Core-contract regression tests (fixer F01 — audit A01 findings).
//!
//! One section per finding; each reproduction assertion was red on the base
//! commit `617822d` before the matching fix landed.

use ndarray::{Array1, Array2, ArrayView1};
use splatsdb::config::SplatsDBConfig;
use splatsdb::splats::SplatStore;

/// Deterministic LCG → `f32` in `[0, 1)` (no RNG dependency needed).
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = Lcg(seed | 1);
    Array2::from_shape_fn((n, dim), |_| rng.next_f32())
}

fn config(dim: usize, max_splats: usize) -> SplatsDBConfig {
    SplatsDBConfig {
        max_splats,
        latent_dim: dim,
        ..Default::default()
    }
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

/// Reference raw L2 distance `||a - b||` (not squared).
fn l2(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Store holding `data` (alpha = kappa = 1) with all indexes built.
fn built_store(cfg: SplatsDBConfig, data: &Array2<f32>) -> SplatStore {
    let mut store = SplatStore::new(cfg);
    store
        .load_splats(data, &ones(data.nrows()), &ones(data.nrows()))
        .expect("load_splats");
    store.build_index();
    store
}

// ── A01-F01: `NeighborResult.distance` is the raw L2 `||q - mu||` ──────────
//
// `SplatStore::new` configures the HRM2 engine with metric "cosine": the
// engine indexes L2-normalized rows and ranks in that normalized space.
// Those distances are candidate hints only; for n >= 100 `find_neighbors_fast`
// used to export them verbatim while n < 100 exported the raw L2.

/// Rows with clearly non-unit norms: raw L2 and L2-in-normalized-space
/// disagree by construction.
fn non_unit_norm_data(n: usize, dim: usize) -> Array2<f32> {
    let base = random_vectors(n, dim, 7);
    let norms = [0.5f32, 1.0, 2.0, 4.0];
    Array2::from_shape_fn((n, dim), |(i, j)| {
        let row = base.row(i);
        let norm = row.dot(&row).sqrt().max(1e-9);
        base[[i, j]] / norm * norms[i % norms.len()]
    })
}

/// Every result of the fast path must carry exactly what the linear path
/// reports for the same index, plus a score derived from that distance.
fn assert_raw_l2_contract(
    store: &mut SplatStore,
    data: &Array2<f32>,
    query: &ArrayView1<f32>,
    k: usize,
    label: &str,
) {
    let n = data.nrows();
    let exact: Vec<(usize, f32)> = store
        .find_neighbors(query, n)
        .into_iter()
        .map(|r| (r.index, r.distance))
        .collect();
    let fast = store.find_neighbors_fast(query, k);
    assert_eq!(fast.len(), k.min(n), "{label}: result count");
    for (pos, r) in fast.iter().enumerate() {
        let expected = l2(query, &data.row(r.index));
        assert!(
            (r.distance - expected).abs() <= 1e-5 * expected.max(1.0),
            "{label}: result {pos} (idx {}) exports distance {}, raw L2 is {}",
            r.index,
            r.distance,
            expected
        );
        let (_, exact_dist) = exact
            .iter()
            .find(|(idx, _)| *idx == r.index)
            .unwrap_or_else(|| panic!("{label}: idx {} missing from the exact path", r.index));
        assert!(
            (r.distance - exact_dist).abs() <= 1e-5 * exact_dist.max(1.0),
            "{label}: idx {} fast {} vs exact {}",
            r.index,
            r.distance,
            exact_dist
        );
        let expected_score = SplatStore::gaussian_score(r.alpha, r.kappa, expected * expected);
        assert!(
            (r.score - expected_score).abs() <= 1e-5 * expected_score.abs() + 1e-12,
            "{label}: idx {} score {} vs {}",
            r.index,
            r.score,
            expected_score
        );
        if pos > 0 {
            assert!(
                fast[pos - 1].distance <= r.distance,
                "{label}: results must be ordered by the exported distance"
            );
        }
    }
}

#[test]
fn f01_fast_path_exports_raw_l2_for_non_unit_norm_data() {
    let dim = 16;
    let data = non_unit_norm_data(150, dim);
    let mut store = built_store(config(dim, 1000), &data);
    let queries = [
        data.row(3).to_owned(),
        data.row(1).to_owned() * 3.0,
        random_vectors(1, dim, 99).row(0).to_owned(),
        Array1::from_vec(vec![0.5; dim]),
    ];
    for (i, query) in queries.iter().enumerate() {
        assert_raw_l2_contract(&mut store, &data, &query.view(), 10, &format!("query {i}"));
    }
}

#[test]
fn f01_distance_contract_holds_on_both_sides_of_the_100_row_threshold() {
    let dim = 16;
    let data = non_unit_norm_data(101, dim);
    let query = data.row(100).to_owned() * 0.75;
    for n in [99usize, 100, 101] {
        let rows = data.slice(ndarray::s![..n, ..]).to_owned();
        let mut store = built_store(config(dim, 1000), &rows);
        let fast = store.find_neighbors_fast(&query.view(), 5);
        assert_eq!(fast.len(), 5, "n = {n}");
        for r in &fast {
            assert!(r.index < n, "n = {n}: idx {} out of range", r.index);
            let expected = l2(&query.view(), &data.row(r.index));
            assert!(
                (r.distance - expected).abs() <= 1e-5 * expected.max(1.0),
                "n = {n}: idx {} exports {}, raw L2 is {}",
                r.index,
                r.distance,
                expected
            );
        }
    }
}

#[test]
fn f01_fast_path_nan_inf_query_does_not_panic() {
    let dim = 16;
    let data = non_unit_norm_data(150, dim);
    let mut store = built_store(config(dim, 1000), &data);
    let mut bad = random_vectors(1, dim, 5).row(0).to_owned();
    bad[0] = f32::NAN;
    bad[1] = f32::INFINITY;
    let fast = store.find_neighbors_fast(&bad.view(), 5);
    // Garbage in → NaN distances out (exactly like the linear path), no panic.
    assert_eq!(fast.len(), 5);
    assert!(fast.iter().all(|r| r.distance.is_nan()));
}

#[test]
fn f01_unit_norm_data_is_unchanged() {
    let dim = 16;
    let raw = random_vectors(150, dim, 11);
    let data = Array2::from_shape_fn((150, dim), |(i, j)| {
        let row = raw.row(i);
        let norm = row.dot(&row).sqrt();
        raw[[i, j]] / norm
    });
    let mut store = built_store(config(dim, 1000), &data);
    let query = data.row(42).to_owned();
    let fast = store.find_neighbors_fast(&query.view(), 10);
    assert_eq!(fast.len(), 10);
    for r in &fast {
        let expected = l2(&query.view(), &data.row(r.index));
        assert!((r.distance - expected).abs() <= 1e-5 * expected.max(1.0));
    }
}

// ── A01-F02: `*_with_hnsw` must respect the unindexed backlog ──────────────
//
// Backend positions are store indices. `add_batch` appends at the backend's
// current length, so appending a new vector while earlier rows are still
// unindexed shifts it onto the wrong position — and the old code then set
// `hnsw_indexed_count = n_active`, hiding the misalignment forever.

/// HNSW + quant + LSH all enabled: fused search goes through the candidate
/// backend and an invalidation has to cover every positional index.
fn config_all_subsystems(dim: usize, max_splats: usize) -> SplatsDBConfig {
    SplatsDBConfig {
        max_splats,
        latent_dim: dim,
        enable_hnsw: true,
        hnsw_m: 8,
        hnsw_ef_construction: 40,
        hnsw_ef_search: 32,
        enable_lsh: true,
        enable_quantization: true,
        ..Default::default()
    }
}

/// A self-query through the fused path must resolve to `expected_idx` at
/// (numerically) zero distance: backend position == store row.
fn assert_self_query_hits(
    store: &SplatStore,
    row: &ArrayView1<f32>,
    expected_idx: usize,
    label: &str,
) {
    let results = store.find_neighbors_fused(row, 5);
    assert!(
        !results.is_empty(),
        "{label}: fused search returned nothing"
    );
    assert_eq!(
        results[0].index, expected_idx,
        "{label}: self-query must resolve to the store row itself"
    );
    assert!(
        results[0].distance < 1e-4,
        "{label}: self-query distance {} for idx {}",
        results[0].distance,
        expected_idx
    );
}

#[test]
fn f02_add_batch_with_hnsw_indexes_backlog_in_positional_order() {
    let dim = 32;
    let mut store = SplatStore::new(config_all_subsystems(dim, 1000));
    let first = random_vectors(50, dim, 42);
    store.add_splat(&first);
    // No build_index: 50 rows are waiting in the backlog.
    assert!(store.hnsw_needs_sync());

    let batch = random_vectors(10, dim, 7);
    assert!(store.add_batch_with_hnsw(&batch, None));
    assert_eq!(store.n_active(), 60);
    assert_eq!(store.hnsw_indexed_count(), 60);
    assert!(!store.hnsw_needs_sync());

    // Backend position i must be store row i for the WHOLE range, not only
    // for the batch that was appended.
    for (row, expected) in [
        (first.row(0), 0usize),
        (first.row(17), 17),
        (batch.row(5), 55),
    ] {
        let query = row.to_owned();
        assert_self_query_hits(&store, &query.view(), expected, "add_batch_with_hnsw");
    }
}

#[test]
fn f02_insert_with_hnsw_single_does_not_skip_the_backlog() {
    let dim = 32;
    let mut store = SplatStore::new(config_all_subsystems(dim, 1000));
    let first = random_vectors(50, dim, 42);
    store.add_splat(&first);

    let vec: Vec<f32> = random_vectors(1, dim, 5).row(0).to_vec();
    let idx = store
        .insert_with_hnsw(&vec, None)
        .expect("insert should succeed");
    assert_eq!(idx, 50);
    assert_eq!(store.hnsw_indexed_count(), 51);
    assert!(!store.hnsw_needs_sync());

    for (row, expected) in [(first.row(0), 0usize), (first.row(49), 49)] {
        let query = row.to_owned();
        assert_self_query_hits(
            &store,
            &query.view(),
            expected,
            "insert_with_hnsw backlog row",
        );
    }
    let query = Array1::from_vec(vec);
    assert_self_query_hits(&store, &query.view(), 50, "insert_with_hnsw new vector");
}

// ── A01-F04: `compact()` / `load_splats()` invalidate every positional index ─
//
// Both mutators move or replace rows in place. The HNSW graph, the HRM2
// engine, the quantized store and the LSH tables index rows by position, so
// they are meaningless afterwards — and `hnsw_needs_sync()` used to keep
// answering false.

#[test]
fn f04_compact_invalidates_positional_indexes() {
    let dim = 16;
    let n = 20;
    let data = random_vectors(n, dim, 3);
    let mut alpha = ones(n);
    // compact() drops alpha < 1e-6 → every later row shifts down by one.
    alpha[0] = 0.0;
    let mut store = SplatStore::new(config_all_subsystems(dim, 1000));
    store.load_splats(&data, &alpha, &ones(n)).unwrap();
    store.build_index();
    assert!(!store.hnsw_needs_sync());

    store.compact();
    assert_eq!(store.n_active(), 19);
    assert!(
        !store.hnsw_is_built(),
        "the pre-compact graph must not be searched"
    );
    assert!(
        store.hnsw_needs_sync(),
        "compact() must leave the backend out of sync until it is rebuilt"
    );

    // Old idx 5 now lives at idx 4.
    let query = data.row(5).to_owned();
    let results = store.find_neighbors_fused(&query.view(), 1);
    assert_eq!(
        results[0].index, 4,
        "fused search after compact must see shifted rows"
    );
    assert!(results[0].distance < 1e-4);

    // A later incremental sync repairs the alignment in positional order.
    assert_eq!(store.hnsw_sync_incremental(), 19);
    assert!(!store.hnsw_needs_sync());
    let results = store.find_neighbors_fused(&query.view(), 1);
    assert_eq!(results[0].index, 4);
    assert!(results[0].distance < 1e-4);
}

#[test]
fn f04_compact_without_removals_keeps_the_indexes() {
    let dim = 16;
    let n = 20;
    let data = random_vectors(n, dim, 3);
    let mut store = SplatStore::new(config_all_subsystems(dim, 1000));
    store.load_splats(&data, &ones(n), &ones(n)).unwrap();
    store.build_index();

    store.compact();
    assert_eq!(store.n_active(), n);
    // Nothing moved: invalidating would only cost a needless rebuild.
    assert!(store.hnsw_is_built());
    assert!(!store.hnsw_needs_sync());
    let query = data.row(9).to_owned();
    let results = store.find_neighbors_fused(&query.view(), 1);
    assert_eq!(results[0].index, 9);
    assert!(results[0].distance < 1e-4);
}

#[test]
fn f04_load_splats_invalidates_hnsw_quant_and_lsh() {
    let dim = 16;
    let n = 20;
    let a = random_vectors(n, dim, 3);
    let mut store = SplatStore::new(config_all_subsystems(dim, 1000));
    store.load_splats(&a, &ones(n), &ones(n)).unwrap();
    store.build_index();
    assert!(!store.hnsw_needs_sync());

    // A different dataset with the SAME row count: a length check alone
    // cannot tell it apart from the indexed one.
    let b = &a + 100.0;
    store.load_splats(&b, &ones(n), &ones(n)).unwrap();
    assert!(!store.hnsw_is_built(), "stale graph must not be searched");
    assert!(
        store.hnsw_needs_sync(),
        "load_splats() must mark the backend stale"
    );

    let query = b.row(7).to_owned();
    let results = store.find_neighbors_fused(&query.view(), 1);
    assert_eq!(results[0].index, 7);
    assert!(results[0].distance < 1e-4);

    store.hnsw_sync_incremental();
    assert!(!store.hnsw_needs_sync());
    let results = store.find_neighbors_fused(&query.view(), 1);
    assert_eq!(results[0].index, 7);
    assert!(results[0].distance < 1e-4);
}

#[test]
fn f04_load_splats_invalidates_the_hrm2_engine() {
    let dim = 16;
    let n = 150;
    let a = random_vectors(n, dim, 3);
    let mut store = SplatStore::new(config(dim, 1000));
    store.load_splats(&a, &ones(n), &ones(n)).unwrap();
    store.build_index();

    let b = &a + 100.0;
    store.load_splats(&b, &ones(n), &ones(n)).unwrap();
    // n >= 100: without invalidation this answers from the previous
    // dataset's clusters.
    let query = b.row(7).to_owned();
    let results = store.find_neighbors_fast(&query.view(), 1);
    assert_eq!(
        results[0].index, 7,
        "find_neighbors_fast must not answer from the previous dataset's index"
    );
    assert!(results[0].distance < 1e-4);
}
