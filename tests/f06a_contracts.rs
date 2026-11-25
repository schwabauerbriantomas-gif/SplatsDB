//! F06a integration contracts (fixer F06a — validator V1_integration NO_GO on
//! `7c45a1d`): backlog visibility of `find_neighbors_fused` / `_fast`, k
//! results with exact duplicates, `(score, id)` ties on HRM2, and the atomic
//! rejection of non-finite rows. Every `f06a_*` test was RED on `7c45a1d`
//! before the matching fix landed (logs: tmp/F06a/red/).

use ndarray::Array2;
use splatsdb::config::SplatsDBConfig;
use splatsdb::splats::SplatStore;
use std::collections::HashSet;

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

fn hnsw_config(dim: usize, max_splats: usize) -> SplatsDBConfig {
    SplatsDBConfig {
        max_splats,
        latent_dim: dim,
        enable_hnsw: true,
        hnsw_m: 16,
        hnsw_ef_construction: 100,
        hnsw_ef_search: 50,
        ..Default::default()
    }
}

fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("splatsdb_f06a_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Ids of a result row, in order.
macro_rules! ids {
    ($r:expr) => {
        $r.iter().map(|n| n.index).collect::<Vec<usize>>()
    };
}

// ── 1. backlog: rows added by `add_splat` (no *_with_hnsw) are searchable ──

#[test]
fn f06a_fused_and_fast_see_the_backlog_left_by_add_splat() {
    let dim = 32;
    let mut store = SplatStore::new(hnsw_config(dim, 1000));
    store.add_splat(&random_vectors(400, dim, 7));
    store.build_index();
    assert!(!store.hnsw_needs_sync());

    let tail = random_vectors(7, dim, 99);
    assert!(store.add_splat(&tail));
    assert!(store.hnsw_needs_sync());
    assert_eq!(store.hnsw_indexed_count(), 400);

    // A query equal to a backlog row is that row, at distance 0.
    for i in 0..7 {
        let q = tail.row(i).to_owned();
        let fused = store.find_neighbors_fused(&q.view(), 3);
        assert_eq!(fused.len(), 3, "fused must answer k with a backlog");
        assert_eq!(fused[0].index, 400 + i, "fused: query == backlog row {i}");
        assert_eq!(fused[0].distance, 0.0, "fused: exact self distance");
        let fast = store.find_neighbors_fast(&q.view(), 3);
        assert_eq!(fast.len(), 3, "fast must answer k with a backlog");
        assert_eq!(fast[0].index, 400 + i, "fast: query == backlog row {i}");
        assert_eq!(fast[0].distance, 0.0, "fast: exact self distance");
    }

    // k=10: backlog rows count towards k, ids are distinct, order is (d, id).
    let q = tail.row(3).to_owned();
    for row in [
        store.find_neighbors_fused(&q.view(), 10),
        store.find_neighbors_fast(&q.view(), 10),
    ] {
        assert_eq!(row.len(), 10);
        let mut unique = ids!(row);
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 10, "no duplicated ids: {:?}", ids!(row));
    }

    // Recall against the exact linear answer on random probes.
    let probes = random_vectors(50, dim, 123);
    let (mut hit, mut total) = (0usize, 0usize);
    for i in 0..50 {
        let q = probes.row(i).to_owned();
        let lin: HashSet<usize> = store
            .find_neighbors(&q.view(), 10)
            .iter()
            .map(|n| n.index)
            .collect();
        let fus: HashSet<usize> = store
            .find_neighbors_fused(&q.view(), 10)
            .iter()
            .map(|n| n.index)
            .collect();
        hit += lin.intersection(&fus).count();
        total += lin.len();
    }
    assert!(
        hit as f64 / total as f64 >= 0.90,
        "fused recall@10 with backlog = {:.4}",
        hit as f64 / total as f64
    );
}

#[test]
fn f06a_rows_added_after_compact_stay_searchable_by_fused_and_fast() {
    let dim = 32;
    let mut store = SplatStore::new(hnsw_config(dim, 1000));
    let mu = random_vectors(600, dim, 21);
    let mut alpha = vec![1.0f32; 600];
    alpha[0] = 0.0;
    alpha[5] = 0.0;
    alpha[599] = 0.0;
    store
        .load_splats(&mu, &alpha, &[1.0f32; 600])
        .expect("load");
    store.compact();
    assert_eq!(store.n_active(), 597);
    assert_eq!(store.hnsw_indexed_count(), 0);
    assert!(store.hnsw_needs_sync());

    // Unindexed store: fused IS the exact linear answer (the original row
    // 300 sits at 298 once rows 0 and 5 were dropped).
    let q = mu.row(300).to_owned();
    let fused = store.find_neighbors_fused(&q.view(), 5);
    let lin = store.find_neighbors(&q.view(), 5);
    assert_eq!(ids!(fused), ids!(lin));
    assert_eq!(fused[0].index, 298);
    assert_eq!(fused[0].distance, 0.0);

    store.build_index();
    let tail = random_vectors(4, dim, 22);
    assert!(store.add_splat(&tail));
    for i in 0..4 {
        let q = tail.row(i).to_owned();
        let fused = store.find_neighbors_fused(&q.view(), 10);
        assert_eq!(fused[0].index, 597 + i, "fused after compact+add: row {i}");
        assert_eq!(fused[0].distance, 0.0);
        let fast = store.find_neighbors_fast(&q.view(), 10);
        assert_eq!(fast.len(), 10);
        assert_eq!(fast[0].index, 597 + i, "fast after compact+add: row {i}");
        assert_eq!(fast[0].distance, 0.0);
    }
}

// ── 2. k results with exact duplicates; ties ordered by (score, id asc) ──

#[test]
fn f06a_fast_and_fused_return_k_with_150_identical_rows() {
    let dim = 16;
    let mut store = SplatStore::new(hnsw_config(dim, 1000));
    let row = random_vectors(1, dim, 3);
    let dups = Array2::from_shape_fn((150, dim), |(_, j)| row[(0, j)]);
    store.add_splat(&dups);
    store.build_index();
    let q = row.row(0).to_owned();
    let expected: Vec<usize> = (0..10).collect();
    for (name, res) in [
        ("fast", store.find_neighbors_fast(&q.view(), 10)),
        ("fused", store.find_neighbors_fused(&q.view(), 10)),
        ("linear", store.find_neighbors(&q.view(), 10)),
    ] {
        assert_eq!(res.len(), 10, "{name}: k=10 over 150 identical rows");
        assert_eq!(ids!(res), expected, "{name}: ties by ascending id");
        assert!(
            res.iter().all(|n| n.distance == 0.0),
            "{name}: exact zero distances"
        );
    }
}

#[test]
fn f06a_ties_are_ordered_by_id_on_fast_fused_and_linear() {
    let dim = 16;
    let mut store = SplatStore::new(hnsw_config(dim, 1000));
    let mut data = random_vectors(200, dim, 11);
    for d in [3usize, 7, 40, 199] {
        for j in 0..dim {
            data[[d, j]] = data[[0, j]];
        }
    }
    store.add_splat(&data);
    store.build_index();
    let q = data.row(0).to_owned();
    let tied = [0usize, 3, 7, 40, 199];
    for k in 1..=5usize {
        for (name, res) in [
            ("fused", store.find_neighbors_fused(&q.view(), k)),
            ("linear", store.find_neighbors(&q.view(), k)),
        ] {
            assert_eq!(res.len(), k);
            assert_eq!(ids!(res), tied[..k].to_vec(), "{name}: k={k}");
        }
        // HRM2 probes clusters (approximate set), but the answer still has k
        // distinct rows, all exact duplicates, in ascending id order.
        let fast = store.find_neighbors_fast(&q.view(), k);
        assert_eq!(fast.len(), k, "fast: k={k}");
        let got = ids!(fast);
        assert!(
            got.iter().all(|i| tied.contains(i)),
            "fast: k={k} returned a non-duplicate: {got:?}"
        );
        assert!(
            got.windows(2).all(|w| w[0] < w[1]),
            "fast: ties by id, k={k}: {got:?}"
        );
        assert!(fast.iter().all(|n| n.distance == 0.0));
    }
}

// ── 3. NaN/inf: rejected BEFORE any state changes ──

#[test]
fn f06a_non_finite_rows_are_rejected_atomically() {
    let dim = 16;
    let mut store = SplatStore::new(hnsw_config(dim, 1000));
    store.add_splat(&random_vectors(10, dim, 5));
    store.build_index();
    let before = store.get_mu().unwrap();
    let version = store.data_version();

    let mut bad = random_vectors(3, dim, 6);
    bad[[1, 4]] = f32::NAN;
    assert!(!store.add_splat(&bad), "a NaN row must be rejected");
    assert_eq!(store.n_active(), 10, "no partial batch");
    assert_eq!(store.data_version(), version, "no version bump");
    assert_eq!(store.hnsw_indexed_count(), 10);
    assert!(!store.hnsw_needs_sync());
    assert_eq!(store.get_mu().unwrap(), before, "mu untouched");

    let mut inf = random_vectors(2, dim, 7);
    inf[[1, 0]] = f32::INFINITY;
    assert!(!store.add_splat(&inf), "an inf row must be rejected");
    assert_eq!(store.n_active(), 10);

    let mut vec = random_vectors(1, dim, 8).row(0).to_vec();
    vec[2] = f32::NEG_INFINITY;
    assert!(
        store.insert(&vec).is_none(),
        "insert must reject a non-finite vector"
    );
    assert_eq!(store.n_active(), 10);

    // The batch-with-HNSW writer: Ok(false), nothing indexed, nothing written.
    let dir = scratch_dir("nan");
    assert_eq!(
        store.try_add_batch_with_hnsw(&bad, Some(dir.to_str().unwrap())),
        Ok(false)
    );
    assert_eq!(store.n_active(), 10);
    assert!(
        !dir.join("hnsw_index.bin").exists(),
        "no index written for a rejected batch"
    );

    // load_splats: Err, state unchanged.
    let mut mu = random_vectors(4, dim, 9);
    mu[[2, 2]] = f32::NAN;
    assert!(store.load_splats(&mu, &[1.0; 4], &[1.0; 4]).is_err());
    assert_eq!(store.n_active(), 10);
    assert_eq!(store.data_version(), version);
    let alpha_bad = [1.0f32, f32::NAN, 1.0, 1.0];
    assert!(store
        .load_splats(&random_vectors(4, dim, 10), &alpha_bad, &[1.0; 4])
        .is_err());
    assert_eq!(store.n_active(), 10);

    // A valid batch still goes through.
    assert!(store.add_splat(&random_vectors(3, dim, 12)));
    assert_eq!(store.n_active(), 13);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.build_index();
    }));
    assert!(r.is_ok(), "build_index must not panic");
    std::fs::remove_dir_all(&dir).ok();
}
