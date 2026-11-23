//! F06a integration contracts (fixer F06a — validator V1_integration NO_GO on
//! `7c45a1d`): backlog visibility of `find_neighbors_fused` / `_fast`, k
//! results with exact duplicates, `(score, id)` ties on HRM2, and the atomic
//! rejection of non-finite rows. Every `f06a_*` test was RED on `7c45a1d`
//! before the matching fix landed (logs: tmp/F06a/red/).

use ndarray::Array2;
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
