//! F02f — a failed HNSW persistence is an error, never a silent success.
//!
//! `build_index_with_save` returned `()` and only warned when
//! `hnsw_index.bin` could not be written, so `append` (the CLI writer that
//! maintains a persisted HNSW index) reported `appended` / exit 0 with a
//! stale index on disk. Fault injection on the real binary: the index FILE
//! is made read-only (the directory stays writable), and separately the
//! shard DIRECTORY is made read-only.

#[path = "cli_durability_common/mod.rs"]
mod common;

use std::fs::Permissions;
use std::path::Path;

use common::*;

fn readonly(path: &Path, on: bool) -> Permissions {
    let original = std::fs::metadata(path).unwrap().permissions();
    let mut perms = original.clone();
    perms.set_readonly(on);
    std::fs::set_permissions(path, perms).unwrap();
    original
}

fn restore(path: &Path, original: Permissions) {
    std::fs::set_permissions(path, original).unwrap();
}

/// Permission-based fault injection is void for root: the test must say so
/// instead of passing vacuously.
fn require_non_root() {
    let euid = std::process::Command::new("id")
        .arg("-u")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    assert_ne!(
        euid, "0",
        "this test injects unwritable files/dirs: run it as a non-root user"
    );
}

/// A 4-row shard plus the `hnsw_index.bin` that makes `append` maintain it.
fn seeded(tag: &str) -> (std::path::PathBuf, String, String) {
    let root = scratch(tag);
    let data = root.join("data");
    let vectors4 = root.join("vectors4.bin");
    write_matrix(&vectors4, 4, DIM, 0.0);
    let extra2 = root.join("extra2.bin");
    write_matrix(&extra2, 2, DIM, 50.0);
    let (vectors4, extra2) = (
        vectors4.to_string_lossy().into_owned(),
        extra2.to_string_lossy().into_owned(),
    );
    let (status, out, err) = cli(&data, DIM, &["index", "--input", &vectors4]);
    assert!(
        status.success() && out.contains("indexed"),
        "index failed: stdout={out}\nstderr={err}"
    );
    let query = query_of(DIM);
    let (status, out, err) = cli(&data, DIM, &["hnsw-search", "--query", &query, "--k", "4"]);
    assert!(status.success(), "hnsw-search failed: stdout={out}\nstderr={err}");
    assert!(out.contains("\"index\""), "hnsw-search printed no rows: {out}");
    assert!(
        err.contains("Saved HNSW index (4 vectors)"),
        "hnsw-search must persist the index: {err}"
    );
    (data, vectors4, extra2)
}

/// FILE not writable: `append` must exit non-zero without `appended` and
/// leave both the shard and the previous index untouched.
#[test]
fn append_does_not_report_success_when_the_hnsw_index_cannot_be_persisted() {
    require_non_root();
    let (data, _vectors4, extra2) = seeded("hnsw-append");
    let (shard, index) = (shard_path(&data), hnsw_path(&data));
    let (shard_before, index_before) = (
        std::fs::read(&shard).unwrap(),
        std::fs::read(&index).unwrap(),
    );

    let original = readonly(&index, true);
    let result = cli(&data, DIM, &["append", "--input", &extra2]);
    restore(&index, original); // first: the cleanup has to work anyway

    let (status, out, err) = result;
    assert_ne!(
        status.code(),
        Some(0),
        "append reported success although the HNSW index could not be written: stdout={out}\nstderr={err}"
    );
    assert!(!out.contains("appended"), "stdout must not claim an append: {out}");
    assert!(err.contains("HNSW"), "the diagnostic must name the failed persist: {err}");
    assert_eq!(
        std::fs::read(&shard).unwrap(),
        shard_before,
        "a failed append must leave the shard untouched"
    );
    assert_eq!(
        std::fs::read(&index).unwrap(),
        index_before,
        "the previous HNSW index must survive the failed persist"
    );

    // Writable again: appended, and the index follows the shard.
    let (status, out, err) = cli(&data, DIM, &["append", "--input", &extra2]);
    assert!(
        status.success() && out.contains("appended"),
        "append failed on a writable data-dir: stdout={out}\nstderr={err}"
    );
    assert_eq!(read_bin(&shard).len(), 6, "the shard grew by the 2 vectors");
    assert_ne!(
        std::fs::read(&index).unwrap(),
        index_before,
        "the persisted index must follow the shard"
    );
    assert!(
        err.contains("Saved HNSW index (6 vectors)"),
        "append must persist the extended index: {err}"
    );
    std::fs::remove_dir_all(data.parent().unwrap()).ok();
}

/// DIRECTORY not writable: `index` must exit non-zero without `indexed` and
/// keep the previous shard and index byte for byte.
#[test]
fn index_on_an_unwritable_shard_dir_exits_nonzero_and_keeps_the_previous_state() {
    require_non_root();
    let (data, _vectors4, _extra2) = seeded("hnsw-index");
    let (shard, index) = (shard_path(&data), hnsw_path(&data));
    let (shard_before, index_before) = (
        std::fs::read(&shard).unwrap(),
        std::fs::read(&index).unwrap(),
    );
    let vectors6 = data.parent().unwrap().join("vectors6.bin");
    write_matrix(&vectors6, 6, DIM, 10.0);
    let vectors6 = vectors6.to_string_lossy().into_owned();
    let vectors_dir = shard.parent().unwrap();

    let original = readonly(vectors_dir, true);
    let result = cli(&data, DIM, &["index", "--input", &vectors6]);
    restore(vectors_dir, original);

    let (status, out, err) = result;
    assert_eq!(
        status.code(),
        Some(1),
        "index reported success on an unwritable shard dir: stdout={out}\nstderr={err}"
    );
    assert!(err.contains("Save error"), "{err}");
    assert!(!out.contains("indexed"), "stdout must not claim an index: {out}");
    assert_eq!(std::fs::read(&shard).unwrap(), shard_before, "the shard");
    assert_eq!(std::fs::read(&index).unwrap(), index_before, "the index");

    let (status, out, err) = cli(&data, DIM, &["index", "--input", &vectors6]);
    assert!(
        status.success() && out.contains("\"n_active\":6"),
        "index failed on a writable data-dir: stdout={out}\nstderr={err}"
    );
    assert_eq!(read_bin(&shard).len(), 6);
    std::fs::remove_dir_all(data.parent().unwrap()).ok();
}

/// A reader is not turned into a failure by an index it cannot cache:
/// `hnsw-search` rebuilds the stale index in memory, answers, and says on
/// stderr that the persist failed — never a silent success either.
#[test]
fn hnsw_search_answers_when_the_stale_index_cannot_be_replaced_and_says_so() {
    require_non_root();
    let (data, _vectors4, _extra2) = seeded("hnsw-search");
    let index = hnsw_path(&data);
    let query = query_of(DIM);
    // The shard grows: the persisted index is stale (4 vectors for 6 splats).
    let vectors6 = data.parent().unwrap().join("vectors6.bin");
    write_matrix(&vectors6, 6, DIM, 10.0);
    let vectors6 = vectors6.to_string_lossy().into_owned();
    let (status, out, err) = cli(&data, DIM, &["index", "--input", &vectors6]);
    assert!(status.success(), "index failed: stdout={out}\nstderr={err}");
    let index_before = std::fs::read(&index).unwrap();

    let original = readonly(&index, true);
    let result = cli(&data, DIM, &["hnsw-search", "--query", &query, "--k", "6"]);
    restore(&index, original);

    let (status, out, err) = result;
    assert!(status.success(), "a read-only command must not fail on the cache: {err}");
    assert!(out.contains("\"index\""), "no rows: {out}");
    assert!(
        err.contains("failed to save HNSW index"),
        "the failed persist must be visible: {err}"
    );
    assert_eq!(
        std::fs::read(&index).unwrap(),
        index_before,
        "the read-only index file is untouched"
    );

    let (status, out, err) = cli(&data, DIM, &["hnsw-search", "--query", &query, "--k", "6"]);
    assert!(status.success() && out.contains("\"index\""), "stdout={out}\nstderr={err}");
    assert!(
        err.contains("Saved HNSW index (6 vectors)"),
        "a writable data-dir gets the rebuilt index: {err}"
    );
    assert_ne!(std::fs::read(&index).unwrap(), index_before);
    std::fs::remove_dir_all(data.parent().unwrap()).ok();
}
