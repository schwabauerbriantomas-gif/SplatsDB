//! F02f — every CLI command that writes into a data-dir takes the same
//! `DataDirLock` `serve` holds, for its whole read→write window.
//!
//! Before the fix only `serve` locked: `index`, `append`, `ingest`, `save`,
//! `verbatim-store`, … ran against the data-dir of a live server, rewrote
//! the shard under it and left `serve` to fail its shutdown fingerprint check
//! (or worse, to overwrite the CLI's work). The lock is the KERNEL's
//! (`flock`): a stale lockfile never blocks, and once the server is gone the
//! very same commands succeed.

#[path = "cli_durability_common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::*;

/// Exit status of a CLI writer refused by the data-dir lock.
const EXIT_LOCKED: i32 = 3;

type Tree = BTreeMap<PathBuf, Vec<u8>>;

/// Every file under `dir` with its bytes.
fn snapshot(dir: &Path) -> Tree {
    fn walk(root: &Path, dir: &Path, out: &mut Tree) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = Tree::new();
    walk(dir, dir, &mut out);
    out
}

fn snapshot_diff(before: &Tree, after: &Tree) -> Vec<String> {
    let mut out = Vec::new();
    for (path, bytes) in before {
        match after.get(path) {
            None => out.push(format!("{} was removed", path.display())),
            Some(now) if now != bytes => out.push(format!(
                "{} changed ({} -> {} bytes)",
                path.display(),
                bytes.len(),
                now.len()
            )),
            _ => {}
        }
    }
    for path in after.keys().filter(|p| !before.contains_key(*p)) {
        out.push(format!("{} was created", path.display()));
    }
    out
}

fn owned(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

fn expect_success(dir: &Path, args: &[&str], token: &str, name: &str) -> Result<(), String> {
    let (status, out, err) = cli(dir, DIM, args);
    if status.success() && out.contains(token) {
        Ok(())
    } else {
        Err(format!(
            "`{name}` did not succeed once the lock was free: exit {:?}, stdout={out}, stderr={err}",
            status.code()
        ))
    }
}

/// The census of CLI writers (see `cli_command_census` in reports/F02f.json):
/// each one refuses — fast, with a diagnostic, exit 3, nothing modified —
/// while a `serve` holds the data-dir, leaves that `serve` intact, and
/// succeeds once it stopped. Read-only commands are not blocked.
#[test]
fn writer_commands_refuse_a_data_dir_a_serve_holds_and_run_once_it_stops() {
    let root = scratch("lock");
    let data = root.join("data");
    let verbatim = root.join("verbatim");
    write_matrix(&shard_path(&data), 5, DIM, 0.0);
    let vectors4 = root.join("vectors4.bin");
    write_matrix(&vectors4, 4, DIM, 10.0);
    let extra2 = root.join("extra2.bin");
    write_matrix(&extra2, 2, DIM, 50.0);
    let raw = root.join("raw.bin");
    write_rows(&raw, &clustered_corpus());
    let queries = root.join("queries.bin");
    write_matrix(&queries, 1, DIM, 0.5);
    let (data_s, raw_s, vectors4_s, extra2_s, queries_s) = (
        data.to_string_lossy().into_owned(),
        raw.to_string_lossy().into_owned(),
        vectors4.to_string_lossy().into_owned(),
        extra2.to_string_lossy().into_owned(),
        queries.to_string_lossy().into_owned(),
    );
    // (command, args, what stdout prints on success, dir for the success run)
    let writers: [(&str, Vec<String>, &str, &Path); 9] = [
        (
            "index",
            owned(&["index", "--input", &vectors4_s]),
            "indexed",
            &data,
        ),
        (
            "append",
            owned(&["append", "--input", &extra2_s]),
            "appended",
            &data,
        ),
        (
            "ingest",
            owned(&["ingest", "--input", &raw_s, "--n-clusters", "6", "--seed", "7"]),
            "ingested",
            &data,
        ),
        (
            "ingest-hierarchical",
            owned(&[
                "ingest-hierarchical",
                "--input",
                &raw_s,
                "-n",
                "3",
                "--min-cluster-size",
                "2",
                "--seed",
                "7",
            ]),
            "ingested",
            &data,
        ),
        ("save", owned(&["save"]), "saved", &data),
        (
            "verbatim-store",
            owned(&["verbatim-store", "--id", "v1", "--text", "hello world"]),
            "stored_verbatim",
            // `verbatim-store` builds its own 64-D store: it only fits an
            // empty data-dir.
            &verbatim,
        ),
        (
            "doc-add",
            owned(&["doc-add", "--id", "d1", "--text", "hello"]),
            "\"status\":\"ok\"",
            &data,
        ),
        (
            "doc-del",
            owned(&["doc-del", "--id", "d1"]),
            "deleted",
            &data,
        ),
        (
            "bench-hnsw",
            owned(&[
                "bench-hnsw",
                "--train",
                &raw_s,
                "--queries",
                &queries_s,
                "--dim",
                "32",
                "--data-dir",
                &data_s,
            ]),
            "\"bench\"",
            &data,
        ),
    ];

    let port = probe_port();
    let mut server = spawn_serve(&data, &root.join("serve.log"), port);
    wait_healthy(port, &mut server);

    // Read-only commands are not blocked by the lock.
    let query = query_of(DIM);
    for args in [
        owned(&["search", "--query", &query, "--k", "2"]),
        owned(&["status"]),
        owned(&["list"]),
    ] {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let (status, out, err) = cli(&data, DIM, &refs);
        assert!(
            status.success(),
            "reader {:?} was blocked while serve runs: stdout={out}\nstderr={err}",
            args
        );
    }

    let before = snapshot(&data);
    let shard_before = std::fs::read(shard_path(&data)).unwrap();
    let lock_before = std::fs::read(data.join(".splatsdb.lock")).unwrap();

    // Every violation is collected, so one run shows the whole census.
    let mut violations = Vec::new();
    for (name, args, token, _) in &writers {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let t0 = Instant::now();
        let (status, out, err) = cli(&data, DIM, &refs);
        let elapsed = t0.elapsed();
        if status.code() != Some(EXIT_LOCKED) {
            violations.push(format!(
                "`{name}` exited {:?} instead of {EXIT_LOCKED} while serve holds the data-dir",
                status.code()
            ));
        }
        if !err.contains("bloqueado") {
            violations.push(format!("`{name}` gave no lock diagnostic: stderr={err}"));
        }
        if out.contains(token) {
            violations.push(format!(
                "`{name}` reported success against a locked data-dir: stdout={out}"
            ));
        }
        if elapsed > Duration::from_secs(20) {
            violations.push(format!(
                "`{name}` took {elapsed:?} to be refused (it must fail fast)"
            ));
        }
        let diff = snapshot_diff(&before, &snapshot(&data));
        if !diff.is_empty() {
            violations.push(format!(
                "`{name}` modified the data-dir of a running serve: {}",
                diff.join(", ")
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "{} violation(s) of the serve data-dir lock:\n  - {}",
        violations.len(),
        violations.join("\n  - ")
    );

    // The refused commands left the server intact: it still answers, its
    // shard and lock are untouched, and it still accepts a durable write.
    assert!(matches!(http(port, "GET", "/health", None), Some((200, _))));
    let (code, body) = http(port, "POST", "/status", None).expect("/status answered");
    assert_eq!(code, 200, "{body}");
    let status: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(status["n_active"], 5, "/status after the refusals: {body}");
    assert_eq!(
        std::fs::read(shard_path(&data)).unwrap(),
        shard_before,
        "no refused writer may touch the shard"
    );
    assert_eq!(
        std::fs::read(data.join(".splatsdb.lock")).unwrap(),
        lock_before,
        "the lock file still names the serve that holds it"
    );
    let far: Vec<f32> = (0..DIM).map(|d| 40.0 + 0.25 * d as f32).collect();
    serve_store(port, &far);

    let code = shutdown(&mut server);
    assert_eq!(
        code,
        Some(0),
        "serve must shut down cleanly after the refused writers: {}",
        stderr_of(&server)
    );
    assert_eq!(
        read_bin(&shard_path(&data)).len(),
        6,
        "the shutdown persists the /store the server accepted"
    );

    // The lock died with the server: the same commands now succeed.
    let mut failures = Vec::new();
    for (name, args, token, dir) in &writers {
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        if let Err(e) = expect_success(dir, &refs, token, name) {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} writer command(s) still failed after serve stopped:\n  - {}",
        failures.len(),
        failures.join("\n  - ")
    );
    assert!(
        data.join(".splatsdb.lock").exists(),
        "the lock file is never removed"
    );
    assert!(
        hnsw_path(&data).exists(),
        "bench-hnsw persisted its index once the lock was free"
    );
    assert!(
        TcpStream::connect(("127.0.0.1", port)).is_err(),
        "no listener may remain on the probe port"
    );
    std::fs::remove_dir_all(&root).ok();
}
