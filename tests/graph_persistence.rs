//! F05c — the MCP knowledge graph is DURABLE (A04-F10) and fails closed.
//!
//! With a data dir (the directory of the SQLite doc store, or
//! `SPLATSDB_GRAPH_DIR`) the graph is loaded at start and every mutation is
//! persisted — atomically — BEFORE the tool answers. Verified here with real
//! `splatsdb mcp` processes over synthetic data:
//!   * SIGKILL after acknowledged mutations: the restart sees every node and
//!     edge, and node ids are never reused;
//!   * a mutation that cannot be persisted answers with an error, is rolled
//!     back and leaves the previous state file byte-for-byte intact;
//!   * an unreadable state file is refused: mutations error out and the file
//!     is never overwritten, truncated or deleted;
//!   * two servers on one directory: the stale one cannot overwrite the
//!     other's nodes;
//!   * without a data dir the graph is explicitly EPHEMERAL and writes nothing.
//!
//! Scratch data lives under TMPDIR only — never in the default `~/.hermes`
//! doc store: every server is started with an explicit `SPLATSDB_DOC_PATH`.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::json;

/// Closed local port: connection refused — nothing listens here.
const NO_SERVICE_URL: &str = "http://127.0.0.1:9/embed";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RESP_TIMEOUT: Duration = Duration::from_secs(120);
const LOG_TIMEOUT: Duration = Duration::from_secs(30);
/// File the graph is persisted to inside the data dir.
const STATE_FILE: &str = "graph_state.json";
/// Prefix of the transient file a save writes before its atomic rename.
const TMP_PREFIX: &str = "graph_state.json.tmp.";

struct Server {
    child: Child,
    stdin: Option<ChildStdin>,
    out: Receiver<String>,
    err: Receiver<String>,
    /// stderr lines already drained.
    err_log: Vec<String>,
}

fn pump<R: Read + Send + 'static>(r: R) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(r).lines() {
            if tx.send(line.unwrap_or_default()).is_err() {
                break;
            }
        }
    });
    rx
}

/// Start `splatsdb mcp` on `doc_db`. The embed service is DOWN and the
/// SimCos fallback is explicitly authorized: graph nodes get deterministic
/// text-hash vectors, which is all persistence needs.
fn spawn(doc_db: &Path, envs: &[(&str, &str)]) -> Server {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_splatsdb"));
    cmd.arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("SPLATSDB_DOC_PATH", doc_db)
        .env("SPLATSDB_EMBED_URL", NO_SERVICE_URL)
        .env("SPLATSDB_ALLOW_SIMCOS", "1")
        .env_remove("SPLATSDB_GRAPH_DIR");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let mut child = cmd.spawn().expect("failed to spawn `splatsdb mcp`");
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    let out = pump(stdout);
    let err = pump(stderr);
    Server {
        child,
        stdin: Some(stdin),
        out,
        err,
        err_log: Vec::new(),
    }
}

impl Server {
    /// Send one JSON-RPC request and return the full response object.
    fn rpc(&mut self, id: u64, method: &str, params: serde_json::Value) -> serde_json::Value {
        let req = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let stdin = self.stdin.as_mut().expect("server already closed");
        stdin
            .write_all(req.to_string().as_bytes())
            .expect("write request");
        stdin.write_all(b"\n").expect("write newline");
        stdin.flush().expect("flush stdin");
        let deadline = Instant::now() + RESP_TIMEOUT;
        loop {
            assert!(
                Instant::now() < deadline,
                "timeout waiting for id={id} ({method})"
            );
            let line = match self.out.recv_timeout(RESP_TIMEOUT) {
                Ok(l) => l,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("stdout idle waiting for id={id} ({method})")
                }
                Err(RecvTimeoutError::Disconnected) => panic!(
                    "server stdout closed before answering id={id} ({method}); stderr:\n{}",
                    self.err_log.join("\n")
                ),
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                return v;
            }
        }
    }

    /// `tools/call` → (is_error, parsed tool payload or the raw error text).
    fn tool(&mut self, id: u64, name: &str, args: serde_json::Value) -> (bool, serde_json::Value) {
        let v = self.rpc(id, "tools/call", json!({ "name": name, "arguments": args }));
        let is_error = v
            .pointer("/result/isError")
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        let text = v
            .pointer("/result/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string();
        (is_error, serde_json::from_str(&text).unwrap_or(json!(text)))
    }

    fn ok(&mut self, id: u64, name: &str, args: serde_json::Value) -> serde_json::Value {
        let (is_error, payload) = self.tool(id, name, args);
        assert!(!is_error, "{name} failed: {payload}");
        payload
    }

    fn wait_stderr_containing(&mut self, needle: &str, timeout: Duration) -> bool {
        if self.err_log.iter().any(|l| l.contains(needle)) {
            return true;
        }
        let deadline = Instant::now() + timeout;
        while let Ok(line) = self
            .err
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            self.err_log.push(line.clone());
            if line.contains(needle) {
                return true;
            }
        }
        false
    }

    /// Assert a startup / warm-start line is on stderr, printing the whole
    /// drained log when it is not.
    fn expect_log(&mut self, needle: &str) {
        assert!(
            self.wait_stderr_containing(needle, LOG_TIMEOUT),
            "stderr lacks {needle:?}; log:\n{}",
            self.err_log.join("\n")
        );
    }

    fn wait_ready(&mut self) -> bool {
        self.wait_stderr_containing("MCP server ready", STARTUP_TIMEOUT)
    }

    fn status(&mut self, id: u64) -> serde_json::Value {
        self.ok(id, "splatsdb_status", json!({}))
    }

    fn stats(&mut self, id: u64) -> serde_json::Value {
        self.ok(id, "splatsdb_graph_stats", json!({}))
    }

    fn add_doc(&mut self, id: u64, text: &str) -> serde_json::Value {
        self.ok(id, "splatsdb_graph_add_doc", json!({ "text": text }))
    }

    fn add_entity(&mut self, id: u64, name: &str) -> serde_json::Value {
        self.ok(
            id,
            "splatsdb_graph_add_entity",
            json!({ "name": name, "entity_type": "concept" }),
        )
    }

    fn add_relation(
        &mut self,
        id: u64,
        source: u64,
        target: u64,
        relation: &str,
    ) -> serde_json::Value {
        self.ok(
            id,
            "splatsdb_graph_add_relation",
            json!({ "source_id": source, "target_id": target, "relation_type": relation, "weight": 0.9 }),
        )
    }

    fn traverse(&mut self, id: u64, start: u64, depth: u64) -> serde_json::Value {
        self.ok(
            id,
            "splatsdb_graph_traverse",
            json!({ "start_id": start, "max_depth": depth }),
        )
    }

    /// Close stdin (server exits on EOF) and reap the process.
    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
    }

    /// SIGKILL right now: no stdin EOF, no shutdown path, no flush-on-exit —
    /// a crash.
    fn kill_now(mut self) {
        self.child.kill().expect("SIGKILL");
        self.child.wait().expect("reap killed server");
    }
}

impl Drop for Server {
    /// No orphans: a failed assertion must not leave a server behind.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splatsdb_mcp_graph_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn state_path(dir: &Path) -> PathBuf {
    dir.join(STATE_FILE)
}

/// Transient files a save left behind (must be none outside a crash).
fn tmp_files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read data dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with(TMP_PREFIX))
        .collect();
    names.sort();
    names
}

fn id_of(v: &serde_json::Value, key: &str) -> u64 {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("no numeric {key} in {v}"))
}

fn text_of(v: &serde_json::Value) -> String {
    v.as_str()
        .map(str::to_string)
        .unwrap_or_else(|| v.to_string())
}

/// Make `dir` read-only for this process. Returns false (and restores the
/// permissions) when they are ignored — root, or a filesystem without modes:
/// the injected failure would be void.
#[cfg(unix)]
fn make_unwritable(dir: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(dir).expect("dir metadata").permissions();
    perms.set_mode(0o555);
    std::fs::set_permissions(dir, perms).expect("chmod read-only");
    match std::fs::File::create(dir.join(".f05c_write_probe")) {
        Ok(probe) => {
            drop(probe);
            restore_writable(dir);
            let _ = std::fs::remove_file(dir.join(".f05c_write_probe"));
            false
        }
        Err(_) => true,
    }
}

#[cfg(unix)]
fn restore_writable(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(dir).expect("dir metadata").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(dir, perms).expect("chmod writable");
}

// ---------------------------------------------------------------------------
// (1) Crash: everything acknowledged is on disk, ids are never reused
// ---------------------------------------------------------------------------

#[test]
fn g01_sigkill_loses_nothing_acknowledged_and_ids_are_never_reused() {
    let dir = scratch("g01");
    let db = dir.join("docs.db");
    let mut a = spawn(&db, &[]);
    a.expect_log("no state file yet");

    let d = a.add_doc(1, "F05c crash doc: the graph must survive SIGKILL");
    assert_eq!(d["persisted"], json!(true), "{d}");
    let doc = id_of(&d, "node_id");
    let e = a.add_entity(2, "Durability");
    assert_eq!(e["persisted"], json!(true), "{e}");
    let ent = id_of(&e, "node_id");
    let r = a.add_relation(3, doc, ent, "MENTIONS");
    assert_eq!(r["persisted"], json!(true), "{r}");
    a.expect_log("graph persisted");

    let stats_a = a.stats(4);
    assert_eq!(stats_a["total_nodes"], json!(2), "{stats_a}");
    assert_eq!(stats_a["total_edges"], json!(1), "{stats_a}");
    assert_eq!(stats_a["documents"], json!(1), "{stats_a}");
    assert_eq!(stats_a["entities"], json!(1), "{stats_a}");

    // Persisted BEFORE the answer: the file is already complete on disk.
    let bytes_a = std::fs::read(state_path(&dir))
        .expect("graph_state.json must exist once a mutation was acknowledged");
    let on_disk: serde_json::Value =
        serde_json::from_slice(&bytes_a).expect("the live file must be complete JSON");
    assert_eq!(on_disk["version"], json!(1), "{on_disk}");
    assert_eq!(on_disk["nodes"].as_array().expect("nodes").len(), 2);
    assert_eq!(on_disk["edges"].as_array().expect("edges").len(), 1);

    // CRASH.
    a.kill_now();
    assert_eq!(
        std::fs::read(state_path(&dir)).expect("state file"),
        bytes_a,
        "the kill must leave the acknowledged state intact"
    );

    // ---- Restart on the same files ----
    let mut b = spawn(&db, &[]);
    b.expect_log("loaded 2 nodes / 1 edges");
    let t = b.traverse(5, doc, 2);
    assert_eq!(t["count"], json!(2), "{t}");
    let nodes = t["nodes"].as_array().expect("nodes");
    assert!(
        nodes
            .iter()
            .any(|n| n["id"] == json!(ent) && n["content"] == "Durability"),
        "the edge doc -> entity must survive the crash: {t}"
    );
    let stats_b = b.stats(6);
    for key in [
        "total_nodes",
        "total_edges",
        "documents",
        "entities",
        "concepts",
    ] {
        assert_eq!(stats_a[key], stats_b[key], "{key}: {stats_a} vs {stats_b}");
    }
    let st = b.status(7);
    assert_eq!(
        st["graph_persistence"],
        json!({
            "persisted": true, "ephemeral": false, "state": "durable",
            "path": state_path(&dir).display().to_string(), "error": null
        }),
        "{st}"
    );

    // Ids are never reused across the crash.
    let e2 = b.add_entity(8, "AfterRestart");
    assert!(id_of(&e2, "node_id") > ent, "{e2}");
    b.shutdown();

    let mut c = spawn(&db, &[]);
    let stats_c = c.stats(1);
    assert_eq!(stats_c["total_nodes"], json!(3), "{stats_c}");
    assert_eq!(stats_c["total_edges"], json!(1), "{stats_c}");
    c.shutdown();
    assert!(tmp_files(&dir).is_empty(), "{:?}", tmp_files(&dir));
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// (2) No data dir → explicitly ephemeral, nothing written
// ---------------------------------------------------------------------------

#[test]
fn g02_without_a_data_dir_the_graph_is_explicitly_ephemeral() {
    // (a) explicit opt-out
    let dir = scratch("g02a");
    let db = dir.join("docs.db");
    let mut a = spawn(&db, &[("SPLATSDB_GRAPH_DIR", "memory")]);
    a.expect_log("graph state: EPHEMERAL");
    let st = a.status(1);
    assert_eq!(
        st["graph_persistence"],
        json!({"persisted": false, "ephemeral": true, "state": "ephemeral", "path": null, "error": null}),
        "{st}"
    );
    let d = a.add_doc(2, "ephemeral by explicit configuration");
    assert_eq!(d["persisted"], json!(false), "{d}");
    assert!(
        !state_path(&dir).exists(),
        "an ephemeral graph must write nothing"
    );
    a.shutdown();
    let mut b = spawn(&db, &[("SPLATSDB_GRAPH_DIR", "memory")]);
    assert_eq!(b.stats(1)["total_nodes"], json!(0), "nothing was persisted");
    b.shutdown();
    let _ = std::fs::remove_dir_all(&dir);

    // (b) derived: the doc store itself could not be opened → no data dir.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let ro = scratch("g02b_ro");
        let locked = ro.join("locked");
        std::fs::create_dir_all(&locked).expect("mkdir");
        // The parent cannot even be created: SQLite init fails → `:memory:`.
        let db = locked.join("sub").join("docs.db");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555))
            .expect("chmod read-only");
        if let Ok(probe) = std::fs::File::create(locked.join(".probe")) {
            drop(probe);
            let _ = std::fs::remove_file(locked.join(".probe"));
            let _ = std::fs::remove_dir_all(&ro);
            eprintln!("g02(b) skipped: directory permissions are ignored by this user");
        } else {
            let mut c = spawn(&db, &[]);
            c.expect_log("SQLite init failed");
            c.expect_log("graph state: EPHEMERAL");
            let st = c.status(1);
            assert_eq!(st["persisted"], json!(false), "{st}");
            assert_eq!(st["graph_persistence"]["state"], json!("ephemeral"), "{st}");
            assert_eq!(st["graph_persistence"]["path"], json!(null), "{st}");
            let d = c.add_doc(2, "ephemeral because the doc store is");
            assert_eq!(d["persisted"], json!(false), "{d}");
            c.shutdown();
            let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));
            assert!(
                !locked.join("sub").exists(),
                "nothing may be created under an ephemeral doc store"
            );
            let _ = std::fs::remove_dir_all(&ro);
        }
    }
}

// ---------------------------------------------------------------------------
// (3) A mutation that cannot be persisted: error + rollback + file intact
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn g03_unpersistable_mutation_is_an_error_rolls_back_and_keeps_the_previous_file() {
    let dir = scratch("g03");
    let db = dir.join("docs.db");
    let mut a = spawn(&db, &[]);
    let first = a.add_doc(1, "first node, persisted");
    let n1 = id_of(&first, "node_id");
    let before = std::fs::read(state_path(&dir)).expect("state file after the first mutation");

    // REAL write failure injected from outside the process: the data dir
    // becomes read-only for the server.
    if !make_unwritable(&dir) {
        eprintln!("g03 skipped: directory permissions are ignored by this user");
        a.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let (is_error, payload) = a.tool(
        2,
        "splatsdb_graph_add_doc",
        json!({ "text": "second node must not be acknowledged" }),
    );
    let msg = text_of(&payload);
    restore_writable(&dir);

    assert!(
        is_error,
        "a mutation that could not be persisted answered success: {payload}"
    );
    for needle in ["not persisted", "rolled back", STATE_FILE] {
        assert!(msg.contains(needle), "{needle:?} missing from: {msg}");
    }
    assert_eq!(
        std::fs::read(state_path(&dir)).expect("state file"),
        before,
        "the previous state must be intact"
    );
    assert!(
        tmp_files(&dir).is_empty(),
        "temp files left behind: {:?}",
        tmp_files(&dir)
    );
    a.expect_log("graph save FAILED");

    // Rolled back in memory: the failed node is invisible and its id unused.
    let st = a.stats(3);
    assert_eq!(st["total_nodes"], json!(1), "{st}");
    let retry = a.add_doc(4, "second node must not be acknowledged");
    assert_eq!(
        id_of(&retry, "node_id"),
        n1 + 1,
        "the failed attempt consumed an id: {retry}"
    );
    assert_eq!(a.stats(5)["total_nodes"], json!(2));
    a.shutdown();

    let mut b = spawn(&db, &[]);
    b.expect_log("loaded 2 nodes / 0 edges");
    b.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// (4) Unreadable state file: fail closed, never overwritten
// ---------------------------------------------------------------------------

#[test]
fn g04_unreadable_state_file_fails_closed_and_is_never_overwritten() {
    let dir = scratch("g04");
    let db = dir.join("docs.db");
    let mut a = spawn(&db, &[]);
    a.add_doc(1, "a valid graph, about to be corrupted");
    a.add_entity(2, "Victim");
    a.shutdown();
    let good = std::fs::read(state_path(&dir)).expect("valid state");

    let mut foreign: serde_json::Value = serde_json::from_slice(&good).expect("valid JSON");
    foreign["version"] = json!(99);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("truncated", good[..good.len() / 2].to_vec()),
        ("garbage", b"{\"version\": 1, \"next_id\": ".to_vec()),
        ("empty", Vec::new()),
        (
            "foreign version",
            serde_json::to_vec(&foreign).expect("serialize"),
        ),
    ];
    for (tag, corrupt) in cases {
        std::fs::write(state_path(&dir), &corrupt).expect("plant corrupt state");

        let mut b = spawn(&db, &[]);
        assert!(
            b.wait_ready(),
            "the server must still start ({tag}); log:\n{}",
            b.err_log.join("\n")
        );
        b.expect_log("graph state file is UNREADABLE");
        if tag == "foreign version" {
            b.expect_log("unsupported state version 99");
        }

        let (is_error, payload) = b.tool(
            1,
            "splatsdb_graph_add_doc",
            json!({ "text": "must be refused while the state file is unreadable" }),
        );
        assert!(
            is_error,
            "{tag}: a mutation over an unreadable state answered success: {payload}"
        );
        assert!(
            text_of(&payload).contains("graph state file is unreadable"),
            "{tag}: {payload}"
        );
        let st = b.status(2);
        assert_eq!(
            st["graph_persistence"]["state"],
            json!("corrupt"),
            "{tag}: {st}"
        );
        assert_eq!(
            st["graph_persistence"]["persisted"],
            json!(false),
            "{tag}: {st}"
        );
        assert_eq!(
            st["graph_persistence"]["path"],
            json!(state_path(&dir).display().to_string()),
            "{tag}: {st}"
        );
        assert!(
            st["graph_persistence"]["error"]
                .as_str()
                .unwrap_or_default()
                .contains(STATE_FILE),
            "{tag}: {st}"
        );
        assert_eq!(
            b.stats(3)["total_nodes"],
            json!(0),
            "{tag}: reads see an empty graph"
        );
        b.shutdown();

        assert_eq!(
            std::fs::read(state_path(&dir)).expect("state file"),
            corrupt,
            "{tag}: the unreadable file was modified"
        );
        assert!(tmp_files(&dir).is_empty(), "{tag}: {:?}", tmp_files(&dir));
    }

    // Recovery is an explicit operator action: move the bad file away.
    std::fs::rename(state_path(&dir), dir.join("graph_state.json.corrupt"))
        .expect("set the bad file aside");
    let mut c = spawn(&db, &[]);
    c.expect_log("no state file yet");
    let d = c.add_doc(1, "fresh graph after the operator set the bad file aside");
    assert_eq!(d["persisted"], json!(true), "{d}");
    c.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// (5) Two servers, one directory: no lost update
// ---------------------------------------------------------------------------

#[test]
fn g05_stale_server_cannot_overwrite_another_writers_nodes() {
    let dir = scratch("g05");
    let db = dir.join("docs.db");
    let mut a = spawn(&db, &[]);
    a.add_doc(1, "written by server A");

    let mut b = spawn(&db, &[]);
    b.expect_log("loaded 1 nodes / 0 edges");

    // A writes again: the file changes under B's feet.
    a.add_doc(2, "written by server A after B started");
    let after_a = std::fs::read(state_path(&dir)).expect("state file");

    let (is_error, payload) = b.tool(
        1,
        "splatsdb_graph_add_doc",
        json!({ "text": "written by the stale server B" }),
    );
    assert!(
        is_error,
        "a stale server overwrote the other writer's nodes: {payload}"
    );
    assert!(text_of(&payload).contains("changed since"), "{payload}");
    assert_eq!(
        std::fs::read(state_path(&dir)).expect("state file"),
        after_a,
        "A's state must be intact"
    );
    assert_eq!(
        b.stats(2)["total_nodes"],
        json!(1),
        "B's refused node was rolled back"
    );
    b.shutdown();
    a.shutdown();

    let mut c = spawn(&db, &[]);
    let st = c.stats(1);
    assert_eq!(
        st["total_nodes"],
        json!(2),
        "both of A's nodes, none of B's: {st}"
    );
    c.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}
