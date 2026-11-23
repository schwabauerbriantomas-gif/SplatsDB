//! Contract tests for the MCP stdio server (fixer F05a) — real `splatsdb mcp`
//! binary driven over JSON-RPC on stdin/stdout.
//!
//! No remote embeddings and no productive ports: every session points
//! `SPLATSDB_EMBED_URL` at a closed local port, or at a fake embed service
//! bound to 127.0.0.1:0 inside the test.
//!
//!   f10_*  A03-F10  SimCos is never a silent fallback; embeddings are labelled
//!   f11_*  A03-F11  doc_del / re-store leave no orphan vectors (tombstones)
//!   f13_*  A03-F13  ephemeral SQLite is visible; service dim mismatch rejected
//!   f14_*  A03-F14  text of live documents after restart
//!   f15_*  A03-F15  ping → {}, unknown tool → JSON-RPC -32602

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// Closed local port: connection refused, nothing productive listens here.
const NO_SERVICE_URL: &str = "http://127.0.0.1:9/embed";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RESP_TIMEOUT: Duration = Duration::from_secs(120);

struct Server {
    child: Child,
    stdin: ChildStdin,
    out: Receiver<String>,
    err: Receiver<String>,
    /// stderr lines already drained (startup happens before the first wait).
    err_log: Vec<String>,
}

fn spawn_server(doc_db: &std::path::Path, envs: &[(&str, &str)]) -> Server {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_splatsdb"));
    cmd.arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("SPLATSDB_DOC_PATH", doc_db)
        .env("SPLATSDB_EMBED_URL", NO_SERVICE_URL);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("failed to spawn `splatsdb mcp`");
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    let out = pump(stdout);
    let err = pump(stderr);
    let mut srv = Server {
        child,
        stdin,
        out,
        err,
        err_log: Vec::new(),
    };
    assert!(
        srv.wait_stderr_containing("MCP server ready", STARTUP_TIMEOUT),
        "server did not start"
    );
    srv
}

fn pump<R: Read + Send + 'static>(r: R) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in std::io::BufRead::lines(std::io::BufReader::new(r)) {
            if tx.send(line.unwrap_or_default()).is_err() {
                break;
            }
        }
    });
    rx
}

impl Server {
    /// Send one JSON-RPC request and return the full response object.
    fn rpc(&mut self, id: u64, method: &str, params: serde_json::Value) -> serde_json::Value {
        let req =
            serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        self.stdin
            .write_all(req.to_string().as_bytes())
            .expect("write request");
        self.stdin.write_all(b"\n").expect("write newline");
        self.stdin.flush().expect("flush stdin");
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
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("server stdout closed before answering id={id} ({method})")
                }
            };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                return v;
            }
        }
    }

    /// `tools/call` → (is_error, parsed tool payload or the raw text).
    fn tool(&mut self, id: u64, name: &str, args: serde_json::Value) -> (bool, serde_json::Value) {
        let v = self.rpc(
            id,
            "tools/call",
            serde_json::json!({ "name": name, "arguments": args }),
        );
        let is_error = v
            .pointer("/result/isError")
            .and_then(|b| b.as_bool())
            .unwrap_or(false);
        let text = v
            .pointer("/result/content/0/text")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string();
        (
            is_error,
            serde_json::from_str(&text).unwrap_or(serde_json::json!(text)),
        )
    }

    fn ok(&mut self, id: u64, name: &str, args: serde_json::Value) -> serde_json::Value {
        let (is_error, payload) = self.tool(id, name, args);
        assert!(!is_error, "{name} failed: {payload}");
        payload
    }

    fn wait_stderr_containing(&mut self, needle: &str, timeout: Duration) -> bool {
        // Lines drained earlier (e.g. startup warnings consumed while waiting
        // for "MCP server ready") are kept in `err_log` and checked first.
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

    fn status(&mut self, id: u64) -> serde_json::Value {
        self.ok(id, "splatsdb_status", serde_json::json!({}))
    }

    fn shutdown(mut self) {
        drop(self.stdin);
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splatsdb_mcp_contracts_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Embedding with two informative components (rest zero): `emb(1.0, 0.3*i)`
/// are pairwise distinct and `emb(1.0, x)` is nearest to the `i` with
/// `0.3*i ≈ x`.
fn emb(dim: usize, a: f64, b: f64) -> serde_json::Value {
    let mut v = vec![0.0; dim];
    v[0] = a;
    v[1] = b;
    serde_json::json!(v)
}

fn ids(results: &serde_json::Value) -> Vec<String> {
    results["results"]
        .as_array()
        .unwrap_or_else(|| panic!("no results array: {results}"))
        .iter()
        .map(|r| r["id"].as_str().unwrap_or("").to_string())
        .collect()
}

fn texts(results: &serde_json::Value) -> Vec<String> {
    results["results"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|r| r["text"].as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Fake embed service on 127.0.0.1:0. A text containing `WRONGDIM` is
/// answered with a 3-dim vector (misconfigured model); everything else with
/// `dim` dims.
fn spawn_fake_embed_server(dim: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
    let url = format!(
        "http://127.0.0.1:{}/embed",
        listener.local_addr().expect("local addr").port()
    );
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let (head_end, body_len) = loop {
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
                    break (
                        pos + 4,
                        head.lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0),
                    );
                }
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break (buf.len(), 0),
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            };
            while buf.len() < head_end + body_len {
                match stream.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
            let text = String::from_utf8_lossy(&buf).to_ascii_lowercase();
            let n = if text.contains("wrongdim") { 3 } else { dim };
            let vector: Vec<f64> = (0..n)
                .map(|i| ((i * 7 + text.len()) % 13) as f64 / 13.0 + 0.1)
                .collect();
            let body = serde_json::json!({ "embeddings": [vector] }).to_string();
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            );
            let _ = stream.flush();
        }
    });
    url
}

// ---------------------------------------------------------------------------
// A03-F15 — JSON-RPC protocol errors
// ---------------------------------------------------------------------------

#[test]
fn f15_ping_returns_empty_object_and_unknown_tool_is_32602() {
    let tmp = scratch("f15");
    let mut srv = spawn_server(&tmp.join("docs.db"), &[]);

    let pong = srv.rpc(1, "ping", serde_json::json!({}));
    assert_eq!(pong["result"], serde_json::json!({}), "ping: {pong}");
    assert!(
        pong.get("error").is_none(),
        "ping must not be an error: {pong}"
    );

    let unknown = srv.rpc(
        2,
        "tools/call",
        serde_json::json!({ "name": "splatsdb_does_not_exist", "arguments": {} }),
    );
    assert_eq!(
        unknown["error"]["code"],
        serde_json::json!(-32602),
        "{unknown}"
    );
    assert!(
        unknown.get("result").is_none(),
        "unknown tool must be a JSON-RPC error, not a tool result: {unknown}"
    );
    assert_eq!(unknown["id"], serde_json::json!(2), "{unknown}");

    // Unknown METHOD keeps its -32601.
    let method = srv.rpc(3, "no/such/method", serde_json::json!({}));
    assert_eq!(
        method["error"]["code"],
        serde_json::json!(-32601),
        "{method}"
    );

    // The stream stays usable afterwards.
    let status = srv.status(4);
    assert!(status["dimension"].as_u64().unwrap_or(0) >= 8, "{status}");
    srv.shutdown();
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// A03-F10 — no silent SimCos fallback (strict mode: not authorized)
// ---------------------------------------------------------------------------

#[test]
fn f10_strict_mode_fails_closed_and_labels_client_embeddings() {
    let tmp = scratch("f10_strict");
    let mut srv = spawn_server(&tmp.join("docs.db"), &[]);

    let status = srv.status(1);
    let dim = status["dimension"].as_u64().expect("dimension") as usize;
    assert_eq!(status["n_active"], serde_json::json!(0), "{status}");

    // No service + no opt-in: every WRITE that needs a text embedding is a
    // controlled error, never a success built on a text-hash vector.
    for (id, name, args) in [
        (
            10u64,
            "splatsdb_store",
            serde_json::json!({ "text": "hello", "id": "d_strict" }),
        ),
        (
            11,
            "splatsdb_doc_add",
            serde_json::json!({ "id": "d_add", "text": "hello" }),
        ),
        (
            12,
            "splatsdb_graph_add_doc",
            serde_json::json!({ "text": "hello" }),
        ),
        (
            13,
            "splatsdb_graph_add_entity",
            serde_json::json!({ "name": "acme", "entity_type": "org" }),
        ),
    ] {
        let (is_error, payload) = srv.tool(id, name, args);
        assert!(
            is_error,
            "{name} must fail closed without embed service: {payload}"
        );
        let msg = payload.to_string();
        assert!(msg.contains("embedding unavailable"), "{name}: {msg}");
        assert!(msg.contains("SPLATSDB_ALLOW_SIMCOS"), "{name}: {msg}");
    }

    // Nothing was indexed by the rejected calls. Status exposes the policy.
    let status = srv.status(20);
    assert_eq!(status["n_active"], serde_json::json!(0), "{status}");
    assert_eq!(status["doc_count"], serde_json::json!(0), "{status}");
    assert_eq!(
        status["embedding_fallback"],
        serde_json::json!("none"),
        "{status}"
    );

    // Client-provided embeddings keep working and are labelled as such.
    let stored = srv.ok(
        30,
        "splatsdb_store",
        serde_json::json!({
            "text": "client embedded",
            "id": "d_client",
            "embedding": emb(dim, 1.0, 0.0)
        }),
    );
    assert_eq!(
        stored["embedding"],
        serde_json::json!("provided"),
        "{stored}"
    );
    assert_eq!(stored["persisted"], serde_json::json!(true), "{stored}");
    assert_eq!(stored["ephemeral"], serde_json::json!(false), "{stored}");

    let found = srv.ok(
        31,
        "splatsdb_search",
        serde_json::json!({ "query": "client embedded", "embedding": emb(dim, 1.0, 0.0) }),
    );
    assert_eq!(found["embedding"], serde_json::json!("provided"), "{found}");
    assert_eq!(ids(&found), vec!["d_client".to_string()], "{found}");

    // Text queries against a NON-empty store fail closed as well.
    for (id, name) in [
        (40u64, "splatsdb_search"),
        (41, "splatsdb_spatial_search"),
        (42, "splatsdb_verbatim_search"),
        (43, "splatsdb_graph_search"),
        (44, "splatsdb_graph_search_entities"),
    ] {
        let (is_error, payload) = srv.tool(id, name, serde_json::json!({ "query": "hello" }));
        assert!(
            is_error,
            "{name} must fail closed without embed service: {payload}"
        );
        let msg = payload.to_string();
        assert!(msg.contains("embedding unavailable"), "{name}: {msg}");
    }
    let status = srv.status(50);
    assert_eq!(status["n_active"], serde_json::json!(1), "{status}");
    srv.shutdown();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn f10_simcos_only_with_explicit_opt_in_and_always_labelled() {
    let tmp = scratch("f10_simcos");
    let mut srv = spawn_server(&tmp.join("docs.db"), &[("SPLATSDB_ALLOW_SIMCOS", "1")]);

    let stored = srv.ok(
        10,
        "splatsdb_store",
        serde_json::json!({ "text": "opted in doc", "id": "d_opt" }),
    );
    assert_eq!(stored["embedding"], serde_json::json!("simcos"), "{stored}");
    let added = srv.ok(
        11,
        "splatsdb_doc_add",
        serde_json::json!({ "id": "d_opt_add", "text": "another opted in doc" }),
    );
    assert_eq!(added["embedding"], serde_json::json!("simcos"), "{added}");
    let found = srv.ok(
        12,
        "splatsdb_search",
        serde_json::json!({ "query": "opted in doc" }),
    );
    assert_eq!(found["embedding"], serde_json::json!("simcos"), "{found}");
    assert!(!ids(&found).is_empty(), "{found}");
    let status = srv.status(13);
    assert_eq!(
        status["embedding_fallback"],
        serde_json::json!("simcos"),
        "{status}"
    );

    // Same binary, opt-in NOT set → strict again (no process-global leak).
    srv.shutdown();
    let mut strict = spawn_server(&tmp.join("other.db"), &[]);
    let (is_error, payload) = strict.tool(
        1,
        "splatsdb_store",
        serde_json::json!({ "text": "x", "id": "d_no" }),
    );
    assert!(
        is_error && payload.to_string().contains("embedding unavailable"),
        "{payload}"
    );
    strict.shutdown();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn f13_real_embedding_is_labelled_and_dim_mismatch_is_rejected() {
    let tmp = scratch("f13_dim");
    let mut probe = spawn_server(&tmp.join("docs.db"), &[]);
    let dim = probe.status(1)["dimension"].as_u64().expect("dimension") as usize;
    probe.shutdown();

    let url = spawn_fake_embed_server(dim);
    // Strict (no SimCos opt-in): the label can only ever be "real".
    let mut srv = spawn_server(
        &tmp.join("docs.db"),
        &[("SPLATSDB_EMBED_URL", url.as_str())],
    );

    // Real service, right dimension → labelled "real".
    let stored = srv.ok(
        10,
        "splatsdb_store",
        serde_json::json!({ "text": "real embedding please", "id": "d_real" }),
    );
    assert_eq!(stored["embedding"], serde_json::json!("real"), "{stored}");
    let found = srv.ok(
        11,
        "splatsdb_search",
        serde_json::json!({ "query": "real embedding please" }),
    );
    assert_eq!(found["embedding"], serde_json::json!("real"), "{found}");
    assert_eq!(ids(&found), vec!["d_real".to_string()], "{found}");

    // Service answering with another dimension (misconfigured model): the
    // vector must be rejected - not zero-padded/truncated and indexed.
    let before = srv.status(20)["n_active"].clone();
    let (is_error, payload) = srv.tool(
        21,
        "splatsdb_store",
        serde_json::json!({ "text": "WRONGDIM probe", "id": "d_bad" }),
    );
    assert!(
        is_error,
        "wrong-dim service vector must be rejected: {payload}"
    );
    assert!(
        payload.to_string().contains("embedding unavailable"),
        "{}",
        payload
    );
    assert!(
        srv.wait_stderr_containing("rejecting", Duration::from_secs(10)),
        "server log must state the rejection"
    );
    let status = srv.status(22);
    assert_eq!(
        status["n_active"], before,
        "a padded vector was indexed: {status}"
    );
    let (_, payload) = srv.tool(23, "splatsdb_doc_get", serde_json::json!({ "id": "d_bad" }));
    assert!(payload.to_string().contains("not found"), "{payload}");
    srv.shutdown();
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// A03-F13 — ephemeral SQLite must be visible
// ---------------------------------------------------------------------------

#[test]
fn f13_ephemeral_doc_store_is_reported_as_not_persisted() {
    let tmp = scratch("f13_eph");
    let mut probe = spawn_server(&tmp.join("ok.db"), &[]);
    let dim = probe.status(1)["dimension"].as_u64().expect("dimension") as usize;
    let ok = probe.ok(2, "splatsdb_store", serde_json::json!({ "text": "persisted doc", "id": "d_ok", "embedding": emb(dim, 1.0, 0.0) }));
    assert_eq!(ok["persisted"], serde_json::json!(true), "{ok}");
    assert_eq!(ok["ephemeral"], serde_json::json!(false), "{ok}");
    let status = probe.status(3);
    assert_eq!(status["persisted"], serde_json::json!(true), "{status}");
    assert_eq!(status["ephemeral"], serde_json::json!(false), "{status}");
    probe.shutdown();

    // A directory as DB path: SQLite cannot open it → in-memory fallback.
    let bad = tmp.join("unopenable.db");
    std::fs::create_dir_all(&bad).expect("create blocker dir");
    let mut srv = spawn_server(&bad, &[]);
    let stored = srv.ok(10, "splatsdb_store", serde_json::json!({ "text": "will vanish", "id": "d_eph", "embedding": emb(dim, 1.0, 0.0) }));
    assert_eq!(
        stored["persisted"],
        serde_json::json!(false),
        "ephemeral store must not claim persisted: {stored}"
    );
    assert_eq!(stored["ephemeral"], serde_json::json!(true), "{stored}");
    let status = srv.status(11);
    assert_eq!(status["persisted"], serde_json::json!(false), "{status}");
    assert_eq!(status["ephemeral"], serde_json::json!(true), "{status}");
    assert!(
        srv.wait_stderr_containing("SQLite init failed", Duration::from_secs(5)),
        "ephemeral fallback must warn on stderr"
    );
    srv.shutdown();
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// A03-F11 — delete / re-store: no orphan vectors, top-k refilled
// ---------------------------------------------------------------------------

#[test]
fn f11_doc_del_and_re_store_leave_no_orphans_before_and_after_restart() {
    let tmp = scratch("f11");
    let doc_db = tmp.join("docs.db");
    let mut dim: usize = 0;

    let doc_emb = |dim: usize, i: f64| emb(dim, 1.0, 0.3 * i);

    {
        let mut srv = spawn_server(&doc_db, &[]);
        dim = srv.status(1)["dimension"].as_u64().expect("dimension") as usize;
        for i in 1..=5 {
            let id = format!("d{i}");
            let stored = srv.ok(
                10 + i,
                "splatsdb_store",
                serde_json::json!({ "text": format!("text of {id} v1"), "id": id, "embedding": doc_emb(dim, i as f64) }),
            );
            assert_eq!(stored["persisted"], serde_json::json!(true), "{stored}");
        }

        let top3 = serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.8), "top_k": 3 });
        assert_eq!(
            ids(&srv.ok(20, "splatsdb_search", top3.clone())),
            vec!["d3", "d2", "d4"],
            "baseline ranking"
        );

        // ---- delete d3: gone from every search path, top-k refilled ----
        srv.ok(30, "splatsdb_doc_del", serde_json::json!({ "id": "d3" }));
        for (id, name, extra) in [
            (31u64, "splatsdb_search", serde_json::json!({})),
            (32, "splatsdb_spatial_search", serde_json::json!({})),
            (33, "splatsdb_verbatim_search", serde_json::json!({})),
        ] {
            let mut args = top3.clone();
            for (k, v) in extra.as_object().unwrap() {
                args[k] = v.clone();
            }
            let res = srv.ok(id, name, args);
            let got = ids(&res);
            assert!(
                !got.iter().any(|i| i.starts_with("vec_")),
                "{name}: orphan vectors leaked: {res}"
            );
            assert!(
                !got.contains(&"d3".to_string()),
                "{name}: deleted doc returned: {res}"
            );
            assert_eq!(
                got.len(),
                3,
                "{name}: top-k must be refilled past the tombstone: {res}"
            );
            assert!(
                texts(&res).iter().all(|t| !t.is_empty()),
                "{name}: empty text: {res}"
            );
        }
        let status = srv.status(34);
        assert_eq!(status["live_vectors"], serde_json::json!(4), "{status}");
        assert_eq!(
            status["tombstoned_vectors"],
            serde_json::json!(1),
            "{status}"
        );

        // Searching with the deleted doc's own vector must not resurrect it.
        let near = srv.ok(
            35,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": doc_emb(dim, 3.0), "top_k": 10 }),
        );
        assert!(!ids(&near).contains(&"d3".to_string()), "{near}");
        assert_eq!(ids(&near).len(), 4, "{near}");

        // ---- re-store d1 with a new vector and text: exactly one hit ----
        srv.ok(
            40,
            "splatsdb_store",
            serde_json::json!({ "text": "text of d1 v2", "id": "d1", "embedding": emb(dim, 0.0, 1.0) }),
        );
        let all = srv.ok(
            41,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 0.0, 1.0), "top_k": 10 }),
        );
        let got = ids(&all);
        assert_eq!(
            got.iter().filter(|i| *i == "d1").count(),
            1,
            "re-store left a duplicate mapping: {all}"
        );
        assert_eq!(got.len(), 4, "{all}");
        assert!(!got.iter().any(|i| i.starts_with("vec_")), "{all}");
        let d1 = all["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == "d1")
            .expect("d1");
        assert_eq!(
            d1["text"],
            serde_json::json!("text of d1 v2"),
            "stale text attributed: {all}"
        );
        // The OLD vector of d1 must no longer surface d1.
        let old = srv.ok(
            42,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": doc_emb(dim, 1.0), "top_k": 10 }),
        );
        assert_eq!(ids(&old).iter().filter(|i| *i == "d1").count(), 1, "{old}");
        assert_eq!(ids(&old).len(), 4, "{old}");
        let status = srv.status(43);
        assert_eq!(status["live_vectors"], serde_json::json!(4), "{status}");
        assert_eq!(
            status["tombstoned_vectors"],
            serde_json::json!(2),
            "{status}"
        );
        assert_eq!(status["doc_count"], serde_json::json!(4), "{status}");

        // ---- spatial: a deleted doc leaves the spatial index ----
        srv.ok(50, "splatsdb_store", serde_json::json!({ "text": "spatial doc", "id": "d_sp", "wing": "w1", "room": "r1", "hall": "fact", "embedding": emb(dim, 0.0, 1.0, ) }));
        let filtered = srv.ok(51, "splatsdb_spatial_search", serde_json::json!({ "query": "q", "wing": "w1", "top_k": 5, "embedding": emb(dim, 0.0, 1.0) }));
        assert_eq!(ids(&filtered), vec!["d_sp".to_string()], "{filtered}");
        srv.ok(52, "splatsdb_doc_del", serde_json::json!({ "id": "d_sp" }));
        let filtered = srv.ok(53, "splatsdb_spatial_search", serde_json::json!({ "query": "q", "wing": "w1", "top_k": 5, "embedding": emb(dim, 0.0, 1.0) }));
        assert!(
            ids(&filtered).is_empty(),
            "spatial index still returns a deleted doc: {filtered}"
        );
        let unfiltered = srv.ok(
            54,
            "splatsdb_spatial_search",
            serde_json::json!({ "query": "q", "top_k": 10, "embedding": emb(dim, 0.0, 1.0) }),
        );
        assert!(
            !ids(&unfiltered).contains(&"d_sp".to_string()),
            "{unfiltered}"
        );
        assert_eq!(ids(&unfiltered).len(), 4, "{unfiltered}");
        srv.shutdown();
    }

    // ---- restart (warm start from SQLite): same contract ----
    // Vectors are not persisted yet (F05b): warm start re-embeds through the
    // (fake, real-labelled) embed service.
    {
        let url = spawn_fake_embed_server(dim);
        let mut srv = spawn_server(&doc_db, &[("SPLATSDB_EMBED_URL", url.as_str())]);
        let all = srv.ok(
            2,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 0.0, 1.0), "top_k": 10 }),
        );
        let got = ids(&all);
        assert!(
            !got.iter().any(|i| i.starts_with("vec_")),
            "orphans after restart: {all}"
        );
        assert!(
            !got.contains(&"d3".to_string()) && !got.contains(&"d_sp".to_string()),
            "deleted docs resurrected: {all}"
        );
        assert_eq!(got.iter().filter(|i| *i == "d1").count(), 1, "{all}");
        assert_eq!(got.len(), 4, "{all}");
        // A03-F14: every live doc returns its text after warm start.
        assert!(
            texts(&all).iter().all(|t| !t.is_empty()),
            "empty text for live doc: {all}"
        );
        assert!(
            all["results"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["text"] == "text of d1 v2"),
            "{all}"
        );
        // A text query goes through the real (fake) service and says so.
        let by_text = srv.ok(
            4,
            "splatsdb_search",
            serde_json::json!({ "query": "anything" }),
        );
        assert_eq!(by_text["embedding"], serde_json::json!("real"), "{by_text}");
        assert_eq!(ids(&by_text).len(), 4, "{by_text}");
        assert!(texts(&by_text).iter().all(|t| !t.is_empty()), "{by_text}");
        let status = srv.status(3);
        assert_eq!(status["live_vectors"], serde_json::json!(4), "{status}");
        assert_eq!(
            status["tombstoned_vectors"],
            serde_json::json!(0),
            "{status}"
        );
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A03-F10 (warm start): without the embed service and without opt-in the
/// server must NOT re-embed persisted docs with a text hash. It comes up with
/// an empty vector store and says so: a visible failure instead of silent
/// garbage ranking.
#[test]
fn f10_warm_start_without_service_indexes_nothing_and_warns() {
    let tmp = scratch("f10_warm");
    let doc_db = tmp.join("docs.db");
    let mut dim: usize = 0;
    {
        let mut srv = spawn_server(&doc_db, &[]);
        dim = srv.status(1)["dimension"].as_u64().expect("dimension") as usize;
        srv.ok(
            2,
            "splatsdb_store",
            serde_json::json!({ "text": "persisted doc", "id": "d_p", "embedding": emb(dim, 1.0, 0.0) }),
        );
        assert_eq!(srv.status(3)["n_active"], serde_json::json!(1));
        srv.shutdown();
    }
    {
        let mut srv = spawn_server(&doc_db, &[]);
        assert!(
            srv.wait_stderr_containing("documents were NOT indexed", Duration::from_secs(20)),
            "warm start must report the skipped docs on stderr"
        );
        let status = srv.status(1);
        assert_eq!(
            status["doc_count"],
            serde_json::json!(1),
            "the doc is still in SQLite: {status}"
        );
        assert_eq!(
            status["n_active"],
            serde_json::json!(0),
            "a doc was indexed with an unauthorized hash vector: {status}"
        );
        let res = srv.ok(
            2,
            "splatsdb_search",
            serde_json::json!({ "query": "persisted doc", "embedding": emb(dim, 1.0, 0.0) }),
        );
        assert!(ids(&res).is_empty(), "{res}");
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
