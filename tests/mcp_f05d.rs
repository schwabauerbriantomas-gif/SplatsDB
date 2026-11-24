//! F05d — MCP contracts closed with real `splatsdb mcp` processes over
//! synthetic data under TMPDIR (never the default `~/.hermes` doc store):
//!
//!   * A04-F11 — `splatsdb_graph_search` ranks by RAW cosine by default; the
//!     degree prior is reported as `graph_boost` and only reorders under an
//!     explicit `boost: true` (`score_mode` names what `score` is);
//!     `splatsdb_graph_add_relation` is idempotent on
//!     (source, target, relation_type);
//!   * A04-F12 — a graph of another dimension than the query is a controlled
//!     error (never "everything is similar at 0.0"), equal scores are ordered
//!     by ascending node id, and a legacy state file mixing dimensions fails
//!     closed without being overwritten;
//!   * A03-F14 — a live document evicted from the 10K text cache still
//!     answers with its text (10_001 documents, top_k = 100);
//!   * A03-F15 — SIGINT / SIGTERM end the server even with stdin open and
//!     idle, in bounded time, with the pipes closed and every acknowledged
//!     mutation still there after a restart.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// Closed local port: connection refused — nothing listens here.
const NO_SERVICE_URL: &str = "http://127.0.0.1:9/embed";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RESP_TIMEOUT: Duration = Duration::from_secs(120);
/// The contract of A03-F15: a signal ends the server within this bound.
const SIGNAL_EXIT_BOUND: Duration = Duration::from_secs(2);
/// File the graph is persisted to inside the data dir.
const STATE_FILE: &str = "graph_state.json";

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

/// Start `splatsdb mcp` on `doc_db`. The embed service is DOWN by default and
/// the SimCos fallback is explicitly authorized, so text-embedding works
/// without a service; tests that need chosen vectors point
/// `SPLATSDB_EMBED_URL` at `spawn_vector_server`.
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
    /// Write one JSON-RPC request without waiting for its response. A closed
    /// pipe is reported, not hidden: a request that could not even be written
    /// says nothing about the server.
    fn send_request(&mut self, id: u64, method: &str, params: Value) -> std::io::Result<()> {
        let req = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let stdin = self.stdin.as_mut().expect("server already closed");
        stdin.write_all(req.to_string().as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()
    }

    /// Write one JSON-RPC request without waiting for its response.
    fn rpc_send(&mut self, id: u64, method: &str, params: Value) {
        self.send_request(id, method, params)
            .expect("write request");
    }

    /// Send one JSON-RPC request and return the full response object.
    fn rpc(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.rpc_send(id, method, params);
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
            let Ok(v) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if v.get("id").and_then(|i| i.as_u64()) == Some(id) {
                return v;
            }
        }
    }

    /// `tools/call` → (is_error, parsed tool payload or the raw error text).
    fn tool(&mut self, id: u64, name: &str, args: Value) -> (bool, Value) {
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

    fn ok(&mut self, id: u64, name: &str, args: Value) -> Value {
        let (is_error, payload) = self.tool(id, name, args);
        assert!(!is_error, "{name} failed: {payload}");
        payload
    }

    /// `tools/call` that MUST fail; returns the error text the client sees.
    fn error(&mut self, id: u64, name: &str, args: Value) -> String {
        let (is_error, payload) = self.tool(id, name, args);
        assert!(is_error, "{name} unexpectedly succeeded: {payload}");
        match payload {
            Value::String(text) => text,
            other => other.to_string(),
        }
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

    fn wait_ready(&mut self) -> bool {
        let ready = self.wait_stderr_containing("MCP server ready", STARTUP_TIMEOUT);
        if !ready {
            eprintln!(
                "server never became ready; stderr:\n{}",
                self.err_log.join("\n")
            );
        }
        ready
    }

    fn status(&mut self, id: u64) -> Value {
        self.ok(id, "splatsdb_status", json!({}))
    }

    fn stats(&mut self, id: u64) -> Value {
        self.ok(id, "splatsdb_graph_stats", json!({}))
    }

    fn add_doc(&mut self, id: u64, text: &str) -> Value {
        self.ok(id, "splatsdb_graph_add_doc", json!({ "text": text }))
    }

    fn add_entity(&mut self, id: u64, name: &str) -> Value {
        self.ok(
            id,
            "splatsdb_graph_add_entity",
            json!({ "name": name, "entity_type": "concept" }),
        )
    }

    fn add_relation(&mut self, id: u64, source: u64, target: u64, relation: &str) -> Value {
        self.ok(
            id,
            "splatsdb_graph_add_relation",
            json!({ "source_id": source, "target_id": target, "relation_type": relation, "weight": 0.9 }),
        )
    }

    /// Send `sig` to the server process (by pid — never by name).
    fn signal(&self, sig: i32) {
        let pid: i32 = self.child.id().try_into().expect("pid fits in i32");
        assert_eq!(
            unsafe { libc::kill(pid, sig) },
            0,
            "kill({pid}, {sig}) failed"
        );
    }

    /// Wait for the exit, for at most `bound`. Returns the status and the
    /// time since `t0`.
    fn exit_within(&mut self, t0: Instant, bound: Duration, sig: i32) -> (ExitStatus, Duration) {
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return (status, t0.elapsed()),
                Ok(None) if t0.elapsed() < bound => thread::sleep(Duration::from_millis(10)),
                Ok(None) => panic!(
                    "server still running {bound:?} after signal {sig}; stderr:\n{}",
                    self.err_log.join("\n")
                ),
                Err(e) => panic!("try_wait: {e}"),
            }
        }
    }

    /// Send `sig` and require a graceful exit within `bound`: the process
    /// must leave on its own (not be killed by the default disposition), and
    /// both pipes must close. Returns how long the exit took.
    fn expect_signal_exit(mut self, sig: i32, bound: Duration) -> Duration {
        let t0 = Instant::now();
        self.signal(sig);
        let (status, took) = self.exit_within(t0, bound, sig);
        assert!(
            status.success(),
            "signal {sig}: expected a graceful exit, got {status:?}"
        );
        assert!(
            self.pipes_closed(bound),
            "stdout/stderr still open {bound:?} after the exit"
        );
        took
    }

    /// Drain stdout until EOF (`bound` at most) and return the lines.
    fn stdout_until_closed(&self, bound: Duration) -> Vec<String> {
        let deadline = Instant::now() + bound;
        let mut lines = Vec::new();
        while let Ok(line) = self
            .out
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            lines.push(line);
        }
        lines
    }

    /// True when both pumps reached EOF: the process left no pipe open.
    fn pipes_closed(&self, bound: Duration) -> bool {
        let deadline = Instant::now() + bound;
        let (mut out_open, mut err_open) = (true, true);
        while out_open || err_open {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            if out_open {
                if let Err(RecvTimeoutError::Disconnected) =
                    self.out.recv_timeout(left.min(Duration::from_millis(20)))
                {
                    out_open = false;
                }
            }
            if err_open {
                if let Err(RecvTimeoutError::Disconnected) =
                    self.err.recv_timeout(left.min(Duration::from_millis(20)))
                {
                    err_open = false;
                }
            }
        }
        !(out_open || err_open)
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
        "splatsdb_mcp_f05d_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn id_of(v: &Value, key: &str) -> u64 {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("no numeric {key} in {v}"))
}

fn state_path(dir: &Path) -> PathBuf {
    dir.join(STATE_FILE)
}

fn cosine_of(result: &Value) -> f64 {
    result["score"]
        .as_f64()
        .unwrap_or_else(|| panic!("no score in {result}"))
}

/// Fake embed service with CHOSEN vectors: a text carrying `V:<x>:<y>` gets
/// `[x, y, 0, …]`; any other text gets `[0, 0, 1, 0, …]`. Always `dim`
/// components — the MCP server refuses another dimension.
fn spawn_vector_server(dim: usize) -> String {
    assert!(dim >= 3, "the vector service needs at least 3 components");
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
            let body =
                String::from_utf8_lossy(&buf[head_end.min(buf.len())..]).to_ascii_lowercase();
            let mut vector = vec![0.0f64; dim];
            vector[2] = 1.0;
            if let Some((x, y)) = parse_xy(&body) {
                vector[0] = x;
                vector[1] = y;
                vector[2] = 0.0;
            }
            let body = json!({ "embeddings": [vector] }).to_string();
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

/// `V:<x>:<y>` of the lowercased request body → `(x, y)`.
fn parse_xy(body: &str) -> Option<(f64, f64)> {
    let rest = body.split("v:").nth(1)?;
    let mut fields = rest.split([':', '"', ' ', ']']);
    let x = fields.next()?.parse().ok()?;
    let y = fields.next()?.parse().ok()?;
    Some((x, y))
}

/// Accepts TCP connections and never answers: a request that needs an
/// embedding stays inside the HTTP call until the client's own timeout
/// (5 s) — or the shutdown signal interrupts it.
fn spawn_black_hole() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
    let url = format!(
        "http://127.0.0.1:{}/embed",
        listener.local_addr().expect("local addr").port()
    );
    thread::spawn(move || {
        // Every accepted stream stays open (and unanswered) for the life of the test.
        let mut held: Vec<TcpStream> = Vec::new();
        held.extend(listener.incoming().flatten());
    });
    url
}

fn embedding_dim() -> usize {
    splatsdb::config::SplatsDBConfig::mcp(None).latent_dim
}

const TEXT_A: &str = "NODE_A V:0.9:0.435889894354067";
const TEXT_B: &str = "NODE_B V:0.75:0.6614378277661477";

/// A (cosine 0.90, no edges), B (cosine 0.75, five edges) and the vector
/// service that maps the texts to those vectors.
fn graph_with_a_popular_b(db: &Path) -> (Server, u64, u64) {
    let url = spawn_vector_server(embedding_dim());
    let mut srv = spawn(db, &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let a = id_of(&srv.add_doc(1, TEXT_A), "node_id");
    let b = id_of(&srv.add_doc(2, TEXT_B), "node_id");
    for i in 0..5u64 {
        let e = id_of(&srv.add_entity(10 + i, &format!("ENT_{i}")), "node_id");
        srv.add_relation(20 + i, b, e, "MENTIONS");
    }
    (srv, a, b)
}

// ---------------------------------------------------------------------------
// A04-F11 — raw cosine by default, degree prior as an opt-in annotation
// ---------------------------------------------------------------------------

#[test]
fn s01_graph_search_ranks_by_raw_cosine_by_default_and_reports_the_boost() {
    let dir = scratch("graph-rank");
    let (mut srv, a, b) = graph_with_a_popular_b(&dir.join("docs.db"));

    let r = srv.ok(
        30,
        "splatsdb_graph_search",
        json!({ "query": "QUERY V:1:0", "k": 10 }),
    );
    let results = r["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2, "{r}");
    assert_eq!(
        results[0]["node_id"], json!(a),
        "A (cos 0.90, 0 edges) must rank first — B (cos 0.75, 5 edges) must not win on popularity: {r}"
    );
    assert_eq!(results[1]["node_id"], json!(b));
    assert!((cosine_of(&results[0]) - 0.90).abs() < 1e-5, "{r}");
    assert!((cosine_of(&results[1]) - 0.75).abs() < 1e-5, "{r}");
    assert_eq!(results[0]["graph_boost"], json!(0.0), "{r}");
    assert!(
        (results[1]["graph_boost"].as_f64().unwrap() - 0.2).abs() < 1e-9,
        "{r}"
    );
    assert_eq!(results[0]["score_mode"], json!("cosine"), "{r}");
    assert_eq!(results[1]["score_mode"], json!("cosine"), "{r}");
    assert_eq!(r["score_mode"], json!("cosine"), "{r}");
    assert_eq!(results[0]["content"], json!(TEXT_A));
    srv.shutdown();
}

#[test]
fn s02_graph_search_reorders_by_the_boost_only_with_explicit_opt_in() {
    let dir = scratch("graph-boost");
    let (mut srv, a, b) = graph_with_a_popular_b(&dir.join("docs.db"));

    // The schema advertises the switch.
    let tools = srv.rpc(25, "tools/list", json!({}));
    let schema = tools["result"]["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .find(|t| t["name"] == json!("splatsdb_graph_search"))
        .expect("splatsdb_graph_search is listed");
    assert!(
        schema["inputSchema"]["properties"]["boost"].is_object(),
        "{schema}"
    );

    let r = srv.ok(
        30,
        "splatsdb_graph_search",
        json!({ "query": "QUERY V:1:0", "k": 10, "boost": true }),
    );
    let results = r["results"].as_array().expect("results array");
    assert_eq!(r["score_mode"], json!("cosine+graph_boost"), "{r}");
    assert_eq!(results[0]["node_id"], json!(b), "{r}");
    assert_eq!(results[1]["node_id"], json!(a), "{r}");
    // Documented formula: score = cosine + min(0.05 × outgoing edges, 0.2).
    assert!((cosine_of(&results[0]) - 0.95).abs() < 1e-5, "{r}");
    assert!((cosine_of(&results[1]) - 0.90).abs() < 1e-5, "{r}");
    assert!(
        (results[0]["graph_boost"].as_f64().unwrap() - 0.2).abs() < 1e-9,
        "{r}"
    );
    assert_eq!(results[0]["score_mode"], json!("cosine+graph_boost"), "{r}");

    // The default is unchanged after an opt-in call.
    let r = srv.ok(
        31,
        "splatsdb_graph_search",
        json!({ "query": "QUERY V:1:0", "k": 10 }),
    );
    assert_eq!(r["results"][0]["node_id"], json!(a), "{r}");
    assert_eq!(r["score_mode"], json!("cosine"), "{r}");
    srv.shutdown();
}

#[test]
fn s03_add_relation_is_idempotent_and_the_degree_stays_at_one_edge() {
    let dir = scratch("relation-dedup");
    let db = dir.join("docs.db");
    let (mut srv, a, b) = graph_with_a_popular_b(&db);

    let first = srv.add_relation(40, a, b, "CITES");
    assert_eq!(first["created"], json!(true), "{first}");
    for i in 41..45u64 {
        let again = srv.add_relation(i, a, b, "CITES");
        assert_eq!(
            again["created"],
            json!(false),
            "a repeated relation was inserted again: {again}"
        );
    }
    let stats = srv.stats(50);
    assert_eq!(
        stats["total_edges"],
        json!(6),
        "5 MENTIONS + 1 CITES: {stats}"
    );
    assert_eq!(
        srv.ok(
            51,
            "splatsdb_graph_search",
            json!({ "query": "QUERY V:1:0", "k": 10 })
        )["results"][1]["graph_boost"],
        json!(0.2),
        "A's degree did not change by repeating the relation"
    );

    // One edge on disk as well.
    srv.shutdown();
    let mut again = spawn(&db, &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    assert_eq!(again.stats(1)["total_edges"], json!(6));
    again.shutdown();
}

// ---------------------------------------------------------------------------
// A04-F12 — dimension is an invariant, ties are deterministic
// ---------------------------------------------------------------------------

#[test]
fn s04_equal_scores_are_ordered_by_ascending_node_id() {
    let dir = scratch("ties");
    let url = spawn_vector_server(embedding_dim());
    let mut srv = spawn(&dir.join("docs.db"), &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let mut ids = Vec::new();
    for i in 0..16u64 {
        ids.push(id_of(
            &srv.add_entity(1 + i, &format!("same {i}")),
            "node_id",
        ));
    }
    ids.sort_unstable();
    let r = srv.ok(
        30,
        "splatsdb_graph_search_entities",
        json!({ "query": "QUERY V:0:1", "k": 16 }),
    );
    let got: Vec<u64> = r["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|e| id_of(e, "id"))
        .collect();
    assert_eq!(got, ids, "equal scores must come out in ascending node id");
    srv.shutdown();
}

/// A valid v1 state file whose nodes have `dims` components each.
fn legacy_state(dims: &[usize]) -> Vec<u8> {
    let nodes: Vec<Value> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let mut embedding = vec![0.0; d];
            embedding[i % d] = 1.0;
            json!({
                "id": i + 1,
                "node_type": if i == 0 { "Document" } else { "Entity" },
                "content": format!("legacy {d}-d node {i}"),
                "embedding": embedding,
                "alpha": 1.0,
                "kappa": 10.0,
            })
        })
        .collect();
    json!({ "version": 1, "next_id": dims.len() + 1, "nodes": nodes, "edges": [] })
        .to_string()
        .into_bytes()
}

#[test]
fn s05_query_of_another_dimension_than_the_graph_is_a_controlled_error() {
    let dir = scratch("dim-query");
    let legacy = legacy_state(&[3, 3]);
    std::fs::write(state_path(&dir), &legacy).expect("write legacy state");

    let url = spawn_vector_server(embedding_dim());
    let mut srv = spawn(&dir.join("docs.db"), &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    assert_eq!(
        srv.stats(1)["total_nodes"],
        json!(2),
        "the 3-d legacy graph must load"
    );

    // The query vector has the STORE's dimension, the graph has 3: an error,
    // never "the most similar" nodes at a silent similarity of 0.0.
    for (i, tool) in ["splatsdb_graph_search", "splatsdb_graph_search_entities"]
        .into_iter()
        .enumerate()
    {
        let err = srv.error(
            10 + i as u64,
            tool,
            json!({ "query": "QUERY V:1:0", "k": 5 }),
        );
        assert!(err.to_lowercase().contains("dimension"), "{tool}: {err}");
    }
    // A node of the store's dimension does not join a 3-d graph either.
    let err = srv.error(20, "splatsdb_graph_add_doc", json!({ "text": "new" }));
    assert!(err.to_lowercase().contains("dimension"), "{err}");
    assert_eq!(
        std::fs::read(state_path(&dir)).unwrap(),
        legacy,
        "a refused mutation must not touch the state file"
    );
    assert_eq!(srv.stats(21)["total_nodes"], json!(2));
    srv.shutdown();
}

#[test]
fn s06_legacy_state_mixing_dimensions_fails_closed_and_is_never_overwritten() {
    let dir = scratch("dim-mixed");
    let mixed = legacy_state(&[3, 5]);
    std::fs::write(state_path(&dir), &mixed).expect("write mixed state");

    let mut srv = spawn(&dir.join("docs.db"), &[]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let status = srv.status(1);
    assert_eq!(
        status["graph_persistence"]["state"],
        json!("corrupt"),
        "{status}"
    );
    let err = srv.error(2, "splatsdb_graph_add_doc", json!({ "text": "not stored" }));
    assert!(err.contains("unreadable"), "{err}");
    assert_eq!(
        std::fs::read(state_path(&dir)).unwrap(),
        mixed,
        "a state file that failed to load must never be overwritten"
    );
    // Still corrupt after the refused mutation, and nothing was persisted.
    assert_eq!(
        srv.status(3)["graph_persistence"]["state"],
        json!("corrupt")
    );
    assert_eq!(srv.stats(4)["total_nodes"], json!(0));
    srv.shutdown();
}

// ---------------------------------------------------------------------------
// A03-F14 — live docs never come back as `text: ""`
// ---------------------------------------------------------------------------

fn doc_record(i: usize, dim: usize) -> (splatsdb::storage::DocumentRecord, Vec<f32>) {
    // Distinct unit vectors on a circle: doc i sits at angle 2π·i / 10001.
    let theta = 2.0 * std::f64::consts::PI * (i as f64) / 10_001.0;
    let mut vector = vec![0.0f32; dim];
    vector[0] = theta.cos() as f32;
    vector[1] = theta.sin() as f32;
    let record = splatsdb::storage::DocumentRecord {
        id: format!("doc_{i:05}"),
        shard_idx: 0,
        vector_idx: i as i64,
        metadata: Some(json!({ "i": i })),
        document: Some(format!("text of doc {i}")),
        deleted: false,
        created_at: 0.0,
        updated_at: 0.0,
    };
    (record, vector)
}

/// The query text whose `spawn_vector_server` vector is the one of doc `i`.
fn query_for(i: usize) -> String {
    let theta = 2.0 * std::f64::consts::PI * (i as f64) / 10_001.0;
    format!("V:{}:{}", theta.cos(), theta.sin())
}

#[test]
fn t01_search_keeps_the_text_of_a_doc_evicted_from_the_10k_cache_with_top_k_100() {
    let dir = scratch("text-10k");
    let db = dir.join("docs.db");
    let dim = embedding_dim();
    {
        // 10_001 live documents WITH their stored vector, written by the
        // library's own writer: the warm start needs no embedding service.
        let store = splatsdb::storage::sqlite_store::SqliteMetadataStore::new(db.clone())
            .expect("open sqlite store");
        for i in 0..10_001 {
            let (record, vector) = doc_record(i, dim);
            store
                .upsert_with_embedding(
                    &record,
                    &vector,
                    splatsdb::storage::sqlite_store::EmbeddingProvenance::Provided,
                )
                .expect("store doc");
        }
    }

    let url = spawn_vector_server(dim);
    let mut srv = spawn(&db, &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));

    let status = srv.status(1);
    assert_eq!(
        status["vector_persistence"]["warm_start"]["documents"],
        json!(10_001),
        "{status}"
    );

    // doc_00000 is the FIRST the warm start cached, so the 10_001st insert
    // evicted it from the 10K FIFO cache: its text can only come from SQLite.
    let r = srv.ok(
        2,
        "splatsdb_search",
        json!({ "query": query_for(0), "top_k": 100 }),
    );
    let results = r["results"].as_array().expect("results");
    assert_eq!(results.len(), 100, "{} results", results.len());
    for result in results {
        assert!(
            !result["text"].as_str().unwrap_or_default().is_empty(),
            "a live document came back with an empty text: {result}"
        );
    }
    assert_eq!(results[0]["id"], json!("doc_00000"), "{r}");
    assert_eq!(results[0]["text"], json!("text of doc 0"), "{r}");
    assert_eq!(results[0]["metadata"]["i"], json!(0), "{r}");

    // A tombstone is not resurrected by the batched row fetch.
    srv.ok(3, "splatsdb_doc_del", json!({ "id": "doc_00001" }));
    let r = srv.ok(
        4,
        "splatsdb_search",
        json!({ "query": query_for(1), "top_k": 100 }),
    );
    let results = r["results"].as_array().expect("results");
    assert_eq!(results.len(), 100, "{} results", results.len());
    assert!(
        results.iter().all(|x| x["id"] != json!("doc_00001")),
        "a soft-deleted document was returned: {r}"
    );
    for result in results {
        assert!(!result["text"].as_str().unwrap_or_default().is_empty());
    }
    srv.shutdown();
}

// ---------------------------------------------------------------------------
// A03-F15 — SIGINT / SIGTERM end the server for real
// ---------------------------------------------------------------------------

fn mutations_confirmed_then_signal(dir: &Path, sig: i32) -> Duration {
    let db = dir.join("docs.db");
    let mut srv = spawn(&db, &[]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    srv.ok(
        1,
        "splatsdb_doc_add",
        json!({ "id": "mem_1", "text": "kept across the signal" }),
    );
    srv.add_doc(2, "graph node kept across the signal");

    // stdin stays OPEN and idle — the server is blocked waiting for a line.
    let took = srv.expect_signal_exit(sig, SIGNAL_EXIT_BOUND);

    // Everything that was acknowledged is still there after a restart.
    let mut again = spawn(&db, &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    let doc = again.ok(1, "splatsdb_doc_get", json!({ "id": "mem_1" }));
    assert_eq!(doc["text"], json!("kept across the signal"), "{doc}");
    let stats = again.stats(2);
    assert_eq!(stats["total_nodes"], json!(1), "{stats}");
    let found = again.ok(
        3,
        "splatsdb_search",
        json!({ "query": "kept across the signal", "top_k": 5 }),
    );
    assert_eq!(found["results"][0]["id"], json!("mem_1"), "{found}");
    again.shutdown();
    took
}

#[test]
fn x01_sigint_exits_within_2s_with_stdin_open_and_confirmed_mutations_survive() {
    let took = mutations_confirmed_then_signal(&scratch("sigint"), libc::SIGINT);
    eprintln!("SIGINT (stdin open, idle): graceful exit after {took:?}");
}

#[test]
fn x02_sigterm_exits_within_2s_with_stdin_open_and_confirmed_mutations_survive() {
    let took = mutations_confirmed_then_signal(&scratch("sigterm"), libc::SIGTERM);
    eprintln!("SIGTERM (stdin open, idle): graceful exit after {took:?}");
}

#[test]
fn x03_sigint_exits_while_a_partial_line_is_being_read() {
    let dir = scratch("sigint-partial");
    let mut srv = spawn(&dir.join("docs.db"), &[]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    // Half a request and no newline: the reader is blocked mid-line, and the
    // client will never finish it.
    srv.stdin
        .as_mut()
        .expect("stdin")
        .write_all(br#"{"jsonrpc":"2.0","id":7,"method":"ping""#)
        .expect("write partial line");
    srv.stdin.as_mut().expect("stdin").flush().expect("flush");
    thread::sleep(Duration::from_millis(400));
    let took = srv.expect_signal_exit(libc::SIGINT, SIGNAL_EXIT_BOUND);
    eprintln!("SIGINT (partial line): graceful exit after {took:?}");
}

#[test]
fn x04_in_flight_request_bounds_its_completion_and_no_new_request_runs_after_sigterm() {
    let dir = scratch("sigterm-inflight");
    let hole = spawn_black_hole();
    let mut srv = spawn(&dir.join("docs.db"), &[("SPLATSDB_EMBED_URL", &hole)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    // A request whose embedding HTTP call does not answer.
    srv.rpc_send(
        7,
        "tools/call",
        json!({ "name": "splatsdb_graph_search", "arguments": { "query": "in flight" } }),
    );
    thread::sleep(Duration::from_millis(400));

    let t0 = Instant::now();
    srv.signal(libc::SIGTERM);
    // A request that arrives AFTER the signal must not run. (A broken pipe
    // here only means the server had already left — never answered either.)
    thread::sleep(Duration::from_millis(100));
    let _ = srv.send_request(8, "ping", json!({}));
    // Bounded by the embedding client's own 5 s timeout at worst — never a hang.
    let (status, took) = srv.exit_within(t0, Duration::from_secs(10), libc::SIGTERM);
    assert!(status.success(), "expected a graceful exit, got {status:?}");
    assert!(srv.pipes_closed(Duration::from_secs(2)), "pipes left open");
    let lines = srv.stdout_until_closed(Duration::from_secs(2));
    for line in &lines {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            assert_ne!(
                v.get("id").and_then(|i| i.as_u64()),
                Some(8),
                "a request sent after SIGTERM was answered: {line}"
            );
        }
    }
    eprintln!("SIGTERM (request in flight): graceful exit after {took:?}; {lines:?}");
}

// ---------------------------------------------------------------------------
// F05e — deterministic shutdown, a real warm start, a no-op graph and invalid
// legacy nodes, all against real `splatsdb mcp` processes (same harness).
//
// A debug server needs ~35 s of CPU to come up (measured, F05e), so the
// in-suite signal cells run `SIGNAL_REPS` times; the ≥20-repetition table is
// executed by the standalone probe `tmp/F05e/probe_signals.py` on the release
// binaries and reported in `reports/F05e.json`. Both pin the same contract:
// rc, exit duration, nothing acknowledged for an incomplete request, and the
// restart + on-disk state of what WAS acknowledged.
// ---------------------------------------------------------------------------

const SIGNAL_REPS: usize = 3;
/// The server abandons a request in flight 1.9 s after the signal; this is
/// that bound plus scheduling slack for a loaded machine.
const IN_FLIGHT_EXIT_BOUND: Duration = Duration::from_millis(3000);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StdinState {
    Idle,
    PartialLine,
}

/// Fake embed service that COUNTS its requests (same vectors as
/// `spawn_vector_server`): `reembedded` may only advance by that count.
fn spawn_counting_vector_server(
    dim: usize,
) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
    let url = format!(
        "http://127.0.0.1:{}/embed",
        listener.local_addr().expect("local addr").port()
    );
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = calls.clone();
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
            let body =
                String::from_utf8_lossy(&buf[head_end.min(buf.len())..]).to_ascii_lowercase();
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut vector = vec![0.0f64; dim];
            vector[2] = 1.0;
            if let Some((x, y)) = parse_xy(&body) {
                vector[0] = x;
                vector[1] = y;
                vector[2] = 0.0;
            }
            let body = json!({ "embeddings": [vector] }).to_string();
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
    (url, calls)
}

/// `legacy` rows WITHOUT a stored vector (inserted FIRST, so a big corpus
/// evicts them from the 10K text cache) plus `with_vector` documents that
/// carry theirs — all written by the library's own writer.
fn seed_docs(db: &Path, with_vector: usize, legacy: usize) {
    let dim = embedding_dim();
    let store = splatsdb::storage::sqlite_store::SqliteMetadataStore::new(db.to_path_buf())
        .expect("open sqlite store");
    {
        use splatsdb::storage::MetadataStore as _;
        for i in 0..legacy {
            let (mut record, _) = doc_record(i, dim);
            record.id = format!("legacy_{i}");
            record.document = Some(format!("text of legacy {i}"));
            store.upsert(&record).expect("store legacy row");
        }
    }
    for i in 0..with_vector {
        let (record, vector) = doc_record(i, dim);
        store
            .upsert_with_embedding(
                &record,
                &vector,
                splatsdb::storage::sqlite_store::EmbeddingProvenance::Provided,
            )
            .expect("store doc");
    }
}

fn warm_start_of(status: &Value) -> &Value {
    &status["vector_persistence"]["warm_start"]
}

fn ids_of(r: &Value) -> Vec<String> {
    r["results"]
        .as_array()
        .unwrap_or_else(|| panic!("no results array: {r}"))
        .iter()
        .map(|x| x["id"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[test]
fn f05e_w01_legacy_rows_are_embedded_by_real_calls_and_the_vectors_are_stored() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let dir = scratch("f05e-w01");
    let db = dir.join("docs.db");
    seed_docs(&db, 3, 2);
    let (url, calls) = spawn_counting_vector_server(embedding_dim());

    // Service up: one REAL call per legacy row, each vector written back.
    let mut srv = spawn(&db, &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let ws = warm_start_of(&srv.status(1)).clone();
    assert_eq!(ws["documents"], json!(5), "{ws}");
    assert_eq!(ws["vectors_loaded"], json!(3), "{ws}");
    assert_eq!(ws["reembedded"], json!(2), "{ws}");
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        2,
        "`reembedded` must equal the real service calls"
    );
    let r = srv.ok(
        2,
        "splatsdb_search",
        json!({ "query": "text of legacy", "top_k": 5 }),
    );
    let ids = ids_of(&r);
    assert!(
        ids.contains(&"legacy_0".into()) && ids.contains(&"legacy_1".into()),
        "{r}"
    );
    srv.shutdown();

    // Restart with the service DOWN: the stored vectors are used, nothing is
    // re-embedded and the service is not called at all.
    let before = calls.load(AtomicOrdering::SeqCst);
    let mut again = spawn(&db, &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    let ws = warm_start_of(&again.status(1)).clone();
    assert_eq!(ws["vectors_loaded"], json!(5), "{ws}");
    assert_eq!(ws["reembedded"], json!(0), "{ws}");
    assert_eq!(ws["unindexed"], json!(0), "{ws}");
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        before,
        "a restart from stored vectors called the service"
    );
    again.shutdown();
}

#[test]
fn f05e_w02_legacy_rows_with_the_service_down_are_quarantined_never_hash_reembedded() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let dir = scratch("f05e-w02");
    let db = dir.join("docs.db");
    seed_docs(&db, 1, 1);
    // Harness default: the SimCos fallback IS authorized, the service is DOWN.
    let mut srv = spawn(&db, &[]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let st = srv.status(1);
    let ws = warm_start_of(&st);
    assert_eq!(ws["documents"], json!(2), "{st}");
    assert_eq!(ws["vectors_loaded"], json!(1), "{st}");
    assert_eq!(
        ws["reembedded"],
        json!(0),
        "no real call happened, so nothing was re-embedded: {st}"
    );
    assert_eq!(ws["unindexed"], json!(1), "{st}");
    // Quarantined: invisible to search, and reported with its id.
    let r = srv.ok(
        2,
        "splatsdb_search",
        json!({ "query": "text of legacy 0", "top_k": 5 }),
    );
    assert!(
        ids_of(&r).iter().all(|id| id != "legacy_0"),
        "a quarantined row was indexed: {r}"
    );
    assert!(
        srv.wait_stderr_containing("legacy_0", Duration::from_secs(5)),
        "the quarantine is not reported with the id: {}",
        srv.err_log.join("\n")
    );
    assert!(
        srv.err_log
            .iter()
            .any(|l| l.contains("QUARANTINED") && l.contains("legacy_0")),
        "{}",
        srv.err_log.join("\n")
    );
    srv.shutdown();
    // The row is untouched: a restart with the service up indexes it once.
    let (url, calls) = spawn_counting_vector_server(embedding_dim());
    let mut again = spawn(&db, &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    let ws = warm_start_of(&again.status(1)).clone();
    assert_eq!(ws["reembedded"], json!(1), "{ws}");
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    again.shutdown();
}

#[test]
fn f05e_w03_legacy_row_in_a_corpus_past_the_10k_cache_is_embedded_once_and_keeps_its_text() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let dir = scratch("f05e-w03");
    let db = dir.join("docs.db");
    seed_docs(&db, 10_001, 1);
    let (url, calls) = spawn_counting_vector_server(embedding_dim());
    let mut srv = spawn(&db, &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let ws = warm_start_of(&srv.status(1)).clone();
    assert_eq!(ws["documents"], json!(10_002), "{ws}");
    assert_eq!(ws["reembedded"], json!(1), "{ws}");
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    // 10_002 documents > the 10K text cache: the text comes from SQLite.
    let r = srv.ok(
        2,
        "splatsdb_search",
        json!({ "query": "text of legacy", "top_k": 100 }),
    );
    let hit = r["results"]
        .as_array()
        .expect("results")
        .iter()
        .find(|x| x["id"] == json!("legacy_0"))
        .unwrap_or_else(|| panic!("legacy_0 must be indexed: {r}"));
    assert_eq!(hit["text"], json!("text of legacy 0"), "{r}");
    srv.shutdown();
    // Restart with the service down: stored vectors only.
    let mut again = spawn(&db, &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    let ws = warm_start_of(&again.status(1)).clone();
    assert_eq!(ws["vectors_loaded"], json!(10_002), "{ws}");
    assert_eq!(ws["reembedded"], json!(0), "{ws}");
    assert_eq!(ws["unindexed"], json!(0), "{ws}");
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        2,
        "one embed for the legacy row + one for the search; none after the restart"
    );
    again.shutdown();
}

/// A v1 state file with the given `(id, embedding-as-json-text)` nodes and
/// two edges 1→2 and 1→3.
fn legacy_state_text(nodes: &[(usize, &str)]) -> String {
    let nodes: Vec<String> = nodes
        .iter()
        .map(|(id, emb)| {
            format!(
                r#"{{"id":{id},"node_type":"Document","content":"node {id}","embedding":{emb},"alpha":1.0,"kappa":10.0}}"#
            )
        })
        .collect();
    format!(
        r#"{{"version":1,"next_id":4,"nodes":[{}],"edges":[{{"source_id":1,"target_id":2,"relation_type":"CITES","weight":1.0}},{{"source_id":1,"target_id":3,"relation_type":"CITES","weight":1.0}}]}}"#,
        nodes.join(",")
    )
}

fn unit_vector_text(dim: usize, hot: usize) -> String {
    let mut v = vec!["0.0"; dim];
    v[hot % dim] = "1.0";
    format!("[{}]", v.join(","))
}

fn assert_legacy_quarantine(dir_tag: &str, bad_embedding: &str) {
    let dir = scratch(dir_tag);
    let dim = embedding_dim();
    let file = legacy_state_text(&[
        (1, &unit_vector_text(dim, 0)),
        (2, &unit_vector_text(dim, 1)),
        (3, bad_embedding),
    ]);
    std::fs::write(state_path(&dir), &file).expect("write legacy state");

    let url = spawn_vector_server(dim);
    let mut srv = spawn(&dir.join("docs.db"), &[("SPLATSDB_EMBED_URL", &url)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    let status = srv.status(1);
    assert_eq!(
        status["graph_persistence"]["state"],
        json!("durable"),
        "one invalid node must not brick the whole graph: {status}"
    );
    assert_eq!(
        status["graph_persistence"]["quarantined_nodes"],
        json!([3]),
        "{status}"
    );
    let stats = srv.stats(2);
    assert_eq!(stats["total_nodes"], json!(2), "{stats}");
    assert_eq!(
        stats["total_edges"],
        json!(1),
        "the edge to the refused node goes with it: {stats}"
    );
    // The graph is usable: a new node joins and is persisted.
    let added = srv.ok(
        3,
        "splatsdb_graph_add_doc",
        json!({ "text": "after the quarantine" }),
    );
    assert_eq!(added["persisted"], json!(true), "{added}");
    // The ORIGINAL bytes survive before anything could overwrite them.
    let backup = dir.join("graph_state.json.pre-quarantine");
    assert_eq!(
        std::fs::read(&backup).expect("byte-exact copy of the original state"),
        file.as_bytes().to_vec()
    );
    assert!(
        srv.wait_stderr_containing("QUARANTINED", Duration::from_secs(5)),
        "{}",
        srv.err_log.join("\n")
    );
    assert!(
        srv.err_log
            .iter()
            .any(|l| l.contains("QUARANTINED") && l.contains("ids=[3]")),
        "{}",
        srv.err_log.join("\n")
    );
    srv.shutdown();
    // Restart: 3 nodes / 1 edge, nothing left to quarantine.
    let mut again = spawn(&dir.join("docs.db"), &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    assert_eq!(again.stats(1)["total_nodes"], json!(3));
    assert_eq!(again.stats(2)["total_edges"], json!(1));
    assert_eq!(
        again.status(3)["graph_persistence"]["quarantined_nodes"],
        json!([])
    );
    again.shutdown();
}

#[test]
fn f05e_g01_legacy_node_with_an_empty_embedding_is_quarantined_not_the_whole_file() {
    assert_legacy_quarantine("f05e-g01", "[]");
}

#[test]
fn f05e_g02_legacy_node_with_a_non_finite_embedding_is_quarantined_not_the_whole_file() {
    let dim = embedding_dim();
    let mut v = vec!["0.0"; dim];
    v[0] = "1e40"; // finite as f64, +inf once stored as f32
    assert_legacy_quarantine("f05e-g02", &format!("[{}]", v.join(",")));
}

#[test]
fn f05e_g03_repeated_relation_is_a_no_op_no_rewrite_no_spurious_outofsync() {
    let dir = scratch("f05e-g03");
    let db = dir.join("docs.db");
    let (mut srv, a, b) = graph_with_a_popular_b(&db);
    let first = srv.add_relation(40, a, b, "CITES");
    assert_eq!(first["created"], json!(true), "{first}");
    // The log line of THIS relation. A bare `created=true` is not enough: the
    // five MENTIONS relations of the fixture say the same, so the wait used to
    // return at the third mutation of the start-up and the count below then
    // included the nine `graph persisted` lines still queued behind it
    // (13 vs 4 — the fixture's own writes, not a rewrite by the no-op).
    let cites = format!("--CITES--> {b} (w=0.9, created=");
    assert!(
        srv.wait_stderr_containing(&format!("{cites}true"), Duration::from_secs(5)),
        "{}",
        srv.err_log.join("\n")
    );
    let persisted_before = srv
        .err_log
        .iter()
        .filter(|l| l.contains("graph persisted"))
        .count();
    let on_disk = std::fs::read(state_path(&dir)).expect("state file");
    // A save writes a transient file and renames it over the state file: a
    // rewrite is a NEW inode even when the bytes come out identical.
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    #[cfg(unix)]
    let inode_before = std::fs::metadata(state_path(&dir))
        .expect("state file metadata")
        .ino();

    // The repeat answers created=false from the live graph: no file write.
    let again = srv.add_relation(41, a, b, "CITES");
    assert_eq!(again["created"], json!(false), "{again}");
    assert!(
        srv.wait_stderr_containing(&format!("{cites}false"), Duration::from_secs(5)),
        "{}",
        srv.err_log.join("\n")
    );
    let persisted_after = srv
        .err_log
        .iter()
        .filter(|l| l.contains("graph persisted"))
        .count();
    assert_eq!(
        persisted_after, persisted_before,
        "the no-op re-wrote the state file"
    );
    assert_eq!(
        std::fs::read(state_path(&dir)).unwrap(),
        on_disk,
        "the no-op changed the state file"
    );
    #[cfg(unix)]
    assert_eq!(
        std::fs::metadata(state_path(&dir))
            .expect("state file metadata")
            .ino(),
        inode_before,
        "the no-op replaced the state file (same bytes, new inode)"
    );

    // Another writer replaced the state file meanwhile: the no-op still
    // answers (it writes nothing); a REAL mutation must still fail closed.
    let mut foreign = on_disk.clone();
    foreign.push(b'\n');
    std::fs::write(state_path(&dir), &foreign).expect("foreign write");
    let (is_error, repeat) = srv.tool(
        42,
        "splatsdb_graph_add_relation",
        json!({ "source_id": a, "target_id": b, "relation_type": "CITES", "weight": 0.9 }),
    );
    assert!(
        !is_error,
        "a no-op was refused because of another writer: {repeat}"
    );
    assert_eq!(repeat["created"], json!(false));
    let err = srv.error(
        43,
        "splatsdb_graph_add_relation",
        json!({ "source_id": b, "target_id": a, "relation_type": "CITES", "weight": 0.9 }),
    );
    assert!(
        err.contains("persisted"),
        "a real mutation must fail closed: {err}"
    );
    assert_eq!(
        std::fs::read(state_path(&dir)).unwrap(),
        foreign,
        "the foreign file was overwritten"
    );
    srv.shutdown();
}

fn stderr_until_closed(srv: &Server, bound: Duration) -> Vec<String> {
    let deadline = Instant::now() + bound;
    let mut lines = srv.err_log.clone();
    while let Ok(line) = srv
        .err
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
    {
        lines.push(line);
    }
    lines
}

#[test]
fn f05e_x05_sigterm_during_a_large_warm_start_exits_cleanly_and_the_store_is_intact() {
    let dir = scratch("f05e-x05");
    let db = dir.join("docs.db");
    seed_docs(&db, 5_000, 0);
    let mut srv = spawn(&db, &[]);
    // The signal lands while the 5,000 stored vectors are still loading.
    assert!(
        srv.wait_stderr_containing("docs loaded", STARTUP_TIMEOUT),
        "no warm-start progress line: {}",
        srv.err_log.join("\n")
    );
    let t0 = Instant::now();
    srv.signal(libc::SIGTERM);
    let (status, took) = srv.exit_within(t0, Duration::from_secs(10), libc::SIGTERM);
    assert!(status.success(), "{status:?}");
    let err = stderr_until_closed(&srv, Duration::from_secs(5));
    assert!(srv.pipes_closed(Duration::from_secs(2)), "pipes left open");
    assert!(
        err.iter()
            .any(|l| l.contains("Warm start interrupted by shutdown signal")),
        "the signal did not land inside the warm start: {err:#?}"
    );
    assert!(
        !err.iter().any(|l| l.contains("MCP server ready")),
        "a shutting-down server announced itself ready: {err:#?}"
    );
    eprintln!("[F05e] SIGTERM during warm start (5000 docs): exit after {took:?}");
    let mut again = spawn(&db, &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    let ws = warm_start_of(&again.status(1)).clone();
    assert_eq!(ws["documents"], json!(5_000), "{ws}");
    assert_eq!(ws["vectors_loaded"], json!(5_000), "{ws}");
    again.shutdown();
}

/// One table cell: `reps` real signals of kind `sig` with the server in
/// `state`, each after two acknowledged mutations. Every run must end by
/// itself, gracefully, inside `SIGNAL_EXIT_BOUND` (= the 2 s contract), with
/// both pipes closed, nothing answered for an incomplete request, and both
/// mutations still on disk after a restart.
fn run_signal_cell(
    db: &Path,
    sig: i32,
    state: StdinState,
    reps: usize,
) -> (usize, Vec<String>, u128) {
    let mut failures = 0;
    let mut notes = Vec::new();
    let mut max_ms = 0;
    for rep in 0..reps {
        // One directory per run: the graph state is `graph_state.json` beside
        // the SQLite file and one directory holds ONE graph (A04-F10), so runs
        // sharing a directory keep each other's nodes and `total_nodes == 1`
        // below could only hold for the first one.
        let run_dir = db.with_extension(format!("run{rep}"));
        std::fs::create_dir_all(&run_dir).expect("create per-run dir");
        let db = run_dir.join("docs.db");
        let mut srv = spawn(&db, &[]);
        if !srv.wait_ready() {
            failures += 1;
            notes.push(format!("rep {rep}: never ready"));
            continue;
        }
        srv.ok(
            1,
            "splatsdb_doc_add",
            json!({ "id": "mem_kept", "text": "kept across the signal" }),
        );
        srv.add_doc(2, "graph node kept across the signal");
        if state == StdinState::PartialLine {
            let stdin = srv.stdin.as_mut().expect("stdin");
            stdin
                .write_all(br#"{"jsonrpc":"2.0","id":7,"method":"ping""#)
                .expect("write partial line");
            stdin.flush().expect("flush");
            thread::sleep(Duration::from_millis(200));
        }
        let t0 = Instant::now();
        srv.signal(sig);
        let mut exited = None;
        while t0.elapsed() < SIGNAL_EXIT_BOUND {
            match srv.child.try_wait() {
                Ok(Some(status)) => {
                    exited = Some((status, t0.elapsed()));
                    break;
                }
                Ok(None) => thread::sleep(Duration::from_millis(5)),
                Err(e) => {
                    notes.push(format!("rep {rep}: try_wait {e}"));
                    break;
                }
            }
        }
        let Some((status, took)) = exited else {
            failures += 1;
            let _ = srv.child.kill();
            let _ = srv.child.wait();
            notes.push(format!(
                "rep {rep}: STILL RUNNING {SIGNAL_EXIT_BOUND:?} after signal {sig} (killed by the test)"
            ));
            continue;
        };
        max_ms = max_ms.max(took.as_millis());
        let mut ok = status.success();
        if !ok {
            notes.push(format!("rep {rep}: exit {status:?}"));
        }
        let answered = srv.stdout_until_closed(Duration::from_secs(2));
        if !answered.is_empty() {
            ok = false;
            notes.push(format!(
                "rep {rep}: answered after the signal: {answered:?}"
            ));
        }
        if !srv.pipes_closed(Duration::from_secs(2)) {
            ok = false;
            notes.push(format!("rep {rep}: pipes still open"));
        }
        if !ok {
            failures += 1;
        }
        // On disk: everything acknowledged before the signal is still there.
        let mut again = spawn(&db, &[]);
        let ready = again.wait_ready();
        let doc = again.ok(1, "splatsdb_doc_get", json!({ "id": "mem_kept" }));
        let nodes = again.stats(2)["total_nodes"].clone();
        again.shutdown();
        if !ready || doc["text"] != json!("kept across the signal") || nodes != json!(1) {
            failures += 1;
            notes.push(format!("rep {rep}: restart state doc={doc} nodes={nodes}"));
        }
    }
    (failures, notes, max_ms)
}

fn signal_table(db_tag: &str, sig: i32, state: StdinState, reps: usize) {
    let db = scratch(db_tag).join("docs.db");
    let (failures, notes, max_ms) = run_signal_cell(&db, sig, state, reps);
    let (sig_name, case) = (
        if sig == libc::SIGINT {
            "SIGINT"
        } else {
            "SIGTERM"
        },
        if state == StdinState::PartialLine {
            "partial-line"
        } else {
            "idle"
        },
    );
    eprintln!(
        "[F05e signal table] sig={sig_name} case={case} N={reps} failures={failures} max_exit_ms={max_ms} rc=0 notes={notes:?}"
    );
    assert!(
        failures == 0 && notes.is_empty(),
        "{sig_name} {case}: {failures}/{reps} failed: {notes:#?}"
    );
}

#[test]
fn f05e_x10a_signal_table_idle_sigint() {
    signal_table(
        "f05e-t-idle-int",
        libc::SIGINT,
        StdinState::Idle,
        SIGNAL_REPS,
    );
}

#[test]
fn f05e_x10b_signal_table_idle_sigterm() {
    signal_table(
        "f05e-t-idle-term",
        libc::SIGTERM,
        StdinState::Idle,
        SIGNAL_REPS,
    );
}

#[test]
fn f05e_x10c_signal_table_partial_line_sigint() {
    signal_table(
        "f05e-t-partial-int",
        libc::SIGINT,
        StdinState::PartialLine,
        SIGNAL_REPS,
    );
}

#[test]
fn f05e_x10d_signal_table_partial_line_sigterm() {
    signal_table(
        "f05e-t-partial-term",
        libc::SIGTERM,
        StdinState::PartialLine,
        SIGNAL_REPS,
    );
}

#[test]
fn f05e_x07_in_flight_mutation_is_abandoned_after_a_bounded_wait_and_never_acked() {
    let dir = scratch("f05e-x07");
    let db = dir.join("docs.db");
    let hole = spawn_black_hole();
    let mut srv = spawn(&db, &[("SPLATSDB_EMBED_URL", &hole)]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    // A mutation whose embedding HTTP call never answers.
    srv.rpc_send(
        7,
        "tools/call",
        json!({ "name": "splatsdb_graph_add_doc", "arguments": { "text": "never acknowledged" } }),
    );
    thread::sleep(Duration::from_millis(300));
    let t0 = Instant::now();
    srv.signal(libc::SIGTERM);
    let (status, took) = srv.exit_within(t0, IN_FLIGHT_EXIT_BOUND, libc::SIGTERM);
    assert!(status.success(), "{status:?}");
    let lines = srv.stdout_until_closed(Duration::from_secs(2));
    let err = stderr_until_closed(&srv, Duration::from_secs(2));
    assert!(srv.pipes_closed(Duration::from_secs(2)), "pipes left open");
    assert!(
        lines.is_empty(),
        "an abandoned request was answered: {lines:?}"
    );
    assert!(
        err.iter().any(|l| l.contains("NO response")),
        "stderr must say the in-flight request was not acknowledged: {err:#?}"
    );
    eprintln!(
        "[F05e] SIGTERM with a mutation in flight: exit after {took:?}, nothing acknowledged"
    );
    // Nothing was persisted for the unacknowledged mutation.
    let mut again = spawn(&db, &[]);
    assert!(again.wait_ready(), "{}", again.err_log.join("\n"));
    assert_eq!(
        again.stats(1)["total_nodes"],
        json!(0),
        "an unacknowledged mutation was persisted"
    );
    again.shutdown();
}

#[test]
fn f05e_x08_second_signal_forces_the_exit_while_a_request_is_in_flight() {
    for sig in [libc::SIGTERM, libc::SIGINT] {
        let dir = scratch("f05e-x08");
        let db = dir.join("docs.db");
        let hole = spawn_black_hole();
        let mut srv = spawn(&db, &[("SPLATSDB_EMBED_URL", &hole)]);
        assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
        srv.rpc_send(
            7,
            "tools/call",
            json!({ "name": "splatsdb_graph_search", "arguments": { "query": "in flight" } }),
        );
        thread::sleep(Duration::from_millis(300));
        srv.signal(sig);
        thread::sleep(Duration::from_millis(300));
        let t0 = Instant::now();
        srv.signal(sig);
        let (status, took) = srv.exit_within(t0, SIGNAL_EXIT_BOUND, sig);
        assert_eq!(
            status.code(),
            Some(128 + sig),
            "the second signal must force the exit (128 + signal), got {status:?}"
        );
        eprintln!("[F05e] second signal {sig}: forced exit after {took:?}");
    }
}

#[test]
fn f05e_x09_eof_after_a_request_and_on_a_partial_line_end_cleanly() {
    for rep in 0..2 {
        let db = scratch(&format!("f05e-x09a-{rep}")).join("docs.db");
        let mut srv = spawn(&db, &[]);
        assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
        assert_eq!(srv.rpc(1, "ping", json!({}))["id"], json!(1));
        let t0 = Instant::now();
        srv.stdin.take();
        let (status, _) = srv.exit_within(t0, Duration::from_secs(10), 0);
        assert!(status.success(), "EOF after a request: {status:?}");
        assert!(srv.pipes_closed(Duration::from_secs(2)));
    }
    for rep in 0..2 {
        let db = scratch(&format!("f05e-x09b-{rep}")).join("docs.db");
        let mut srv = spawn(&db, &[]);
        assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
        let stdin = srv.stdin.as_mut().expect("stdin");
        stdin
            .write_all(br#"{"jsonrpc":"2.0","id":7,"method":"ping""#)
            .expect("write partial line");
        stdin.flush().expect("flush");
        thread::sleep(Duration::from_millis(200));
        let t0 = Instant::now();
        srv.stdin.take();
        let (status, _) = srv.exit_within(t0, Duration::from_secs(10), 0);
        assert!(status.success(), "EOF on a partial line: {status:?}");
        let lines = srv.stdout_until_closed(Duration::from_secs(2));
        for line in &lines {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                assert_ne!(
                    v.get("id").and_then(|i| i.as_u64()),
                    Some(7),
                    "an incomplete request was answered at EOF: {line}"
                );
            }
        }
        assert!(srv.pipes_closed(Duration::from_secs(2)));
    }
}

#[test]
fn f05e_x11_signal_delivered_to_a_non_main_thread_still_ends_the_server() {
    let db = scratch("f05e-x11").join("docs.db");
    let mut srv = spawn(&db, &[]);
    assert!(srv.wait_ready(), "{}", srv.err_log.join("\n"));
    srv.add_doc(1, "to get the worker threads going");
    let pid = srv.child.id() as i32;
    let mut tids: Vec<i32> = std::fs::read_dir(format!("/proc/{pid}/task"))
        .expect("list threads")
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()))
        .filter(|&t| t != pid)
        .collect();
    tids.sort_unstable();
    if tids.is_empty() {
        eprintln!(
            "[F05e] SKIP: the server has one thread only; nothing but the main thread to target"
        );
        srv.shutdown();
        return;
    }
    let tid = tids[tids.len() - 1];
    let t0 = Instant::now();
    let rc = unsafe {
        libc::syscall(
            libc::SYS_tgkill,
            pid as libc::c_long,
            tid as libc::c_long,
            libc::SIGINT as libc::c_long,
        )
    };
    assert_eq!(rc, 0, "tgkill");
    let (status, took) = srv.exit_within(t0, SIGNAL_EXIT_BOUND, libc::SIGINT);
    assert!(status.success(), "{status:?}");
    eprintln!(
        "[F05e] SIGINT delivered to thread {tid} (of {} threads): graceful exit after {took:?}",
        tids.len() + 1
    );
}
