//! F05b — persistence contracts of the MCP doc store: the embedding vector
//! and its provenance are persisted with the document (A03-F10, persistence
//! half), a restart loads them without re-embedding, legacy databases
//! migrate without loss, bad blobs are quarantined, and the spatial
//! metadata of `splatsdb_store` survives a restart (A04-F14).
//!
//! Every "restart" is a real `splatsdb mcp` process over the same SQLite
//! file. The embed service is either DOWN (closed port 9 → connection
//! refused) or a local fake that COUNTS each `/embed` request, so
//! "zero re-embeddings" is a measured number, not a reading of the code.

use rusqlite::{params, Connection, OptionalExtension};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Closed local port: connection refused — nothing listens here.
const NO_SERVICE_URL: &str = "http://127.0.0.1:9/embed";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RESP_TIMEOUT: Duration = Duration::from_secs(120);
/// Warm-start lines are printed before "MCP server ready", so they are
/// already drained when a server is returned; this only covers lines still
/// in flight.
const LOG_TIMEOUT: Duration = Duration::from_secs(30);

/// Document table exactly as the pre-F05b schema created it (verbatim DDL of
/// `SqliteMetadataStore::new` at 3703d1f): no embedding columns.
const LEGACY_DDL: &str = "CREATE TABLE IF NOT EXISTS documents (
             id TEXT PRIMARY KEY,
             shard_idx INTEGER,
             vector_idx INTEGER,
             metadata TEXT,
             document TEXT,
             deleted INTEGER DEFAULT 0,
             created_at REAL,
             updated_at REAL
         );
         CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted);
         CREATE INDEX IF NOT EXISTS idx_vector ON documents(vector_idx);";

struct Server {
    child: Child,
    stdin: ChildStdin,
    out: Receiver<String>,
    err: Receiver<String>,
    /// stderr lines already drained.
    err_log: Vec<String>,
}

fn spawn_server(doc_db: &Path, embed_url: &str, envs: &[(&str, &str)]) -> Server {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_splatsdb"));
    cmd.arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("SPLATSDB_DOC_PATH", doc_db)
        .env("SPLATSDB_EMBED_URL", embed_url)
        // Strict unless a test opts in explicitly.
        .env_remove("SPLATSDB_ALLOW_SIMCOS");
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
        let req = serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        });
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
        // Lines drained earlier (e.g. the warm-start report consumed while
        // waiting for "MCP server ready") are kept in `err_log` and checked
        // first.
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

    /// Assert a warm-start / startup line is on stderr, printing the whole
    /// drained log when it is not.
    fn expect_log(&mut self, needle: &str) {
        assert!(
            self.wait_stderr_containing(needle, LOG_TIMEOUT),
            "stderr lacks {needle:?}; log:\n{}",
            self.err_log.join("\n")
        );
    }

    fn status(&mut self, id: u64) -> serde_json::Value {
        self.ok(id, "splatsdb_status", serde_json::json!({}))
    }

    /// Close stdin (server exits on EOF) and reap the process.
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

    /// SIGKILL right now: no stdin EOF, no shutdown path — a crash.
    fn kill_now(mut self) {
        self.child.kill().expect("SIGKILL");
        self.child.wait().expect("reap killed server");
    }
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splatsdb_mcp_persistence_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Client-provided embedding with two informative components (rest zero).
fn provided(dim: usize, a: f64, b: f64) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = a as f32;
    v[1] = b as f32;
    v
}

fn emb(dim: usize, a: f64, b: f64) -> serde_json::Value {
    serde_json::json!(provided(dim, a, b))
}

/// Little-endian f32 bytes: exactly what must be in the `embedding` BLOB.
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ids(results: &serde_json::Value) -> Vec<String> {
    results["results"]
        .as_array()
        .unwrap_or_else(|| panic!("no results array: {results}"))
        .iter()
        .map(|r| r["id"].as_str().unwrap_or("").to_string())
        .collect()
}

/// Store dimension, learned from the store's own error message: a 2-dim
/// embedding is rejected before anything is written or requested.
fn probe_dim(srv: &mut Server, id: u64) -> usize {
    let (is_error, payload) = srv.tool(
        id,
        "splatsdb_store",
        serde_json::json!({ "text": "dim probe", "embedding": [0.5, 0.5] }),
    );
    assert!(is_error, "a 2-dim embedding must be rejected: {payload}");
    let msg = payload
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| payload.to_string());
    msg.split("expected ")
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or_else(|| panic!("cannot parse the store dimension from: {msg}"))
}

/// Fake embed service that COUNTS requests. It answers `dim` dims once the
/// test has stored the store dimension in `dim_cell` (0 = unknown yet →
/// HTTP 500, i.e. an unusable service).
fn spawn_counting_embed_server(dim_cell: Arc<AtomicUsize>) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
    let url = format!(
        "http://127.0.0.1:{}/embed",
        listener.local_addr().expect("local addr").port()
    );
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            counter.fetch_add(1, Ordering::SeqCst);
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
            let dim = dim_cell.load(Ordering::SeqCst);
            if dim == 0 {
                let _ = stream.write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                let _ = stream.flush();
                continue;
            }
            let text = String::from_utf8_lossy(&buf).to_ascii_lowercase();
            let vector: Vec<f64> = (0..dim)
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
    (url, hits)
}

// ---------------------------------------------------------------------------
// Direct SQLite inspection (the server is down whenever these are used)
// ---------------------------------------------------------------------------

fn open_db(path: &Path) -> Connection {
    Connection::open(path).expect("open doc db")
}

#[derive(Debug)]
struct StoredRow {
    deleted: i64,
    blob: Option<Vec<u8>>,
    source: Option<String>,
}

fn stored_row(db: &Connection, id: &str) -> StoredRow {
    db.query_row(
        "SELECT deleted, embedding, embedding_source FROM documents WHERE id=?1",
        params![id],
        |r| {
            Ok(StoredRow {
                deleted: r.get(0)?,
                blob: r.get(1)?,
                source: r.get(2)?,
            })
        },
    )
    .optional()
    .unwrap_or_else(|e| panic!("reading {id}: {e}"))
    .unwrap_or_else(|| panic!("row {id} is missing"))
}

fn has_column(db: &Connection, column: &str) -> bool {
    column_count(db, column) > 0
}

fn column_count(db: &Connection, column: &str) -> usize {
    let mut stmt = db.prepare("PRAGMA table_info(documents)").expect("pragma");
    let mut rows = stmt.query([]).expect("pragma rows");
    let mut n = 0;
    while let Some(row) = rows.next().expect("pragma row") {
        if row.get::<_, String>(1).expect("column name") == column {
            n += 1;
        }
    }
    n
}

/// A row written by the pre-F05b schema (no embedding columns at all).
fn insert_legacy(db: &Connection, id: &str, metadata: &str, document: &str, deleted: i32) {
    db.execute(
        "INSERT INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
         VALUES (?1, 0, 0, ?2, ?3, ?4, 1.0, 1.0)",
        params![id, metadata, document, deleted],
    )
    .expect("insert legacy row");
}

// ---------------------------------------------------------------------------
// (2) Warm start without re-embedding — embed service DOWN
// ---------------------------------------------------------------------------

/// A restart with the embed service DOWN (and no SimCos opt-in) searches
/// over the STORED vectors: every document comes back, each one labelled
/// with the provenance it was stored with, and the warm start reports
/// `re-embedded: 0`.
#[test]
fn p05_restart_with_embed_service_down_searches_from_stored_vectors() {
    let tmp = scratch("p05_service_down");
    let doc_db = tmp.join("docs.db");
    let dim_cell = Arc::new(AtomicUsize::new(0));
    let (url, hits) = spawn_counting_embed_server(Arc::clone(&dim_cell));
    let dim;
    let all = [
        "p_real_1", "p_real_2", "p_real_3", "p_prov_1", "p_prov_2", "p_prov_3", "p_add_1",
    ];
    // ---- Session A: service UP. 4 real + 3 provided vectors. ----
    {
        let mut srv = spawn_server(&doc_db, &url, &[]);
        dim = probe_dim(&mut srv, 1);
        dim_cell.store(dim, Ordering::SeqCst);
        for (i, text) in ["real doc one", "real doc two", "real doc three"]
            .iter()
            .enumerate()
        {
            let r = srv.ok(
                10 + i as u64,
                "splatsdb_store",
                serde_json::json!({ "id": format!("p_real_{}", i + 1), "text": text }),
            );
            assert_eq!(r["embedding"], serde_json::json!("real"), "{r}");
        }
        assert_eq!(hits.load(Ordering::SeqCst), 3, "3 real docs = 3 requests");
        for (i, (a, b)) in [(1.0, 0.0), (1.0, 0.6), (0.2, 1.0)].into_iter().enumerate() {
            let r = srv.ok(
                20 + i as u64,
                "splatsdb_store",
                serde_json::json!({
                    "id": format!("p_prov_{}", i + 1),
                    "text": format!("provided doc {}", i + 1),
                    "embedding": emb(dim, a, b)
                }),
            );
            assert_eq!(r["embedding"], serde_json::json!("provided"), "{r}");
            assert_eq!(r["persisted"], serde_json::json!(true), "{r}");
        }
        assert_eq!(
            hits.load(Ordering::SeqCst),
            3,
            "a provided embedding makes no request"
        );
        let r = srv.ok(
            30,
            "splatsdb_doc_add",
            serde_json::json!({ "id": "p_add_1", "text": "doc added through doc_add" }),
        );
        assert_eq!(r["embedding"], serde_json::json!("real"), "{r}");
        assert_eq!(hits.load(Ordering::SeqCst), 4);
        let st = srv.status(31);
        assert_eq!(st["n_active"], serde_json::json!(7), "{st}");
        srv.shutdown();
    }
    // On disk: every row carries an f32 BLOB of exactly dim*4 bytes and its
    // provenance label, in the SAME row as the document.
    {
        let db = open_db(&doc_db);
        for id in all {
            let row = stored_row(&db, id);
            assert_eq!(row.deleted, 0, "{id}");
            assert_eq!(
                row.blob.as_ref().map(Vec::len),
                Some(dim * 4),
                "{id}: the vector is not persisted as a dim*4 f32 BLOB: {:?}",
                row.blob.as_ref().map(Vec::len)
            );
            assert!(row.source.is_some(), "{id}: provenance label missing");
        }
        let want = f32_bytes(&provided(dim, 1.0, 0.0));
        assert_eq!(
            stored_row(&db, "p_prov_1").blob.as_deref(),
            Some(want.as_slice()),
            "the client's f32 vector must be stored verbatim (little-endian)"
        );
        assert_eq!(stored_row(&db, "p_real_1").source.as_deref(), Some("real"));
        assert_eq!(
            stored_row(&db, "p_prov_1").source.as_deref(),
            Some("provided")
        );
        assert_eq!(stored_row(&db, "p_add_1").source.as_deref(), Some("real"));
    }
    // ---- Session B: embed service DOWN, strict mode (no SimCos opt-in). ----
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 7/7 documents");
        srv.expect_log("stored vectors loaded: 7");
        srv.expect_log("re-embedded: 0");
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({
                "query": "the query text is irrelevant: the embedding is provided",
                "embedding": emb(dim, 1.0, 0.0),
                "top_k": 20
            }),
        );
        let mut got = ids(&res);
        got.sort();
        let mut want: Vec<String> = all.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(
            got, want,
            "search after a restart with the service down: {res}"
        );
        assert_eq!(
            res["results"][0]["id"],
            serde_json::json!("p_prov_1"),
            "the nearest neighbour must come from the STORED vector: {res}"
        );
        assert_eq!(
            res["embedding"],
            serde_json::json!("provided"),
            "the top-level label stays the QUERY embedding's source: {res}"
        );
        for r in res["results"].as_array().expect("results array") {
            let id = r["id"].as_str().expect("id");
            let expect = if id.starts_with("p_prov_") {
                "stored_provided"
            } else {
                "stored_real"
            };
            assert_eq!(
                r["embedding_source"],
                serde_json::json!(expect),
                "per-result provenance of {id}: {r}"
            );
            assert!(
                !r["text"].as_str().unwrap_or("").is_empty(),
                "text lost for {id}: {r}"
            );
        }
        let st = srv.status(2);
        assert_eq!(st["n_active"], serde_json::json!(7), "{st}");
        assert_eq!(st["live_vectors"], serde_json::json!(7), "{st}");
        assert_eq!(st["doc_count"], serde_json::json!(7), "{st}");
        let ws = &st["vector_persistence"]["warm_start"];
        assert_eq!(ws["vectors_loaded"], serde_json::json!(7), "{st}");
        assert_eq!(ws["stored_real"], serde_json::json!(4), "{st}");
        assert_eq!(ws["stored_simcos"], serde_json::json!(0), "{st}");
        assert_eq!(ws["stored_provided"], serde_json::json!(3), "{st}");
        assert_eq!(ws["reembedded"], serde_json::json!(0), "{st}");
        assert_eq!(ws["unindexed"], serde_json::json!(0), "{st}");
        assert_eq!(ws["quarantined"], serde_json::json!(0), "{st}");
        assert_eq!(
            st["vector_persistence"]["stored_vectors"],
            serde_json::json!(7),
            "{st}"
        );
        assert_eq!(
            st["vector_persistence"]["schema_ready"],
            serde_json::json!(true),
            "{st}"
        );
        // Still fail-closed (A03-F10): a NEW document needs an embedding and
        // the service is down.
        let (is_error, payload) = srv.tool(
            3,
            "splatsdb_store",
            serde_json::json!({ "id": "p_new", "text": "no embedding available for this one" }),
        );
        assert!(
            is_error,
            "storing without any embedding must fail: {payload}"
        );
        assert!(
            payload
                .as_str()
                .unwrap_or_default()
                .contains("embedding unavailable"),
            "{payload}"
        );
        assert_eq!(srv.status(4)["doc_count"], serde_json::json!(7));
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// (2) Warm start without re-embedding — service UP, requests counted
// ---------------------------------------------------------------------------

/// With the service UP the pre-F05b warm start re-embedded every document.
/// Now the embed service's own request counter must not move on restart.
#[test]
fn p05_restart_with_service_up_makes_zero_embed_requests() {
    let tmp = scratch("p05_zero_requests");
    let doc_db = tmp.join("docs.db");
    let dim_cell = Arc::new(AtomicUsize::new(0));
    let (url, hits) = spawn_counting_embed_server(Arc::clone(&dim_cell));
    let dim;
    {
        let mut srv = spawn_server(&doc_db, &url, &[]);
        dim = probe_dim(&mut srv, 1);
        dim_cell.store(dim, Ordering::SeqCst);
        for (i, text) in ["r one", "r two", "r three"].iter().enumerate() {
            let r = srv.ok(
                10 + i as u64,
                "splatsdb_store",
                serde_json::json!({ "id": format!("r_{}", i + 1), "text": text }),
            );
            assert_eq!(r["embedding"], serde_json::json!("real"), "{r}");
        }
        srv.ok(
            20,
            "splatsdb_store",
            serde_json::json!({ "id": "q_1", "text": "provided doc", "embedding": emb(dim, 1.0, 0.0) }),
        );
        assert_eq!(hits.load(Ordering::SeqCst), 4);
        srv.shutdown();
    }
    // ---- Restart with the service UP: zero requests. ----
    {
        let mut srv = spawn_server(&doc_db, &url, &[]);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            4,
            "the warm start made embed requests although every vector is stored"
        );
        srv.expect_log("reloaded 4/4 documents");
        srv.expect_log("stored vectors loaded: 4");
        srv.expect_log("re-embedded: 0");
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({ "query": "query embedded by the service", "top_k": 10 }),
        );
        assert_eq!(
            hits.load(Ordering::SeqCst),
            5,
            "exactly ONE request: the query embedding"
        );
        let mut got = ids(&res);
        got.sort();
        assert_eq!(
            got,
            vec![
                "q_1".to_string(),
                "r_1".to_string(),
                "r_2".to_string(),
                "r_3".to_string()
            ],
            "{res}"
        );
        // A new document is the only thing that legitimately needs the service.
        srv.ok(
            2,
            "splatsdb_store",
            serde_json::json!({ "id": "r_4", "text": "r four" }),
        );
        assert_eq!(hits.load(Ordering::SeqCst), 6);
        srv.shutdown();
    }
    // ---- Second restart: still zero requests. ----
    {
        let mut srv = spawn_server(&doc_db, &url, &[]);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            6,
            "second restart: zero requests"
        );
        srv.expect_log("reloaded 5/5 documents");
        srv.expect_log("re-embedded: 0");
        let st = srv.status(1);
        let ws = &st["vector_persistence"]["warm_start"];
        assert_eq!(ws["vectors_loaded"], serde_json::json!(5), "{st}");
        assert_eq!(ws["stored_real"], serde_json::json!(4), "{st}");
        assert_eq!(ws["stored_provided"], serde_json::json!(1), "{st}");
        assert_eq!(ws["reembedded"], serde_json::json!(0), "{st}");
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// (3) Legacy schema migration
// ---------------------------------------------------------------------------

/// A database written by the pre-vector schema opens without loss: the two
/// columns are added in place (once, however many times it is opened), the
/// documents stay readable, strict mode leaves them unindexed with a
/// warning, and with the service up they are re-embedded — the only case
/// where a re-embed is legitimate — and backfilled.
#[test]
fn p05_legacy_schema_without_vector_columns_migrates_without_loss() {
    let tmp = scratch("p05_legacy");
    let doc_db = tmp.join("docs.db");
    {
        let db = open_db(&doc_db);
        db.execute_batch(LEGACY_DDL)
            .expect("create the legacy schema");
        insert_legacy(
            &db,
            "leg_1",
            r#"{"text":"legacy one","wing":"leg_wing","room":"leg_room","hall":"leg_hall"}"#,
            "legacy one",
            0,
        );
        insert_legacy(&db, "leg_2", r#"{"text":"legacy two"}"#, "legacy two", 0);
        insert_legacy(
            &db,
            "leg_del",
            r#"{"text":"deleted legacy"}"#,
            "deleted legacy",
            1,
        );
        assert!(
            !has_column(&db, "embedding") && !has_column(&db, "embedding_source"),
            "the fixture must be the pre-F05b schema"
        );
    }
    let dim_cell = Arc::new(AtomicUsize::new(0));
    let (url, hits) = spawn_counting_embed_server(Arc::clone(&dim_cell));
    let dim;
    // ---- Session 1: service DOWN, strict. Opens, readable, nothing indexed. ----
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        dim = probe_dim(&mut srv, 1);
        dim_cell.store(dim, Ordering::SeqCst);
        srv.expect_log("migrated legacy schema");
        srv.expect_log("documents were NOT indexed");
        let st = srv.status(2);
        assert_eq!(
            st["doc_count"],
            serde_json::json!(2),
            "no document lost: {st}"
        );
        assert_eq!(st["n_active"], serde_json::json!(0), "{st}");
        let ws = &st["vector_persistence"]["warm_start"];
        assert_eq!(ws["unindexed"], serde_json::json!(2), "{st}");
        assert_eq!(ws["reembedded"], serde_json::json!(0), "{st}");
        assert_eq!(ws["quarantined"], serde_json::json!(0), "{st}");
        assert_eq!(
            st["vector_persistence"]["schema_ready"],
            serde_json::json!(true),
            "{st}"
        );
        let g = srv.ok(3, "splatsdb_doc_get", serde_json::json!({ "id": "leg_1" }));
        assert_eq!(g["text"], serde_json::json!("legacy one"), "{g}");
        assert_eq!(g["metadata"]["wing"], serde_json::json!("leg_wing"), "{g}");
        let g = srv.ok(4, "splatsdb_doc_get", serde_json::json!({ "id": "leg_2" }));
        assert_eq!(g["text"], serde_json::json!("legacy two"), "{g}");
        srv.shutdown();
    }
    {
        let db = open_db(&doc_db);
        assert!(has_column(&db, "embedding") && has_column(&db, "embedding_source"));
        for id in ["leg_1", "leg_2"] {
            let row = stored_row(&db, id);
            assert!(
                row.blob.is_none() && row.source.is_none(),
                "strict mode must not invent a vector for {id}: {row:?}"
            );
        }
        assert_eq!(
            stored_row(&db, "leg_del").deleted,
            1,
            "soft-deleted row untouched"
        );
    }
    // ---- Session 2: service UP → the 2 legacy rows are re-embedded and backfilled. ----
    {
        let mut srv = spawn_server(&doc_db, &url, &[]);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "exactly the two legacy rows without a stored vector are re-embedded"
        );
        srv.expect_log("reloaded 2/2 documents");
        srv.expect_log("re-embedded: 2");
        let st = srv.status(1);
        assert_eq!(st["n_active"], serde_json::json!(2), "{st}");
        let ws = &st["vector_persistence"]["warm_start"];
        assert_eq!(ws["reembedded"], serde_json::json!(2), "{st}");
        assert_eq!(ws["vectors_loaded"], serde_json::json!(0), "{st}");
        let res = srv.ok(
            2,
            "splatsdb_spatial_search",
            serde_json::json!({
                "query": "legacy",
                "wing": "leg_wing",
                "embedding": emb(dim, 1.0, 0.0),
                "top_k": 5
            }),
        );
        assert_eq!(ids(&res), vec!["leg_1".to_string()], "{res}");
        srv.shutdown();
    }
    {
        let db = open_db(&doc_db);
        for id in ["leg_1", "leg_2"] {
            let row = stored_row(&db, id);
            assert_eq!(
                row.blob.as_ref().map(Vec::len),
                Some(dim * 4),
                "{id}: the re-embedded vector was not backfilled"
            );
            assert_eq!(row.source.as_deref(), Some("real"), "{id}");
        }
        assert!(
            stored_row(&db, "leg_del").blob.is_none(),
            "deleted rows are not backfilled"
        );
    }
    // ---- Session 3: service DOWN again → the backfilled vectors load directly. ----
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 2/2 documents");
        srv.expect_log("re-embedded: 0");
        assert_eq!(srv.status(1)["n_active"], serde_json::json!(2));
        let res = srv.ok(
            2,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 5 }),
        );
        let mut got = ids(&res);
        got.sort();
        assert_eq!(got, vec!["leg_1", "leg_2"], "{res}");
        srv.shutdown();
    }
    // Idempotent: three opens later each column exists exactly once and no
    // row was lost or added.
    {
        let db = open_db(&doc_db);
        assert_eq!(column_count(&db, "embedding"), 1);
        assert_eq!(column_count(&db, "embedding_source"), 1);
        let n: i64 = db
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
            .expect("count");
        assert_eq!(n, 3, "no row lost, no row added");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// (4) Validation / quarantine
// ---------------------------------------------------------------------------

/// A stored vector that is not a valid `dim`-dimensional finite f32 vector
/// is QUARANTINED with a per-document diagnostic: not indexed, never
/// resized or rewritten, the document itself stays readable, and the other
/// documents are unaffected.
#[test]
fn p05_corrupt_stored_vectors_are_quarantined_never_resized() {
    let tmp = scratch("p05_quarantine");
    let doc_db = tmp.join("docs.db");
    let dim;
    let all = ["c_ok", "c_len", "c_nan", "c_dim", "c_src"];
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        dim = probe_dim(&mut srv, 1);
        for (i, id) in all.iter().enumerate() {
            let r = srv.ok(
                10 + i as u64,
                "splatsdb_store",
                serde_json::json!({
                    "id": id,
                    "text": format!("doc {id}"),
                    "embedding": emb(dim, 1.0, 0.1 * i as f64)
                }),
            );
            assert_eq!(r["persisted"], serde_json::json!(true), "{r}");
        }
        srv.shutdown();
    }
    // Corrupt four blobs in place and snapshot all five rows.
    let snapshot: Vec<(String, Option<Vec<u8>>, Option<String>)> = {
        let db = open_db(&doc_db);
        db.execute(
            "UPDATE documents SET embedding=?1 WHERE id=?2",
            params![vec![1u8, 2, 3, 4, 5, 6, 7], "c_len"],
        )
        .expect("tamper c_len");
        let mut nan = vec![0u8; dim * 4];
        nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        db.execute(
            "UPDATE documents SET embedding=?1 WHERE id=?2",
            params![nan, "c_nan"],
        )
        .expect("tamper c_nan");
        db.execute(
            "UPDATE documents SET embedding=?1 WHERE id=?2",
            params![vec![0u8; (dim + 1) * 4], "c_dim"],
        )
        .expect("tamper c_dim");
        db.execute(
            "UPDATE documents SET embedding_source=?1 WHERE id=?2",
            params!["tensor", "c_src"],
        )
        .expect("tamper c_src");
        all.iter()
            .map(|id| {
                let r = stored_row(&db, id);
                (id.to_string(), r.blob, r.source)
            })
            .collect()
    };
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 1/5 documents");
        srv.expect_log("stored vectors loaded: 1");
        srv.expect_log("re-embedded: 0");
        srv.expect_log("quarantined: 4");
        for (id, why) in [
            ("c_len", "not a multiple of 4"),
            ("c_nan", "non-finite"),
            ("c_dim", "dimension mismatch"),
            ("c_src", "unknown provenance label"),
        ] {
            let found = srv
                .err_log
                .iter()
                .any(|l| l.contains("QUARANTINED") && l.contains(&format!("{id}: {why}")));
            assert!(
                found,
                "no diagnostic for {id} ({why}); log:\n{}",
                srv.err_log.join("\n")
            );
        }
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 10 }),
        );
        assert_eq!(
            ids(&res),
            vec!["c_ok".to_string()],
            "only the valid vector is searchable: {res}"
        );
        let st = srv.status(2);
        assert_eq!(st["n_active"], serde_json::json!(1), "{st}");
        assert_eq!(
            st["doc_count"],
            serde_json::json!(5),
            "no document deleted: {st}"
        );
        let ws = &st["vector_persistence"]["warm_start"];
        assert_eq!(ws["quarantined"], serde_json::json!(4), "{st}");
        assert_eq!(ws["unindexed"], serde_json::json!(0), "{st}");
        let g = srv.ok(3, "splatsdb_doc_get", serde_json::json!({ "id": "c_nan" }));
        assert_eq!(
            g["text"],
            serde_json::json!("doc c_nan"),
            "a quarantined vector must not make the document unreadable: {g}"
        );
        srv.shutdown();
    }
    // Nothing was resized, truncated, NULLed or re-labelled on disk.
    {
        let db = open_db(&doc_db);
        for (id, blob, source) in &snapshot {
            let r = stored_row(&db, id);
            assert_eq!(r.blob, *blob, "{id}: the blob was rewritten");
            assert_eq!(r.source, *source, "{id}: the label was rewritten");
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// (4) doc_del / re-store (with the F05a tombstones)
// ---------------------------------------------------------------------------

/// `doc_del` retires vector + provenance in the same UPDATE as the
/// tombstone; a re-store under the same id replaces them; a restart loads
/// exactly what survived.
#[test]
fn p05_doc_del_retires_vector_and_provenance_re_store_replaces_it() {
    let tmp = scratch("p05_del");
    let doc_db = tmp.join("docs.db");
    let dim;
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        dim = probe_dim(&mut srv, 1);
        srv.ok(
            10,
            "splatsdb_store",
            serde_json::json!({
                "id": "d_1",
                "text": "first version of d_1",
                "embedding": emb(dim, 1.0, 0.0)
            }),
        );
        srv.ok(
            11,
            "splatsdb_store",
            serde_json::json!({ "id": "d_2", "text": "doc two", "embedding": emb(dim, 0.9, 0.1) }),
        );
        srv.ok(12, "splatsdb_doc_del", serde_json::json!({ "id": "d_1" }));
        let st = srv.status(13);
        assert_eq!(st["live_vectors"], serde_json::json!(1), "{st}");
        assert_eq!(st["tombstoned_vectors"], serde_json::json!(1), "{st}");
        let res = srv.ok(
            14,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 5 }),
        );
        assert_eq!(ids(&res), vec!["d_2".to_string()], "{res}");
        srv.shutdown();
    }
    let a2 = provided(dim, 0.9, 0.1);
    let b1 = provided(dim, 0.0, 1.0);
    {
        let db = open_db(&doc_db);
        let r = stored_row(&db, "d_1");
        assert_eq!(r.deleted, 1, "the F05a tombstone is kept: {r:?}");
        assert!(
            r.blob.is_none() && r.source.is_none(),
            "doc_del must retire vector + provenance: {r:?}"
        );
        let want = f32_bytes(&a2);
        assert_eq!(
            stored_row(&db, "d_2").blob.as_deref(),
            Some(want.as_slice())
        );
        assert_eq!(stored_row(&db, "d_2").source.as_deref(), Some("provided"));
    }
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 1/2 documents");
        srv.expect_log("re-embedded: 0");
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 5 }),
        );
        assert_eq!(ids(&res), vec!["d_2".to_string()], "{res}");
        // Re-store under the SAME id: exactly one vector, the new one.
        let r = srv.ok(
            2,
            "splatsdb_store",
            serde_json::json!({
                "id": "d_1",
                "text": "second version of d_1",
                "embedding": emb(dim, 0.0, 1.0)
            }),
        );
        assert_eq!(r["persisted"], serde_json::json!(true), "{r}");
        let st = srv.status(3);
        assert_eq!(st["live_vectors"], serde_json::json!(2), "{st}");
        assert_eq!(st["doc_count"], serde_json::json!(2), "{st}");
        let res = srv.ok(
            4,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 0.0, 1.0), "top_k": 5 }),
        );
        assert_eq!(res["results"][0]["id"], serde_json::json!("d_1"), "{res}");
        assert_eq!(
            res["results"][0]["text"],
            serde_json::json!("second version of d_1"),
            "{res}"
        );
        assert_eq!(
            res["results"][0]["embedding_source"],
            serde_json::json!("provided"),
            "{res}"
        );
        assert_eq!(
            res["results"][1]["embedding_source"],
            serde_json::json!("stored_provided"),
            "{res}"
        );
        let mut got = ids(&res);
        got.sort();
        assert_eq!(got, vec!["d_1", "d_2"], "{res}");
        srv.shutdown();
    }
    {
        let db = open_db(&doc_db);
        let r = stored_row(&db, "d_1");
        assert_eq!(r.deleted, 0, "{r:?}");
        let want = f32_bytes(&b1);
        assert_eq!(
            r.blob.as_deref(),
            Some(want.as_slice()),
            "the re-store must replace the vector"
        );
        assert_eq!(r.source.as_deref(), Some("provided"));
    }
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 2/2 documents");
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 0.0, 1.0), "top_k": 5 }),
        );
        assert_eq!(res["results"][0]["id"], serde_json::json!("d_1"), "{res}");
        assert_eq!(
            res["results"][0]["embedding_source"],
            serde_json::json!("stored_provided"),
            "{res}"
        );
        let mut got = ids(&res);
        got.sort();
        assert_eq!(got, vec!["d_1", "d_2"], "{res}");
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// (1) Transaction / crash evidence
// ---------------------------------------------------------------------------

/// Document and vector live in ONE SQLite statement. When SQLite aborts it
/// (injected failure inside the write) neither exists afterwards — no
/// document without vector, no vector without document; and when the process
/// is SIGKILLed right after the durable `stored` answer, both are there on
/// the next start.
#[test]
fn p05_failed_or_killed_write_leaves_neither_doc_without_vector_nor_vector_without_doc() {
    let tmp = scratch("p05_crash");
    let doc_db = tmp.join("docs.db");
    let dim;
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        dim = probe_dim(&mut srv, 1);
        srv.ok(
            10,
            "splatsdb_store",
            serde_json::json!({ "id": "k_1", "text": "doc one", "embedding": emb(dim, 0.0, 1.0) }),
        );
        srv.shutdown();
    }
    // Injected failure INSIDE the write: SQLite aborts any insert of
    // `k_abort`, so the statement that carries document + vector fails whole.
    {
        let db = open_db(&doc_db);
        db.execute_batch(
            "CREATE TRIGGER f05b_abort BEFORE INSERT ON documents WHEN NEW.id='k_abort'
             BEGIN SELECT RAISE(ABORT,'injected f05b failure'); END;",
        )
        .expect("install trigger");
    }
    {
        // SimCos opt-in only so doc_add can produce a vector while the
        // service is down; a STORED vector still takes precedence over it.
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[("SPLATSDB_ALLOW_SIMCOS", "1")]);
        srv.expect_log("reloaded 1/1 documents");
        srv.expect_log("re-embedded: 0");
        let (is_error, payload) = srv.tool(
            1,
            "splatsdb_doc_add",
            serde_json::json!({ "id": "k_abort", "text": "must not survive" }),
        );
        assert!(is_error, "doc_add must report the failed write: {payload}");
        let r = srv.ok(
            2,
            "splatsdb_store",
            serde_json::json!({
                "id": "k_store_abort",
                "text": "must not persist",
                "embedding": emb(dim, 1.0, 0.0)
            }),
        );
        assert_eq!(
            r["persisted"],
            serde_json::json!(false),
            "a failed SQLite write can never be reported as persisted: {r}"
        );
        assert_eq!(r["ephemeral"], serde_json::json!(false), "{r}");
        let st = srv.status(3);
        assert_eq!(st["live_vectors"], serde_json::json!(2), "{st}");
        assert_eq!(
            st["doc_count"],
            serde_json::json!(1),
            "SQLite holds only k_1: {st}"
        );
        let res = srv.ok(
            4,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 10 }),
        );
        assert_eq!(
            ids(&res),
            vec!["k_store_abort".to_string(), "k_1".to_string()],
            "k_abort (unmapped by F05a) must not be searchable: {res}"
        );
        srv.shutdown();
    }
    {
        let db = open_db(&doc_db);
        let n: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE id IN ('k_abort','k_store_abort')",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(
            n, 0,
            "the aborted statement left a row behind (document without vector or vector without document)"
        );
        let r = stored_row(&db, "k_1");
        assert_eq!(r.blob.as_ref().map(Vec::len), Some(dim * 4));
    }
    // Restart (strict, service down): only k_1 comes back, from its stored
    // vector. Then a crash right after a durable `stored` answer.
    let k2 = provided(dim, 1.0, 0.0);
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 1/1 documents");
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 10 }),
        );
        assert_eq!(ids(&res), vec!["k_1".to_string()], "{res}");
        let r = srv.ok(
            2,
            "splatsdb_store",
            serde_json::json!({
                "id": "k_2",
                "text": "written right before the crash",
                "embedding": emb(dim, 1.0, 0.0)
            }),
        );
        assert_eq!(r["persisted"], serde_json::json!(true), "{r}");
        srv.kill_now();
    }
    {
        let db = open_db(&doc_db);
        let r = stored_row(&db, "k_2");
        let want = f32_bytes(&k2);
        assert_eq!(r.deleted, 0, "{r:?}");
        assert_eq!(
            r.blob.as_deref(),
            Some(want.as_slice()),
            "the vector acknowledged as stored was lost by the crash: {r:?}"
        );
        assert_eq!(r.source.as_deref(), Some("provided"));
    }
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 2/2 documents");
        srv.expect_log("re-embedded: 0");
        let res = srv.ok(
            1,
            "splatsdb_search",
            serde_json::json!({ "query": "q", "embedding": emb(dim, 1.0, 0.0), "top_k": 5 }),
        );
        assert_eq!(res["results"][0]["id"], serde_json::json!("k_2"), "{res}");
        assert_eq!(
            res["results"][0]["text"],
            serde_json::json!("written right before the crash"),
            "{res}"
        );
        let mut got = ids(&res);
        got.sort();
        assert_eq!(got, vec!["k_1", "k_2"], "{res}");
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// A04-F14 — spatial metadata of `splatsdb_store`
// ---------------------------------------------------------------------------

/// wing/room/hall given to `splatsdb_store` are persisted exactly like
/// `doc_add` does: they survive a restart, `spatial_info` sees them and the
/// filtered search returns the document. A re-store without spatial keys
/// drops them on disk as well.
#[test]
fn p05_store_wing_room_hall_survives_restart() {
    let tmp = scratch("p05_spatial");
    let doc_db = tmp.join("docs.db");
    let dim;
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        dim = probe_dim(&mut srv, 1);
        let r = srv.ok(
            10,
            "splatsdb_store",
            serde_json::json!({
                "id": "s_wing",
                "text": "doc stored with spatial keys",
                "wing": "s_wing",
                "room": "s_room",
                "hall": "s_hall",
                "embedding": emb(dim, 1.0, 0.0)
            }),
        );
        assert_eq!(r["persisted"], serde_json::json!(true), "{r}");
        srv.ok(
            11,
            "splatsdb_store",
            serde_json::json!({
                "id": "s_plain",
                "text": "doc stored without spatial keys",
                "embedding": emb(dim, 0.9, 0.1)
            }),
        );
        srv.shutdown();
    }
    {
        let db = open_db(&doc_db);
        let meta: String = db
            .query_row(
                "SELECT metadata FROM documents WHERE id='s_wing'",
                [],
                |r| r.get(0),
            )
            .expect("metadata");
        let meta: serde_json::Value = serde_json::from_str(&meta).expect("metadata json");
        assert_eq!(meta["wing"], serde_json::json!("s_wing"), "{meta}");
        assert_eq!(meta["room"], serde_json::json!("s_room"), "{meta}");
        assert_eq!(meta["hall"], serde_json::json!("s_hall"), "{meta}");
    }
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 2/2 documents");
        srv.expect_log("registered 1 spatial metadata entries");
        let info = srv.ok(1, "splatsdb_spatial_info", serde_json::json!({}));
        assert_eq!(
            info["documents_with_spatial_metadata"],
            serde_json::json!(1),
            "{info}"
        );
        assert_eq!(
            info["wings"],
            serde_json::json!({ "s_wing": ["s_room"] }),
            "{info}"
        );
        assert_eq!(info["halls"], serde_json::json!(["s_hall"]), "{info}");
        let res = srv.ok(
            2,
            "splatsdb_spatial_search",
            serde_json::json!({
                "query": "q",
                "wing": "s_wing",
                "embedding": emb(dim, 1.0, 0.0),
                "top_k": 5
            }),
        );
        assert_eq!(
            ids(&res),
            vec!["s_wing".to_string()],
            "filtered search after the restart: {res}"
        );
        // A re-store without spatial keys drops the registration — on disk too.
        srv.ok(
            3,
            "splatsdb_store",
            serde_json::json!({
                "id": "s_wing",
                "text": "re-stored without spatial keys",
                "embedding": emb(dim, 1.0, 0.0)
            }),
        );
        let res = srv.ok(
            4,
            "splatsdb_spatial_search",
            serde_json::json!({
                "query": "q",
                "wing": "s_wing",
                "embedding": emb(dim, 1.0, 0.0),
                "top_k": 5
            }),
        );
        assert!(ids(&res).is_empty(), "{res}");
        srv.shutdown();
    }
    {
        let mut srv = spawn_server(&doc_db, NO_SERVICE_URL, &[]);
        srv.expect_log("reloaded 2/2 documents");
        let info = srv.ok(1, "splatsdb_spatial_info", serde_json::json!({}));
        assert_eq!(
            info["documents_with_spatial_metadata"],
            serde_json::json!(0),
            "{info}"
        );
        srv.shutdown();
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
