//! Characterization tests for the MCP server warm-start spatial registration
//! block (`run_mcp_server`, `src/mcp_server.rs`) — the span that the repowise
//! extract-method plan `refac4_c7b69cacc17990c0271a` moves into a helper.
//!
//! These tests pin the OBSERVABLE behavior of that block at the process level
//! (the only level it is reachable at: `run_mcp_server()` takes no parameters
//! and owns its own stdin loop). They must pass BEFORE and AFTER the
//! extract-method refactoring — that is what "characterization" means here.
//!
//! Pinned behavior:
//!   1. A doc whose SQLite metadata carries wing/room/hall is re-registered
//!      during warm start after a restart; `splatsdb_spatial_search` finds it
//!      through the wing filter.
//!   2. The exact stderr message "Warm start: registered N spatial metadata
//!      entries ... (SPLATDB_SPATIAL_WARMSTART=on)" is emitted when enabled.
//!   3. With SPLATDB_SPATIAL_WARMSTART=0 the exact "spatial registration
//!      skipped" message is emitted and the wing filter matches nothing.
//!
//! The assertions are embedder-independent on purpose: spatial registration is
//! metadata-driven, so they hold for both the real embed service and the
//! SimCos fallback used in CI.

use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

const RESP_TIMEOUT: Duration = Duration::from_secs(120);
/// Server startup takes ~30s on this machine (store init is CPU-bound), so
/// warm-start stderr messages must be waited for generously.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout_rx: Receiver<String>,
    stderr_rx: Receiver<String>,
}

fn spawn_server(doc_db: &std::path::Path, skip_spatial: bool) -> Server {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_splatsdb"));
    cmd.arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("SPLATSDB_DOC_PATH", doc_db);
    if skip_spatial {
        cmd.env("SPLATDB_SPATIAL_WARMSTART", "0");
    }
    let mut child = cmd.spawn().expect("failed to spawn `splatsdb mcp`");
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let (tx_out, rx_out) = mpsc::channel();
    thread::spawn(move || {
        let reader = std::io::BufReader::new(stdout);
        for line in std::io::BufRead::lines(reader) {
            match line {
                Ok(l) => {
                    if tx_out.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let (tx_err, rx_err) = mpsc::channel();
    thread::spawn(move || {
        let reader = std::io::BufReader::new(stderr);
        for line in std::io::BufRead::lines(reader) {
            match line {
                Ok(l) => {
                    if tx_err.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    Server {
        child,
        stdin,
        stdout_rx: rx_out,
        stderr_rx: rx_err,
    }
}

impl Server {
    /// Send a `tools/call` request and return the parsed tool result JSON
    /// (the value inside `result.content[0].text`).
    fn call_tool(&mut self, id: u64, name: &str, args: serde_json::Value) -> serde_json::Value {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        });
        self.stdin
            .write_all(req.to_string().as_bytes())
            .expect("write request");
        self.stdin.write_all(b"\n").expect("write newline");
        self.stdin.flush().expect("flush stdin");

        let deadline = Instant::now() + RESP_TIMEOUT;
        loop {
            if Instant::now() > deadline {
                panic!("timeout waiting for tools/call response id={id} ({name})");
            }
            let line = match self.stdout_rx.recv_timeout(RESP_TIMEOUT) {
                Ok(l) => l,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("stdout idle for {RESP_TIMEOUT:?} waiting for id={id} ({name})")
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("server stdout closed before answering id={id} ({name})")
                }
            };
            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue, // non-JSON line on stdout: skip
            };
            if v.get("id").and_then(|i| i.as_u64()) != Some(id) {
                continue;
            }
            let is_error = v
                .pointer("/result/isError")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            assert!(
                !is_error,
                "tools/call {name} returned isError=true: {}",
                v.pointer("/result/content/0/text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("?")
            );
            let text = v
                .pointer("/result/content/0/text")
                .and_then(|t| t.as_str())
                .unwrap_or_else(|| panic!("missing content text in response for {name}: {v}"));
            return serde_json::from_str(text).unwrap_or_else(|e| {
                panic!("tool result for {name} is not valid JSON: {e}: {text}")
            });
        }
    }

    /// Drain stderr for up to `timeout` looking for `needle`.
    /// Returns true as soon as a matching line is seen.
    fn wait_stderr_containing(&self, needle: &str, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            match self
                .stderr_rx
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(line) => {
                    if line.contains(needle) {
                        return true;
                    }
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        false
    }

    /// Close stdin (server exits on EOF) and reap the process.
    fn shutdown(mut self) {
        drop(self.stdin);
        let deadline = Instant::now() + Duration::from_secs(30);
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

#[test]
fn warm_start_spatial_registration_characterization() {
    let tmp = std::env::temp_dir().join(format!(
        "splatsdb_warmstart_char_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let doc_db = tmp.join("docs.db");

    // ---- Session A: store a doc carrying wing/room/hall, then stop. ----
    {
        let mut srv = spawn_server(&doc_db, false);
        let res = srv.call_tool(
            1,
            "splatsdb_doc_add",
            serde_json::json!({
                "id": "mem_char_1",
                "text": "characterization doc for warm start spatial registration",
                "metadata": "{\"wing\":\"char_wing\",\"room\":\"char_room\",\"hall\":\"char_hall\"}"
            }),
        );
        assert_eq!(
            res["ok"],
            serde_json::json!(true),
            "doc_add did not succeed: {res}"
        );
        srv.shutdown();
    }

    // ---- Session B: restart with warm start ON (default). ----
    // The doc must be re-registered into the spatial index and the exact
    // "registered N spatial metadata entries" line must appear on stderr.
    {
        let mut srv = spawn_server(&doc_db, false);
        assert!(
            srv.wait_stderr_containing("registered 1 spatial metadata entries", STARTUP_TIMEOUT),
            "warm start did not report spatial registration of the stored doc"
        );
        let res = srv.call_tool(
            2,
            "splatsdb_spatial_search",
            serde_json::json!({"query": "warm start", "wing": "char_wing", "top_k": 5}),
        );
        let results = res["results"]
            .as_array()
            .unwrap_or_else(|| panic!("spatial_search returned no results array: {res}"));
        assert_eq!(
            results.len(),
            1,
            "expected exactly 1 result for wing=char_wing after restart: {res}"
        );
        assert_eq!(
            results[0]["id"],
            serde_json::json!("mem_char_1"),
            "spatial search returned the wrong doc after restart: {res}"
        );
        srv.shutdown();
    }

    // ---- Session C: restart with SPLATDB_SPATIAL_WARMSTART=0. ----
    // The exact "spatial registration skipped" line must appear and the wing
    // filter must match nothing (metadata was not re-registered).
    {
        let mut srv = spawn_server(&doc_db, true);
        assert!(
            srv.wait_stderr_containing(
                "spatial registration skipped (SPLATDB_SPATIAL_WARMSTART=0)",
                STARTUP_TIMEOUT
            ),
            "skip mode did not emit the exact 'spatial registration skipped' message"
        );
        let res = srv.call_tool(
            3,
            "splatsdb_spatial_search",
            serde_json::json!({"query": "warm start", "wing": "char_wing", "top_k": 5}),
        );
        let results = res["results"].as_array().expect("results array");
        assert!(
            results.is_empty(),
            "spatial filter matched a doc although warm start was skipped: {res}"
        );
        srv.shutdown();
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
