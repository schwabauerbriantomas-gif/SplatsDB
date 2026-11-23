//! Shared black-box helpers for the F02f CLI durability tests.
//!
//! `cli_energy`, `cli_write_lock` and `cli_hnsw_persist` include this file
//! with `#[path]`; cargo never compiles it on its own. Every test drives the
//! real `splatsdb` binary. A `serve` binds ONLY to the probe range
//! 18121-18129 (never production ports) on a scratch data-dir under
//! `$TMPDIR`; every child is terminated by PID, also on unwind, so a failing
//! assertion never leaves an orphan on a probe port.

// Each of the three test binaries uses a different subset of this file.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::json;

pub const DIM: usize = 32;
const PROBE_BASE: u16 = 18121;
const PROBE_COUNT: usize = 9;
static PROBE_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Next port of the probe range; panics if the range is exhausted or the
/// port is taken, so a test can never fall back to an arbitrary (production)
/// port.
pub fn probe_port() -> u16 {
    let n = PROBE_NEXT.fetch_add(1, Ordering::SeqCst);
    assert!(
        n < PROBE_COUNT,
        "probe port range {PROBE_BASE}..={} exhausted",
        PROBE_BASE + PROBE_COUNT as u16 - 1
    );
    let port = PROBE_BASE + n as u16;
    assert!(
        TcpStream::connect(("127.0.0.1", port)).is_err(),
        "probe port {port} is already in use"
    );
    port
}

pub fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splatdb-f02f-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `[u64 rows][u64 cols][f32 LE]`: the on-disk shard / `.bin` ingest format,
/// filled with a deterministic ramp.
pub fn write_matrix(path: &Path, rows: usize, cols: usize, base: f32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut buf = Vec::with_capacity(16 + rows * cols * 4);
    buf.extend_from_slice(&(rows as u64).to_le_bytes());
    buf.extend_from_slice(&(cols as u64).to_le_bytes());
    for i in 0..rows * cols {
        buf.extend_from_slice(&(base + i as f32 * 0.01).to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

/// `[u64 rows][u64 cols][f32 LE]` from explicit rows.
pub fn write_rows(path: &Path, rows: &[Vec<f32>]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let cols = rows[0].len();
    let mut buf = Vec::with_capacity(16 + rows.len() * cols * 4);
    buf.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(cols as u64).to_le_bytes());
    for r in rows {
        for x in r {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, buf).unwrap();
}

pub fn shard_path(dir: &Path) -> PathBuf {
    dir.join("data").join("vectors").join("default.bin")
}

pub fn hnsw_path(dir: &Path) -> PathBuf {
    dir.join("hnsw_index.bin")
}

pub fn landscape_path(dir: &Path) -> PathBuf {
    dir.join("energy").join("landscape.bin")
}

pub fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Rows of a `[u64 rows][u64 cols][f32 LE]` file.
pub fn read_bin(path: &Path) -> Vec<Vec<f32>> {
    let b = std::fs::read(path).unwrap();
    let rows = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    let cols = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| f32_at(&b, 16 + (r * cols + c) * 4))
                .collect()
        })
        .collect()
}

/// The F02e1 corpus: six well-separated cluster centres with a different
/// spread each — so `ingest` derives a different kappa per cluster — and
/// eight noisy chunks around every centre, in `DIM` dimensions.
pub fn centres() -> Vec<Vec<f32>> {
    const CENTRES: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.1],
        [-1.3, 0.7, 0.0],
        [0.6, -1.4, 0.0],
    ];
    CENTRES
        .iter()
        .map(|c| {
            let mut v = vec![0.0f32; DIM];
            v[..3].copy_from_slice(c);
            v
        })
        .collect()
}

/// The raw chunks `ingest` clusters: what `centres` describes, with noise.
pub fn clustered_corpus() -> Vec<Vec<f32>> {
    const SPREADS: [f64; 6] = [0.02, 0.5, 0.1, 0.15, 0.3, 0.05];
    // xorshift64* + Box–Muller: deterministic without an RNG crate.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut normal = |sigma: f64| -> f64 {
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let x = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((x >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let u = next().max(1e-12);
        let v = next();
        sigma * (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let mut rows = Vec::with_capacity(6 * 8);
    for (c, centre) in centres().iter().enumerate() {
        for _ in 0..8 {
            rows.push(
                centre
                    .iter()
                    .map(|m| m + normal(SPREADS[c]) as f32)
                    .collect::<Vec<f32>>(),
            );
        }
    }
    rows
}

/// A `serve` child that is SIGKILLed by PID on unwind if still running. Its
/// stderr goes to `log` (a pipe would block the server once full).
pub struct Running {
    pub child: Option<Child>,
    pub log: PathBuf,
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = Command::new("kill")
                    .args(["-9", &child.id().to_string()])
                    .status();
                let _ = child.wait();
            }
        }
    }
}

pub fn spawn_serve(data_dir: &Path, log: &Path, port: u16) -> Running {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_splatsdb"));
    cmd.arg("--data-dir")
        .arg(data_dir)
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("serve")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env_remove("SPLATSDB_API_KEY")
        .env_remove("SPLATSDB_ALLOW_UNAUTHENTICATED")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(std::fs::File::create(log).unwrap()));
    let child = cmd.spawn().expect("spawn splatsdb serve");
    eprintln!(
        "[f02f] serve pid={} port={port} data-dir={}",
        child.id(),
        data_dir.display()
    );
    Running {
        child: Some(child),
        log: log.to_path_buf(),
    }
}

pub fn stderr_of(server: &Running) -> String {
    std::fs::read_to_string(&server.log).unwrap_or_default()
}

/// SIGTERM by PID (graceful shutdown: shard persisted, WAL truncated), then
/// SIGKILL by PID only if it is still alive after 60 s. Returns the exit code.
pub fn shutdown(server: &mut Running) -> Option<i32> {
    let child = server.child.as_mut().expect("server already shut down");
    let _ = Command::new("kill")
        .args(["-s", "TERM", &child.id().to_string()])
        .status();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                server.child = None;
                return status.code();
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            _ => {
                let _ = Command::new("kill")
                    .args(["-9", &child.id().to_string()])
                    .status();
                let code = child.wait().ok().and_then(|s| s.code());
                server.child = None;
                return code;
            }
        }
    }
}

fn dechunk(mut rest: &[u8]) -> String {
    let mut out = Vec::new();
    while let Some(nl) = rest.windows(2).position(|w| w == b"\r\n".as_slice()) {
        let size =
            usize::from_str_radix(String::from_utf8_lossy(&rest[..nl]).trim(), 16).unwrap_or(0);
        rest = &rest[nl + 2..];
        if size == 0 || rest.len() < size {
            break;
        }
        out.extend_from_slice(&rest[..size]);
        let after = &rest[size..];
        rest = after.strip_prefix(b"\r\n").unwrap_or(after);
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Minimal HTTP/1.1 client. Returns (status, body).
pub fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let body = body.unwrap_or("");
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(20)))
        .ok()?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).ok()?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n".as_slice())?;
    let headers = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
    let rest = &raw[split + 4..];
    let text = if headers.contains("transfer-encoding: chunked") {
        dechunk(rest)
    } else {
        String::from_utf8_lossy(rest).into_owned()
    };
    let status = String::from_utf8_lossy(&raw[..split])
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some((status, text))
}

pub fn wait_healthy(port: u16, server: &mut Running) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child.as_mut().unwrap().try_wait() {
            panic!(
                "serve on :{port} exited early with {status}; stderr:\n{}",
                stderr_of(server)
            );
        }
        if matches!(http(port, "GET", "/health", None), Some((200, _))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!(
        "serve on :{port} never answered /health; stderr:\n{}",
        stderr_of(server)
    );
}

/// `POST /store` of one embedding; asserts the 200.
pub fn serve_store(port: u16, embedding: &[f32]) {
    let body = json!({ "text": "f02f-new-vector", "embedding": embedding }).to_string();
    let (code, text) = http(port, "POST", "/store", Some(&body)).expect("/store answered");
    assert_eq!(code, 200, "POST /store on :{port}: {text}");
}

/// Wait for a child to exit; SIGKILL by PID and panic if it does not within
/// `timeout`. Returns (status, stdout, stderr).
pub fn wait_exit(mut child: Child, timeout: Duration) -> (ExitStatus, String, String) {
    let deadline = Instant::now() + timeout;
    let pid = child.id();
    let (status, timed_out) = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break (st, false);
        }
        if Instant::now() > deadline {
            let _ = Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status();
            break (child.wait().expect("wait after SIGKILL"), true);
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut out);
    }
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut err);
    }
    eprintln!(
        "[f02f] pid={pid} exited {status} timed_out={timed_out}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
    );
    assert!(
        !timed_out,
        "splatsdb pid {pid} did not exit within {timeout:?}; stdout:\n{out}\nstderr:\n{err}"
    );
    (status, out, err)
}

/// Run the real CLI with `--data-dir <data_dir> --dim <dim>` and the command
/// `args`; returns (status, stdout, stderr).
pub fn cli(data_dir: &Path, dim: usize, args: &[&str]) -> (ExitStatus, String, String) {
    let child = Command::new(env!("CARGO_BIN_EXE_splatsdb"))
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--dim")
        .arg(dim.to_string())
        .args(args)
        .env_remove("SPLATSDB_API_KEY")
        .env_remove("SPLATSDB_ALLOW_UNAUTHENTICATED")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn splatsdb");
    eprintln!(
        "[f02f] splatsdb --data-dir {} --dim {dim} {} (pid {})",
        data_dir.display(),
        args.join(" "),
        child.id()
    );
    wait_exit(child, Duration::from_secs(180))
}

/// The vector as a `--query` value: f32 `Display` round-trips exactly.
pub fn csv(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{x}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn query_of(dim: usize) -> String {
    csv(&vec![0.1; dim])
}
