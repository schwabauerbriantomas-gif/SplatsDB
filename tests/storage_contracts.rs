//! F02b — storage/HTTP/CLI lifecycle contracts (A03-F01/F04/F06/F12/F17).
//!
//! Black-box: every test drives the real `splatsdb` binary. Servers bind ONLY
//! to the probe range 18081-18089 (never production ports) on scratch
//! data-dirs under `$TMPDIR`; every child is terminated by PID, also on
//! unwind, so a failing assertion never leaves an orphan on a probe port.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const DIM: usize = 32;
const PROBE_BASE: u16 = 18081;
const PROBE_COUNT: usize = 9;
static PROBE_NEXT: AtomicUsize = AtomicUsize::new(0);

/// Next port of the probe range; panics if the range is exhausted, so a test
/// can never fall back to an arbitrary (production) port.
fn probe_port() -> u16 {
    let n = PROBE_NEXT.fetch_add(1, Ordering::SeqCst);
    assert!(
        n < PROBE_COUNT,
        "probe port range {PROBE_BASE}..={} exhausted",
        PROBE_BASE + PROBE_COUNT as u16 - 1
    );
    PROBE_BASE + n as u16
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splatdb-f02b-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `[u64 rows][u64 cols][f32 LE]`: the on-disk shard / `.bin` ingest format.
fn write_matrix(path: &Path, rows: usize, cols: usize, base: f32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut buf = Vec::with_capacity(16 + rows * cols * 4);
    buf.extend_from_slice(&(rows as u64).to_le_bytes());
    buf.extend_from_slice(&(cols as u64).to_le_bytes());
    for i in 0..rows * cols {
        buf.extend_from_slice(&(base + i as f32 * 0.01).to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

fn shard_path(dir: &Path) -> PathBuf {
    dir.join("data").join("vectors").join("default.bin")
}

fn wal_path(dir: &Path) -> PathBuf {
    dir.join("wal").join("store.wal")
}

/// A valid WAL (`[u32 BE len][JSON]`) with `n` "store" entries.
fn write_wal(dir: &Path, n: usize) {
    let mut buf = Vec::new();
    for i in 0..n {
        let vector: Vec<String> = (0..DIM)
            .map(|j| format!("{}", 0.01 * (i * DIM + j) as f32))
            .collect();
        let body = format!(
            r#"{{"lsn":{i},"timestamp":0.0,"operation":"store","data":{{"id":"w{i}","auto_id":{i},"idx":{i},"vector":[{}]}}}}"#,
            vector.join(",")
        );
        buf.extend_from_slice(&(body.len() as u32).to_be_bytes());
        buf.extend_from_slice(body.as_bytes());
    }
    std::fs::create_dir_all(wal_path(dir).parent().unwrap()).unwrap();
    std::fs::write(wal_path(dir), buf).unwrap();
}

/// Payloads of the intact entries of a WAL file.
fn wal_entries(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let n = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        if i + 4 + n > bytes.len() {
            break;
        }
        out.push(String::from_utf8_lossy(&bytes[i + 4..i + 4 + n]).into_owned());
        i += 4 + n;
    }
    out
}

/// Child that is SIGKILLed by PID on unwind if still running.
struct Running(Option<Child>);

impl Running {
    fn child(&mut self) -> &mut Child {
        self.0.as_mut().unwrap()
    }

    fn take(&mut self) -> Child {
        self.0.take().unwrap()
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = Command::new("kill")
                    .args(["-9", &child.id().to_string()])
                    .status();
                let _ = child.wait();
            }
        }
    }
}

fn spawn_serve(data_dir: &Path, port: u16) -> Child {
    let child = Command::new(env!("CARGO_BIN_EXE_splatsdb"))
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn splatsdb serve");
    eprintln!("[f02b] serve pid={} port={port} started", child.id());
    child
}

fn read_all(child: &mut Child) -> (String, String) {
    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut out);
    }
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut err);
    }
    (out, err)
}

/// Wait for the child to exit (SIGKILL + panic on timeout).
/// Returns (status, stdout, stderr).
fn finish(mut child: Child, timeout: Duration) -> (ExitStatus, String, String) {
    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let pid = child.id();
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if Instant::now() > deadline {
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
            timed_out = true;
            break child.wait().expect("wait after SIGKILL");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let (out, err) = read_all(&mut child);
    eprintln!("[f02b] pid={pid} exited {status} timed_out={timed_out}");
    assert!(
        !timed_out,
        "splatsdb pid {pid} did not exit within {timeout:?}; stderr:\n{err}"
    );
    (status, out, err)
}

fn terminate(server: &mut Running) {
    let pid = server.child().id().to_string();
    let _ = Command::new("kill").args(["-s", "TERM", &pid]).status();
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
fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
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

fn wait_healthy(port: u16, server: &mut Running) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child().try_wait() {
            let (out, err) = read_all(server.child());
            panic!("serve on :{port} exited early with {status}; stdout:\n{out}\nstderr:\n{err}");
        }
        if matches!(http(port, "GET", "/health", None), Some((200, _))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("serve on :{port} never answered /health");
}

fn store(port: u16, text: &str) -> (u16, String) {
    let body = format!(r#"{{"text":"{text}"}}"#);
    http(port, "POST", "/store", Some(&body)).expect("/store answered")
}

#[test]
fn a1_serve_fails_closed_on_an_unreadable_shard_and_touches_nothing() {
    let dir = scratch("a1");
    // 100 bytes of garbage: the header promises an impossible shape
    std::fs::create_dir_all(shard_path(&dir).parent().unwrap()).unwrap();
    std::fs::write(shard_path(&dir), [0x78u8; 100]).unwrap();
    write_wal(&dir, 3);
    let shard_before = std::fs::read(shard_path(&dir)).unwrap();
    let wal_before = std::fs::read(wal_path(&dir)).unwrap();
    let port = probe_port();
    let (status, _out, err) = finish(spawn_serve(&dir, port), Duration::from_secs(30));
    assert_ne!(
        status.code(),
        Some(0),
        "serve must not start on an unreadable shard; stderr:\n{err}"
    );
    assert!(err.contains("fail-closed"), "stderr:\n{err}");
    assert_eq!(
        std::fs::read(shard_path(&dir)).unwrap(),
        shard_before,
        "the unreadable shard must survive byte for byte"
    );
    assert_eq!(
        std::fs::read(wal_path(&dir)).unwrap(),
        wal_before,
        "the WAL must stay untouched when the shard never loaded"
    );
    assert!(
        TcpStream::connect(("127.0.0.1", port)).is_err(),
        "nothing may listen on the probe port"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a2_second_serve_on_the_same_data_dir_fails_fast_and_the_lock_dies_with_the_process() {
    let dir = scratch("a2");
    write_matrix(&shard_path(&dir), 5, DIM, 0.0);
    let p1 = probe_port();
    let p2 = probe_port();
    let mut first = Running(Some(spawn_serve(&dir, p1)));
    wait_healthy(p1, &mut first);

    // segundo serve sobre el MISMO data-dir
    let (status, _out, err) = finish(spawn_serve(&dir, p2), Duration::from_secs(30));
    assert_ne!(
        status.code(),
        Some(0),
        "a second serve must not start; stderr:\n{err}"
    );
    assert!(err.contains("bloqueado"), "stderr:\n{err}");
    assert!(TcpStream::connect(("127.0.0.1", p2)).is_err());
    // el primero sigue sirviendo
    assert!(matches!(http(p1, "GET", "/health", None), Some((200, _))));

    // SIGTERM: salida ordenada y el lock se va con el proceso
    terminate(&mut first);
    let (status, _out, err) = finish(first.take(), Duration::from_secs(60));
    assert!(
        status.success(),
        "clean shutdown must exit 0; stderr:\n{err}"
    );

    // el lockfile que quedó en disco NO es un bloqueo permanente
    assert!(
        dir.join(".splatsdb.lock").exists(),
        "the lock file is never removed"
    );
    let mut third = Running(Some(spawn_serve(&dir, p2)));
    wait_healthy(p2, &mut third);
    terminate(&mut third);
    let (status, _out, err) = finish(third.take(), Duration::from_secs(60));
    assert!(status.success(), "stderr:\n{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a3_shutdown_never_overwrites_a_shard_modified_outside_the_server() {
    let dir = scratch("a3");
    write_matrix(&shard_path(&dir), 5, DIM, 0.0);
    let port = probe_port();
    let mut server = Running(Some(spawn_serve(&dir, port)));
    wait_healthy(port, &mut server);

    // mutación externa válida (como haría la CLI): el shard pasa a 7 filas
    write_matrix(&shard_path(&dir), 7, DIM, 100.0);
    let external = std::fs::read(shard_path(&dir)).unwrap();
    for i in 0..3 {
        let (code, body) = store(port, &format!("doc {i}"));
        assert_eq!(code, 200, "{body}");
    }

    terminate(&mut server);
    let (status, _out, err) = finish(server.take(), Duration::from_secs(60));
    eprintln!("[f02b] a3 shutdown exit status: {status}");
    assert!(
        !status.success(),
        "a shutdown that could not persist must not exit 0 (status {status})"
    );
    assert!(err.contains("cambió en disco"), "stderr:\n{err}");
    assert_eq!(
        std::fs::read(shard_path(&dir)).unwrap(),
        external,
        "the external shard must not be overwritten by the stale in-memory state"
    );
    let kept = wal_entries(&wal_path(&dir))
        .iter()
        .filter(|e| e.contains(r#""operation":"store""#))
        .count();
    assert_eq!(kept, 3, "the accepted stores must stay in the WAL");

    // reapertura sobre ese data-dir: no se adivina, fail-closed
    let (status, _out, err) = finish(spawn_serve(&dir, port), Duration::from_secs(30));
    assert_ne!(
        status.code(),
        Some(0),
        "reopen must fail closed; stderr:\n{err}"
    );
    assert!(err.contains("cambió en disco"), "stderr:\n{err}");
    assert_eq!(std::fs::read(shard_path(&dir)).unwrap(), external);
    std::fs::remove_dir_all(&dir).ok();
}

use serde_json::Value;

fn status_json(port: u16) -> Value {
    let (code, body) = http(port, "POST", "/status", None).expect("/status answered");
    assert_eq!(code, 200, "{body}");
    serde_json::from_str(&body).unwrap_or_else(|e| panic!("/status is not JSON ({e}): {body}"))
}

#[test]
fn b1_store_is_acknowledged_before_it_is_durable_and_says_so() {
    let dir = scratch("b1");
    let port = probe_port();
    let mut server = Running(Some(spawn_serve(&dir, port)));
    wait_healthy(port, &mut server);

    let v = status_json(port);
    assert_eq!(
        v["persistence"], "wal",
        "a serve with --data-dir is never ephemeral: {v}"
    );
    assert_eq!(v["durable"], true, "nothing accepted yet: {v}");

    // CHECKPOINT_EVERY = 32: las primeras 31 quedan aceptadas en el WAL pero
    // sin fsync → la respuesta no puede afirmar durabilidad.
    for i in 1..32 {
        let (code, body) = store(port, &format!("doc {i}"));
        assert_eq!(code, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "stored", "{body}");
        assert_eq!(v["durable"], false, "store #{i} was not fsynced: {body}");
    }
    let v = status_json(port);
    assert_eq!(v["durable"], false, "{v}");
    assert_eq!(v["wal_pending"], 31, "{v}");

    // la 32ª hace el group commit: sync_all terminó ANTES de este 200
    let (code, body) = store(port, "doc 32");
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["durable"], true, "{body}");
    let v = status_json(port);
    assert_eq!(v["durable"], true, "{v}");
    assert_eq!(v["wal_pending"], 0, "{v}");

    terminate(&mut server);
    let (status, _out, err) = finish(server.take(), Duration::from_secs(60));
    assert!(
        status.success(),
        "clean shutdown must exit 0; stderr:\n{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn b2_serve_with_a_data_dir_never_starts_ephemeral_when_the_wal_cannot_open() {
    let dir = scratch("b2");
    write_matrix(&shard_path(&dir), 5, DIM, 0.0);
    // un DIRECTORIO donde tiene que estar el WAL
    std::fs::create_dir_all(wal_path(&dir)).unwrap();
    let port = probe_port();
    let (status, _out, err) = finish(spawn_serve(&dir, port), Duration::from_secs(30));
    assert_ne!(
        status.code(),
        Some(0),
        "serve must abort instead of answering /store 200 with no WAL; stderr:\n{err}"
    );
    assert!(err.contains("fail-closed"), "stderr:\n{err}");
    assert!(TcpStream::connect(("127.0.0.1", port)).is_err());
    std::fs::remove_dir_all(&dir).ok();
}
