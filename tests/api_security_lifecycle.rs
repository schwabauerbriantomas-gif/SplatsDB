//! F02c — API auth (A05-F10) and GPU dataset lifecycle (A03-F05) contracts
//! against the REAL `splatsdb` binary.
//!
//! Every server binds ONLY to a port of the probe range 18091-18099 (never a
//! production port) on a scratch data-dir under `$TMPDIR/F02c`, and every
//! child is terminated by PID — also on unwind — so a failing assertion never
//! leaves an orphan on a probe port. The two non-loopback cases that do
//! listen (explicit opt-in without a key / with a key) bind `0.0.0.0` for the
//! shortest possible window over an EMPTY scratch store.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const DIM: usize = 32;
const PROBE_BASE: u16 = 18091;
const PROBE_COUNT: usize = 9;
static PROBE_NEXT: AtomicUsize = AtomicUsize::new(0);

const KEY_ENV: &str = "SPLATSDB_API_KEY";
const OPTIN_ENV: &str = "SPLATSDB_ALLOW_UNAUTHENTICATED";
const KEY: &str = "f02c-probe-key";

/// Next free port of the probe range; panics if the range is exhausted or
/// something already listens on the candidate, so a test can never fall back
/// to an arbitrary (production) port.
fn probe_port() -> u16 {
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

/// Scratch data-dir, always under `$TMPDIR/F02c`.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("F02c").join(format!(
        "{tag}-{}-{}",
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
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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

/// `(env var, value)` applied to the child; `None` REMOVES the variable, so a
/// key inherited from the test runner can never mask a "no key" case.
type Env<'a> = [(&'a str, Option<OsString>)];

fn no_key() -> Vec<(&'static str, Option<OsString>)> {
    vec![(KEY_ENV, None), (OPTIN_ENV, None)]
}

fn with_key(k: &str) -> Vec<(&'static str, Option<OsString>)> {
    vec![(KEY_ENV, Some(k.into())), (OPTIN_ENV, None)]
}

fn opt_in(v: &str) -> Vec<(&'static str, Option<OsString>)> {
    vec![(KEY_ENV, None), (OPTIN_ENV, Some(v.into()))]
}

fn spawn_serve(data_dir: &Path, port: u16, host: &str, gpu: bool, env: &Env) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_splatsdb"));
    cmd.arg("--data-dir")
        .arg(data_dir)
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("serve")
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string());
    if gpu {
        cmd.arg("--gpu");
    }
    for (k, v) in env {
        match v {
            Some(v) => {
                cmd.env(k, v);
            }
            None => {
                cmd.env_remove(k);
            }
        }
    }
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn splatsdb serve");
    eprintln!(
        "[f02c] serve pid={} port={port} host={host} gpu={gpu}",
        child.id()
    );
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
    eprintln!("[f02c] pid={pid} exited {status} timed_out={timed_out}");
    assert!(
        !timed_out,
        "splatsdb pid {pid} did not exit within {timeout:?}; stdout:\n{out}\nstderr:\n{err}"
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

/// Minimal HTTP/1.1 client with extra request headers. Returns (status, body).
fn http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    headers: &[(&str, &str)],
) -> Option<(u16, String)> {
    let body = body.unwrap_or("");
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(20)))
        .ok()?;
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(k);
        req.push_str(": ");
        req.push_str(v);
        req.push_str("\r\n");
    }
    req.push_str("\r\n");
    req.push_str(body);
    stream.write_all(req.as_bytes()).ok()?;
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
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = server.child().try_wait() {
            let (out, err) = read_all(server.child());
            panic!("serve on :{port} exited early with {status}; stdout:\n{out}\nstderr:\n{err}");
        }
        if matches!(http(port, "GET", "/health", None, &[]), Some((200, _))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("serve on :{port} never answered /health");
}

/// `POST /status` with an optional `Authorization` value → (status, JSON).
fn status(port: u16, authorization: Option<&str>) -> (u16, serde_json::Value) {
    let hdrs: Vec<(&str, &str)> = authorization
        .map(|a| vec![("Authorization", a)])
        .unwrap_or_default();
    let (code, body) = http(port, "POST", "/status", None, &hdrs).expect("/status answered");
    let v = serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body.clone()));
    (code, v)
}

/// Start `serve` and expect it to exit ON ITS OWN with a diagnosis, without
/// ever listening. Returns stderr for the caller's assertions.
fn serve_must_refuse(dir: &Path, port: u16, host: &str, env: &Env, tag: &str) -> String {
    let mut server = Running(Some(spawn_serve(dir, port, host, false, env)));
    let (status, out, err) = finish(server.take(), Duration::from_secs(30));
    eprintln!("[f02c] {tag}: exit={status}\n--- stdout ---\n{out}\n--- stderr ---\n{err}");
    assert_eq!(
        status.code(),
        Some(1),
        "{tag}: expected exit code 1, got {status}; stdout:\n{out}\nstderr:\n{err}"
    );
    assert!(
        !out.contains("listening"),
        "{tag}: a refused startup must never announce a listener; stdout:\n{out}"
    );
    assert!(
        TcpStream::connect(("127.0.0.1", port)).is_err(),
        "{tag}: something is listening on probe port {port}"
    );
    err
}

fn nothing_was_created(dir: &Path, tag: &str) {
    let entries: Vec<_> = std::fs::read_dir(dir).unwrap().collect();
    assert!(
        entries.is_empty(),
        "{tag}: a refused startup must not touch the data-dir; found {} entries",
        entries.len()
    );
}

#[test]
fn a1_empty_api_key_aborts_startup_with_a_diagnosis_and_never_listens() {
    let dir = scratch("a1");
    let port = probe_port();
    let env: Vec<(&str, Option<OsString>)> =
        vec![(KEY_ENV, Some(OsString::new())), (OPTIN_ENV, None)];
    let err = serve_must_refuse(&dir, port, "127.0.0.1", &env, "a1");
    assert!(
        err.contains("SPLATSDB_API_KEY") && err.contains("EMPTY"),
        "stderr must name the empty key: {err}"
    );
    nothing_was_created(&dir, "a1");
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(unix)]
#[test]
fn a2_non_utf8_api_key_aborts_startup_with_a_diagnosis_and_never_listens() {
    use std::os::unix::ffi::OsStringExt;
    let dir = scratch("a2");
    let port = probe_port();
    // 0xC3 0x28 is an ill-formed UTF-8 sequence (continuation byte missing).
    let env: Vec<(&str, Option<OsString>)> = vec![
        (KEY_ENV, Some(OsString::from_vec(vec![0xC3, 0x28, 0x41]))),
        (OPTIN_ENV, None),
    ];
    let err = serve_must_refuse(&dir, port, "127.0.0.1", &env, "a2");
    assert!(
        err.contains("SPLATSDB_API_KEY") && err.contains("UTF-8"),
        "stderr must name the non-UTF-8 key: {err}"
    );
    nothing_was_created(&dir, "a2");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a3_non_loopback_bind_without_a_key_is_refused() {
    let dir = scratch("a3");
    let port = probe_port();
    let err = serve_must_refuse(&dir, port, "0.0.0.0", &no_key(), "a3");
    assert!(
        err.contains("SPLATSDB_ALLOW_UNAUTHENTICATED") && err.contains("0.0.0.0"),
        "stderr must explain the opt-in and the address: {err}"
    );
    nothing_was_created(&dir, "a3");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a4_non_loopback_bind_without_a_key_needs_the_explicit_opt_in_to_serve() {
    // Both halves share one probe port: the first never binds (it exits
    // before listening), the second does.
    let port = probe_port();
    // Anything but exactly "1" is NOT an opt-in.
    let dir = scratch("a4-refuse");
    let err = serve_must_refuse(&dir, port, "0.0.0.0", &opt_in("true"), "a4-optin-true");
    assert!(err.contains("SPLATSDB_ALLOW_UNAUTHENTICATED"), "{err}");
    std::fs::remove_dir_all(&dir).ok();

    // Exactly "1" → serves, and keeps warning loudly about it.
    let dir = scratch("a4-serve");
    let mut server = Running(Some(spawn_serve(
        &dir,
        port,
        "0.0.0.0",
        false,
        &opt_in("1"),
    )));
    wait_healthy(port, &mut server);
    let (code, body) = status(port, None);
    assert_eq!(
        code, 200,
        "opted-in unauthenticated serve must answer: {body}"
    );
    terminate(&mut server);
    let (_, _, err) = finish(server.take(), Duration::from_secs(30));
    assert!(
        err.contains("WARNING") && err.contains("SPLATSDB_ALLOW_UNAUTHENTICATED"),
        "the opt-in must still warn on stderr: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a5_with_a_key_wrong_or_malformed_authorization_never_authenticates() {
    let dir = scratch("a5");
    let port = probe_port();
    let mut server = Running(Some(spawn_serve(
        &dir,
        port,
        "127.0.0.1",
        false,
        &with_key(KEY),
    )));
    wait_healthy(port, &mut server);
    let right = format!("Bearer {KEY}");
    for (label, value) in [
        ("no header", None),
        ("empty bearer", Some("Bearer ")),
        ("two spaces", Some("Bearer  f02c-probe-key")),
        ("lowercase scheme", Some("bearer f02c-probe-key")),
        ("wrong key", Some("Bearer wrong")),
        ("key prefix", Some("Bearer f02c-probe-ke")),
    ] {
        let (code, body) = status(port, value);
        assert_eq!(
            code, 401,
            "{label}: {value:?} must NOT authenticate ({body})"
        );
    }
    let (code, before) = status(port, Some(&right));
    assert_eq!(code, 200, "the right key must authenticate");
    let n_before = before["n_active"].as_u64().expect("n_active");

    // A rejected /store must not change the store.
    let body = r#"{"text":"must-not-be-stored"}"#;
    let (code, resp) = http(
        port,
        "POST",
        "/store",
        Some(body),
        &[("Authorization", "Bearer wrong")],
    )
    .expect("/store answered");
    assert_eq!(code, 401, "a wrong key must not store ({resp})");
    let (code, unchanged) = status(port, Some(&right));
    assert_eq!(code, 200);
    assert_eq!(unchanged["n_active"].as_u64(), Some(n_before));

    let (code, resp) = http(
        port,
        "POST",
        "/store",
        Some(body),
        &[("Authorization", &right)],
    )
    .expect("/store answered");
    assert_eq!(code, 200, "the right key stores: {resp}");
    let (code, after) = status(port, Some(&right));
    assert_eq!(code, 200);
    assert_eq!(after["n_active"].as_u64(), Some(n_before + 1));

    // /health stays open (liveness probe).
    assert_eq!(
        http(port, "GET", "/health", None, &[]).expect("health").0,
        200
    );

    terminate(&mut server);
    finish(server.take(), Duration::from_secs(30));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a6_loopback_without_a_key_keeps_serving() {
    let dir = scratch("a6");
    let port = probe_port();
    let mut server = Running(Some(spawn_serve(&dir, port, "127.0.0.1", false, &no_key())));
    wait_healthy(port, &mut server);
    let (code, body) = status(port, None);
    assert_eq!(code, 200, "loopback dev mode must keep working: {body}");
    terminate(&mut server);
    let (_, out, err) = finish(server.take(), Duration::from_secs(30));
    assert!(
        !err.contains("WARNING"),
        "loopback without a key must not warn: {err}"
    );
    assert!(out.contains("listening"), "{out}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a7_non_loopback_bind_with_a_key_requires_the_bearer() {
    let dir = scratch("a7");
    let port = probe_port();
    let mut server = Running(Some(spawn_serve(
        &dir,
        port,
        "0.0.0.0",
        false,
        &with_key(KEY),
    )));
    wait_healthy(port, &mut server);
    let (code, _) = status(port, None);
    assert_eq!(code, 401, "non-loopback with a key: no header → 401");
    let (code, _) = status(port, Some("Bearer nope"));
    assert_eq!(code, 401);
    let (code, _) = status(port, Some(&format!("Bearer {KEY}")));
    assert_eq!(code, 200);
    terminate(&mut server);
    finish(server.take(), Duration::from_secs(30));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a8_opt_in_with_a_key_configured_still_requires_the_bearer() {
    let dir = scratch("a8");
    let port = probe_port();
    let env: Vec<(&str, Option<OsString>)> =
        vec![(KEY_ENV, Some(KEY.into())), (OPTIN_ENV, Some("1".into()))];
    let mut server = Running(Some(spawn_serve(&dir, port, "127.0.0.1", false, &env)));
    wait_healthy(port, &mut server);
    let (code, _) = status(port, None);
    assert_eq!(
        code, 401,
        "the opt-in only concerns a MISSING key: with a key the bearer is mandatory"
    );
    let (code, _) = status(port, Some(&format!("Bearer {KEY}")));
    assert_eq!(code, 200);
    terminate(&mut server);
    let (_, _, err) = finish(server.take(), Duration::from_secs(30));
    assert!(
        !err.contains("Authentication is DISABLED"),
        "a configured key must never be reported as disabled: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ─── GPU dataset lifecycle (A03-F05) ───────────────────────────

#[cfg(feature = "cuda")]
fn search(port: u16, embedding: &[f32], top_k: usize) -> (u16, String) {
    let body =
        serde_json::json!({"query": "probe", "embedding": embedding, "top_k": top_k}).to_string();
    http(port, "POST", "/search", Some(&body), &[]).expect("/search answered")
}

/// `serve --gpu` boots on a 16-vector shard (so the exact GPU index IS built),
/// then 10 `/store` land. Every stored vector must immediately be the top-1
/// for its own embedding — before the fix every `/search` after the first
/// `/store` was answered from the frozen startup dataset.
#[cfg(feature = "cuda")]
#[test]
fn g1_serve_gpu_search_sees_every_stored_vector_without_a_restart() {
    if !splatsdb::gpu::is_cuda_available() {
        eprintln!("[f02c] g1 SKIPPED: CUDA is not available on this host");
        return;
    }
    let dir = scratch("g1");
    let port = probe_port();
    const N0: usize = 16;
    write_matrix(
        &dir.join("data").join("vectors").join("default.bin"),
        N0,
        DIM,
        0.0,
    );
    let mut server = Running(Some(spawn_serve(&dir, port, "127.0.0.1", true, &no_key())));
    wait_healthy(port, &mut server);

    // Unmutated store: the startup fast path answers (backend == "gpu").
    let probe: Vec<f32> = vec![0.5; DIM];
    let (code, body) = search(port, &probe, 3);
    assert_eq!(code, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("search json");
    assert_eq!(
        v["backend"], "gpu",
        "expected the startup GPU fast path on the unmutated store: {body}"
    );
    assert_eq!(v["results"].as_array().map(Vec::len), Some(3), "{body}");

    for i in 0..10usize {
        let mut emb = vec![0.0f32; DIM];
        // Far from the startup cloud (values 0.00 .. 5.11): unique top-1.
        emb[i % DIM] = 60.0 + i as f32;
        let body =
            serde_json::json!({"text": format!("g1-vector-{i}"), "embedding": emb}).to_string();
        let (code, resp) = http(port, "POST", "/store", Some(&body), &[]).expect("/store answered");
        assert_eq!(code, 200, "/store #{i}: {resp}");
        let (code, resp) = search(port, &emb, 1);
        assert_eq!(code, 200, "{resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("search json");
        assert_eq!(
            v["results"][0]["index"].as_u64(),
            Some((N0 + i) as u64),
            "store #{i} is invisible to /search — the GPU snapshot is stale (A03-F05): {resp}"
        );
    }

    terminate(&mut server);
    let (_, _, err) = finish(server.take(), Duration::from_secs(30));
    assert!(
        err.contains("GPU exact index listo"),
        "the server never built its GPU index, so this run did not exercise the GPU path: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
