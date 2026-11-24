//! F02d — `/search` id space, topic and confidence contracts with a `field/`
//! corpus, against the REAL `splatsdb` binary.
//!
//! With a `field/` data-dir (what `splatsdb ingest` leaves behind) the startup
//! GPU fast path of `serve --gpu` indexed the `field.raw` CHUNKS, while the CPU
//! reference, the GPU batch worker and `rescore=gaussian` search the store's
//! SPLATS. The same query on the same snapshot then answered chunk ids before
//! the first `/store` and splat ids after it, and every splat-space row got the
//! topic of ANOTHER splat (its id was looked up in the chunk → splat table).
//!
//! Contract pinned here (same snapshot ⇒ same rows on every route):
//! * `/search` ids are splat ids — the space `/store` appends to (WAL idx).
//!   Chunk ids stay the contract of `/field-search`.
//! * order: ascending L2; `score` is the squared L2 distance on every L2 route.
//! * `topic` is the splat the result IS; `confidence` is 1 − d(q, mu)/R_splat
//!   for L2 rows and is ABSENT for `rescore=gaussian` rows (their score is
//!   alpha·exp(−kappa·d²), not a distance).
//!
//! Every server binds a port of the probe range 18101-18109 (never a
//! production port) on a scratch data-dir under `$TMPDIR/F02d`, and every
//! child is terminated by PID — also on unwind.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const DIM: usize = 32;
/// Splats: rows of the `default` shard (mu) — the id space of `/search`.
const K: usize = 16;
/// Chunks per splat in the `raw` shard / field CSR.
const PER: usize = 2;
/// Chunks: rows of the `raw` shard — the id space of `/field-search`.
const N: usize = K * PER;
const TOP_K: usize = 5;
const PROBE_BASE: u16 = 18101;
const PROBE_COUNT: usize = 9;
static PROBE_NEXT: AtomicUsize = AtomicUsize::new(0);

/// `(index, score, topic, confidence)` of one `/search` row.
type Row = (u64, f64, Option<u32>, Option<f64>);

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

/// Scratch data-dir, always under `$TMPDIR/F02d`.
fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir()
        .join("F02d")
        .join(format!("{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ─── Deterministic field/ corpus ───────────────────────────────────────

/// Splat `j`: one axis, a distinct scale per splat in a scrambled order, so
/// the exact ranking is a non-trivial permutation of the ids.
fn splat(j: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[j] = 1.0 + 0.037 * ((7 * j + 3) % K) as f32;
    v
}

/// Splat that owns chunk `c`. `(7c + 3) % K == c` has no solution, so a splat
/// id mistaken for a chunk id yields a wrong owner for EVERY splat.
fn owner(c: usize) -> usize {
    (7 * c + 3) % K
}

/// Chunk `c`: its splat plus a small, non-zero offset.
fn chunk(c: usize) -> Vec<f32> {
    let mut v = splat(owner(c));
    v[(3 * c + 1) % DIM] += 0.05 * ((c % 5) as f32 - 2.0) + 0.017;
    v[(5 * c + 7) % DIM] += 0.025 * ((c % 3) as f32 - 1.0) - 0.011;
    v
}

/// Per-splat radius: the farthest member chunk (what `persist_field` writes).
fn radii() -> Vec<f32> {
    (0..K)
        .map(|j| {
            (0..N)
                .filter(|&c| owner(c) == j)
                .map(|c| {
                    chunk(c)
                        .iter()
                        .zip(splat(j))
                        .map(|(x, m)| (x - m) * (x - m))
                        .sum::<f32>()
                        .sqrt()
                })
                .fold(0.0f32, f32::max)
        })
        .collect()
}

/// Known query: splat 0 plus 0.01 on another axis — equal to no row, closer to
/// splat 0 than its radius (a non-trivial confidence) and tie-free.
fn query() -> Vec<f32> {
    let mut q = splat(0);
    q[1] += 0.01;
    q
}

/// Stored far away from the corpus: its own top-1, outside the query's top-k.
fn stored() -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 200.0;
    v[1] = 0.5;
    v[2] = 0.25;
    v
}

/// Squared L2 distance in f64 over the f32 rows.
fn d2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum()
}

/// Exact brute-force top-k over `rows`: `(row index, d²)`, ascending. The
/// contract is compared in ORDER, so a tie in the reference is a bug of the
/// corpus, not of the server.
fn exact_topk(rows: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (i, d2(r, q)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(k);
    for w in all.windows(2) {
        assert!(w[0].1 < w[1].1, "tie in the exact reference: {all:?}");
    }
    all
}

/// `[u64 rows][u64 cols][f32 LE]`: the on-disk shard format.
fn write_shard(path: &Path, rows: &[Vec<f32>]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut buf = Vec::with_capacity(16 + rows.len() * DIM * 4);
    buf.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(DIM as u64).to_le_bytes());
    for r in rows {
        for x in r {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, buf).unwrap();
}

/// The on-disk layout `splatsdb ingest` + `persist_field` leave behind: shard
/// `default` = the K splats, shard `raw` = the N chunks, the CSR memberships
/// (PER chunks per splat) and the per-splat radii. Returns the splats.
fn write_field_corpus(dir: &Path) -> Vec<Vec<f32>> {
    let mu: Vec<Vec<f32>> = (0..K).map(splat).collect();
    write_shard(&dir.join("data").join("vectors").join("default.bin"), &mu);
    write_shard(
        &dir.join("data").join("vectors").join("raw.bin"),
        &(0..N).map(chunk).collect::<Vec<_>>(),
    );
    let mut members = Vec::new();
    members.extend_from_slice(&(N as u64).to_le_bytes());
    members.extend_from_slice(&(K as u64).to_le_bytes());
    for c in 0..=K {
        members.extend_from_slice(&((c * PER) as u64).to_le_bytes());
    }
    for j in 0..K {
        for c in 0..N {
            if owner(c) == j {
                members.extend_from_slice(&(c as u32).to_le_bytes());
            }
        }
    }
    let mut radii_bin = Vec::new();
    radii_bin.extend_from_slice(&(K as u64).to_le_bytes());
    for r in radii() {
        radii_bin.extend_from_slice(&r.to_le_bytes());
    }
    std::fs::create_dir_all(dir.join("field")).unwrap();
    std::fs::write(dir.join("field").join("members.idx"), members).unwrap();
    std::fs::write(dir.join("field").join("radii.bin"), radii_bin).unwrap();
    mu
}

// ─── Row contract helpers ──────────────────────────────────────────────

fn rows(v: &Value) -> Vec<Row> {
    v["results"]
        .as_array()
        .unwrap_or_else(|| panic!("results array missing: {v}"))
        .iter()
        .map(|r| {
            (
                r["index"].as_u64().expect("index"),
                r["score"].as_f64().expect("score"),
                r["topic"].as_u64().and_then(|t| u32::try_from(t).ok()),
                r["confidence"].as_f64(),
            )
        })
        .collect()
}

/// One L2 route against the exact ranking of the same snapshot: ids and order
/// equal it, `score` is d², `topic` is the splat id and `confidence` follows
/// 1 − d/R_splat (absent when the splat has no radius).
fn assert_rows_match(rows: &[Row], exact: &[(usize, f64)], radii: &[f32], what: &str) {
    assert_eq!(
        rows.len(),
        exact.len(),
        "{what}: {} rows, the exact ranking has {}",
        rows.len(),
        exact.len()
    );
    for ((index, score, topic, confidence), (exact_index, d2)) in rows.iter().zip(exact) {
        assert_eq!(
            *index, *exact_index as u64,
            "{what}: ids/order differ from the exact splat ranking {exact:?}: {rows:?}"
        );
        assert!(
            (score - d2).abs() <= 1e-3 + 1e-4 * d2,
            "{what}: score {score} is not the squared L2 distance {d2}: {rows:?}"
        );
        assert_eq!(
            *topic,
            Some(*exact_index as u32),
            "{what}: topic must be the splat the result is: {rows:?}"
        );
        let expected = radii
            .get(*exact_index)
            .copied()
            .filter(|r| *r > 0.0)
            .map(|r| ((1.0 - d2.sqrt() / f64::from(r)).clamp(0.0, 1.0) * 1000.0).round() / 1000.0);
        match (confidence, expected) {
            (None, None) => {}
            (Some(got), Some(want)) => assert!(
                (got - want).abs() <= 0.002,
                "{what}: confidence {got}, expected 1 − d/R = {want}: {rows:?}"
            ),
            _ => panic!("{what}: confidence {confidence:?}, expected {expected:?}: {rows:?}"),
        }
    }
}

/// Two routes, same snapshot and mode: identical ids, order and topics; scores
/// and confidences equal up to f32 accumulation noise.
fn assert_rows_eq(a: &[Row], b: &[Row], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: {} vs {} rows", a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(
            (x.0, x.2),
            (y.0, y.2),
            "{what}: ids/order/topics differ: {a:?} vs {b:?}"
        );
        assert!(
            (x.1 - y.1).abs() <= 1e-3 + 1e-4 * y.1.abs(),
            "{what}: scores differ: {a:?} vs {b:?}"
        );
        match (x.3, y.3) {
            (None, None) => {}
            (Some(c1), Some(c2)) => assert!(
                (c1 - c2).abs() <= 0.002,
                "{what}: confidences differ: {a:?} vs {b:?}"
            ),
            _ => panic!("{what}: confidence presence differs: {a:?} vs {b:?}"),
        }
    }
}

// ─── Real server ───────────────────────────────────────────────────────

/// Child that is SIGKILLed by PID on unwind if still running. Its stderr goes
/// to `log` (a pipe would block the server once full).
struct Running {
    child: Option<Child>,
    log: PathBuf,
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

fn spawn_serve(data_dir: &Path, port: u16, gpu: bool) -> Running {
    let log = data_dir.join(if gpu {
        "serve-gpu.log"
    } else {
        "serve-cpu.log"
    });
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
        .stderr(Stdio::from(std::fs::File::create(&log).unwrap()));
    if gpu {
        cmd.arg("--gpu");
    }
    Running {
        child: Some(cmd.spawn().expect("spawn splatsdb serve")),
        log,
    }
}

fn stderr_of(server: &Running) -> String {
    std::fs::read_to_string(&server.log).unwrap_or_default()
}

/// SIGTERM by PID (graceful shutdown: shard persisted, WAL truncated), then
/// SIGKILL by PID only if it is still alive after 30 s. Returns the exit code.
fn shutdown(server: &mut Running) -> Option<i32> {
    let child = server.child.as_mut().expect("server already shut down");
    let _ = Command::new("kill")
        .args(["-s", "TERM", &child.id().to_string()])
        .status();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                server.child = None;
                return status.code();
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
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
fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<(u16, String)> {
    let body = body.unwrap_or("");
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(20)))
        .ok()?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
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

/// `POST /search` (optionally `?rescore=gaussian`) → the parsed response.
fn search(port: u16, gaussian: bool, embedding: &[f32], top_k: usize) -> Value {
    let path = if gaussian {
        "/search?rescore=gaussian"
    } else {
        "/search"
    };
    let body = json!({ "query": "f02d-probe", "embedding": embedding, "top_k": top_k }).to_string();
    let (code, text) = http(port, "POST", path, Some(&body))
        .unwrap_or_else(|| panic!("/search on :{port} did not answer"));
    assert_eq!(code, 200, "POST {path} on :{port}: {text}");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("POST {path} json: {e}: {text}"))
}

fn store(port: u16, embedding: &[f32]) {
    let body = json!({ "text": "f02d-stored", "embedding": embedding }).to_string();
    let (code, text) = http(port, "POST", "/store", Some(&body)).expect("/store answered");
    assert_eq!(code, 200, "POST /store on :{port}: {text}");
}

/// The stored vector is its own top-1 at the next splat id, on any route.
fn assert_stored_is_top_1(port: u16) {
    let hit = rows(&search(port, false, &stored(), 1));
    assert_eq!(hit.len(), 1, "top_k=1 must return one row: {hit:?}");
    assert_eq!(
        hit[0].0, K as u64,
        "the stored vector is not splat {K}: {hit:?}"
    );
    assert!(hit[0].1.abs() <= 1e-6, "d²(q, q) must be 0: {hit:?}");
    assert_eq!(
        hit[0].2,
        Some(K as u32),
        "topic of the stored splat: {hit:?}"
    );
}

/// With uniform alpha/kappa (a shard-loaded store) the gaussian ranking IS the
/// exact L2 ranking: same ids, same order.
fn assert_gaussian_ranking(rows: &[Row], exact: &[(usize, f64)], what: &str) {
    assert_eq!(rows.len(), exact.len(), "{what}: {rows:?}");
    for ((index, score, _, _), (exact_index, _)) in rows.iter().zip(exact) {
        assert_eq!(
            *index, *exact_index as u64,
            "{what}: ids/order differ from the exact splat ranking {exact:?}: {rows:?}"
        );
        assert!(
            *score > 0.0,
            "{what}: gaussian scores must be > 0: {rows:?}"
        );
    }
    assert!(
        rows.windows(2).all(|w| w[0].1 >= w[1].1),
        "{what}: gaussian scores must be descending: {rows:?}"
    );
}

// ─── CPU reference (every build) ───────────────────────────────────────

/// The CPU route with a field/ corpus: `/search` answers splat ids in exact
/// L2 order with the splat's own topic and confidence, before AND after a
/// `/store`. Before F02d every row carried `chunk_cluster[splat_id]` as its
/// topic (and that other splat's radius for the confidence).
#[test]
fn cpu_search_with_a_field_corpus_returns_splat_ids_with_their_own_topic() {
    let dir = scratch("cpu");
    let port = probe_port();
    let mu = write_field_corpus(&dir);
    let radii = radii();
    let (q, v) = (query(), stored());
    let exact = exact_topk(&mu, &q, TOP_K);

    let mut server = spawn_serve(&dir, port, false);
    wait_healthy(port, &mut server);

    let before = search(port, false, &q, TOP_K);
    assert_eq!(
        before["backend"],
        Value::Null,
        "no fast path without --gpu: {before}"
    );
    assert_rows_match(&rows(&before), &exact, &radii, "CPU before /store");
    assert_gaussian_ranking(
        &rows(&search(port, true, &q, TOP_K)),
        &exact,
        "CPU gaussian before /store",
    );

    store(port, &v);
    let after = search(port, false, &q, TOP_K);
    assert_rows_match(&rows(&after), &exact, &radii, "CPU after /store");
    assert_rows_eq(
        &rows(&before),
        &rows(&after),
        "CPU before vs after an unrelated /store",
    );
    assert_stored_is_top_1(port);

    let code = shutdown(&mut server);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&server)
    );
    assert!(
        stderr_of(&server).contains("campo splat cargado"),
        "the field/ corpus was not loaded"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// `rescore=gaussian` with a field/ corpus: splat ids, the splat's own topic
/// and NO confidence — its score is alpha·exp(−kappa·d²), not a distance.
/// Before F02d the confidence was `1 − sqrt(score)/R`: a score of 0.0 (very
/// far) came out as a perfect 1.0.
#[test]
fn gaussian_rescore_with_a_field_corpus_has_the_splat_topic_and_no_distance_confidence() {
    let dir = scratch("gauss");
    let port = probe_port();
    let mu = write_field_corpus(&dir);
    let (q, v) = (query(), stored());
    let exact = exact_topk(&mu, &q, TOP_K);

    let mut server = spawn_serve(&dir, port, false);
    wait_healthy(port, &mut server);

    let before = rows(&search(port, true, &q, TOP_K));
    assert_gaussian_ranking(&before, &exact, "gaussian before /store");
    for (index, _, topic, confidence) in &before {
        assert_eq!(
            *topic,
            Some(*index as u32),
            "topic must be the splat the result is: {before:?}"
        );
        assert_eq!(
            *confidence, None,
            "a gaussian score is not a distance: no confidence may be derived from it: {before:?}"
        );
    }

    store(port, &v);
    let after = rows(&search(port, true, &q, TOP_K));
    assert_rows_eq(
        &before,
        &after,
        "gaussian before vs after an unrelated /store",
    );
    assert_gaussian_ranking(&after, &exact, "gaussian after /store");

    let code = shutdown(&mut server);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&server)
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ─── GPU fast path vs batch worker vs CPU (real CUDA) ──────────────────

/// `serve --gpu` on a field/ corpus. Before the first `/store` the startup
/// fast path answers (backend "gpu"); after it the batch worker does. Both
/// must return the same rows as the exact splat ranking AND as a CPU server on
/// an identical data-dir. Before F02d the fast path answered the nearest
/// `field.raw` CHUNKS: other ids, other distances, other vectors.
#[cfg(feature = "cuda")]
#[test]
fn gpu_search_with_a_field_corpus_agrees_with_cpu_before_and_after_a_store() {
    if !splatsdb::gpu::is_cuda_available() {
        eprintln!("[f02d] SKIPPED: CUDA is not available on this host");
        return;
    }
    let gpu_dir = scratch("gpu");
    let cpu_dir = scratch("cpu-ref");
    let (port_gpu, port_cpu) = (probe_port(), probe_port());
    let mu = write_field_corpus(&gpu_dir);
    assert_eq!(
        mu,
        write_field_corpus(&cpu_dir),
        "both servers must boot the same dataset"
    );
    let radii = radii();
    let (q, v) = (query(), stored());
    let exact = exact_topk(&mu, &q, TOP_K);

    let mut gpu = spawn_serve(&gpu_dir, port_gpu, true);
    wait_healthy(port_gpu, &mut gpu);
    let mut cpu = spawn_serve(&cpu_dir, port_cpu, false);
    wait_healthy(port_cpu, &mut cpu);

    // Unmutated store: the startup fast path answers.
    let fast = search(port_gpu, false, &q, TOP_K);
    assert_eq!(
        fast["backend"], "gpu",
        "expected the startup GPU fast path: {fast}"
    );
    assert_rows_match(&rows(&fast), &exact, &radii, "GPU fast path before /store");
    assert_rows_eq(
        &rows(&fast),
        &rows(&search(port_cpu, false, &q, TOP_K)),
        "GPU fast path vs CPU, before /store",
    );
    let gauss_before = rows(&search(port_gpu, true, &q, TOP_K));
    assert_gaussian_ranking(&gauss_before, &exact, "GPU gaussian before /store");
    assert_rows_eq(
        &gauss_before,
        &rows(&search(port_cpu, true, &q, TOP_K)),
        "GPU gaussian vs CPU gaussian, before /store",
    );

    // Mutation: the startup key goes stale — the GPU index is a snapshot of
    // ONE dataset version (A03-F05). Before F06a nothing rebuilt it and every
    // later /search took the batch worker for good; F06a rebuilds the exact
    // index once per data_version, so the fast path answers again with the
    // NEW snapshot. Either route may answer; a stale one may not: the rows
    // must be the exact answer and the CPU route's.
    store(port_gpu, &v);
    store(port_cpu, &v);
    assert_stored_is_top_1(port_gpu);
    let after = search(port_gpu, false, &q, TOP_K);
    assert!(
        after["backend"] == Value::Null || after["backend"] == "gpu",
        "after a mutation only the batch worker or a rebuilt GPU index may answer (A03-F05): {after}"
    );
    assert_rows_match(&rows(&after), &exact, &radii, "GPU batch after /store");
    assert_rows_eq(
        &rows(&fast),
        &rows(&after),
        "GPU before vs after an unrelated /store",
    );
    assert_rows_eq(
        &rows(&after),
        &rows(&search(port_cpu, false, &q, TOP_K)),
        "GPU batch vs CPU, after /store",
    );
    let gauss_after = rows(&search(port_gpu, true, &q, TOP_K));
    assert_rows_eq(
        &gauss_after,
        &rows(&search(port_cpu, true, &q, TOP_K)),
        "GPU gaussian vs CPU gaussian, after /store",
    );

    let code = shutdown(&mut gpu);
    assert_eq!(
        code,
        Some(0),
        "serve --gpu must exit 0 after SIGTERM: {}",
        stderr_of(&gpu)
    );
    let code = shutdown(&mut cpu);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&cpu)
    );
    let err = stderr_of(&gpu);
    assert!(
        err.contains("GPU exact index listo") && err.contains("campo splat cargado"),
        "this run did not exercise the GPU fast path on a field corpus: {err}"
    );
    assert!(
        err.contains(&format!("GPU exact index listo ({K} vectores")),
        "the GPU index must hold the {K} splats, not the {N} chunks: {err}"
    );
    std::fs::remove_dir_all(&gpu_dir).ok();
    std::fs::remove_dir_all(&cpu_dir).ok();
}

// ─── F02e1: exact ties resolve by (score, then ascending id) on every route ───

/// Splats of the tie corpus: more than the 256 threads of a kernel block
/// hold, so the kernel's thread-major scan order (thread 0: ids 0, 256, …;
/// then thread 1: ids 1, 257, …) is NOT the id order.
const TIE_N: usize = 300;

/// One splat per axis, so every squared distance is a single exact product.
/// ids {1, 2, 3, 256, 257, 258} share d² = 4.25 (a six-way tie for the
/// top-5 after id 0 at d² = 1.25); every other id has its own d² ≥ 16.25.
fn tie_row(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    match i {
        0 => v[3] = 1.0,
        1 => v[0] = 2.0,
        2 => v[0] = -2.0,
        3 => v[1] = 2.0,
        256 => v[1] = -2.0,
        257 => v[2] = 2.0,
        258 => v[2] = -2.0,
        i => v[7 + i % 24] = 4.0 + 0.03125 * i as f32,
    }
    v
}

/// Query on the one axis no splat uses (+0.25 on every d²): all ties survive.
fn tie_query() -> Vec<f32> {
    let mut q = vec![0.0f32; DIM];
    q[31] = 0.5;
    q
}

/// Stored far away: its own top-1 at the next splat id, outside every top-k.
fn tie_far() -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[31] = 60.0;
    v
}

/// The layout `ingest` leaves behind for the tie corpus: shard `default` =
/// the splats, plus an energy landscape with `alpha[257] = 1000` (kappa = 1),
/// so the gaussian ranking differs from L2: [257, 0, 1, 2, 3].
fn write_tie_corpus(dir: &Path) {
    let mu: Vec<Vec<f32>> = (0..TIE_N).map(tie_row).collect();
    write_shard(&dir.join("data").join("vectors").join("default.bin"), &mu);
    let mut alpha = vec![1.0f32; TIE_N];
    alpha[257] = 1000.0;
    let kappa = vec![1.0f32; TIE_N];
    let mut buf = Vec::new();
    buf.extend_from_slice(&(TIE_N as u64).to_le_bytes());
    buf.extend_from_slice(&(DIM as u64).to_le_bytes());
    for r in &mu {
        for x in r {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(TIE_N as u64).to_le_bytes());
    for a in &alpha {
        buf.extend_from_slice(&a.to_le_bytes());
    }
    buf.extend_from_slice(&(TIE_N as u64).to_le_bytes());
    for k in &kappa {
        buf.extend_from_slice(&k.to_le_bytes());
    }
    std::fs::create_dir_all(dir.join("energy")).unwrap();
    std::fs::write(dir.join("energy").join("landscape.bin"), buf).unwrap();
}

/// Contractual reference over the tie corpus (+ `extra` rows appended by
/// `/store`): `(id, score)` by score (d² ascending / gaussian descending),
/// then ascending id.
fn tie_reference(gaussian: bool, extra: &[Vec<f32>]) -> Vec<(u64, f64)> {
    let mut rows: Vec<Vec<f32>> = (0..TIE_N).map(tie_row).collect();
    rows.extend_from_slice(extra);
    let q = tie_query();
    let mut all: Vec<(usize, f64)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let d2 = d2(r, &q);
            let score = if gaussian {
                (if i == 257 { 1000.0 } else { 1.0 }) * (-d2).exp()
            } else {
                d2
            };
            (i, score)
        })
        .collect();
    all.sort_by(|a, b| {
        let by_score = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
        let by_score = if gaussian {
            by_score.reverse()
        } else {
            by_score
        };
        by_score.then(a.0.cmp(&b.0))
    });
    all.truncate(TOP_K);
    all.into_iter().map(|(i, s)| (i as u64, s)).collect()
}

/// `/search` rows == the contractual reference: ids in (score, id) order and
/// scores within f32/`__expf` noise.
fn assert_tie_rows(v: &Value, gaussian: bool, extra: &[Vec<f32>], what: &str) {
    let got = rows(v);
    let want = tie_reference(gaussian, extra);
    assert_eq!(
        got.iter().map(|r| r.0).collect::<Vec<_>>(),
        want.iter().map(|r| r.0).collect::<Vec<_>>(),
        "{what}: ids/order must be (score, then ascending id): got {got:?} want {want:?}"
    );
    for ((_, score, _, _), (_, want_score)) in got.iter().zip(&want) {
        assert!(
            (score - want_score).abs() <= 1e-9 + 1e-5 * want_score.abs(),
            "{what}: score {score}, expected {want_score}: {got:?}"
        );
    }
}

/// CPU route (`serve` without `--gpu`), every build: the six-way tie
/// {1, 2, 3, 256, 257, 258} for the four places behind id 0 resolves to the
/// four lowest ids — in L2 and in gaussian — before and after a `/store`.
#[test]
fn exact_ties_resolve_by_id_on_the_cpu_route() {
    let dir = scratch("ties-cpu");
    let port = probe_port();
    write_tie_corpus(&dir);
    let (q, far) = (tie_query(), tie_far());

    let mut server = spawn_serve(&dir, port, false);
    wait_healthy(port, &mut server);

    assert_tie_rows(
        &search(port, false, &q, TOP_K),
        false,
        &[],
        "CPU L2 before /store",
    );
    assert_tie_rows(
        &search(port, true, &q, TOP_K),
        true,
        &[],
        "CPU gaussian before /store",
    );

    store(port, &far);
    assert_tie_rows_ordered(
        &search(port, false, &q, TOP_K),
        false,
        std::slice::from_ref(&far),
        "CPU L2 after /store",
    );
    assert_tie_rows_ordered(
        &search(port, true, &q, TOP_K),
        true,
        std::slice::from_ref(&far),
        "CPU gaussian after /store",
    );

    let code = shutdown(&mut server);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&server)
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Rows of an ANN route — CPU after the index is built (HNSW candidates +
/// exact re-rank, DECISIONS D2): the ORDER is contractual, (score, id), but
/// which tied rows the candidate set holds follows the backend's recall.
fn assert_tie_rows_ordered(v: &Value, gaussian: bool, _extra: &[Vec<f32>], what: &str) {
    let got = rows(v);
    assert_eq!(got.len(), TOP_K, "{what}: {got:?}");
    for w in got.windows(2) {
        let better = if gaussian {
            w[0].1 >= w[1].1
        } else {
            w[0].1 <= w[1].1
        };
        assert!(
            better && (w[0].1 != w[1].1 || w[0].0 < w[1].0),
            "{what}: rows must be (score, then ascending id): {got:?}"
        );
    }
}

/// Real CUDA: the startup fast path (l2_topk kernel), the GPU batch worker
/// (l2_topk / gaussian_topk kernels, before and after a `/store`) and the
/// CPU route all answer the contractual order — and the same SET: the
/// kernel's thread-major tie order returned {…, 257, 258} where the contract
/// wants {…, 3, 256}.
#[cfg(feature = "cuda")]
#[test]
fn exact_ties_resolve_by_id_on_the_gpu_fast_path_the_gpu_batch_and_the_cpu_route() {
    if !splatsdb::gpu::is_cuda_available() {
        eprintln!("[f02e1] SKIPPED: CUDA is not available on this host");
        return;
    }
    let gpu_dir = scratch("ties-gpu");
    let cpu_dir = scratch("ties-cpu-ref");
    let (port_gpu, port_cpu) = (probe_port(), probe_port());
    write_tie_corpus(&gpu_dir);
    write_tie_corpus(&cpu_dir);
    let (q, far) = (tie_query(), tie_far());

    let mut gpu = spawn_serve(&gpu_dir, port_gpu, true);
    wait_healthy(port_gpu, &mut gpu);
    let mut cpu = spawn_serve(&cpu_dir, port_cpu, false);
    wait_healthy(port_cpu, &mut cpu);

    // Unmutated store: the startup fast path answers L2 (backend "gpu");
    // `rescore=gaussian` always goes through the batch worker.
    let fast = search(port_gpu, false, &q, TOP_K);
    assert_eq!(
        fast["backend"], "gpu",
        "expected the startup GPU fast path: {fast}"
    );
    assert_tie_rows(&fast, false, &[], "GPU fast path L2 before /store");
    assert_tie_rows(
        &search(port_gpu, true, &q, TOP_K),
        true,
        &[],
        "GPU batch gaussian before /store",
    );
    assert_tie_rows(
        &search(port_cpu, false, &q, TOP_K),
        false,
        &[],
        "CPU L2 before /store",
    );
    assert_tie_rows(
        &search(port_cpu, true, &q, TOP_K),
        true,
        &[],
        "CPU gaussian before /store",
    );

    // Mutation: the startup key goes stale — the GPU index is a snapshot of
    // ONE dataset version (A03-F05). Before F06a nothing rebuilt it and every
    // later /search took the batch worker for good; F06a rebuilds the exact
    // index once per data_version, so the fast path answers again with the
    // NEW snapshot. Either route may answer; a stale one may not: the rows
    // must be the exact answer and the CPU route's.
    store(port_gpu, &far);
    store(port_cpu, &far);
    let after = search(port_gpu, false, &q, TOP_K);
    assert!(
        after["backend"] == Value::Null || after["backend"] == "gpu",
        "after a mutation only the batch worker or a rebuilt GPU index may answer (A03-F05): {after}"
    );
    assert_tie_rows(
        &after,
        false,
        std::slice::from_ref(&far),
        "GPU batch L2 after /store",
    );
    assert_tie_rows(
        &search(port_gpu, true, &q, TOP_K),
        true,
        std::slice::from_ref(&far),
        "GPU batch gaussian after /store",
    );
    assert_tie_rows_ordered(
        &search(port_cpu, false, &q, TOP_K),
        false,
        std::slice::from_ref(&far),
        "CPU L2 after /store",
    );
    assert_tie_rows_ordered(
        &search(port_cpu, true, &q, TOP_K),
        true,
        std::slice::from_ref(&far),
        "CPU gaussian after /store",
    );

    let code = shutdown(&mut gpu);
    assert_eq!(
        code,
        Some(0),
        "serve --gpu must exit 0 after SIGTERM: {}",
        stderr_of(&gpu)
    );
    let code = shutdown(&mut cpu);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&cpu)
    );
    let err = stderr_of(&gpu);
    assert!(
        err.contains(&format!("GPU exact index listo ({TIE_N} vectores")),
        "the GPU index must hold the {TIE_N} splats: {err}"
    );
    std::fs::remove_dir_all(&gpu_dir).ok();
    std::fs::remove_dir_all(&cpu_dir).ok();
}
