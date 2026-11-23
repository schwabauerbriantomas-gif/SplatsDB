//! F02e1 — `serve` reloads the persisted energy state (alpha/kappa) together
//! with mu, against the REAL `splatsdb` binary and the REAL `ingest` CLI.
//!
//! `ingest` persists the splat centroids (`data/vectors/default.bin`) AND the
//! energy landscape (`energy/landscape.bin`: alpha/kappa per splat). Before
//! F02e1 `serve` loaded only mu through `add_splat`, so after every restart
//! the splats answered with the init alpha/kappa (1.0) whatever `ingest` had
//! written, and `?rescore=gaussian` scored with those fabricated values.
//!
//! Contract pinned here:
//! * `?rescore=gaussian` ranks by `alpha_i · exp(−kappa_i · ‖q − mu_i‖²)` with
//!   the PERSISTED alpha/kappa — same ids, order and scores before `/store`,
//!   after it and after a restart of the same data-dir (CPU and `--gpu`).
//! * `/search` (L2) is unaffected by the energy state (contract A5: energy
//!   never reorders the L2 route).
//! * The rows `/store` appended keep the init alpha/kappa they had in memory;
//!   `serve`'s own compact rewrites the shard and leaves the landscape alone.
//!
//! Every server binds a port of the probe range 18111-18119 (never a
//! production port) on a scratch data-dir under `$TMPDIR/F02e1`, and every
//! child is terminated by PID — also on unwind.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const DIM: usize = 32;
/// Splats (clusters) of the ingested corpus — rows of the `default` shard.
const K: usize = 6;
/// Chunks per cluster of the raw corpus.
const PER: usize = 8;
const TOP_K: usize = 5;
const PROBE_BASE: u16 = 18111;
const PROBE_COUNT: usize = 9;
static PROBE_NEXT: AtomicUsize = AtomicUsize::new(0);

/// `(index, score)` of one `/search` row.
type Pair = (u64, f64);

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

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir()
        .join("F02e1")
        .join(format!("{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ─── Deterministic raw corpus ─────────────────────────────────────────

/// Cluster centres on the first three axes (the other 27 are 0) and the
/// per-cluster spread. `ingest` derives `kappa = 1/variance` (clamped to
/// 0.1..=100) and `alpha = cluster_size / n`, so a TIGHT cluster (kappa 100)
/// next to a LOOSE one (kappa ≈ 4) makes the gaussian ranking depart from
/// L2 for a query between them — the signature the lost energy state has.
const CENTRES: [[f32; 3]; K] = [
    [0.0, 0.0, 0.0],
    [1.2, 0.0, 0.0],
    [0.0, 1.5, 0.0],
    [0.0, 0.0, 1.1],
    [-1.3, 0.7, 0.0],
    [0.6, -1.4, 0.0],
];
const SPREADS: [f64; K] = [0.02, 0.5, 0.1, 0.15, 0.3, 0.05];

/// xorshift64* — a tiny explicit PRNG so the corpus needs no RNG crate.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Box–Muller (the second value of the pair is discarded: still deterministic).
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64().max(1e-12);
        let v = self.next_f64();
        sigma * (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Cluster centres + `PER` noisy chunks each: what `ingest` clusters.
fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let centers: Vec<Vec<f32>> = CENTRES
        .iter()
        .map(|c| {
            let mut v = vec![0.0f32; DIM];
            v[..3].copy_from_slice(c);
            v
        })
        .collect();
    let mut chunks = Vec::with_capacity(K * PER);
    for (c, centre) in centers.iter().enumerate() {
        for _ in 0..PER {
            let chunk: Vec<f32> = centre
                .iter()
                .map(|m| m + rng.normal(SPREADS[c]) as f32)
                .collect();
            chunks.push(chunk);
        }
    }
    (centers, chunks)
}

/// `[u64 rows][u64 cols][f32 LE]`: the on-disk vector format.
fn write_bin(path: &Path, rows: &[Vec<f32>]) {
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

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn read_bin(path: &Path) -> Vec<Vec<f32>> {
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

/// `energy/landscape.bin` as `persistence::save_energy_state` writes it:
/// `[u64 rows][u64 cols][f32 mu][u64 n_alpha][f32 alpha][u64 n_kappa][f32 kappa]`.
fn read_landscape(path: &Path) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    let u64at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize;
    let (rows, cols) = (u64at(0), u64at(8));
    let mu: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| f32_at(&b, 16 + (r * cols + c) * 4))
                .collect()
        })
        .collect();
    let o = 16 + rows * cols * 4;
    let n_alpha = u64at(o);
    let alpha: Vec<f32> = (0..n_alpha).map(|i| f32_at(&b, o + 8 + i * 4)).collect();
    let o = o + 8 + n_alpha * 4;
    let n_kappa = u64at(o);
    let kappa: Vec<f32> = (0..n_kappa).map(|i| f32_at(&b, o + 8 + i * 4)).collect();
    assert_eq!(
        o + 8 + n_kappa * 4,
        b.len(),
        "trailing bytes in landscape.bin"
    );
    (mu, alpha, kappa)
}

// ─── References ───────────────────────────────────────────────────────

fn d2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum()
}

/// Server-faithful gaussian top-k: the score in f32 exactly like
/// `SplatStore::gaussian_score` and the `gaussian_topk` kernel
/// (`alpha · exp(−kappa · d²)`, so a far splat underflows to 0.0 and ties
/// with every other such splat), score descending, ties → ascending id (the
/// contract of `gpu::cpu_gaussian_topk`).
fn gaussian_topk(
    mu: &[Vec<f32>],
    alpha: &[f32],
    kappa: &[f32],
    q: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = mu
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let d2 = d2(m, q) as f32;
            (i, alpha[i] * (-(kappa[i] * d2)).exp())
        })
        .collect();
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

/// Exact L2 top-k: `(id, d²)`, ascending, ties → ascending id.
fn l2_topk(mu: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = mu.iter().enumerate().map(|(i, m)| (i, d2(m, q))).collect();
    all.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

// ─── Real server ──────────────────────────────────────────────────────

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

fn search(port: u16, gaussian: bool, embedding: &[f32], top_k: usize) -> Value {
    let path = if gaussian {
        "/search?rescore=gaussian"
    } else {
        "/search"
    };
    let body =
        json!({ "query": "f02e1-probe", "embedding": embedding, "top_k": top_k }).to_string();
    let (code, text) = http(port, "POST", path, Some(&body))
        .unwrap_or_else(|| panic!("/search on :{port} did not answer"));
    assert_eq!(code, 200, "POST {path} on :{port}: {text}");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("POST {path} json: {e}: {text}"))
}

fn store(port: u16, embedding: &[f32]) {
    let body = json!({ "text": "f02e1-new-vector", "embedding": embedding }).to_string();
    let (code, text) = http(port, "POST", "/store", Some(&body)).expect("/store answered");
    assert_eq!(code, 200, "POST /store on :{port}: {text}");
}

fn pairs(v: &Value) -> Vec<Pair> {
    v["results"]
        .as_array()
        .unwrap_or_else(|| panic!("results array missing: {v}"))
        .iter()
        .map(|r| {
            (
                r["index"].as_u64().expect("index"),
                r["score"].as_f64().expect("score"),
            )
        })
        .collect()
}

/// Rows against the reference: same ids in the same order, scores within
/// f32 / `__expf` noise of the reference.
fn assert_rows(rows: &[Pair], exact: &[(usize, f64)], what: &str) {
    assert_eq!(rows.len(), exact.len(), "{what}: {rows:?} vs {exact:?}");
    for ((index, score), (exact_index, exact_score)) in rows.iter().zip(exact) {
        assert_eq!(
            *index, *exact_index as u64,
            "{what}: ids/order differ from the persisted-alpha/kappa ranking {exact:?}: {rows:?}"
        );
        assert!(
            (score - exact_score).abs() <= 1e-9 + 1e-4 * exact_score.abs(),
            "{what}: score {score}, expected {exact_score}: {rows:?}"
        );
    }
}

fn copy_dir(src: &Path, dst: &Path) {
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let target = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// What [`seed`] leaves behind.
struct Seed {
    dir: PathBuf,
    mu: Vec<Vec<f32>>,
    alpha: Vec<f32>,
    kappa: Vec<f32>,
    /// Discriminating query (see [`seed`]).
    q: Vec<f32>,
    /// Vector a later `/store` appends, far from every splat.
    far: Vec<f32>,
}

/// The seed data-dir: real CLI `ingest` (shard + energy landscape). Returns
/// `(dir, mu, alpha, kappa, query, far vector)`.
fn seed() -> Seed {
    let dir = scratch("seed");
    let (centers, chunks) = corpus();
    write_bin(&dir.join("raw.bin"), &chunks);
    let out = Command::new(env!("CARGO_BIN_EXE_splatsdb"))
        .arg("--data-dir")
        .arg(&dir)
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("ingest")
        .arg("--input")
        .arg(dir.join("raw.bin"))
        .arg("--n-clusters")
        .arg(K.to_string())
        .arg("--seed")
        .arg("7")
        .output()
        .expect("run splatsdb ingest");
    assert!(
        out.status.success(),
        "ingest failed: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let mu = read_bin(&dir.join("data").join("vectors").join("default.bin"));
    let (mu_l, alpha, kappa) = read_landscape(&dir.join("energy").join("landscape.bin"));
    assert_eq!(mu_l, mu, "the landscape's mu must be the shard's mu");
    assert_eq!(mu.len(), K, "ingest must leave {K} splats: {mu:?}");
    assert!(
        alpha.iter().any(|a| *a != 1.0) || kappa.iter().any(|k| *k != 1.0),
        "the persisted energy state must differ from the init (1.0) to be observable: alpha={alpha:?} kappa={kappa:?}"
    );
    eprintln!("[f02e1] persisted energy state: alpha={alpha:?} kappa={kappa:?}");
    // Discriminating query: `t` units from centre `a` towards centre `b`.
    // Its gaussian ranking under the persisted alpha/kappa must differ from
    // the L2 ranking (what alpha=kappa=1 yields) and from index order, and
    // its top-6 scores must be robust: non-zero or exactly 0.0 (underflow),
    // never in the near-underflow band, and separated by > 1e-3 or tied at
    // 0.0 — so f32 rounding cannot reorder them on any route.
    let q = (0..K)
        .flat_map(|a| (0..K).filter(move |b| *b != a).map(move |b| (a, b)))
        .flat_map(|(a, b)| [0.3, 0.5, 0.7, 0.9, 1.2, 1.6].map(move |t| (a, b, t)))
        .find_map(|(a, b, t)| {
            let d = d2(&centers[a], &centers[b]).sqrt();
            let q: Vec<f32> = (0..DIM)
                .map(|i| centers[a][i] + ((t / d) as f32) * (centers[b][i] - centers[a][i]))
                .collect();
            let g = gaussian_topk(&mu, &alpha, &kappa, &q, TOP_K + 1);
            let ids: Vec<usize> = g[..TOP_K].iter().map(|r| r.0).collect();
            let l2: Vec<usize> = l2_topk(&mu, &q, TOP_K).iter().map(|r| r.0).collect();
            let s: Vec<f64> = g.iter().map(|r| f64::from(r.1)).collect();
            let robust = s[0] > 0.0
                && s.iter().all(|x| *x == 0.0 || *x >= 1e-30)
                && s.windows(2)
                    .all(|w| (w[0] == 0.0 && w[1] == 0.0) || (w[0] - w[1]).abs() > 1e-3 * w[0]);
            let discriminating = ids != l2 && ids != (0..TOP_K).collect::<Vec<_>>();
            (robust && discriminating).then_some(q)
        })
        .expect("no discriminating query for this corpus");
    let far: Vec<f32> = (0..DIM).map(|d| 40.0 + 0.25 * d as f32).collect();
    Seed {
        dir,
        mu,
        alpha,
        kappa,
        q,
        far,
    }
}

// ─── The lifecycle (every build) ──────────────────────────────────────

/// One `serve` lifecycle on a copy of the seed: gaussian + L2 before a
/// `/store`, after it, and after a restart of the same data-dir.
fn assert_energy_survives_a_restart(gpu: bool) {
    let Seed {
        dir: seed_dir,
        mu,
        alpha,
        kappa,
        q,
        far,
    } = seed();
    let port = probe_port();
    let dir = scratch(if gpu { "gpu" } else { "cpu" });
    copy_dir(&seed_dir, &dir);
    let f64s = |r: Vec<(usize, f32)>| -> Vec<(usize, f64)> {
        r.into_iter().map(|(i, s)| (i, f64::from(s))).collect()
    };
    let gauss = f64s(gaussian_topk(&mu, &alpha, &kappa, &q, TOP_K));
    let l2 = l2_topk(&mu, &q, TOP_K);
    // After the /store: the appended row carries the init alpha/kappa.
    let mut mu_post = mu.clone();
    mu_post.push(far.clone());
    let mut alpha_post = alpha.clone();
    alpha_post.push(1.0);
    let mut kappa_post = kappa.clone();
    kappa_post.push(1.0);
    let gauss_post = f64s(gaussian_topk(&mu_post, &alpha_post, &kappa_post, &q, TOP_K));
    let l2_post = l2_topk(&mu_post, &q, TOP_K);
    let ids = |r: &[Pair]| r.iter().map(|p| p.0).collect::<Vec<_>>();
    eprintln!(
        "[f02e1] reference: gaussian={gauss:?} l2={:?}",
        l2.iter().map(|r| r.0).collect::<Vec<_>>()
    );

    let mut server = spawn_serve(&dir, port, gpu);
    wait_healthy(port, &mut server);

    let before_g = pairs(&search(port, true, &q, TOP_K));
    let first = search(port, false, &q, TOP_K);
    let before_l = pairs(&first);
    eprintln!(
        "[f02e1] serve{} gaussian before /store = {before_g:?}",
        if gpu { " --gpu" } else { "" }
    );
    assert_rows(&before_g, &gauss, "gaussian before /store");
    assert_rows(&before_l, &l2, "L2 before /store");
    assert_ne!(
        ids(&before_g),
        ids(&before_l),
        "the query must discriminate the gaussian ranking from L2"
    );
    assert_ne!(
        ids(&before_g),
        (0..TOP_K as u64).collect::<Vec<_>>(),
        "a gaussian top-k in index order is the degenerate signature of the lost energy state"
    );
    if gpu {
        assert_eq!(
            first["backend"], "gpu",
            "expected the startup GPU fast path: {first}"
        );
    }

    store(port, &far);
    let after_g = pairs(&search(port, true, &q, TOP_K));
    assert_rows(&after_g, &gauss_post, "gaussian after /store");
    assert_rows(
        &pairs(&search(port, false, &q, TOP_K)),
        &l2_post,
        "L2 after /store",
    );
    let hit = pairs(&search(port, true, &far, 1));
    assert_eq!(
        hit[0].0, K as u64,
        "the stored vector is its own gaussian top-1: {hit:?}"
    );

    let code = shutdown(&mut server);
    let log1 = stderr_of(&server);
    assert_eq!(code, Some(0), "serve must exit 0 after SIGTERM: {log1}");
    assert!(
        log1.contains(&format!(
            "energy state cargado (alpha/kappa de {K}/{K} splats"
        )),
        "the persisted energy state was not reported as loaded: {log1}"
    );
    // The shutdown compacted the shard (K+1 rows) and left the landscape alone.
    assert_eq!(
        read_bin(&dir.join("data").join("vectors").join("default.bin")).len(),
        K + 1
    );
    let (_, alpha_disk, kappa_disk) = read_landscape(&dir.join("energy").join("landscape.bin"));
    assert_eq!(
        (alpha_disk, kappa_disk),
        (alpha, kappa),
        "the landscape must not be rewritten by serve"
    );

    // Restart on the same data-dir: the energy state must come back from disk.
    let mut server = spawn_serve(&dir, port, gpu);
    wait_healthy(port, &mut server);
    let restart_g = pairs(&search(port, true, &q, TOP_K));
    assert_rows(&restart_g, &gauss_post, "gaussian after restart");
    eprintln!(
        "[f02e1] serve{} gaussian after restart = {restart_g:?}",
        if gpu { " --gpu" } else { "" }
    );
    assert_eq!(
        restart_g, after_g,
        "a restart must not change the gaussian rows"
    );
    assert_rows(
        &pairs(&search(port, false, &q, TOP_K)),
        &l2_post,
        "L2 after restart",
    );
    let code = shutdown(&mut server);
    let log2 = stderr_of(&server);
    assert_eq!(code, Some(0), "serve must exit 0 after SIGTERM: {log2}");
    assert!(
        log2.contains(&format!(
            "energy state cargado (alpha/kappa de {K}/{} splats",
            K + 1
        )),
        "after the restart the energy state of the {K} ingested splats must be loaded: {log2}"
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&seed_dir).ok();
}

/// CPU build / `serve` without `--gpu`: `?rescore=gaussian` ranks with the
/// alpha/kappa `ingest` persisted — before a `/store`, after it, and after a
/// restart. Before F02e1 every restart reset them to 1.0.
#[test]
fn serve_reloads_the_persisted_energy_state_across_a_restart() {
    assert_energy_survives_a_restart(false);
}

/// `serve --gpu` (fast path + GPU batch worker): same contract on a CUDA host.
#[cfg(feature = "cuda")]
#[test]
fn gpu_serve_reloads_the_persisted_energy_state_across_a_restart() {
    if !splatsdb::gpu::is_cuda_available() {
        eprintln!("[f02e1] SKIPPED: CUDA is not available on this host");
        return;
    }
    assert_energy_survives_a_restart(true);
}
