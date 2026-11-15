//! Characterization for `load_field` + `/field-search` (api_server).
//!
//! No test reached the field/ corpus path: `load_field` is private and only
//! exercised via `run_server` when `data_dir/field/` exists. These tests pin
//! the exact-sweep response shape (with deterministic pruning math) and the
//! disabled-path response BEFORE `load_field` is split into parsing helpers.
//!
//! Data layout (all LE):
//! - `field/members.idx`: 16-byte header (n: u64, k: u64) + (k+1)×8 CSR
//!   offsets (u64) + n×4 member ids (u32)
//! - `field/radii.bin`: 8-byte header (kr: u64) + kr×4 radii (f32)
//! - SQLite shards `raw` (n×dim) and `default` (k×dim) via
//!   `SplatsDBPersistence::save_vectors`

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const DIM: usize = 1024; // SplatsDBConfig::default().latent_dim

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
    listener.local_addr().expect("local addr").port()
}

/// Minimal HTTP/1.1 client: one request per connection (Connection: close).
fn http_request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let body = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).expect("write");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read");
    let status: u16 = response
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .expect("status line");
    let body = response.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

fn wait_healthy(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        assert!(
            Instant::now() < deadline,
            "server did not become healthy on port {port}"
        );
        if let Ok(s) = TcpStream::connect(("127.0.0.1", port)) {
            drop(s);
            let (status, body) = http_request(port, "GET", "/health", None);
            if status == 200 && body.contains("\"ok\"") {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn boot_server(data_dir: &std::path::Path) -> u16 {
    let port = free_port();
    let dir = data_dir.to_str().unwrap().to_string();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let _ = rt.block_on(splatsdb::api_server::run_server(
            "127.0.0.1",
            port,
            splatsdb::config::SplatsDBConfig::default(),
            &dir,
            false, // CPU: sin índice GPU
        ));
    });
    wait_healthy(port);
    port
}

fn u64le(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

fn seed_shards(
    data_dir: &std::path::Path,
    raw: &ndarray::Array2<f32>,
    splats: &ndarray::Array2<f32>,
) {
    let persist = splatsdb::storage::persistence::SplatsDBPersistence::with_backend(
        data_dir.to_str().unwrap(),
        splatsdb::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .expect("persist backend");
    persist.save_vectors(raw, "raw").expect("save raw");
    persist
        .save_vectors(splats, "default")
        .expect("save default");
}

/// Happy path: valid field/ + shards → exact bounds sweep over a synthetic
/// corpus whose pruning math is hand-computed:
/// - raw rows: [0.0×dim], [0.1×dim], [0.2×dim]; query zeros → d² = 0, 10.24, 40.96
/// - splat mus: [0.05×dim], [0.15×dim]; radii [1.0, 1.0]
/// - CSR: chunk0 = {0,1}, chunk1 = {2}; offsets [0,2,3]
/// → d_mu ≈ [1.6, 4.8]; lower bound of chunk1 ≈ (4.8−1)² ≈ 14.44 > kth 10.24
///   after chunk0 fills top-2 → chunk1 pruned: clusters_visited=1,
///   vectors_scanned=2, results [{0,~0}, {1,~10.24}].
#[test]
fn field_search_exact_sweep_characterization() {
    let tmp = std::env::temp_dir().join(format!(
        "splatsdb_field_char_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(tmp.join("field")).expect("create data dir");

    let mut raw = ndarray::Array2::<f32>::zeros((3, DIM));
    raw.row_mut(1).fill(0.1);
    raw.row_mut(2).fill(0.2);
    let mut splats = ndarray::Array2::<f32>::zeros((2, DIM));
    splats.row_mut(0).fill(0.05);
    splats.row_mut(1).fill(0.15);
    seed_shards(&tmp, &raw, &splats);

    // members.idx: n=3, k=2, offsets [0,2,3], ids [0,1,2]
    let mut members = Vec::new();
    members.extend_from_slice(&u64le(3));
    members.extend_from_slice(&u64le(2));
    for off in [0u64, 2, 3] {
        members.extend_from_slice(&u64le(off));
    }
    for id in [0u32, 1, 2] {
        members.extend_from_slice(&id.to_le_bytes());
    }
    std::fs::write(tmp.join("field/members.idx"), &members).expect("write members.idx");

    // radii.bin: kr=2, radii [1.0, 1.0]
    let mut radii = Vec::new();
    radii.extend_from_slice(&u64le(2));
    radii.extend_from_slice(&1.0f32.to_le_bytes());
    radii.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(tmp.join("field/radii.bin"), &radii).expect("write radii.bin");

    let port = boot_server(&tmp);
    let query = format!(
        "{{\"embedding\": [{}], \"top_k\": 2}}",
        vec!["0.0"; DIM].join(",")
    );
    let (status, body) = http_request(port, "POST", "/field-search", Some(&query));
    assert_eq!(status, 200, "field-search status: {body}");

    let v: serde_json::Value = serde_json::from_str(&body).expect("response json");
    assert_eq!(v["backend"], "field_exact_bounds", "backend marker: {v}");
    assert_eq!(v["k"], 2, "k echoed: {v}");
    assert_eq!(v["n_vectors"], 3, "n echoed: {v}");
    assert_eq!(v["n_splats"], 2, "n_splats echoed: {v}");
    assert_eq!(
        v["clusters_visited"], 1,
        "chunk1 must be pruned by the bound: {v}"
    );
    assert_eq!(v["vectors_scanned"], 2, "scanned only chunk0 members: {v}");
    let results = v["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2, "top_k=2: {v}");
    assert_eq!(results[0]["index"], 0, "nearest index: {v}");
    let s0 = results[0]["score"].as_f64().unwrap();
    assert!(s0.abs() < 1e-3, "row0 is the query itself, d²≈0: {s0}");
    assert_eq!(results[1]["index"], 1, "second index: {v}");
    let s1 = results[1]["score"].as_f64().unwrap();
    assert!(
        (s1 - 10.24).abs() < 0.05,
        "row1 d² = 1024·0.1² ≈ 10.24: {s1}"
    );
}

/// Disabled paths: without field/ (or with a corrupt members.idx that makes
/// load_field bail) the route answers 404 with the exact guidance message.
#[test]
fn field_search_disabled_characterization() {
    // Case A: no field/ at all.
    let tmp_a = std::env::temp_dir().join(format!(
        "splatsdb_field_off_a_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp_a).expect("create dir A");
    let port_a = boot_server(&tmp_a);
    let (status, body) = http_request(
        port_a,
        "POST",
        "/field-search",
        Some(&format!(
            "{{\"embedding\": [{}]}}",
            vec!["0.0"; DIM].join(",")
        )),
    );
    assert_eq!(status, 404, "no field → 404, got {status}: {body}");
    assert!(
        body.contains("no hay field/ en este data-dir"),
        "guidance message: {body}"
    );

    // Case B: members.idx truncated (header asks for 52 bytes, file has 24).
    let tmp_b = std::env::temp_dir().join(format!(
        "splatsdb_field_off_b_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(tmp_b.join("field")).expect("create dir B");
    let mut members = Vec::new();
    members.extend_from_slice(&u64le(3));
    members.extend_from_slice(&u64le(2));
    members.extend_from_slice(&u64le(0)); // only 1 of 3 offsets
    std::fs::write(tmp_b.join("field/members.idx"), &members).expect("write truncated");
    let mut radii = Vec::new();
    radii.extend_from_slice(&u64le(2));
    radii.extend_from_slice(&1.0f32.to_le_bytes());
    radii.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(tmp_b.join("field/radii.bin"), &radii).expect("write radii");
    let port_b = boot_server(&tmp_b);
    let (status, body) = http_request(
        port_b,
        "POST",
        "/field-search",
        Some(&format!(
            "{{\"embedding\": [{}]}}",
            vec!["0.0"; DIM].join(",")
        )),
    );
    assert_eq!(
        status, 404,
        "truncated members.idx disables field-search: {status}: {body}"
    );
    assert!(
        body.contains("no hay field/ en este data-dir"),
        "same guidance message: {body}"
    );
}
