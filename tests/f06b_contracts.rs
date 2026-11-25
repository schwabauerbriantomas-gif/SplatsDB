//! F06b integration contracts: content-fingerprinted HNSW snapshots and
//! honest, mode-consistent WAL acknowledgements.

use ndarray::Array2;
use splatsdb::config::SplatsDBConfig;
use splatsdb::splats::SplatStore;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const DIM: usize = 4;

fn vectors(n: usize, seed: u64) -> Array2<f32> {
    let mut state = seed | 1;
    Array2::from_shape_fn((n, DIM), |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / (1u64 << 24) as f32
    })
}

fn config() -> SplatsDBConfig {
    SplatsDBConfig {
        latent_dim: DIM,
        max_splats: 100,
        enable_hnsw: true,
        hnsw_m: 8,
        hnsw_ef_construction: 50,
        hnsw_ef_search: 20,
        ..Default::default()
    }
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "splatsdb-f06b-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assert_self_match(store: &SplatStore, data: &Array2<f32>) {
    for i in 0..data.nrows() {
        let q = data.row(i).to_owned();
        let linear = store.find_neighbors(&q.view(), 1);
        let fused = store.find_neighbors_fused(&q.view(), 1);
        assert_eq!(linear[0].index, i);
        assert_eq!(linear[0].distance, 0.0);
        assert_eq!(
            fused[0].index, i,
            "fused returned stale row {} for query {i}",
            fused[0].index
        );
        assert_eq!(fused[0].distance, 0.0);
    }
}

#[test]
fn f06b_same_shape_content_change_rebuilds_before_fused_search() {
    let dir = scratch("stale-content");
    let data_dir = dir.to_str().unwrap();
    let a = vectors(40, 11);
    let mut first = SplatStore::new(config());
    assert!(first.add_splat(&a));
    assert!(first.build_index_if_needed(data_dir));

    let b = vectors(40, 99);
    let mut second = SplatStore::new(config());
    assert!(second.add_splat(&b));
    assert!(second.build_index_if_needed(data_dir));
    assert_self_match(&second, &b);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn f06b_legacy_snapshot_is_rebuilt_and_saved_with_versioned_fingerprint() {
    let dir = scratch("legacy");
    let data_dir = dir.to_str().unwrap();
    let data = vectors(40, 22);
    let mut store = SplatStore::new(config());
    assert!(store.add_splat(&data));
    assert!(store.build_index_if_needed(data_dir));
    let path = dir.join("hnsw_index.bin");

    // Remove the v2 trailer: an old, un fingerprinted snapshot is readable
    // but cannot authorize `is_built` over same-shape content.
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() > 28);
    std::fs::write(&path, &bytes[..bytes.len() - 28]).unwrap();
    let mut restarted = SplatStore::new(config());
    assert!(restarted.add_splat(&data));
    assert!(restarted.build_index_if_needed(data_dir));
    assert_self_match(&restarted, &data);

    let upgraded = std::fs::read(&path).unwrap();
    assert!(upgraded.len() >= 28);
    assert_eq!(
        u64::from_le_bytes(
            upgraded[upgraded.len() - 28..upgraded.len() - 20]
                .try_into()
                .unwrap()
        ),
        2,
        "legacy snapshot was not upgraded to the versioned format"
    );
    assert!(
        upgraded[upgraded.len() - 20..].iter().any(|b| *b != 0),
        "content fingerprint is absent/zero"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn f06b_malformed_snapshot_fails_closed_and_preserves_exact_linear_results() {
    let dir = scratch("malformed");
    let data_dir = dir.to_str().unwrap();
    let data = vectors(16, 33);
    let mut store = SplatStore::new(config());
    assert!(store.add_splat(&data));
    assert!(store.build_index_if_needed(data_dir));
    let path = dir.join("hnsw_index.bin");
    let valid = std::fs::read(&path).unwrap();
    let mut damaged = valid.clone();
    damaged[0] ^= 0x5a; // invalid magic: load/rebuild must fail, never authorize the graph
    std::fs::write(&path, &damaged).unwrap();
    assert!(store.try_build_index_if_needed(data_dir).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), damaged);
    assert_self_match(&store, &data);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn f06b_power_loss_mode_acknowledges_only_after_sync() {
    let dir = scratch("power-mode");
    let port = 18282_u16;
    let server = spawn_mode(port, &dir, "power-loss");
    let initial = json_http(port, "POST", "/status", None);
    assert_eq!(initial["durability_mode"], "power-loss");
    assert!(initial["durable"].as_bool().unwrap());
    assert!(initial["crash_durable"].as_bool().unwrap());
    assert!(initial["power_loss_durable"].as_bool().unwrap());
    assert_eq!(initial["wal_pending"], 0);

    let body = r#"{"id":"power-1","text":"one","embedding":[0,0.25,0.5,0.75]}"#;
    let stored = json_http(port, "POST", "/store", Some(body));
    assert_eq!(stored["status"], "stored");
    assert_eq!(stored["durability_mode"], "power-loss");
    assert!(stored["durable"].as_bool().unwrap());
    assert!(stored["crash_durable"].as_bool().unwrap());
    assert!(stored["power_loss_durable"].as_bool().unwrap());
    assert_eq!(stored["wal_pending"], 0);
    drop(server);
    std::fs::remove_dir_all(dir).unwrap();
}

struct Running(Option<Child>);

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

fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let payload = body.unwrap_or("");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let status = response
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("invalid HTTP response: {response}"));
    let body = response.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
    (status, body)
}

fn json_http(port: u16, method: &str, path: &str, body: Option<&str>) -> serde_json::Value {
    let (status, text) = http(port, method, path, body);
    assert_eq!(status, 200, "{method} {path}: {text}");
    serde_json::from_str(&text).unwrap()
}

fn spawn(port: u16, dir: &Path) -> Running {
    spawn_mode(port, dir, "process-crash")
}

fn spawn_mode(port: u16, dir: &Path, mode: &str) -> Running {
    let mut child = Command::new(env!("CARGO_BIN_EXE_splatsdb"))
        .arg("--data-dir")
        .arg(dir)
        .arg("--dim")
        .arg(DIM.to_string())
        .arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .arg("--wal-mode")
        .arg(mode)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(
            child.try_wait().unwrap().is_none(),
            "serve exited before health"
        );
        if TcpStream::connect(("127.0.0.1", port)).is_ok()
            && http(port, "GET", "/health", None).0 == 200
        {
            return Running(Some(child));
        }
        assert!(Instant::now() < deadline, "serve did not become healthy");
        std::thread::sleep(Duration::from_millis(25));
    }
}

fn assert_process_crash(v: &serde_json::Value, pending: u64) {
    assert_eq!(v["durability_mode"], "process-crash", "{v}");
    assert_eq!(v["persistence"], "wal", "{v}");
    assert_eq!(v["durable"], true, "{v}");
    assert_eq!(v["crash_durable"], true, "{v}");
    assert_eq!(v["power_loss_durable"], false, "{v}");
    assert_eq!(v["wal_pending"], pending, "{v}");
}

#[test]
fn f06b_process_crash_ack_and_status_are_consistent() {
    let dir = scratch("ack-mode");
    let port = 18281_u16;
    let server = spawn(port, &dir);
    assert_process_crash(&json_http(port, "POST", "/status", None), 0);

    let embedding: Vec<String> = (0..DIM).map(|i| (i as f32 * 0.25).to_string()).collect();
    let body = format!(
        r#"{{"id":"f06b-1","text":"one","embedding":[{}]}}"#,
        embedding.join(",")
    );
    let stored = json_http(port, "POST", "/store", Some(&body));
    assert_eq!(stored["status"], "stored", "{stored}");
    assert_eq!(stored["id"], "f06b-1", "{stored}");
    assert_process_crash(&stored, 1);
    assert_process_crash(&json_http(port, "POST", "/status", None), 1);

    drop(server);
    std::fs::remove_dir_all(dir).unwrap();
}
