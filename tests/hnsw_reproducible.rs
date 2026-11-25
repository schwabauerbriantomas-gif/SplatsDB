use ndarray::Array2;
use splatsdb::hnsw_index::HNSWIndex;
use splatsdb::interfaces::VectorIndex;

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn identical_hnsw_state_has_identical_bytes_across_processes() {
    let mut index = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
    index.build(Array2::from_shape_fn((1000, 4), |(row, col)| {
        ((row * 13 + col * 7) % 251) as f32 / 7.0 - 0.5
    }));
    index.remove(&[5, 17, 77, 800, 999]);

    let path = std::env::temp_dir().join(format!("hnsw-reproducible-{}.bin", std::process::id()));
    index.save(&path).expect("save deterministic HNSW fixture");
    let bytes = std::fs::read(&path).expect("read saved HNSW fixture");
    std::fs::remove_file(&path).expect("remove saved HNSW fixture");
    let hash = fnv1a64(&bytes);
    println!("HNSW_CANONICAL_FNV1A64={hash:016x} bytes={}", bytes.len());

    // Pins v2 bytes, not only load/search results: HashSet iteration order and
    // the content fingerprint must never leak nondeterminism into the file.
    // Regenerate deliberately if the versioned format changes again.
    assert_eq!(hash, 0xbf0f4cad99987c0e, "canonical HNSW v2 bytes changed");
}
