//! Shared helpers for CLI command handlers.

use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};

use splatsdb::{
    loaders::load_energy_landscape,
    storage::persistence::{SplatsDBPersistence, StorageBackend},
    storage::shard_guard::{try_lock_data_dir, DataDirLock},
    SplatStore, SplatsDBConfig,
};

pub fn make_config(dim: usize, max_splats: usize) -> SplatsDBConfig {
    SplatsDBConfig {
        latent_dim: dim,
        max_splats,
        ..Default::default()
    }
}

pub fn resolve_backend(name: &str) -> StorageBackend {
    match name {
        "json" | "json-file" => StorageBackend::JsonFile,
        "sqlite" => StorageBackend::Sqlite,
        _ => StorageBackend::Sqlite,
    }
}

pub fn make_persistence(
    data_dir: &str,
    backend: &str,
    rw: bool,
) -> Result<SplatsDBPersistence, Box<dyn std::error::Error + Send + Sync>> {
    SplatsDBPersistence::with_backend(data_dir, resolve_backend(backend), rw)
}

pub fn load_or_create_store(
    data_dir: &str,
    config: &SplatsDBConfig,
) -> (SplatStore, Option<SplatsDBPersistence>) {
    load_shard_store(data_dir, config, "default")
}

/// Load a store populated from a specific shard (issue #4).
///
/// `search`/`search-file` previously always read the "default" shard, so
/// vectors ingested with `index -s <name>` were silently invisible. This
/// variant loads the named shard; when the shard does not exist we fail
/// loudly instead of returning an empty (silent) result set.
pub fn load_shard_store(
    data_dir: &str,
    config: &SplatsDBConfig,
    shard: &str,
) -> (SplatStore, Option<SplatsDBPersistence>) {
    match try_load_shard_store(data_dir, config, shard) {
        Ok(loaded) => loaded,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(2);
        }
    }
}

/// Result-returning core of [`load_shard_store`] (no `exit`, testable).
///
/// Fail-closed (A03-F17): a data-dir that cannot be opened and a shard that
/// exists but cannot be loaded into the store — unreadable, or rejected by
/// `add_splat` (dimension/capacity mismatch) — are an `Err`. They never
/// degrade to an empty store that answers "0 results" with exit 0.
///
/// F02f: it also applies the energy state (alpha/kappa) `ingest` persisted
/// for the shard — see [`load_cli_energy_state`].
pub fn try_load_shard_store(
    data_dir: &str,
    config: &SplatsDBConfig,
    shard: &str,
) -> Result<(SplatStore, Option<SplatsDBPersistence>), String> {
    let persist = make_persistence(data_dir, "sqlite", true)
        .map_err(|e| format!("cannot open data-dir '{}': {}", data_dir, e))?;
    let mut store = SplatStore::new(config.clone());

    match persist.load_vectors(shard) {
        Ok(Some(vectors)) => {
            eprintln!(
                "[splatsdb] Loaded {} vectors from shard '{}'",
                vectors.nrows(),
                shard
            );
            if !store.add_splat(&vectors) {
                return Err(format!(
                    "shard '{}' holds {}x{} vectors but the store was configured with --dim {} \
                     and --max-splats {}: dimension or capacity mismatch (nothing was modified)",
                    shard,
                    vectors.nrows(),
                    vectors.ncols(),
                    config.latent_dim,
                    config.max_splats
                ));
            }
            // The energy state `ingest` persisted — the same policy serve
            // applies (F02f): without it every CLI command reported the init
            // alpha/kappa for vectors that have a derived energy state.
            load_cli_energy_state(&mut store, &vectors, data_dir, shard)?;
            // Try loading persisted HNSW first; only build from scratch if needed
            store.build_index_if_needed(data_dir);
            Ok((store, Some(persist)))
        }
        Ok(None) => {
            if shard != "default" {
                let known = persist.list_shards();
                return Err(format!(
                    "shard '{}' not found. Available shards: {:?}",
                    shard, known
                ));
            }
            Ok((store, Some(persist)))
        }
        Err(e) => Err(format!("failed to load shard '{}': {}", shard, e)),
    }
}

/// Exit status of a CLI writer refused because another process holds the
/// data-dir lock (a live `serve`, another writer).
pub const EXIT_DATA_DIR_LOCKED: u8 = 3;

/// Take the data-dir lock a CLI writer must hold for its whole read→write
/// window — the same `flock` `serve` holds (F02e1). While another process
/// holds it, this prints why and exits with [`EXIT_DATA_DIR_LOCKED`] BEFORE
/// anything was read or modified; the lock is released with the process.
pub fn exit_if_data_dir_locked(data_dir: &str, writer: &str) -> DataDirLock {
    let path = Path::new(data_dir);
    if let Err(e) = std::fs::create_dir_all(path) {
        eprintln!("Error: cannot create data-dir '{}': {}", data_dir, e);
        std::process::exit(1);
    }
    match try_lock_data_dir(path, writer) {
        Ok(lock) => lock,
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
            eprintln!("Error: {}", e);
            std::process::exit(EXIT_DATA_DIR_LOCKED.into());
        }
        Err(e) => {
            eprintln!("Error: cannot lock data-dir '{}': {}", data_dir, e);
            std::process::exit(1);
        }
    }
}

/// What a CLI writer reports when the HNSW index could not be persisted:
/// exit 1 and no success output (F02f). `on_disk` states exactly what was
/// and was not written.
pub fn exit_on_hnsw_persist_error(command: &str, data_dir: &str, e: String, on_disk: &str) -> ! {
    eprintln!(
        "Save error: {} NOT persisted: the HNSW index could not be written to {} ({}); {}",
        command, data_dir, e, on_disk
    );
    std::process::exit(1)
}

/// Apply the energy state (alpha/kappa) `ingest` persisted for the 'default'
/// shard to a CLI-loaded store — the policy `serve` applies since F02e1.
///
/// Before F02f the CLI only fed the shard to `add_splat`, so every command
/// reported the init values (1.0 / 10.0) for splats `ingest` had persisted
/// with a derived energy state, and `serve` and the CLI disagreed on the
/// same data-dir after a restart.
///
/// * no `energy/landscape.bin` → the init values (as serve);
/// * a landscape that does not describe this shard (another dataset, or rows
///   it never saw) → ignored / applied as a prefix (as serve);
/// * unreadable or corrupt → `Err`: fail closed like an unreadable shard,
///   never a silent fallback to the init values.
fn load_cli_energy_state(
    store: &mut SplatStore,
    vectors: &Array2<f32>,
    data_dir: &str,
    shard: &str,
) -> Result<(), String> {
    let path = Path::new(data_dir).join("energy").join("landscape.bin");
    if !path.is_file() {
        return Ok(());
    }
    let landscape = load_energy_landscape(path.to_str().unwrap_or_default()).map_err(|e| {
        format!(
            "energy state '{}' is unreadable ({}): refusing to answer with the init alpha/kappa \
             — re-run `ingest`, or delete the file to fall back to the init values (nothing was \
             modified)",
            path.display(),
            e
        )
    })?;
    let m = landscape.matching_prefix(vectors);
    if m == 0 {
        eprintln!(
            "[splatsdb] WARNING: {} does not describe shard '{}' ({}x{} vs {}x{}): it is IGNORED \
             and alpha/kappa stay at the init values",
            path.display(),
            shard,
            landscape.mu.nrows(),
            landscape.mu.ncols(),
            vectors.nrows(),
            vectors.ncols()
        );
        return Ok(());
    }
    let n = vectors.nrows();
    let loaded = store
        .load_energy(&landscape.alpha[..m], &landscape.kappa[..m])
        .map_err(|e| format!("energy state '{}' cannot be applied: {}", path.display(), e))?;
    eprintln!(
        "[splatsdb] energy state cargado (alpha/kappa de {}/{} splats desde {})",
        loaded, n, path.display()
    );
    Ok(())
}

pub fn parse_query(s: &str) -> Result<Array1<f32>, String> {
    let vals: Result<Vec<f32>, _> = s.split(',').map(|v| v.trim().parse()).collect();
    match vals {
        Ok(v) => Ok(Array1::from(v)),
        Err(e) => Err(format!("Invalid query vector: {}", e)),
    }
}

pub fn resolve_query(query: Option<&str>, query_file: Option<&PathBuf>) -> Array1<f32> {
    match (query, query_file) {
        (Some(q), _) => parse_query(q).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }),
        (_, Some(f)) => load_query_bin(f).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }),
        (None, None) => {
            eprintln!("Error: provide --query or --query-file");
            std::process::exit(1);
        }
    }
}

pub fn load_query_bin(path: &PathBuf) -> Result<Array1<f32>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "rows overflow")?;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "cols overflow")?;

    if rows != 1 {
        return Err(format!("Query file must have exactly 1 row, got {}", rows));
    }

    // The header is untrusted input: validate element count and on-disk size
    // BEFORE allocating, so a hostile/truncated `.bin` fails with a clean
    // error instead of an OOM.
    let total = validate_bin_header(path, rows, cols)?;

    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok(Array1::from(data))
}

/// Validate the (untrusted) `rows`/`cols` header of a `.bin` vector file
/// BEFORE allocating for it.
///
/// Caps the element count at 1e9 (same limit as `persistence::load_vectors`)
/// and requires the file to actually contain the header-promised payload
/// (16-byte header + `rows*cols` little-endian f32), so a hostile or
/// truncated file fails with a clean `Result` error instead of an OOM or a
/// short read. Returns the validated element count.
fn validate_bin_header(path: &std::path::Path, rows: usize, cols: usize) -> Result<usize, String> {
    const MAX_ELEMS: u64 = 1_000_000_000;
    const HEADER_BYTES: u64 = 16; // rows: u64 LE + cols: u64 LE

    let total = (rows as u64)
        .checked_mul(cols as u64)
        .ok_or_else(|| "overflow in rows*cols".to_string())?;
    if total > MAX_ELEMS {
        return Err(format!(
            "Refusing excessive allocation: {}x{} = {} elements (cap {})",
            rows, cols, total, MAX_ELEMS
        ));
    }
    let expected = HEADER_BYTES
        + total
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| "overflow in expected file size".to_string())?;
    let actual = std::fs::metadata(path)
        .map_err(|e| format!("Cannot stat {}: {}", path.display(), e))?
        .len();
    if actual < expected {
        return Err(format!(
            "Truncated file {}: header promises {}x{} f32 ({} bytes including header), file has {} bytes",
            path.display(),
            rows,
            cols,
            expected,
            actual
        ));
    }
    Ok(total as usize)
}

pub fn load_vectors_bin(path: &PathBuf) -> Result<Array2<f32>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "rows overflow")?;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "cols overflow")?;

    // The header is untrusted input: validate element count and on-disk size
    // BEFORE allocating, so a hostile/truncated `.bin` fails with a clean
    // error instead of an OOM.
    let total = validate_bin_header(path, rows, cols)?;

    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

pub fn format_neighbors(results: &[splatsdb::splats::NeighborResult], format: &str) -> String {
    match format {
        "json" => {
            let entries: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "index": r.index,
                        "distance": (r.distance * 10000.0).round() / 10000.0,
                        "alpha": (r.alpha * 10000.0).round() / 10000.0,
                        "kappa": (r.kappa * 10000.0).round() / 10000.0,
                    })
                })
                .collect();
            serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".into())
        }
        "compact" => results
            .iter()
            .map(|r| format!("{}:{:.4}", r.index, r.distance))
            .collect::<Vec<_>>()
            .join(" "),
        _ => {
            let mut out = String::from("Index\tDistance\tAlpha\t\tKappa\n");
            out.push_str("—\t—\t—\t—\n");
            for r in results {
                out.push_str(&format!(
                    "{}\t{:.4}\t\t{:.4}\t\t{:.4}\n",
                    r.index, r.distance, r.alpha, r.kappa
                ));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "splatdb-cli-helpers-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn write_header_payload(path: &PathBuf, rows: u64, cols: u64, f32_elems: &[f32]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&rows.to_le_bytes());
        buf.extend_from_slice(&cols.to_le_bytes());
        buf.extend_from_slice(bytemuck::cast_slice(f32_elems));
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn load_vectors_bin_rejects_truncated_and_oversized_headers() {
        let dir = temp_dir("vectors-bin");
        std::fs::create_dir_all(&dir).unwrap();

        // 1) Alloc bomb: header promises 50_000 x 50_000 = 2.5e9 elements
        //    (over the 1e9 cap) with a 16-byte file -> clean error, no OOM.
        let p = dir.join("bomb.bin");
        write_header_payload(&p, 50_000, 50_000, &[]);
        let err = load_vectors_bin(&p).unwrap_err();
        assert!(
            err.contains("Refusing excessive allocation"),
            "got: {}",
            err
        );

        // 2) Truncated: header promises 2x3 f32 but payload is cut short.
        let p = dir.join("truncated.bin");
        write_header_payload(&p, 2, 3, &[1.0]);
        let err = load_vectors_bin(&p).unwrap_err();
        assert!(err.contains("Truncated file"), "got: {}", err);

        // 3) Happy path: full 2x3 payload loads with correct shape/values.
        let p = dir.join("ok.bin");
        write_header_payload(&p, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = load_vectors_bin(&p).unwrap();
        assert_eq!(v.dim(), (2, 3));
        assert_eq!(v[[1, 2]], 6.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_query_bin_rejects_truncated_and_oversized_headers() {
        let dir = temp_dir("query-bin");
        std::fs::create_dir_all(&dir).unwrap();

        // Alloc bomb: 1 x 2_000_000_000 f32 header, tiny file.
        let p = dir.join("bomb.bin");
        write_header_payload(&p, 1, 2_000_000_000, &[]);
        let err = load_query_bin(&p).unwrap_err();
        assert!(
            err.contains("Refusing excessive allocation"),
            "got: {}",
            err
        );

        // Truncated: 1x4 promised, only 2 f32 present.
        let p = dir.join("truncated.bin");
        write_header_payload(&p, 1, 4, &[1.0, 2.0]);
        let err = load_query_bin(&p).unwrap_err();
        assert!(err.contains("Truncated file"), "got: {}", err);

        // Happy path.
        let p = dir.join("ok.bin");
        write_header_payload(&p, 1, 3, &[0.5, 1.5, 2.5]);
        let q = load_query_bin(&p).unwrap();
        assert_eq!(q.len(), 3);
        assert_eq!(q[2], 2.5);

        std::fs::remove_dir_all(&dir).ok();
    }
    #[test]
    fn try_load_shard_store_fails_closed_on_mismatch_and_unreadable_shards() {
        let dir = temp_dir("shard-store");
        std::fs::create_dir_all(&dir).unwrap();
        let p = SplatsDBPersistence::new(dir.to_str().unwrap(), false).unwrap();
        let vectors = Array2::from_shape_fn((2, 32), |(i, j)| (i * 32 + j) as f32);
        p.save_vectors(&vectors, "default").unwrap();

        // configuración correcta: carga
        let (store, _) =
            try_load_shard_store(dir.to_str().unwrap(), &make_config(32, 10), "default").unwrap();
        assert_eq!(store.n_active(), 2);

        // --dim que no coincide: Err, nunca un store vacío con 0 resultados
        let err = try_load_shard_store(dir.to_str().unwrap(), &make_config(16, 10), "default")
            .err()
            .expect("try_load_shard_store must fail closed");
        assert!(err.contains("mismatch"), "{err}");
        // capacidad insuficiente
        let err = try_load_shard_store(dir.to_str().unwrap(), &make_config(32, 1), "default")
            .err()
            .expect("try_load_shard_store must fail closed");
        assert!(err.contains("mismatch"), "{err}");

        // shard ilegible
        let shard = dir.join("data").join("vectors").join("default.bin");
        std::fs::write(&shard, [7u8; 40]).unwrap();
        let err = try_load_shard_store(dir.to_str().unwrap(), &make_config(32, 10), "default")
            .err()
            .expect("try_load_shard_store must fail closed");
        assert!(err.contains("failed to load shard"), "{err}");

        // data-dir que no se puede abrir (un archivo donde va el directorio)
        let bogus = dir.join("not-a-dir");
        std::fs::write(&bogus, b"x").unwrap();
        let err = try_load_shard_store(bogus.to_str().unwrap(), &make_config(32, 10), "default")
            .err()
            .expect("try_load_shard_store must fail closed");
        assert!(err.contains("cannot open data-dir"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── F02f: the CLI loader applies the persisted energy state like serve ──

    /// The F02e1 corpus: six cluster centres with a different spread each
    /// (so `ingest` derives a different kappa per cluster), eight noisy
    /// chunks around every centre, in 32 dimensions.
    fn energy_corpus() -> Array2<f32> {
        const K: usize = 6;
        const CENTRES: [[f32; 3]; K] = [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.1],
            [-1.3, 0.7, 0.0],
            [0.6, -1.4, 0.0],
        ];
        const SPREADS: [f64; K] = [0.02, 0.5, 0.1, 0.15, 0.3, 0.05];
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
        let mut rows = Vec::with_capacity(K * 8);
        for (c, centre) in CENTRES.iter().enumerate() {
            for _ in 0..8 {
                let mut row = vec![0.0f32; 32];
                for (j, m) in centre.iter().enumerate() {
                    row[j] = m + normal(SPREADS[c]) as f32;
                }
                rows.extend(row);
            }
        }
        Array2::from_shape_vec((K * 8, 32), rows).unwrap()
    }

    /// `load_or_create_store` (every CLI reader/writer) feeds
    /// `try_load_shard_store`: it must load the alpha/kappa `ingest`
    /// persisted — exactly as serve does since F02e1.
    #[test]
    fn try_load_shard_store_loads_the_persisted_energy_state_like_serve() {
        use ndarray::s;
        let config = make_config(32, 1000);
        let (init_alpha, init_kappa) = (config.init_alpha as f32, config.init_kappa as f32);
        let mut ingested = SplatStore::new(config.clone());
        match ingested.ingest_with_transformer(&energy_corpus(), 6, 7) {
            Ok(_) => {}
            Err(e) => panic!("ingest_with_transformer: {e}"),
        }
        let (mu, alpha, kappa) = (
            ingested.get_mu().unwrap(),
            ingested.get_alpha().unwrap(),
            ingested.get_kappa().unwrap(),
        );
        assert!(
            alpha.iter().any(|a| *a != init_alpha) && kappa.iter().any(|k| *k != init_kappa),
            "the corpus must derive an energy state that differs from the init values"
        );
        let dir = temp_dir("energy-cli");
        {
            let persist = make_persistence(dir.to_str().unwrap(), "sqlite", false).unwrap();
            persist.save_vectors(&mu, "default").unwrap();
            persist
                .save_energy_state(&mu, &alpha.to_vec(), &kappa.to_vec())
                .unwrap();
        }

        // 1) The CLI loader and the library contract (`load_splats`) agree.
        let (cli, _) = try_load_shard_store(dir.to_str().unwrap(), &config, "default").unwrap();
        assert_eq!(cli.get_alpha().unwrap(), alpha, "the CLI loader must load alpha");
        assert_eq!(cli.get_kappa().unwrap(), kappa, "the CLI loader must load kappa");
        let mut reference = SplatStore::new(config.clone());
        reference
            .load_splats(&mu, alpha.as_slice().unwrap(), kappa.as_slice().unwrap())
            .unwrap();
        let mut q = vec![0.0f32; 32];
        q[..3].copy_from_slice(&[0.6, 0.0, 0.0]); // between centre 0 and centre 1
        let q = Array1::from(q);
        let ranking = |s: &SplatStore| -> Vec<(usize, u32)> {
            s.find_neighbors_gaussian(&q.view(), 6)
                .iter()
                .map(|n| (n.index, n.score.to_bits()))
                .collect()
        };
        assert_eq!(
            ranking(&cli),
            ranking(&reference),
            "the CLI loader must rank like load_splats"
        );
        let mut lost = SplatStore::new(config.clone());
        lost.load_splats(&mu, &[init_alpha; 6], &[init_kappa; 6])
            .unwrap();
        assert_ne!(
            ranking(&lost),
            ranking(&reference),
            "the query must tell the persisted state from the init values"
        );

        // 2) No landscape: the init values, exactly as in serve.
        let bare = temp_dir("energy-absent");
        make_persistence(bare.to_str().unwrap(), "sqlite", false)
            .unwrap()
            .save_vectors(&mu, "default")
            .unwrap();
        let (store, _) = try_load_shard_store(bare.to_str().unwrap(), &config, "default").unwrap();
        assert!(store.get_alpha().unwrap().iter().all(|a| *a == init_alpha));
        assert!(store.get_kappa().unwrap().iter().all(|k| *k == init_kappa));

        // 3) A corrupt landscape fails closed, like an unreadable shard.
        std::fs::write(bare.join("energy").join("landscape.bin"), [0x78u8; 40]).unwrap();
        let err = try_load_shard_store(bare.to_str().unwrap(), &config, "default")
            .err()
            .expect("a corrupt energy state must not degrade to init values");
        assert!(err.contains("energy state"), "{err}");

        // 4) A landscape that describes another dataset is ignored, not applied.
        make_persistence(bare.to_str().unwrap(), "sqlite", false)
            .unwrap()
            .save_energy_state(&mu.mapv(|x| x + 5.0), &alpha.to_vec(), &kappa.to_vec())
            .unwrap();
        let (store, _) = try_load_shard_store(bare.to_str().unwrap(), &config, "default").unwrap();
        assert!(store.get_alpha().unwrap().iter().all(|a| *a == init_alpha));

        // 5) A row the landscape does not know keeps the init values.
        let mut grown = Array2::zeros((7, 32));
        grown.slice_mut(s![..6, ..]).assign(&mu);
        grown.row_mut(6).assign(&Array1::from(vec![9.0f32; 32]));
        make_persistence(dir.to_str().unwrap(), "sqlite", false)
            .unwrap()
            .save_vectors(&grown, "default")
            .unwrap();
        let (store, _) = try_load_shard_store(dir.to_str().unwrap(), &config, "default").unwrap();
        assert_eq!(store.get_alpha().unwrap().slice(s![..6]), alpha.slice(s![..6]));
        assert_eq!(store.get_kappa().unwrap().slice(s![..6]), kappa.slice(s![..6]));
        assert_eq!(store.get_alpha().unwrap()[6], init_alpha);
        assert_eq!(store.get_kappa().unwrap()[6], init_kappa);

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&bare);
    }
}
