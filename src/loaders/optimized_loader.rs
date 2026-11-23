//! Optimized data loader. Ported from splatsdb Python.

use ndarray::Array2;
use std::io::{Read, Write};

const MAX_LOAD_ELEMENTS: usize = 1_000_000_000;

/// Loads vector data from binary format (rows: u64, cols: u64, f32 data).
pub fn load_vectors_bin(path: &str) -> Result<Array2<f32>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "rows overflow")?;

    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "cols overflow")?;

    let total = rows.checked_mul(cols).ok_or("overflow in rows*cols")?;
    if total > MAX_LOAD_ELEMENTS {
        return Err(format!(
            "Allocation too large: {} elements (max {})",
            total, MAX_LOAD_ELEMENTS
        ));
    }

    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

/// Saves vectors to binary format — atomically (A05-F09).
///
/// The destination is never truncated in place. The bytes go to a NEW
/// uniquely named sibling (`persistence::create_unique_sibling`, opened with
/// `create_new`), are `sync_all`'d, the temp file is `rename`d over `path`
/// and the directory is fsync'd. A crash, ENOSPC or EIO mid-write leaves the
/// previous file complete and removes the temp file.
///
/// Before this, `File::create(path)` truncated the live shard first: any
/// failure afterwards (including the `as_slice().unwrap()` panic on a
/// non-standard layout) had already destroyed the last valid version.
pub fn save_vectors_bin(path: &str, vectors: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = vectors.dim();
    let dest = std::path::Path::new(path);
    // Standard-layout arrays are written straight from memory; any other
    // layout (a sliced/transposed view) is serialised row by row instead of
    // panicking.
    let owned: Vec<u8>;
    let bytes: &[u8] = if let Some(flat) = vectors.as_slice() {
        bytemuck::cast_slice(flat)
    } else {
        owned = vectors.iter().flat_map(|v| v.to_ne_bytes()).collect();
        owned.as_slice()
    };
    let (tmp_path, mut file) = crate::storage::persistence::create_unique_sibling(dest, "tmp")
        .map_err(|e| format!("Cannot write {}: {}", path, e))?;
    let result = (|| -> std::io::Result<()> {
        file.write_all(&rows.to_le_bytes())?;
        file.write_all(&cols.to_le_bytes())?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, dest)?;
        crate::storage::persistence::sync_dir(&crate::storage::persistence::parent_dir(dest))
    })();
    if let Err(err) = result {
        // `path` is untouched: drop the partial temp file.
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!("Cannot write {}: {}", path, err));
    }
    Ok(())
}

/// Persisted energy landscape — `energy/landscape.bin`, exactly as
/// `storage::persistence::SplatsDBPersistence::save_energy_state` writes it:
/// `[u64 rows][u64 cols][f32 mu, row-major][u64 n_alpha][f32 alpha]
/// [u64 n_kappa][f32 kappa]`, little-endian.
///
/// Before F02e1 nothing in the crate read this file back: `ingest`, `save`
/// and `verbatim store` wrote it and every reload (`serve`, the CLI)
/// answered with the init alpha/kappa.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyLandscape {
    /// Splat centres the state belongs to: `rows × cols`.
    pub mu: Array2<f32>,
    /// Per-splat amplitudes `[rows]`.
    pub alpha: Vec<f32>,
    /// Per-splat concentrations `[rows]`.
    pub kappa: Vec<f32>,
}

impl EnergyLandscape {
    /// Path of the landscape inside a data-dir.
    pub fn path_in(data_dir: &std::path::Path) -> std::path::PathBuf {
        data_dir.join("energy").join("landscape.bin")
    }

    /// Number of leading rows whose mu equals `shard`'s row-wise — `0` when
    /// the state belongs to another dataset (dim or first row differ).
    pub fn matching_prefix(&self, shard: &Array2<f32>) -> usize {
        if self.mu.ncols() != shard.ncols() {
            return 0;
        }
        let m = self.mu.nrows().min(shard.nrows());
        (0..m)
            .all(|i| self.mu.row(i) == shard.row(i))
            .then_some(m)
            .unwrap_or(0)
    }
}

/// Largest `rows*cols` / vector length accepted from an untrusted header.
const MAX_ENERGY_ELEMENTS: usize = MAX_LOAD_ELEMENTS;

fn read_u64(file: &mut std::fs::File, what: &str) -> Result<u64, String> {
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error ({what}): {e}"))?;
    Ok(u64::from_le_bytes(buf8))
}

fn read_f32s(file: &mut std::fs::File, n: usize, what: &str) -> Result<Vec<f32>, String> {
    let mut data = vec![0.0f32; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error ({what}): {e}"))?;
    Ok(data)
}

fn header_len(value: u64, what: &str) -> Result<usize, String> {
    let n = usize::try_from(value).map_err(|_| format!("{what} overflow"))?;
    if n > MAX_ENERGY_ELEMENTS {
        return Err(format!(
            "Allocation too large: {what} = {n} elements (max {MAX_ENERGY_ELEMENTS})"
        ));
    }
    Ok(n)
}

/// Load `energy/landscape.bin`.
///
/// The header is untrusted input: every count is validated against the
/// on-disk size BEFORE allocating, and the file must be exactly as long as
/// its header promises — a truncated file, trailing bytes, a count that does
/// not match the rows, or a non-finite / negative alpha·kappa or mu value is
/// an `Err` (never a partially loaded state). The caller decides the policy
/// (`serve` fails closed, like with an unreadable shard).
pub fn load_energy_landscape(path: &str) -> Result<EnergyLandscape, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let size = file
        .metadata()
        .map_err(|e| format!("Cannot stat {}: {}", path, e))?
        .len();
    let rows = header_len(read_u64(&mut file, "rows")?, "rows")?;
    let cols = header_len(read_u64(&mut file, "cols")?, "cols")?;
    let mu_len = rows.checked_mul(cols).ok_or("overflow in rows*cols")?;
    if mu_len > MAX_ENERGY_ELEMENTS {
        return Err(format!(
            "Allocation too large: {rows}x{cols} elements (max {MAX_ENERGY_ELEMENTS})"
        ));
    }
    // The counts follow the mu payload: skip over it (already validated
    // against the on-disk size above).
    std::io::Seek::seek(
        &mut file,
        std::io::SeekFrom::Current((mu_len as u64 * 4) as i64),
    )
    .map_err(|e| format!("Seek error: {e}"))?;
    let n_alpha = header_len(read_u64(&mut file, "n_alpha")?, "n_alpha")?;
    // Same for the alpha payload before the kappa count.
    std::io::Seek::seek(
        &mut file,
        std::io::SeekFrom::Current((n_alpha as u64 * 4) as i64),
    )
    .map_err(|e| format!("Seek error: {e}"))?;
    let n_kappa = header_len(read_u64(&mut file, "n_kappa")?, "n_kappa")?;
    let floats =
        u64::try_from(mu_len + n_alpha + n_kappa).map_err(|_| "overflow in element count")?;
    let expected = 32 + floats * 4;
    if size != expected {
        return Err(format!(
            "{} is {size} bytes, its header promises {expected} (truncated or trailing garbage)",
            path
        ));
    }
    if n_alpha != rows || n_kappa != rows {
        return Err(format!(
            "{} holds {n_alpha} alpha / {n_kappa} kappa values for {rows} splats",
            path
        ));
    }
    // Back to the payload start for the reads themselves.
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(16))
        .map_err(|e| format!("Seek error: {e}"))?;
    let mu_flat = read_f32s(&mut file, mu_len, "mu")?;
    // Payload order is mu | count | alpha | count | kappa:
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Current(8))
        .map_err(|e| format!("Seek error: {e}"))?;
    let alpha = read_f32s(&mut file, n_alpha, "alpha")?;
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Current(8))
        .map_err(|e| format!("Seek error: {e}"))?;
    let kappa = read_f32s(&mut file, n_kappa, "kappa")?;
    let mu =
        Array2::from_shape_vec((rows, cols), mu_flat).map_err(|e| format!("Shape error: {}", e))?;
    if mu.iter().any(|v| !v.is_finite()) {
        return Err(format!("{path} holds a non-finite mu value"));
    }
    for (name, v) in [("alpha", &alpha), ("kappa", &kappa)] {
        if v.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(format!(
                "{path} holds a non-finite or negative {name} value"
            ));
        }
    }
    Ok(EnergyLandscape { mu, alpha, kappa })
}

#[cfg(test)]
mod tests {
    #[test]
    fn load_energy_landscape_roundtrips_save_energy_state() {
        use crate::storage::persistence::{SplatsDBPersistence, StorageBackend};
        use ndarray::s;
        let dir = std::env::temp_dir().join(format!("F02e1-loader-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let persist =
            SplatsDBPersistence::with_backend(dir.to_str().unwrap(), StorageBackend::Sqlite, false)
                .unwrap();
        let mu = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32 - 2.0);
        let alpha = [0.5f32, 0.25, 1.0 / 3.0];
        let kappa = [0.1f32, 5.0, 100.0];
        persist.save_energy_state(&mu, &alpha, &kappa).unwrap();
        let path = EnergyLandscape::path_in(&dir);
        let l = load_energy_landscape(path.to_str().unwrap()).unwrap();
        assert_eq!(l.mu, mu);
        assert_eq!(l.alpha, alpha);
        assert_eq!(l.kappa, kappa);
        assert_eq!(l.matching_prefix(&mu), 3);
        let mut bigger = Array2::zeros((4, 4));
        bigger.slice_mut(s![..3, ..]).assign(&mu);
        assert_eq!(l.matching_prefix(&bigger), 3);
        let other = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32);
        assert_eq!(l.matching_prefix(&other), 0);
        assert_eq!(l.matching_prefix(&Array2::zeros((3, 5))), 0);
        // reject: truncated / trailing / hostile / mismatched / non-finite / negative
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap())
            .unwrap_err()
            .contains("bytes"));
        let mut trailing = bytes.clone();
        trailing.push(0);
        std::fs::write(&path, &trailing).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap()).is_err());
        let mut bomb = bytes.clone();
        bomb[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &bomb).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap())
            .unwrap_err()
            .contains("Allocation too large"));
        let mut mismatch = bytes.clone();
        let o = 16 + 3 * 4 * 4;
        mismatch[o..o + 8].copy_from_slice(&2u64.to_le_bytes());
        // n_alpha=2 / n_kappa=4 keeps the total size, so the size check passes
        // and the count check is the one that fires.
        mismatch[o + 8 + 2 * 4..o + 8 + 2 * 4 + 8].copy_from_slice(&4u64.to_le_bytes());
        std::fs::write(&path, &mismatch).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap())
            .unwrap_err()
            .contains("alpha /"));
        let mut nan = bytes.clone();
        nan[16 + 3 * 4 * 4 + 8..16 + 3 * 4 * 4 + 8 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&path, &nan).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap())
            .unwrap_err()
            .contains("non-finite"));
        let mut negative = bytes.clone();
        let o = 16 + 3 * 4 * 4 + 8 + 3 * 4 + 8;
        negative[o..o + 4].copy_from_slice(&(-1.0f32).to_le_bytes());
        std::fs::write(&path, &negative).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap())
            .unwrap_err()
            .contains("negative"));
        let mut nan_mu = bytes.clone();
        nan_mu[16..20].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&path, &nan_mu).unwrap();
        assert!(load_energy_landscape(path.to_str().unwrap())
            .unwrap_err()
            .contains("non-finite mu"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("F02c")
            .join(format!("savebin-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entries(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn save_vectors_bin_roundtrips_an_empty_array_and_overwrites_atomically() {
        let dir = scratch("empty");
        let path = dir.join("vectors.bin");
        let old = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f32);
        save_vectors_bin(path.to_str().unwrap(), &old).unwrap();
        // Header-only file (0 rows), replacing the previous one by rename.
        let empty = Array2::<f32>::zeros((0, 3));
        save_vectors_bin(path.to_str().unwrap(), &empty).unwrap();
        let after = load_vectors_bin(path.to_str().unwrap()).unwrap();
        assert_eq!(after.dim(), (0, 3));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 16);
        // Atomic overwrite: only the destination is left, no temp file.
        assert_eq!(entries(&dir), vec!["vectors.bin".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `CHILD` carries the target path in the re-executed test binary.
    #[cfg(unix)]
    const EFBIG_CHILD: &str = "SPLATSDB_TEST_SAVE_BIN_EFBIG_CHILD";

    /// 4096x64 f32 = 1 MiB payload.
    #[cfg(unix)]
    fn efbig_payload() -> Array2<f32> {
        Array2::from_shape_fn((4096, 64), |(i, _)| i as f32)
    }

    #[cfg(unix)]
    #[test]
    fn save_vectors_bin_mid_write_failure_keeps_the_previous_file_and_no_temp() {
        use std::os::unix::process::CommandExt;

        // Child role (re-executed by the parent with RLIMIT_FSIZE = 64 KiB):
        // the 1 MiB write fails mid-way with EFBIG.
        if let Some(target) = std::env::var_os(EFBIG_CHILD) {
            let r = save_vectors_bin(target.to_str().unwrap(), &efbig_payload());
            eprintln!("child save result: {r:?}");
            std::process::exit(0);
        }

        let dir = scratch("efbig");
        let path = dir.join("vectors.bin");
        let old = Array2::from_shape_fn((2, 3), |(i, j)| (i * 3 + j) as f32);
        save_vectors_bin(path.to_str().unwrap(), &old).unwrap();

        let out = unsafe {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "loaders::optimized_loader::tests::save_vectors_bin_mid_write_failure_keeps_the_previous_file_and_no_temp",
                    "--nocapture",
                ])
                .env(EFBIG_CHILD, &path)
                .pre_exec(|| {
                    libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
                    let rl = libc::rlimit {
                        rlim_cur: 64 * 1024,
                        rlim_max: libc::RLIM_INFINITY,
                    };
                    libc::setrlimit(libc::RLIMIT_FSIZE, &rl);
                    Ok(())
                })
                .output()
                .expect("spawn the child writer")
        };
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        eprintln!("child writer: status={} stderr={err}", out.status);
        assert!(out.status.success(), "child writer failed: {err}");
        assert!(
            err.contains("child save result: Err("),
            "the child's write must fail mid-way: {err}"
        );
        // The previous file survived intact ...
        let after = load_vectors_bin(path.to_str().unwrap())
            .expect("the previous file must still load after a failed save");
        assert_eq!(after, old);
        // ... and the partial temp file was removed.
        assert_eq!(entries(&dir), vec!["vectors.bin".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vectors = Array2::from_shape_vec((2, 3), data).unwrap();
        save_vectors_bin("test_loader.bin", &vectors).unwrap();
        let loaded = load_vectors_bin("test_loader.bin").unwrap();
        assert_eq!(loaded.dim(), (2, 3));
        assert_eq!(&loaded, &vectors);
        std::fs::remove_file("test_loader.bin").ok();
    }
}
