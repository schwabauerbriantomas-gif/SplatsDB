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

#[cfg(test)]
mod tests {
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
