//! Persistence layer — vectors (binary), metadata (pluggable backend), energy state, WAL.
//!
//! Supports any structured database via the `MetadataStore` trait.
//! Backend selection via `StorageBackend` enum or feature flags.

use bytemuck::cast_slice;
use ndarray::Array2;
use std::path::{Component, Path, PathBuf};

use super::metadata_store::{DocumentRecord, MetadataStore};
use super::wal::WriteAheadLog;

/// Available storage backends.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum StorageBackend {
    /// SQLite — default, production-ready.
    #[default]
    Sqlite,
    /// JSON file — simple, no native deps. Good for dev/testing.
    JsonFile,
    /// Custom backend registered at runtime.
    Custom(String),
}

/// SplatsDB Persistence manager.
///
/// Binary vector storage is always file-based (shards).
/// Metadata storage is pluggable via `MetadataStore` trait.
pub struct SplatsDBPersistence {
    pub storage_path: PathBuf,
    pub wal: Option<WriteAheadLog>,
    meta: Box<dyn MetadataStore>,
}

/// Validate that a shard name contains no path separators or traversal sequences.
fn validate_shard_name(name: &str) -> std::io::Result<()> {
    if name.contains('/') || name.contains('\\') || name.contains("..") || name.contains('\0') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid shard name",
        ));
    }
    Ok(())
}

/// Parent directory of `path` (`.` for a bare file name).
pub(crate) fn parent_dir(path: &Path) -> PathBuf {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Make a rename inside `dir` durable: fsync the directory itself.
///
/// `rename` is atomic, but the new directory entry only lives in the page
/// cache until the directory is fsynced — without this step a power loss
/// can bring back the pre-rename name (D4 / A03-F02).
///
/// Filesystems that cannot sync a directory answer `EINVAL` / `ENOTSUP`: the
/// rename stays atomic there, it just cannot be made durable explicitly, so
/// the condition is reported once on stderr instead of failing every write.
pub(crate) fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(not(unix))]
    {
        // Directory fsync is a POSIX notion; nothing to do elsewhere.
        let _ = dir;
        Ok(())
    }
    #[cfg(unix)]
    {
        let file = std::fs::File::open(dir)?;
        match file.sync_all() {
            Ok(()) => Ok(()),
            Err(e) if matches!(e.raw_os_error(), Some(libc::EINVAL) | Some(libc::ENOTSUP)) => {
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[storage] WARNING: directory {} cannot be fsynced ({}); renames there stay atomic but cannot be made durable explicitly",
                        dir.display(),
                        e
                    );
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

/// Create a new, uniquely named sibling of `path` and return its path plus
/// an append-mode handle to it.
///
/// The name embeds the pid, a process-wide counter and a random value, and
/// is opened with `create_new` (`O_CREAT | O_EXCL`): it can neither collide
/// with another concurrent caller nor be pre-created by someone else — a
/// symlink planted under a predictable name makes `create_new` fail instead
/// of being followed (CWE-377 / CWE-59, A05-F09).
pub(crate) fn create_unique_sibling(
    path: &Path,
    kind: &str,
) -> std::io::Result<(PathBuf, std::fs::File)> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed".to_string());
    let mut last = None;
    for _ in 0..64 {
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp_path = path.with_file_name(format!(
            "{file_name}.{kind}.{}.{seq}.{:016x}",
            std::process::id(),
            rand::random::<u64>()
        ));
        match std::fs::OpenOptions::new()
            .append(true)
            .create_new(true)
            .open(&tmp_path)
        {
            Ok(file) => return Ok((tmp_path, file)),
            // The name is taken: it is NOT ours, leave it alone and try another.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        std::io::Error::other("create_unique_sibling: could not allocate a unique name")
    }))
}

/// Atomically write `bytes` to `path`.
///
/// Contract (crate-internal, used by `save_vectors`, `save_energy_state` and
/// `json_store::persist`):
///
/// 1. Writes to a NEW sibling temp file `<file_name>.tmp.<pid>.<seq>.<random>`
///    in the same directory as `path`, opened with `create_new`. The name is
///    unique per call (not just per process), so concurrent callers never
///    share a temp file and a planted symlink/name is never followed.
/// 2. `write_all` + `sync_all` so the payload reaches stable storage.
/// 3. `rename` over `path` — atomic on Unix and Windows (replaces the
///    destination), so readers never observe a partially written file and a
///    crash mid-write leaves the previous contents intact.
/// 4. fsync of the parent directory, so the rename itself survives a power
///    loss (without it the old name can come back).
///
/// On any failure before the rename the temp file is removed and `path` is
/// left untouched; a failed directory fsync is reported as an error (the
/// rename itself already happened).
pub fn atomic_write(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    let (tmp_path, mut file) = create_unique_sibling(path, "tmp")?;

    let result = (|| -> std::io::Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, path)?;
        sync_dir(&parent_dir(path))
    })();

    if let Err(err) = result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    Ok(())
}

impl SplatsDBPersistence {
    /// Create with a specific backend.
    pub fn with_backend(
        storage_path: &str,
        backend: StorageBackend,
        enable_wal: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let resolved = validate_path(storage_path)?;

        // Create directory structure
        for dir in &[
            "data/vectors",
            "data/metadata",
            "index",
            "wal/checkpoint",
            "energy",
        ] {
            std::fs::create_dir_all(resolved.join(dir))?;
        }

        let meta: Box<dyn MetadataStore> = match backend {
            StorageBackend::Sqlite => {
                let db_path = resolved.join("data").join("metadata").join("metadata.db");
                Box::new(super::sqlite_store::SqliteMetadataStore::new(db_path)?)
            }
            StorageBackend::JsonFile => {
                let json_path = resolved.join("data").join("metadata").join("metadata.json");
                Box::new(super::json_store::JsonMetadataStore::new(json_path)?)
            }
            StorageBackend::Custom(name) => {
                return Err(format!(
                    "Custom backend '{}' not registered. Use `with_custom_store()` instead.",
                    name
                )
                .into());
            }
        };

        let wal = if enable_wal {
            let wal_path = resolved.join("wal").join("wal.log");
            Some(WriteAheadLog::new(wal_path.to_str().unwrap_or(""), 50)?)
        } else {
            None
        };

        Ok(Self {
            storage_path: resolved,
            wal,
            meta,
        })
    }

    /// Create with default backend (SQLite).
    pub fn new(
        storage_path: &str,
        enable_wal: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_backend(storage_path, StorageBackend::Sqlite, enable_wal)
    }

    /// Create with a custom MetadataStore implementation.
    pub fn with_custom_store(
        storage_path: &str,
        meta_store: Box<dyn MetadataStore>,
        enable_wal: bool,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let resolved = validate_path(storage_path)?;
        for dir in &[
            "data/vectors",
            "data/metadata",
            "index",
            "wal/checkpoint",
            "energy",
        ] {
            std::fs::create_dir_all(resolved.join(dir))?;
        }
        let wal = if enable_wal {
            let wal_path = resolved.join("wal").join("wal.log");
            Some(WriteAheadLog::new(wal_path.to_str().unwrap_or(""), 50)?)
        } else {
            None
        };
        Ok(Self {
            storage_path: resolved,
            wal,
            meta: meta_store,
        })
    }

    /// Get the active backend name.
    pub fn backend_name(&self) -> &'static str {
        self.meta.backend_name()
    }

    // ---- Vector Storage (file-based) ----

    pub fn save_vectors(&self, vectors: &Array2<f32>, shard_name: &str) -> std::io::Result<()> {
        validate_shard_name(shard_name)?;
        let path = self
            .storage_path
            .join("data")
            .join("vectors")
            .join(format!("{}.bin", shard_name));
        // Buffer the full payload, then write it atomically: a crash mid-save
        // leaves the previous shard file intact instead of a truncated one.
        let mut buf = Vec::with_capacity(16 + vectors.len() * std::mem::size_of::<f32>());
        buf.extend_from_slice(&(vectors.nrows() as u64).to_le_bytes());
        buf.extend_from_slice(&(vectors.ncols() as u64).to_le_bytes());
        for row in vectors.rows() {
            let slice = row.as_slice().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "non-contiguous row")
            })?;
            let bytes: &[u8] = cast_slice(slice);
            buf.extend_from_slice(bytes);
        }
        atomic_write(&path, &buf)
    }

    pub fn load_vectors(&self, shard_name: &str) -> std::io::Result<Option<Array2<f32>>> {
        validate_shard_name(shard_name)?;
        let path = self
            .storage_path
            .join("data")
            .join("vectors")
            .join(format!("{}.bin", shard_name));
        if !path.exists() {
            return Ok(None);
        }
        let mut file = std::fs::File::open(path)?;
        use std::io::Read;
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let rows: usize = u64::from_le_bytes(buf8)
            .try_into()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        file.read_exact(&mut buf8)?;
        let cols: usize = u64::from_le_bytes(buf8)
            .try_into()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let total = rows.checked_mul(cols).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "dimension overflow")
        })?;
        if total > 1_000_000_000 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "refusing excessive allocation",
            ));
        }
        let mut data = vec![0.0f32; total];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
        file.read_exact(bytes)?;
        Ok(Some(
            ndarray::Array2::from_shape_vec((rows, cols), data)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        ))
    }

    pub fn list_shards(&self) -> Vec<String> {
        let dir = self.storage_path.join("data").join("vectors");
        std::fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
                    .filter_map(|e| {
                        e.path()
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    // ---- Metadata (delegates to MetadataStore) ----

    pub fn save_metadata(
        &self,
        doc_id: &str,
        shard_idx: i64,
        vector_idx: i64,
        metadata: Option<&serde_json::Value>,
        document: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        let record = DocumentRecord {
            id: doc_id.into(),
            shard_idx,
            vector_idx,
            metadata: metadata.cloned(),
            document: document.map(|s| s.into()),
            deleted: false,
            created_at: now,
            updated_at: now,
        };
        self.meta.upsert(&record)
    }

    pub fn get_metadata(
        &self,
        doc_id: &str,
    ) -> Result<
        Option<std::collections::HashMap<String, serde_json::Value>>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        match self.meta.get(doc_id)? {
            Some(record) => Ok(Some(record.to_hashmap())),
            None => Ok(None),
        }
    }

    pub fn soft_delete(
        &self,
        doc_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.soft_delete(doc_id)
    }

    pub fn hard_delete(
        &self,
        doc_id: &str,
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.hard_delete(doc_id)
    }

    pub fn get_all_ids(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.list_ids(include_deleted)
    }

    pub fn count_documents(
        &self,
        include_deleted: bool,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.count(include_deleted)
    }

    // ---- Direct Document Access (for Verbatim Storage) ----

    /// Upsert a document record directly.
    pub fn upsert_doc(
        &self,
        record: &DocumentRecord,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.meta.upsert(record)
    }

    /// Get a document record by ID.
    pub fn get_doc(
        &self,
        doc_id: &str,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.get(doc_id)
    }

    /// Find a document by vector index.
    ///
    /// Delegates to the backend: SQLite uses the `idx_vector` index (see
    /// `SqliteMetadataStore::find_by_vector_idx` and its parity test); other
    /// backends fall back to the generic O(N) scan.
    pub fn find_doc_by_vector_idx(
        &self,
        vector_idx: i64,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.find_by_vector_idx(vector_idx)
    }

    // ---- Energy State (file-based) ----

    pub fn save_energy_state(
        &self,
        mu: &Array2<f32>,
        alpha: &[f32],
        kappa: &[f32],
    ) -> std::io::Result<()> {
        let path = self.storage_path.join("energy").join("landscape.bin");
        // Buffer the full payload, then write it atomically: a crash mid-save
        // leaves the previous energy state intact instead of a truncated one.
        let mut buf = Vec::with_capacity(
            32 + (mu.len() + alpha.len() + kappa.len()) * std::mem::size_of::<f32>(),
        );
        buf.extend_from_slice(&(mu.nrows() as u64).to_le_bytes());
        buf.extend_from_slice(&(mu.ncols() as u64).to_le_bytes());
        for row in mu.rows() {
            let s = row.as_slice().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "non-contiguous row")
            })?;
            let bytes: &[u8] = cast_slice(s);
            buf.extend_from_slice(bytes);
        }
        buf.extend_from_slice(&(alpha.len() as u64).to_le_bytes());
        buf.extend_from_slice(cast_slice(alpha));
        buf.extend_from_slice(&(kappa.len() as u64).to_le_bytes());
        buf.extend_from_slice(cast_slice(kappa));
        atomic_write(&path, &buf)
    }

    // ---- Backup ----

    pub fn backup(
        &self,
        backup_path: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let dest = PathBuf::from(backup_path);
        if dest.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err("Path traversal detected in backup_path".into());
        }
        self.meta.flush()?;
        std::fs::create_dir_all(&dest)?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let backup_dest = dest.join(format!("m2m_backup_{}", ts));
        copy_dir_recursive(&self.storage_path, &backup_dest, 0)?;
        Ok(backup_dest.to_string_lossy().into_owned())
    }

    // ---- WAL ----

    pub fn checkpoint(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref wal) = self.wal {
            wal.checkpoint()?;
        }
        Ok(())
    }

    pub fn close(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.meta.flush()?;
        if let Some(ref wal) = self.wal {
            wal.close()?;
        }
        Ok(())
    }
}

fn validate_path(storage_path: &str) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let path = PathBuf::from(storage_path);
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err("Path traversal detected in storage_path".into());
    }
    Ok(path.canonicalize().unwrap_or(path))
}

fn copy_dir_recursive(src: &Path, dst: &Path, depth: u32) -> std::io::Result<()> {
    if depth > 128 {
        return Err(std::io::Error::other("directory depth limit exceeded"));
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() && !src_path.is_symlink() {
            copy_dir_recursive(&src_path, &dst_path, depth + 1)?;
        } else if src_path.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_target(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "splatdb-atomic-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("target.bin")
    }

    fn dir_entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// A05-F09 — the temp file must never be a predictable, pre-existing
    /// name: a symlink planted at `<file>.tmp.<pid>` must not be written
    /// through.
    #[cfg(unix)]
    #[test]
    fn atomic_write_does_not_follow_a_precreated_tmp_symlink() {
        let target = temp_target("symlink");
        let dir = target.parent().unwrap();
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, b"victim-original").unwrap();
        let planted = dir.join(format!("target.bin.tmp.{}", std::process::id()));
        std::os::unix::fs::symlink(&victim, &planted).unwrap();

        atomic_write(&target, b"payload").unwrap();

        assert_eq!(
            std::fs::read(&victim).unwrap(),
            b"victim-original",
            "atomic_write wrote through the planted symlink"
        );
        assert!(
            !target.symlink_metadata().unwrap().file_type().is_symlink(),
            "destination must be a regular file"
        );
        assert_eq!(std::fs::read(&target).unwrap(), b"payload");
        std::fs::remove_dir_all(dir).ok();
    }

    /// A03-F08 / A05-F09 — concurrent callers in the same process (same pid)
    /// must not share a temp file.
    #[test]
    fn atomic_write_concurrent_callers_do_not_share_a_temp_file() {
        let target = temp_target("concurrent");
        let dir = target.parent().unwrap().to_path_buf();
        let payload_len = 256 * 1024;
        std::thread::scope(|s| {
            for t in 0..8u8 {
                let target = &target;
                s.spawn(move || {
                    // 48 fsync-bound writes keep this in unit-test range; on the old
                    // code writers already collided within their first two iterations.
                    for i in 0..6usize {
                        let payload = vec![t.wrapping_add(i as u8); payload_len];
                        atomic_write(target, &payload)
                            .unwrap_or_else(|e| panic!("writer {t} iteration {i}: {e}"));
                    }
                });
            }
        });
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), payload_len, "torn final content");
        assert!(
            final_bytes.iter().all(|&b| b == final_bytes[0]),
            "final file mixes two payloads"
        );
        let leftovers: Vec<String> = dir_entries(&dir)
            .into_iter()
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A failure after the temp file was written must leave no temp behind
    /// and must not touch the destination.
    #[test]
    fn atomic_write_failure_leaves_no_temp_and_keeps_the_destination() {
        let base = temp_target("failure");
        let dir = base.parent().unwrap();
        let target = dir.join("subdir");
        std::fs::create_dir(&target).unwrap();

        // rename(file -> existing directory) fails after the temp was
        // written and fsynced.
        let err = atomic_write(&target, b"payload").unwrap_err();

        assert!(target.is_dir(), "destination was replaced");
        assert_eq!(dir_entries(dir), ["subdir"], "temp not cleaned up ({err})");
        std::fs::remove_dir_all(dir).ok();
    }

    /// A03-F02 — traced at integration time with the LD_PRELOAD fsync
    /// tracer: must show `fsync(temp)`, `rename`, then `fsync(parent dir)`.
    #[test]
    fn fsync_probe_atomic_write_syncs_file_then_parent_dir() {
        let target = temp_target("fsync-probe");
        let dir = target.parent().unwrap().to_path_buf();
        atomic_write(&target, b"probe-payload").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"probe-payload");
        assert_eq!(dir_entries(&dir), ["target.bin"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
