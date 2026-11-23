//! Guards against two writers clobbering each other's shard (A03-F12).
//!
//! - [`DataDirLock`]: exclusive advisory lock (kernel `flock`) that `serve`
//!   holds on `{data-dir}/.splatsdb.lock` for its whole lifetime, so two
//!   `serve` processes never load, replay and rewrite the same data-dir.
//!   The decision is the KERNEL lock — never the file's existence or the PID
//!   it contains: a lockfile left behind by a dead process never blocks, and
//!   the lock dies with the process even on SIGKILL.
//! - [`ShardFingerprint`]: content identity (byte length + two 64-bit
//!   multiplicative hashes) of the shard file. `Durability` records it when
//!   a WAL generation starts and refuses to overwrite the shard if the
//!   content changed since — the CLI (or anything else) may rewrite the
//!   shard while `serve` runs.
//! - [`WalBase`]: the fingerprint the current WAL generation was opened
//!   against, kept next to the WAL in `wal/store.wal.base`, so the NEXT start
//!   can tell "entries already in the shard" from "shard replaced under an
//!   un-compacted WAL" instead of guessing from `idx`.
//!
//! The hashes are non-cryptographic: they detect accidental modification (a
//! rewritten shard), not adversarial tampering.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::persistence::atomic_write;

/// Lock file inside the data-dir. It is never removed: the kernel lock, not
/// the file, is what excludes a second `serve`.
pub const LOCK_FILE: &str = ".splatsdb.lock";
/// Sidecar of `wal/store.wal`: the shard this WAL generation was opened against.
pub const WAL_BASE_FILE: &str = "store.wal.base";

/// Content identity of a shard file: byte length plus two independent
/// 64-bit multiplicative hashes of the content. mtime and inode are
/// deliberately NOT part of it: rewriting identical bytes is not a conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardFingerprint {
    pub len: u64,
    pub hash_a: u64,
    pub hash_b: u64,
}

/// Fingerprint of `path`, or `None` if the file does not exist.
pub fn fingerprint(path: &Path) -> std::io::Result<Option<ShardFingerprint>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut hash_a: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hash_b: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut len: u64 = 0;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            let b = u64::from(byte);
            hash_a = (hash_a ^ b).wrapping_mul(0x0000_0100_0000_01b3);
            hash_b = (hash_b ^ b).wrapping_mul(0xd6e8_feb8_6659_fd93);
        }
        len += n as u64;
    }
    Ok(Some(ShardFingerprint {
        len,
        hash_a,
        hash_b,
    }))
}

/// Shard `default` a WAL generation was opened against (`None` = there was
/// no shard file when the generation started).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalBase {
    pub shard: Option<ShardFingerprint>,
}

/// Read the WAL base sidecar. `Ok(None)` = no sidecar (legacy data-dir).
pub fn read_wal_base(path: &Path) -> std::io::Result<Option<WalBase>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    match serde_json::from_slice::<WalBase>(&bytes) {
        Ok(base) => Ok(Some(base)),
        Err(e) => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!("{} is not a valid WAL base record: {e}", path.display()),
        )),
    }
}

/// Atomically replace the WAL base sidecar (tmp + fsync + rename + dir fsync).
pub fn write_wal_base(path: &Path, base: &WalBase) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(base).map_err(std::io::Error::other)?;
    atomic_write(path, &bytes)
}

/// Exclusive lock of a data-dir, held for as long as this value lives.
pub struct DataDirLock {
    file: File,
}

/// Linux errno values for "this filesystem cannot flock". The lock is then
/// unavailable (not held): callers keep going with a warning instead of
/// refusing to serve — the shard fingerprint guard still applies.
fn lock_unsupported(e: &std::io::Error) -> bool {
    const EINVAL: i32 = 22;
    const ENOLCK: i32 = 37;
    const ENOSYS: i32 = 38;
    const EOPNOTSUPP: i32 = 95;
    matches!(
        e.raw_os_error(),
        Some(EINVAL) | Some(ENOLCK) | Some(ENOSYS) | Some(EOPNOTSUPP)
    )
}

impl DataDirLock {
    /// Lock file path inside `data_dir`.
    pub fn path(data_dir: &Path) -> PathBuf {
        data_dir.join(LOCK_FILE)
    }

    /// Take the exclusive lock, or fail with [`ErrorKind::WouldBlock`] while
    /// another `serve` holds it.
    pub fn acquire(data_dir: &Path) -> std::io::Result<Self> {
        let path = Self::path(data_dir);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(e) => {
                let e: std::io::Error = e.into();
                if e.kind() == ErrorKind::WouldBlock {
                    let holder = std::fs::read_to_string(&path)
                        .ok()
                        .and_then(|s| s.lines().next().map(str::trim).map(str::to_string))
                        .unwrap_or_else(|| "holder unknown".to_string());
                    return Err(std::io::Error::new(
                        ErrorKind::WouldBlock,
                        format!(
                            "data-dir {} ya está bloqueado por otro serve ({holder}); dos serve \
                             sobre el mismo data-dir se sobrescriben el shard",
                            data_dir.display()
                        ),
                    ));
                }
                if !lock_unsupported(&e) {
                    return Err(e);
                }
                eprintln!(
                    "[splatsdb] WARNING: flock no disponible en {} ({e}): sin exclusividad entre \
                     serve; el guard de shard sigue activo",
                    path.display()
                );
            }
        }

        // Our PID is informational only (it names the holder in the error
        // above); it never decides whether the lock is taken.
        let _ = file.set_len(0);
        let _ = file.seek(SeekFrom::Start(0));
        let _ = file.write_all(format!("pid={}\n", std::process::id()).as_bytes());
        Ok(Self { file })
    }
}

impl Drop for DataDirLock {
    fn drop(&mut self) {
        // Also released by the kernel when the process dies.
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "splatdb-shard-guard-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn lock_is_exclusive_between_threads_and_released_on_drop_or_error() {
        let dir = temp_dir("threads");
        let a = DataDirLock::acquire(&dir).expect("first acquire");
        let err = DataDirLock::acquire(&dir)
            .err()
            .expect("a second holder must be refused");
        assert_eq!(err.kind(), ErrorKind::WouldBlock, "{err}");
        assert!(err.to_string().contains("bloqueado"), "{err}");
        drop(a);
        // After the holder is gone (normal drop, an error path or a dead
        // process) the same lock file must not block anyone.
        let b = DataDirLock::acquire(&dir).expect("acquire after drop");
        drop(b);
        assert!(DataDirLock::acquire(&dir).is_ok(), "third acquire");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stale_lock_file_with_a_dead_pid_does_not_block() {
        let dir = temp_dir("stale");
        std::fs::write(DataDirLock::path(&dir), "pid=999999\n").unwrap();
        let lock = DataDirLock::acquire(&dir).expect("a stale lockfile must not block");
        let text = std::fs::read_to_string(DataDirLock::path(&dir)).unwrap();
        assert!(
            text.contains(&format!("pid={}", std::process::id())),
            "{text}"
        );
        drop(lock);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fingerprint_tracks_content_not_path_or_size_alone() {
        let dir = temp_dir("fingerprint");
        let a = dir.join("a.bin");
        assert_eq!(fingerprint(&a).unwrap(), None, "missing file");
        std::fs::write(&a, b"hello").unwrap();
        let f1 = fingerprint(&a).unwrap().unwrap();
        // Identical bytes in another file (other inode/mtime) -> same identity.
        let b = dir.join("b.bin");
        std::fs::write(&b, b"hello").unwrap();
        assert_eq!(fingerprint(&b).unwrap(), Some(f1));
        // Same length, one byte changed -> different identity.
        std::fs::write(&a, b"hellO").unwrap();
        assert_ne!(fingerprint(&a).unwrap(), Some(f1));
        // Larger than one read buffer.
        let big = dir.join("big.bin");
        std::fs::write(&big, vec![7u8; 200_000]).unwrap();
        let fb = fingerprint(&big).unwrap().unwrap();
        assert_eq!(fb.len, 200_000);
        let mut other = vec![7u8; 200_000];
        other[199_999] = 8;
        std::fs::write(&big, other).unwrap();
        assert_ne!(fingerprint(&big).unwrap(), Some(fb));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wal_base_roundtrip_missing_and_corrupt() {
        let dir = temp_dir("wal-base");
        let p = dir.join(WAL_BASE_FILE);
        assert_eq!(read_wal_base(&p).unwrap(), None);
        write_wal_base(&p, &WalBase { shard: None }).unwrap();
        assert_eq!(read_wal_base(&p).unwrap(), Some(WalBase { shard: None }));
        let fp = ShardFingerprint {
            len: 3,
            hash_a: 1,
            hash_b: 2,
        };
        write_wal_base(&p, &WalBase { shard: Some(fp) }).unwrap();
        assert_eq!(
            read_wal_base(&p).unwrap(),
            Some(WalBase { shard: Some(fp) })
        );
        std::fs::write(&p, b"{not json").unwrap();
        assert!(
            read_wal_base(&p).is_err(),
            "a corrupt base record is an error, not a silent reset"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
