//! Write-Ahead Log for durability.
//! Binary format: [4-byte big-endian length][JSON payload].
//!
//! The on-disk format is unchanged (no CRC, no version header): every
//! pre-existing WAL stays readable byte for byte.
//!
//! Crash-safety contract:
//! - An entry is ONE `write_all` of `[len][payload]`, handed to the kernel
//!   before `log_operation` returns `Ok`. A failed write is rolled back to
//!   the end of the last complete entry, so nothing is ever appended behind
//!   a torn one.
//! - Opening a WAL with a torn tail (incomplete length prefix, incomplete or
//!   unparseable payload) recovers the valid prefix ONLY. The discarded bytes
//!   are copied to a sibling backup and fsynced BEFORE the file is cut, and
//!   the event is reported on stderr and through
//!   [`WriteAheadLog::recovery_info`].
//! - `checkpoint()` / `close()` fsync the file; a rewrite (`truncate`) is
//!   followed by an fsync of the parent directory, like every rename in
//!   `persistence::atomic_write`.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::persistence::{create_unique_sibling, parent_dir, sync_dir};

/// Largest entry payload accepted from a length prefix (sanity bound).
const MAX_ENTRY_BYTES: u64 = 100_000_000;

/// A single WAL entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WALEntry {
    pub lsn: u64,
    pub timestamp: f64,
    pub operation: String,
    pub data: serde_json::Value,
}

/// A torn tail repaired by [`WriteAheadLog::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalRecovery {
    /// Bytes removed from the end of the file.
    pub discarded_bytes: u64,
    /// Verbatim copy of those bytes, fsynced before the WAL was cut.
    pub backup_path: PathBuf,
}

/// Write-Ahead Log ensuring durability of operations.
pub struct WriteAheadLog {
    path: PathBuf,
    inner: Mutex<WALInner>,
    /// `Some` iff the file had a torn tail that was repaired by [`Self::new`].
    recovery: Option<WalRecovery>,
}

struct WALInner {
    file: File,
    lsn: u64,
    /// Offset just past the last complete entry.
    good_offset: u64,
    /// A write failed and its rollback failed too: the tail is torn, so
    /// appending would only bury the new entries behind garbage.
    poisoned: bool,
    /// Test-only fault injection.
    #[cfg(test)]
    fail_next_entry_after: Option<usize>,
}

/// How the file ends after the last valid entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    Clean,
    PartialLengthPrefix,
    InvalidLength,
    PartialPayload,
    UnparseablePayload,
}

struct Scan {
    entries: Vec<WALEntry>,
    /// End of the last complete entry == start of the torn tail.
    good_bytes: u64,
    file_bytes: u64,
    tail: Tail,
}

impl Scan {
    fn torn(&self) -> bool {
        self.good_bytes < self.file_bytes
    }

    fn cut(entries: Vec<WALEntry>, good_bytes: u64, file_bytes: u64, tail: Tail) -> Self {
        Self {
            entries,
            good_bytes,
            file_bytes,
            tail,
        }
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// `[4-byte big-endian length][JSON payload]` as ONE buffer, so the length
/// prefix and the payload always travel in the same write.
fn encode(entry: &WALEntry) -> std::io::Result<Vec<u8>> {
    let serialized = serde_json::to_vec(entry)?;
    let len: u32 = serialized
        .len()
        .try_into()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "entry too large"))?;
    let mut record = Vec::with_capacity(4 + serialized.len());
    record.extend_from_slice(&len.to_be_bytes());
    record.extend_from_slice(&serialized);
    Ok(record)
}

/// Copy `[from, to)` of `path` into a new sibling file and fsync it (plus
/// its directory) so the bytes survive before anything is cut.
fn salvage_tail(path: &Path, from: u64, to: u64) -> std::io::Result<PathBuf> {
    let mut src = File::open(path)?;
    src.seek(SeekFrom::Start(from))?;
    let (backup_path, mut dst) = create_unique_sibling(path, "torn")?;
    let copied = (|| -> std::io::Result<()> {
        let mut remaining = to - from;
        let mut buf = [0u8; 65536];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            let n = src.read(&mut buf[..want])?;
            if n == 0 {
                return Err(std::io::Error::other(
                    "WAL shrank while its torn tail was being salvaged",
                ));
            }
            dst.write_all(&buf[..n])?;
            remaining -= n as u64;
        }
        dst.sync_all()
    })();
    if let Err(e) = copied {
        let _ = std::fs::remove_file(&backup_path);
        return Err(e);
    }
    sync_dir(&parent_dir(path))?;
    Ok(backup_path)
}

/// Shrink the file to its valid prefix and make the cut durable.
fn cut_to(path: &Path, good_bytes: u64) -> std::io::Result<()> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(good_bytes)?;
    file.sync_all()
}

impl WriteAheadLog {
    /// Open (creating if needed) the WAL at `path` for append.
    ///
    /// A torn tail is repaired first: the valid prefix is kept, the
    /// discarded bytes are backed up next to the file and a WARNING goes to
    /// stderr. If the backup cannot be written the WAL is left untouched and
    /// an error is returned — a damaged WAL is never cut without its
    /// salvage copy.
    ///
    /// `_sync_interval` is accepted for API compatibility only: every entry
    /// is flushed to the kernel before its `log_operation` returns, so an
    /// acknowledged entry can never be stranded in a userspace buffer.
    pub fn new(path: &str, _sync_interval: u32) -> std::io::Result<Self> {
        let pb = PathBuf::from(path);
        if let Some(parent) = pb.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let scan = Self::scan(&pb)?;
        let mut recovery = None;
        if scan.torn() {
            let discarded_bytes = scan.file_bytes - scan.good_bytes;
            let backup_path = salvage_tail(&pb, scan.good_bytes, scan.file_bytes)?;
            cut_to(&pb, scan.good_bytes)?;
            eprintln!(
                "[wal] WARNING: {} has a torn tail ({:?}): recovered the {} valid \
                 entries, discarded {discarded_bytes} byte(s) kept verbatim in {}",
                pb.display(),
                scan.tail,
                scan.entries.len(),
                backup_path.display()
            );
            recovery = Some(WalRecovery {
                discarded_bytes,
                backup_path,
            });
        }

        let file = OpenOptions::new().create(true).append(true).open(&pb)?;
        let lsn = scan
            .entries
            .last()
            .map(|e| e.lsn.saturating_add(1))
            .unwrap_or(0);
        Ok(Self {
            inner: Mutex::new(WALInner {
                file,
                lsn,
                good_offset: scan.good_bytes,
                poisoned: false,
                #[cfg(test)]
                fail_next_entry_after: None,
            }),
            recovery,
            path: pb,
        })
    }

    /// The torn tail repaired by [`Self::new`], if the file had one.
    pub fn recovery_info(&self) -> Option<&WalRecovery> {
        self.recovery.as_ref()
    }

    /// Log an operation. Returns the LSN assigned.
    ///
    /// `Ok` means the entry reached the kernel (page cache): a process crash
    /// cannot lose it. Power-loss durability is the job of [`Self::checkpoint`].
    /// On `Err` that entry — and only that entry — is rolled back, so the
    /// file again ends at the last complete entry and later appends stay
    /// readable.
    pub fn log_operation(&self, op: &str, data: serde_json::Value) -> std::io::Result<u64> {
        let mut inner = self.inner.lock();
        let entry = WALEntry {
            lsn: inner.lsn,
            timestamp: now_secs(),
            operation: op.to_string(),
            data,
        };
        inner.append_entry(&entry)?;
        let lsn = inner.lsn;
        inner.lsn = inner.lsn.saturating_add(1);
        Ok(lsn)
    }

    /// Flush and fsync for durability: appends a `checkpoint` record and
    /// `sync_all`s the file, so everything logged before is on stable
    /// storage once this returns `Ok`.
    pub fn checkpoint(&self) -> std::io::Result<()> {
        let mut inner = self.inner.lock();
        let entry = WALEntry {
            lsn: inner.lsn,
            timestamp: now_secs(),
            operation: "checkpoint".to_string(),
            data: serde_json::json!({ "lsn_at_checkpoint": inner.lsn }),
        };
        inner.append_entry(&entry)?;
        inner.lsn = inner.lsn.saturating_add(1);
        inner.file.sync_all()
    }

    /// Recover all entries from the WAL: the valid prefix, i.e. everything
    /// up to the first torn or unparseable entry.
    pub fn recover(&self) -> std::io::Result<Vec<WALEntry>> {
        Ok(Self::scan(&self.path)?.entries)
    }

    /// Truncate entries before the given LSN.
    ///
    /// Crash-safe rewrite: the surviving entries go to a new sibling temp
    /// file that is fsynced and renamed over the WAL — `inner.file` is only
    /// replaced after EVERY fallible step succeeded, so a failure anywhere
    /// leaves the original file and handle untouched. The rename is then
    /// made durable with an fsync of the parent directory.
    ///
    /// A torn tail (bytes after the last valid entry) would be dropped by the
    /// rewrite: it is backed up first, exactly like in [`Self::new`].
    ///
    /// # TOCTOU safety
    ///
    /// Acquires `self.inner` lock BEFORE scanning, so nothing can be logged
    /// between reading and rewriting.
    pub fn truncate(&self, before_lsn: u64) -> std::io::Result<()> {
        let mut inner = self.inner.lock();

        let scan = Self::scan(&self.path)?;
        if scan.torn() {
            let discarded_bytes = scan.file_bytes - scan.good_bytes;
            let backup_path = salvage_tail(&self.path, scan.good_bytes, scan.file_bytes)?;
            eprintln!(
                "[wal] WARNING: rewriting {} drops a torn tail ({:?}): {discarded_bytes} \
                 byte(s) kept verbatim in {}",
                self.path.display(),
                scan.tail,
                backup_path.display()
            );
        }
        let remaining: Vec<&WALEntry> = scan
            .entries
            .iter()
            .filter(|e| e.lsn >= before_lsn)
            .collect();

        // 1. Build the surviving WAL in a new, uniquely named sibling that is
        //    opened in append mode: after the rename this very handle IS the
        //    live WAL, so no reopen is needed.
        let (tmp_path, file) = create_unique_sibling(&self.path, "tmp")?;
        let built = (|| -> std::io::Result<(File, u64)> {
            let mut writer = BufWriter::new(file);
            let mut bytes = 0u64;
            for entry in &remaining {
                let record = encode(entry)?;
                writer.write_all(&record)?;
                bytes += record.len() as u64;
            }
            writer.flush()?;
            writer.get_ref().sync_all()?;
            let file = writer.into_inner().map_err(|e| e.into_error())?;
            Ok((file, bytes))
        })();
        let (file, written) = match built {
            Ok(v) => v,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(e);
            }
        };

        // 2. Swap atomically. Nothing fallible sits between the rename and
        //    the handle swap, so the writer can never be left pointing at the
        //    unlinked old file.
        if let Err(e) = std::fs::rename(&tmp_path, &self.path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
        inner.file = file;
        inner.good_offset = written;
        inner.poisoned = false;

        // 3. Make the rename itself durable.
        sync_dir(&parent_dir(&self.path))
    }

    /// Close the WAL: fsync everything logged so far.
    pub fn close(&self) -> std::io::Result<()> {
        self.inner.lock().file.sync_all()
    }

    /// Read the valid prefix of the file: the entries up to the first
    /// anomaly. Everything from that anomaly on is the "torn tail".
    fn scan(path: &Path) -> std::io::Result<Scan> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Scan::cut(Vec::new(), 0, 0, Tail::Clean));
            }
            Err(e) => return Err(e),
        };
        let file_bytes = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut offset: u64 = 0;
        loop {
            let start = offset;
            let mut len_buf = [0u8; 4];
            let mut got = 0usize;
            while got < 4 {
                match reader.read(&mut len_buf[got..]) {
                    Ok(0) => break,
                    Ok(n) => got += n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }
            if got == 0 {
                return Ok(Scan::cut(entries, start, start, Tail::Clean));
            }
            if got < 4 {
                return Ok(Scan::cut(
                    entries,
                    start,
                    file_bytes,
                    Tail::PartialLengthPrefix,
                ));
            }
            let len = u32::from_be_bytes(len_buf) as u64;
            if len == 0 || len > MAX_ENTRY_BYTES {
                return Ok(Scan::cut(entries, start, file_bytes, Tail::InvalidLength));
            }
            // Never allocate (or wait for) more than the file actually holds.
            if start + 4 + len > file_bytes {
                return Ok(Scan::cut(entries, start, file_bytes, Tail::PartialPayload));
            }
            let mut payload = vec![0u8; len as usize];
            reader.read_exact(&mut payload)?;
            match serde_json::from_slice::<WALEntry>(&payload) {
                Ok(entry) => {
                    entries.push(entry);
                    offset = start + 4 + len;
                }
                Err(_) => {
                    return Ok(Scan::cut(
                        entries,
                        start,
                        file_bytes,
                        Tail::UnparseablePayload,
                    ));
                }
            }
        }
    }

    /// Test-only fault injection: the next entry is cut after `bytes` bytes
    /// and reported as a failed write (simulates ENOSPC / EIO mid-entry).
    #[cfg(test)]
    pub(crate) fn debug_fail_next_entry_after(&self, bytes: usize) {
        self.inner.lock().fail_next_entry_after = Some(bytes);
    }
}

impl WALInner {
    fn append_entry(&mut self, entry: &WALEntry) -> std::io::Result<()> {
        if self.poisoned {
            return Err(std::io::Error::other(
                "WAL poisoned: an earlier write failure could not be rolled back",
            ));
        }
        let record = encode(entry)?;
        if let Err(e) = self.write_record(&record) {
            // Roll back to the last complete entry. Every earlier entry was
            // written to the kernel before it was acknowledged, so the file
            // on disk is never shorter than `good_offset`.
            if let Err(rb) = self.file.set_len(self.good_offset) {
                self.poisoned = true;
                return Err(std::io::Error::new(
                    e.kind(),
                    format!(
                        "{e}; rolling back to offset {} failed ({rb}) — WAL poisoned",
                        self.good_offset
                    ),
                ));
            }
            return Err(e);
        }
        self.good_offset += record.len() as u64;
        Ok(())
    }

    fn write_record(&mut self, record: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if let Some(n) = self.fail_next_entry_after.take() {
            let n = n.min(record.len());
            self.file.write_all(&record[..n])?;
            return Err(std::io::Error::other("injected WAL write failure"));
        }
        self.file.write_all(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_wal_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "splatdb-wal-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("wal.log")
    }

    fn dir_names(path: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn ops_of(wal: &WriteAheadLog) -> Vec<String> {
        wal.recover()
            .unwrap()
            .into_iter()
            .map(|e| e.operation)
            .collect()
    }

    /// `[len][first 100 of 300 payload bytes]`: what a crash / ENOSPC in the
    /// middle of an entry leaves behind.
    fn partial_payload_tail() -> Vec<u8> {
        let payload = vec![b'{'; 300];
        let mut tail = (payload.len() as u32).to_be_bytes().to_vec();
        tail.extend_from_slice(&payload[..100]);
        tail
    }

    /// Two valid entries, then `tail` appended verbatim by a raw writer.
    /// Returns (path, byte length of the two valid entries).
    fn wal_with_tail(tag: &str, tail: &[u8]) -> (PathBuf, u64) {
        use std::io::Write;
        let path = temp_wal_path(tag);
        let p = path.to_str().unwrap();
        {
            let wal = WriteAheadLog::new(p, 100).unwrap();
            wal.log_operation("op0", serde_json::json!({ "i": 0 }))
                .unwrap();
            wal.log_operation("op1", serde_json::json!({ "i": 1 }))
                .unwrap();
            wal.close().unwrap();
        }
        let good = std::fs::metadata(&path).unwrap().len();
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(tail).unwrap();
        f.sync_all().unwrap();
        (path, good)
    }

    #[test]
    fn log_checkpoint_recover_roundtrip() {
        let path = temp_wal_path("roundtrip");
        let wal = WriteAheadLog::new(path.to_str().unwrap(), 100).unwrap();

        let lsn0 = wal
            .log_operation("insert", serde_json::json!({ "i": 0 }))
            .unwrap();
        let lsn1 = wal
            .log_operation("delete", serde_json::json!({ "i": 1 }))
            .unwrap();
        assert_eq!(lsn0, 0);
        assert_eq!(lsn1, 1);

        // checkpoint() flushes and fsyncs; recover() must see both ops plus
        // the checkpoint entry.
        wal.checkpoint().unwrap();
        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].operation, "insert");
        assert_eq!(entries[1].operation, "delete");
        assert_eq!(entries[2].operation, "checkpoint");

        // LSN continues monotonically after reload.
        drop(wal);
        let wal = WriteAheadLog::new(path.to_str().unwrap(), 100).unwrap();
        let lsn3 = wal
            .log_operation("insert", serde_json::json!({ "i": 3 }))
            .unwrap();
        assert_eq!(lsn3, 3);

        wal.close().unwrap();
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A03-F03 — torn tail: a complete length prefix followed by a partial
    /// payload (crash / ENOSPC in the middle of an entry). After the file is
    /// reopened, the next append must NOT land behind the garbage.
    #[test]
    fn torn_tail_partial_payload_does_not_poison_next_append() {
        use std::io::Write;
        let path = temp_wal_path("torn");
        let p = path.to_str().unwrap();
        {
            let wal = WriteAheadLog::new(p, 100).unwrap();
            wal.log_operation("op0", serde_json::json!({ "i": 0 }))
                .unwrap();
            wal.log_operation("op1", serde_json::json!({ "i": 1 }))
                .unwrap();
            wal.close().unwrap();
        }
        let good_len = std::fs::metadata(&path).unwrap().len();

        // Crash mid-write: length prefix of a third entry + 100 of its 300 bytes.
        let payload = vec![b'{'; 300];
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&(payload.len() as u32).to_be_bytes()).unwrap();
        f.write_all(&payload[..100]).unwrap();
        f.sync_all().unwrap();
        drop(f);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), good_len + 104);

        // Next session: reopen, append one entry, close, read back.
        let wal = WriteAheadLog::new(p, 100).unwrap();
        wal.log_operation("op2", serde_json::json!({ "i": 2 }))
            .unwrap();
        wal.close().unwrap();
        let entries = wal.recover().unwrap();
        let ops: Vec<&str> = entries.iter().map(|e| e.operation.as_str()).collect();
        assert_eq!(
            ops,
            ["op0", "op1", "op2"],
            "torn tail poisoned the WAL: entries read back = {ops:?}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn torn_tail_is_cut_to_the_valid_prefix_and_backed_up() {
        let tail = partial_payload_tail();
        let (path, good) = wal_with_tail("torn-backup", &tail);
        let wal = WriteAheadLog::new(path.to_str().unwrap(), 100).unwrap();

        let info = wal.recovery_info().expect("the torn tail must be reported");
        assert_eq!(info.discarded_bytes, tail.len() as u64);
        assert_eq!(
            std::fs::read(&info.backup_path).unwrap(),
            tail,
            "backup must be verbatim"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            good,
            "WAL not cut to its valid prefix"
        );
        assert_eq!(ops_of(&wal), ["op0", "op1"]);

        // The next append lands right behind the valid prefix and survives.
        let lsn = wal
            .log_operation("op2", serde_json::json!({ "i": 2 }))
            .unwrap();
        assert_eq!(lsn, 2, "LSN continues from the last valid entry");
        wal.close().unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() > good);
        assert_eq!(ops_of(&wal), ["op0", "op1", "op2"]);

        let names = dir_names(&path);
        assert!(
            names.iter().any(|n| n.contains(".torn.")),
            "no backup file: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains(".tmp.")),
            "temp file left behind: {names:?}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Every way a tail can be torn: partial length prefix, frame with a
    /// garbage payload, impossible length, and — what the old writer used to
    /// do — a complete entry appended BEHIND a torn one. Only the valid
    /// prefix is recovered and everything else is kept in the backup.
    #[test]
    fn every_kind_of_torn_tail_recovers_only_the_valid_prefix() {
        let behind = encode(&WALEntry {
            lsn: 2,
            timestamp: 0.0,
            operation: "behind".to_string(),
            data: serde_json::json!({}),
        })
        .unwrap();
        let mut garbage = 64u32.to_be_bytes().to_vec();
        garbage.extend_from_slice(&[0xFF; 64]);
        let mut appended_behind = partial_payload_tail();
        appended_behind.extend_from_slice(&behind);
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("torn-prefix", vec![0, 0]),
            ("torn-garbage", garbage),
            ("torn-badlen", 0xFFFF_FFFFu32.to_be_bytes().to_vec()),
            ("torn-behind", appended_behind),
        ];

        for (tag, tail) in cases {
            let (path, good) = wal_with_tail(tag, &tail);
            let wal = WriteAheadLog::new(path.to_str().unwrap(), 100).unwrap();
            let info = wal
                .recovery_info()
                .unwrap_or_else(|| panic!("{tag}: torn tail not reported"));
            assert_eq!(info.discarded_bytes, tail.len() as u64, "{tag}");
            assert_eq!(std::fs::read(&info.backup_path).unwrap(), tail, "{tag}");
            assert_eq!(std::fs::metadata(&path).unwrap().len(), good, "{tag}");
            assert_eq!(ops_of(&wal), ["op0", "op1"], "{tag}");

            wal.log_operation("op2", serde_json::json!({ "i": 2 }))
                .unwrap();
            wal.close().unwrap();
            assert_eq!(
                ops_of(&wal),
                ["op0", "op1", "op2"],
                "{tag}: append after the repair"
            );
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }

    /// Compatibility: a healthy WAL must be reopened byte for byte.
    #[test]
    fn intact_wal_is_reopened_untouched() {
        let path = temp_wal_path("intact");
        let p = path.to_str().unwrap();
        let wal = WriteAheadLog::new(p, 100).unwrap();
        for i in 0..3 {
            wal.log_operation("op", serde_json::json!({ "i": i }))
                .unwrap();
        }
        wal.checkpoint().unwrap();
        wal.close().unwrap();
        let before = std::fs::read(&path).unwrap();

        let wal = WriteAheadLog::new(p, 100).unwrap();
        assert!(wal.recovery_info().is_none(), "intact WAL reported as torn");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "intact WAL was rewritten"
        );
        let lsn = wal
            .log_operation("op", serde_json::json!({ "i": 3 }))
            .unwrap();
        assert_eq!(lsn, 4, "3 ops + 1 checkpoint");
        wal.close().unwrap();
        assert_eq!(ops_of(&wal).len(), 5);
        assert!(!dir_names(&path).iter().any(|n| n.contains(".torn.")));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A03-F03 (in-session half): a write that fails mid-entry must not
    /// leave a torn entry in front of the next append.
    #[test]
    fn failed_write_is_rolled_back_and_the_wal_stays_appendable() {
        let path = temp_wal_path("rollback");
        let p = path.to_str().unwrap();
        let wal = WriteAheadLog::new(p, 100).unwrap();
        wal.log_operation("op0", serde_json::json!({ "i": 0 }))
            .unwrap();
        wal.log_operation("op1", serde_json::json!({ "i": 1 }))
            .unwrap();
        let good = std::fs::metadata(&path).unwrap().len();

        // ENOSPC 12 bytes into the next entry.
        wal.debug_fail_next_entry_after(12);
        assert!(wal
            .log_operation("torn", serde_json::json!({ "i": 2 }))
            .is_err());
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            good,
            "torn bytes were not rolled back"
        );
        assert_eq!(ops_of(&wal), ["op0", "op1"]);

        // The WAL keeps working: the next entry is not buried behind garbage.
        let lsn = wal
            .log_operation("op2", serde_json::json!({ "i": 2 }))
            .unwrap();
        assert_eq!(lsn, 2);
        wal.close().unwrap();
        assert_eq!(ops_of(&wal), ["op0", "op1", "op2"]);

        // A fresh reader of the same file agrees.
        drop(wal);
        let wal = WriteAheadLog::new(p, 100).unwrap();
        assert!(wal.recovery_info().is_none());
        assert_eq!(ops_of(&wal), ["op0", "op1", "op2"]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A03-F02 — traced at integration time with the LD_PRELOAD fsync
    /// tracer: must show an `fsync` on the WAL file for `checkpoint()` and
    /// another one for `close()`.
    #[test]
    fn fsync_probe_checkpoint_and_close_hit_the_wal_file() {
        let path = temp_wal_path("fsync-probe");
        let p = path.to_str().unwrap();
        let wal = WriteAheadLog::new(p, 100).unwrap();
        wal.log_operation("op0", serde_json::json!({ "i": 0 }))
            .unwrap();
        wal.checkpoint().unwrap();
        wal.log_operation("op1", serde_json::json!({ "i": 1 }))
            .unwrap();
        wal.close().unwrap();
        drop(wal);
        assert_eq!(
            WriteAheadLog::new(p, 100).unwrap().recover().unwrap().len(),
            3
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
