//! SQLite implementation of MetadataStore.

use super::metadata_store::{DocumentRecord, MetadataStore};
use rusqlite::types::Value;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Column holding the persisted embedding vector (little-endian f32 BLOB).
const EMBEDDING_COL: &str = "embedding";
/// Column holding the provenance label of `EMBEDDING_COL`.
const EMBEDDING_SOURCE_COL: &str = "embedding_source";
/// The 8 columns every `documents` row had before the vector columns
/// (positional order of `SqliteMetadataStore::row_to_record`).
const DOC_COLUMNS: &str =
    "id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at";

/// Where a persisted embedding vector came from. Stored next to the blob so
/// that a warm start reports the provenance instead of guessing (A03-F10).
/// A `SimCos` vector is never promoted: it is stored, reloaded and labelled
/// `simcos` for as long as it exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingProvenance {
    Real,
    SimCos,
    Provided,
}

impl EmbeddingProvenance {
    pub fn label(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::SimCos => "simcos",
            Self::Provided => "provided",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label {
            "real" => Some(Self::Real),
            "simcos" => Some(Self::SimCos),
            "provided" => Some(Self::Provided),
            _ => None,
        }
    }
}

/// A stored vector, decoded and validated against the live store dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredEmbedding {
    pub vector: Vec<f32>,
    pub source: EmbeddingProvenance,
}

/// Why a persisted vector was refused instead of being loaded. The blob is
/// never resized, truncated or "repaired": it is quarantined as it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoredEmbeddingError {
    /// A provenance label without a vector.
    MissingBlob,
    /// `embedding` BLOB without a provenance label.
    MissingSource,
    UnknownSource(String),
    /// The column holds something that is not a BLOB (e.g. TEXT).
    NotABlob(&'static str),
    /// Byte length is not a multiple of 4 (not a sequence of f32).
    BadLength(usize),
    Empty,
    /// Decodes to another dimension than the live store.
    DimensionMismatch {
        got: usize,
        expected: usize,
    },
    NonFinite,
}

impl std::fmt::Display for StoredEmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingBlob => write!(f, "provenance label stored without a vector blob"),
            Self::MissingSource => write!(f, "vector stored without a provenance label"),
            Self::UnknownSource(s) => write!(f, "unknown provenance label {s:?}"),
            Self::NotABlob(t) => write!(f, "vector column holds {t}, not a BLOB"),
            Self::BadLength(n) => write!(f, "blob length {n} is not a multiple of 4 (f32)"),
            Self::Empty => write!(f, "empty vector blob"),
            Self::DimensionMismatch { got, expected } => {
                write!(
                    f,
                    "dimension mismatch: stored {got} dims, store expects {expected}"
                )
            }
            Self::NonFinite => write!(f, "vector contains non-finite values (NaN/inf)"),
        }
    }
}

impl std::error::Error for StoredEmbeddingError {}

/// Encode a vector as the little-endian f32 BLOB that is persisted. Input
/// that is empty or not finite is refused before it reaches SQLite.
fn encode_stored_embedding(vector: &[f32]) -> Result<Vec<u8>, String> {
    if vector.is_empty() {
        return Err("embedding is empty: refused".into());
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err("embedding contains non-finite values (NaN/inf): refused".into());
    }
    Ok(vector.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// Decode and validate a persisted vector against the live store dimension
/// `expected_dim`. Nothing is ever resized, truncated or re-labelled: a
/// value that is not a finite `expected_dim`-long f32 vector with a known
/// provenance label is an error that names the reason.
pub fn decode_stored_embedding(
    blob: &Value,
    source: Option<&str>,
    expected_dim: usize,
) -> Result<StoredEmbedding, StoredEmbeddingError> {
    let bytes = match blob {
        Value::Blob(bytes) => bytes.as_slice(),
        Value::Null => return Err(StoredEmbeddingError::MissingBlob),
        Value::Text(_) => return Err(StoredEmbeddingError::NotABlob("TEXT")),
        Value::Integer(_) => return Err(StoredEmbeddingError::NotABlob("INTEGER")),
        Value::Real(_) => return Err(StoredEmbeddingError::NotABlob("REAL")),
    };
    if bytes.is_empty() {
        return Err(StoredEmbeddingError::Empty);
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(StoredEmbeddingError::BadLength(bytes.len()));
    }
    let mut vector = Vec::with_capacity(bytes.len() / 4);
    let mut i = 0;
    while i < bytes.len() {
        vector.push(f32::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
        ]));
        i += 4;
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(StoredEmbeddingError::NonFinite);
    }
    if vector.len() != expected_dim {
        return Err(StoredEmbeddingError::DimensionMismatch {
            got: vector.len(),
            expected: expected_dim,
        });
    }
    let label = source.ok_or(StoredEmbeddingError::MissingSource)?;
    let source = EmbeddingProvenance::parse(label)
        .ok_or_else(|| StoredEmbeddingError::UnknownSource(label.to_string()))?;
    Ok(StoredEmbedding { vector, source })
}

fn has_column(conn: &Connection, column: &str) -> Result<bool, rusqlite::Error> {
    let mut stmt = conn.prepare("PRAGMA table_info(documents)")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Add the embedding columns to a pre-F05b `documents` table, in place.
/// Returns `true` when both columns already existed (nothing done).
///
/// Both `ALTER`s run in ONE transaction — a crash leaves either both or
/// neither — and each is guarded by its own existence check, so the call is
/// idempotent whatever it finds. No row is rewritten and none is dropped:
/// legacy rows simply keep NULL in both columns.
fn ensure_embedding_columns(conn: &mut Connection) -> Result<bool, rusqlite::Error> {
    let tx = conn.transaction()?;
    let mut already_present = true;
    for column in [EMBEDDING_COL, EMBEDDING_SOURCE_COL] {
        if !has_column(&tx, column)? {
            let decl = if column == EMBEDDING_COL {
                "BLOB"
            } else {
                "TEXT"
            };
            tx.execute_batch(&format!(
                "ALTER TABLE documents ADD COLUMN {column} {decl};"
            ))?;
            already_present = false;
        }
    }
    tx.commit()?;
    Ok(already_present)
}

/// SQLite-backed metadata store with a single persistent connection guarded
/// by a `Mutex`.
///
/// # Concurrency semantics (honest)
///
/// Within one process there is exactly ONE connection shared under a `Mutex`:
/// operations serialize — readers never observe torn state and there is no
/// `SQLITE_BUSY` between threads of this process.
///
/// Across processes (CLI + MCP server over the same `.db` file) SQLite's
/// file locking applies: WAL mode allows one writer + N concurrent readers;
/// the rollback journal (fallback when WAL is unavailable, e.g. on filesystems
/// without shared-memory support) allows a single accessor at a time. Either
/// way, `busy_timeout=5000ms` makes lock contention retry for up to 5s
/// instead of failing immediately with `SQLITE_BUSY`. Durability:
/// `synchronous=NORMAL` under WAL (commits are durable on checkpoint / power
/// loss may lose the last transactions but the DB stays consistent); FULL
/// default fsync per commit when running on the rollback journal.
///
/// # Embedding vectors (F05b)
///
/// Each row carries its embedding as a little-endian f32 BLOB
/// (`embedding`) plus the provenance it was produced with
/// (`embedding_source`: real / simcos / provided). `upsert_with_embedding`
/// writes document, metadata, vector and label in ONE statement, so no
/// reader and no crash can observe one without the other.
pub struct SqliteMetadataStore {
    conn: Mutex<Connection>,
    /// True for `:memory:`: nothing written here survives the process.
    ephemeral: bool,
    /// True once `documents` carries the embedding columns (native schema,
    /// or after the idempotent legacy migration). While false no vector can
    /// be persisted: `upsert_with_embedding` retries the migration and then
    /// fails closed.
    embedding_schema: AtomicBool,
}

impl SqliteMetadataStore {
    pub fn new(db_path: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // `:memory:` has no parent directory — and must not materialize one.
        if let Some(parent) = db_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut conn = Connection::open(&db_path)?;
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        // WAL can silently fail (returns the old mode) on some filesystems
        // (e.g. network mounts without shared-memory support). Read the mode
        // back and fall back to the default journal if it did not stick.
        let journal_mode: String =
            conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
        if journal_mode.eq_ignore_ascii_case("wal") {
            // Only enable NORMAL durability under WAL: with the rollback
            // journal, NORMAL still fsyncs on every commit.
            conn.pragma_update(None, "synchronous", "NORMAL")?;
        } else {
            eprintln!(
                "[sqlite] WAL not available (mode={journal_mode:?}), keeping rollback journal"
            );
        }
        conn.pragma_update(None, "cache_size", -64000)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                 id TEXT PRIMARY KEY,
                 shard_idx INTEGER,
                 vector_idx INTEGER,
                 metadata TEXT,
                 document TEXT,
                 deleted INTEGER DEFAULT 0,
                 created_at REAL,
                 updated_at REAL,
                 embedding BLOB,
                 embedding_source TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted);
             CREATE INDEX IF NOT EXISTS idx_vector ON documents(vector_idx);",
        )?;
        // A database written before the vector columns existed gains them
        // here, in place. A migration that cannot run (e.g. a locked or
        // read-only file) is NOT fatal: documents stay readable, vectors are
        // simply not persisted until a write retries the migration.
        let embedding_schema = match ensure_embedding_columns(&mut conn) {
            Ok(true) => {
                eprintln!(
                    "[sqlite] embedding persistence ready (columns {EMBEDDING_COL}/{EMBEDDING_SOURCE_COL})"
                );
                true
            }
            Ok(false) => {
                eprintln!(
                    "[sqlite] migrated legacy schema: added columns {EMBEDDING_COL}/{EMBEDDING_SOURCE_COL} to documents (existing rows untouched)"
                );
                true
            }
            Err(e) => {
                eprintln!(
                    "[sqlite] WARNING: cannot add the embedding columns to the legacy schema ({e}): documents stay readable but embedding vectors will NOT be persisted"
                );
                false
            }
        };
        Ok(Self {
            conn: Mutex::new(conn),
            ephemeral: db_path.to_string_lossy() == ":memory:",
            embedding_schema: AtomicBool::new(embedding_schema),
        })
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<DocumentRecord> {
        let meta_str: Option<String> = row.get(3)?;
        Ok(DocumentRecord {
            id: row.get(0)?,
            shard_idx: row.get(1)?,
            vector_idx: row.get(2)?,
            metadata: meta_str.and_then(|s| serde_json::from_str(&s).ok()),
            document: row.get(4)?,
            deleted: row.get::<_, i32>(5)? != 0,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    }

    fn connection(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Connection>, Box<dyn std::error::Error + Send + Sync>>
    {
        self.conn.lock().map_err(|e| format!("mutex: {e}").into())
    }

    /// Current journal mode (e.g. "wal", "delete") — exposed for tests.
    pub fn journal_mode(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        Ok(mode)
    }

    /// True when the store lives in process memory (`:memory:`): writes
    /// succeed but survive nothing.
    pub fn is_ephemeral(&self) -> bool {
        self.ephemeral
    }

    /// True when the `documents` table can persist embedding vectors.
    pub fn embedding_schema_ready(&self) -> bool {
        self.embedding_schema.load(Ordering::Relaxed)
    }

    /// Persist a document TOGETHER with its embedding vector and its
    /// provenance (A03-F10, persistence half).
    ///
    /// One row, ONE SQLite statement — i.e. one transaction: no reader and
    /// no crash can observe the new document without its vector, or a vector
    /// without its document. A statement that fails leaves both absent.
    /// `INSERT OR REPLACE` retires any previous version of the row
    /// (soft-deleted or not): a re-store never inherits an old vector.
    pub fn upsert_with_embedding(
        &self,
        record: &DocumentRecord,
        embedding: &[f32],
        source: EmbeddingProvenance,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let blob = encode_stored_embedding(embedding)?;
        let mut conn = self.connection()?;
        if !self.embedding_schema.load(Ordering::Relaxed) {
            // The startup migration may have lost a race on a busy database:
            // retry it here, then fail closed.
            ensure_embedding_columns(&mut conn).map_err(|e| {
                format!("embedding persistence unavailable: legacy schema migration failed: {e}")
            })?;
            self.embedding_schema.store(true, Ordering::Relaxed);
        }
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at, embedding, embedding_source)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                record.id,
                record.shard_idx,
                record.vector_idx,
                record.metadata.as_ref().map(|m| m.to_string()),
                record.document,
                record.deleted as i32,
                record.created_at,
                record.updated_at,
                blob,
                source.label()
            ],
        )?;
        Ok(())
    }

    /// Overwrite ONLY the stored vector and provenance of a live document —
    /// used by the warm start to backfill a legacy row it had to embed.
    pub fn set_embedding(
        &self,
        doc_id: &str,
        embedding: &[f32],
        source: EmbeddingProvenance,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let blob = encode_stored_embedding(embedding)?;
        let conn = self.connection()?;
        if !self.embedding_schema.load(Ordering::Relaxed) {
            return Err("embedding persistence unavailable: legacy schema was not migrated".into());
        }
        let updated = conn.execute(
            "UPDATE documents SET embedding=?1, embedding_source=?2 WHERE id=?3 AND deleted=0",
            params![blob, source.label(), doc_id],
        )?;
        if updated == 0 {
            return Err(format!("document {doc_id} is not a live row").into());
        }
        Ok(())
    }

    /// Live (non-deleted) documents with their stored vector, ascending id,
    /// in ONE query — what the MCP warm start iterates. `expected_dim` is the
    /// live store dimension: a blob that decodes to another dimension is
    /// reported in `embedding_error`, never resized.
    pub fn list_with_embeddings(
        &self,
        limit: usize,
        expected_dim: usize,
    ) -> Result<Vec<StoredDoc>, Box<dyn std::error::Error + Send + Sync>> {
        let sql = if self.embedding_schema.load(Ordering::Relaxed) {
            format!(
                "SELECT {DOC_COLUMNS}, {EMBEDDING_COL}, {EMBEDDING_SOURCE_COL}
                 FROM documents WHERE deleted=0 ORDER BY id LIMIT ?1"
            )
        } else {
            format!("SELECT {DOC_COLUMNS}, NULL, NULL FROM documents WHERE deleted=0 ORDER BY id LIMIT ?1")
        };
        let conn = self.connection()?;
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let record = Self::row_to_record(row)?;
            let blob: Value = row.get(8)?;
            let source: Option<String> = row.get(9)?;
            Ok((record, blob, source))
        })?;
        let mut docs = Vec::new();
        for row in rows {
            let (record, blob, source) = row?;
            let (embedding, embedding_error) = if matches!(blob, Value::Null) {
                (None, None)
            } else {
                match decode_stored_embedding(&blob, source.as_deref(), expected_dim) {
                    Ok(stored) => (Some(stored), None),
                    Err(e) => (None, Some(e)),
                }
            };
            docs.push(StoredDoc {
                record,
                embedding,
                embedding_error,
            });
        }
        Ok(docs)
    }

    /// Live documents that carry a stored vector (structural count; the
    /// blobs are not decoded here).
    pub fn stored_vector_count(&self) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        if !self.embedding_schema.load(Ordering::Relaxed) {
            return Ok(0);
        }
        let conn = self.connection()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE deleted=0 AND embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }
}

/// One persisted document plus the outcome of reading its stored vector.
#[derive(Debug)]
pub struct StoredDoc {
    pub record: DocumentRecord,
    /// `Some` when a valid vector was decoded.
    pub embedding: Option<StoredEmbedding>,
    /// `Some(diagnostic)` when a vector exists in the row but was refused
    /// (quarantined).
    pub embedding_error: Option<StoredEmbeddingError>,
}

impl MetadataStore for SqliteMetadataStore {
    /// Insert or replace a document record — WITHOUT a vector: this is the
    /// CLI / HTTP writer, which does not know about embeddings.
    ///
    /// F05c: the document columns are replaced as before, but a vector that
    /// is already stored is no longer collateral damage. When the row exists
    /// and keeps its text, `embedding` and `embedding_source` stay untouched
    /// (byte-for-byte): a metadata-only update must not cost a re-embedding.
    /// The vector is retired only when it would lie — the text changed — or
    /// when the row stops being live; the next warm start re-embeds it.
    /// `upsert_with_embedding` replaces a vector explicitly and `soft_delete`
    /// retires it. One statement, one transaction.
    fn upsert(
        &self,
        record: &DocumentRecord,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        // Ask the schema, not only the cached flag: on a database whose
        // in-place migration could not run the columns may be absent (nothing
        // to keep) or already there (a startup race lost the flag).
        let keeps_vectors = self.embedding_schema.load(Ordering::Relaxed)
            || (has_column(&conn, EMBEDDING_COL)? && has_column(&conn, EMBEDDING_SOURCE_COL)?);
        let sql = if keeps_vectors {
            "INSERT INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(id) DO UPDATE SET
                 shard_idx=excluded.shard_idx,
                 vector_idx=excluded.vector_idx,
                 metadata=excluded.metadata,
                 document=excluded.document,
                 deleted=excluded.deleted,
                 created_at=excluded.created_at,
                 updated_at=excluded.updated_at,
                 embedding=CASE
                     WHEN documents.document IS excluded.document AND excluded.deleted=0
                     THEN documents.embedding
                     ELSE NULL
                 END,
                 embedding_source=CASE
                     WHEN documents.document IS excluded.document AND excluded.deleted=0
                     THEN documents.embedding_source
                     ELSE NULL
                 END"
        } else {
            "INSERT OR REPLACE INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"
        };
        conn.execute(
            sql,
            params![
                record.id,
                record.shard_idx,
                record.vector_idx,
                record.metadata.as_ref().map(|m| m.to_string()),
                record.document,
                record.deleted as i32,
                record.created_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    fn get(
        &self,
        doc_id: &str,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare_cached(
            "SELECT id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at
             FROM documents WHERE id=?1 AND deleted=0",
        )?;
        let result = stmt.query_row(params![doc_id], Self::row_to_record);
        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        let conn = self.connection()?;
        // The tombstone and the retirement of vector + provenance are the
        // SAME statement (F05b): a deleted document never keeps a stored
        // vector behind.
        let sql = if self.embedding_schema.load(Ordering::Relaxed) {
            "UPDATE documents SET deleted=1, updated_at=?1, embedding=NULL, embedding_source=NULL WHERE id=?2"
        } else {
            "UPDATE documents SET deleted=1, updated_at=?1 WHERE id=?2"
        };
        let affected = conn.execute(sql, params![now, doc_id])?;
        Ok(affected > 0)
    }

    fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        let affected = conn.execute("DELETE FROM documents WHERE id=?1", params![doc_id])?;
        Ok(affected > 0)
    }

    fn list_ids(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        let sql = if include_deleted {
            "SELECT id FROM documents"
        } else {
            "SELECT id FROM documents WHERE deleted=0"
        };
        let mut stmt = conn.prepare_cached(sql)?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(ids)
    }

    fn count(
        &self,
        include_deleted: bool,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        let sql = if include_deleted {
            "SELECT COUNT(*) FROM documents"
        } else {
            "SELECT COUNT(*) FROM documents WHERE deleted=0"
        };
        // rusqlite ≥0.41 ya no implementa FromSql para usize (SQLite es i64)
        let count: i64 = conn.query_row(sql, [], |row| row.get(0))?;
        Ok(count as usize)
    }

    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "sqlite"
    }

    /// Indexed lookup via `idx_vector`: O(log N + k) where k = duplicates of
    /// this `vector_idx`, instead of the O(N·log N) default scan.
    ///
    /// `vector_idx` is NOT unique (a soft-deleted doc keeps its slot and
    /// `n_active()` can hand the same index to a new doc), so duplicates are
    /// possible. Parity with the default trait scan is guaranteed by
    /// construction: `ORDER BY id LIMIT 1` picks the lexicographically-first
    /// matching non-deleted row, which is exactly the row the default scan
    /// hits first — `list_ids` is served by the covering index on the TEXT
    /// PRIMARY KEY, which returns `id`s in ascending lexicographic order
    /// (verified empirically; see `find_by_vector_idx_sql_matches_scan`).
    fn find_by_vector_idx(
        &self,
        vector_idx: i64,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare_cached(
            "SELECT id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at
             FROM documents
             WHERE vector_idx = ?1 AND deleted = 0
             ORDER BY id
             LIMIT 1",
        )?;
        let result = stmt.query_row(params![vector_idx], Self::row_to_record);
        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference O(N) scan, verbatim copy of the default trait method — the
    /// SQL path must match this exactly (same doc) on every query.
    fn scan_reference(store: &SqliteMetadataStore, vector_idx: i64) -> Option<DocumentRecord> {
        let ids = store.list_ids(true).unwrap();
        for id in ids {
            if let Some(record) = store.get(&id).unwrap() {
                if record.vector_idx == vector_idx && !record.deleted {
                    return Some(record);
                }
            }
        }
        None
    }

    fn temp_store(tag: &str) -> SqliteMetadataStore {
        let dir = std::env::temp_dir().join(format!(
            "splatsdb-sqlite-test-{}-{}",
            tag,
            std::process::id()
        ));
        SqliteMetadataStore::new(dir.join("meta.db")).unwrap()
    }

    /// P1 parity: indexed SQL path vs O(N) scan over a random dataset with
    /// deliberate vector_idx collisions and soft-deletes — same doc, every time.
    #[test]
    fn find_by_vector_idx_sql_matches_scan() {
        use rand::prelude::SliceRandom;
        use rand::Rng;
        use rand::SeedableRng;
        let store = temp_store("parity");
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE);

        // 300 docs, vector_idx in [0,50) → guaranteed collisions; random ids
        // (not insertion-ordered) so lexicographic order differs from insert
        // order; ~20% soft-deleted so ties can be broken by deleted flags.
        let mut ids: Vec<String> = (0..300).map(|i| format!("doc_{i:05}")).collect();
        ids.shuffle(&mut rng);
        for (i, id) in ids.iter().enumerate() {
            let record = DocumentRecord {
                id: id.clone(),
                shard_idx: 0,
                vector_idx: rng.gen_range(0..50),
                metadata: Some(serde_json::json!({ "n": i })),
                document: Some(format!("content {i}")),
                deleted: false,
                created_at: i as f64,
                updated_at: i as f64,
            };
            store.upsert(&record).unwrap();
            if rng.gen_bool(0.2) {
                store.soft_delete(id).unwrap();
            }
        }

        // Every idx value incl. values with collisions, plus misses (50..60).
        for target in 0..60 {
            let sql = store.find_by_vector_idx(target).unwrap();
            let scan = scan_reference(&store, target);
            assert_eq!(
                sql, scan,
                "parity broken for vector_idx={target}: SQL={sql:?} scan={scan:?}"
            );
        }
    }

    /// P2: journal mode must be read back after setting — WAL if the
    /// filesystem supports it, otherwise the rollback journal, but never a
    /// silently-assumed mode.
    #[test]
    fn journal_mode_is_verified() {
        let store = temp_store("journal");
        let mode = store.journal_mode().unwrap().to_ascii_lowercase();
        assert!(
            mode == "wal" || mode == "delete",
            "unexpected journal mode {mode:?}"
        );
        if mode == "wal" {
            // Round-trip still works under WAL.
            let rec = DocumentRecord {
                id: "wal_probe".into(),
                shard_idx: 0,
                vector_idx: 7,
                metadata: None,
                document: Some("ok".into()),
                deleted: false,
                created_at: 1.0,
                updated_at: 1.0,
            };
            store.upsert(&rec).unwrap();
            assert_eq!(store.get("wal_probe").unwrap().unwrap().vector_idx, 7);
        }
    }

    // -----------------------------------------------------------------------
    // F05b — embedding persistence
    // -----------------------------------------------------------------------

    /// Pre-F05b `documents` DDL, verbatim: a database written by it has no
    /// embedding columns.
    const LEGACY_DDL: &str = "CREATE TABLE documents (
             id TEXT PRIMARY KEY,
             shard_idx INTEGER,
             vector_idx INTEGER,
             metadata TEXT,
             document TEXT,
             deleted INTEGER DEFAULT 0,
             created_at REAL,
             updated_at REAL
         );";

    fn unique_dir(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "splatsdb-sqlite-emb-{tag}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn record(id: &str, text: &str, vector_idx: i64) -> DocumentRecord {
        DocumentRecord {
            id: id.into(),
            shard_idx: 0,
            vector_idx,
            metadata: Some(serde_json::json!({ "text": text })),
            document: Some(text.into()),
            deleted: false,
            created_at: 1.0,
            updated_at: 1.0,
        }
    }

    fn column_count(db: &Connection, column: &str) -> usize {
        let mut stmt = db.prepare("PRAGMA table_info(documents)").expect("pragma");
        let mut rows = stmt.query([]).expect("pragma rows");
        let mut n = 0;
        while let Some(row) = rows.next().expect("pragma row") {
            if row.get::<_, String>(1).expect("column name") == column {
                n += 1;
            }
        }
        n
    }

    fn row_count(db: &Connection) -> i64 {
        db.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
            .expect("row count")
    }

    /// F05b (3): a pre-vector database opens without loss, gains both
    /// columns exactly once (idempotent) and then persists and returns
    /// vectors byte-exact.
    #[test]
    fn legacy_schema_migrates_in_place_and_is_idempotent() {
        let dir = unique_dir("legacy");
        let db_path = dir.join("meta.db");
        {
            let raw = Connection::open(&db_path).expect("create db");
            raw.execute_batch(LEGACY_DDL).expect("legacy ddl");
            for (i, id) in ["leg_1", "leg_2"].iter().enumerate() {
                raw.execute(
                    "INSERT INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
                     VALUES (?1, 0, ?2, ?3, ?4, 0, 1.0, 1.0)",
                    params![
                        id,
                        i as i64,
                        format!(r#"{{"text":"{id} text"}}"#),
                        format!("{id} text")
                    ],
                )
                .expect("legacy insert");
            }
            assert!(!has_column(&raw, EMBEDDING_COL).expect("pragma"));
            assert!(!has_column(&raw, EMBEDDING_SOURCE_COL).expect("pragma"));
        }

        let store = SqliteMetadataStore::new(db_path.clone()).expect("open legacy db");
        assert!(store.embedding_schema_ready());
        assert_eq!(store.count(false).expect("count"), 2, "a row was lost");
        assert_eq!(
            store
                .get("leg_1")
                .expect("get")
                .expect("row")
                .document
                .as_deref(),
            Some("leg_1 text")
        );
        let docs = store.list_with_embeddings(100, 4).expect("list");
        assert_eq!(docs.len(), 2);
        assert!(
            docs.iter()
                .all(|d| d.embedding.is_none() && d.embedding_error.is_none()),
            "a legacy row has no vector and no error: {docs:?}"
        );
        assert_eq!(store.stored_vector_count().expect("stored count"), 0);

        // Second and third open: the migration is a no-op, each column
        // exists exactly once, no row touched.
        let store = SqliteMetadataStore::new(db_path.clone()).expect("reopen");
        drop(SqliteMetadataStore::new(db_path.clone()).expect("reopen again"));
        let raw = Connection::open(&db_path).expect("inspect");
        assert_eq!(column_count(&raw, EMBEDDING_COL), 1);
        assert_eq!(column_count(&raw, EMBEDDING_SOURCE_COL), 1);
        assert_eq!(row_count(&raw), 2);

        // Byte-exact round trip through the migrated table.
        let vector = [0.25f32, -1.5, 3.25e-7, 1.0];
        store
            .upsert_with_embedding(
                &record("new_1", "new one", 7),
                &vector,
                EmbeddingProvenance::SimCos,
            )
            .expect("upsert with vector");
        let docs = store.list_with_embeddings(100, 4).expect("list");
        let new_1 = docs.iter().find(|d| d.record.id == "new_1").expect("new_1");
        let stored = new_1.embedding.as_ref().expect("vector");
        assert_eq!(stored.vector, vector);
        assert_eq!(stored.source, EmbeddingProvenance::SimCos, "never promoted");
        assert_eq!(new_1.record.vector_idx, 7);
        let blob: Vec<u8> = raw
            .query_row(
                "SELECT embedding FROM documents WHERE id='new_1'",
                [],
                |r| r.get(0),
            )
            .expect("blob");
        assert_eq!(
            blob,
            vector
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
            "stored as little-endian f32"
        );
        assert_eq!(store.stored_vector_count().expect("stored count"), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// F05b (1)/(4): the write is ONE statement — when SQLite aborts it
    /// neither the document nor the vector exists; non-finite input never
    /// reaches SQLite; doc_del retires vector + provenance; a bad blob is
    /// refused with a diagnostic and never repaired.
    #[test]
    fn embedding_write_is_atomic_and_bad_blobs_are_rejected_not_repaired() {
        let dir = unique_dir("atomic");
        let db_path = dir.join("meta.db");
        let store = SqliteMetadataStore::new(db_path.clone()).expect("store");
        {
            let raw = Connection::open(&db_path).expect("raw");
            raw.execute_batch(
                "CREATE TRIGGER f05b_abort BEFORE INSERT ON documents WHEN NEW.id='abort'
                 BEGIN SELECT RAISE(ABORT,'injected failure'); END;",
            )
            .expect("trigger");
        }
        let raw = Connection::open(&db_path).expect("raw");
        let two = [1.0f32, 0.0];
        assert!(
            store
                .upsert_with_embedding(
                    &record("abort", "aborted", 0),
                    &two,
                    EmbeddingProvenance::Real
                )
                .is_err(),
            "the injected SQLite failure must surface"
        );
        assert!(store
            .upsert_with_embedding(
                &record("nan", "non-finite", 1),
                &[f32::NAN, 0.0],
                EmbeddingProvenance::Real
            )
            .is_err());
        assert!(store
            .upsert_with_embedding(
                &record("inf", "non-finite", 1),
                &[f32::INFINITY, 0.0],
                EmbeddingProvenance::Provided
            )
            .is_err());
        assert!(store
            .upsert_with_embedding(
                &record("empty", "empty", 1),
                &[],
                EmbeddingProvenance::Provided
            )
            .is_err());
        assert_eq!(
            store.count(true).expect("count"),
            0,
            "neither a document without vector nor a vector without document"
        );
        assert_eq!(row_count(&raw), 0);

        store
            .upsert_with_embedding(
                &record("d1", "doc one", 0),
                &two,
                EmbeddingProvenance::Provided,
            )
            .expect("write");
        store
            .upsert_with_embedding(&record("d2", "doc two", 1), &two, EmbeddingProvenance::Real)
            .expect("write");
        assert_eq!(store.stored_vector_count().expect("count"), 2);

        // doc_del: tombstone and vector retirement are the same UPDATE.
        assert!(store.soft_delete("d1").expect("soft delete"));
        let row: (i64, Option<Vec<u8>>, Option<String>) = raw
            .query_row(
                "SELECT deleted, embedding, embedding_source FROM documents WHERE id='d1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .expect("row");
        assert_eq!(
            row,
            (1, None, None),
            "doc_del must retire vector + provenance"
        );
        assert_eq!(store.list_with_embeddings(10, 2).expect("list").len(), 1);

        // A re-upsert through the vector-less trait method replaces the
        // whole row: the old vector (which may describe another text) is
        // not inherited.
        store
            .upsert(&record("d2", "doc two, rewritten elsewhere", 1))
            .expect("upsert");
        let docs = store.list_with_embeddings(10, 2).expect("list");
        assert!(
            docs[0].embedding.is_none() && docs[0].embedding_error.is_none(),
            "{docs:?}"
        );

        // Validation: every rejection names a reason.
        let valid: Vec<u8> = two.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut nan = valid.clone();
        nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let decode = |blob: Value, source: Option<&str>| decode_stored_embedding(&blob, source, 2);
        assert_eq!(
            decode(Value::Blob(valid.clone()), Some("simcos")),
            Ok(StoredEmbedding {
                vector: two.to_vec(),
                source: EmbeddingProvenance::SimCos
            })
        );
        assert_eq!(
            decode(Value::Blob(valid.clone()), Some("real")),
            Ok(StoredEmbedding {
                vector: two.to_vec(),
                source: EmbeddingProvenance::Real
            })
        );
        assert_eq!(
            decode(Value::Blob(vec![0u8; 7]), Some("real")),
            Err(StoredEmbeddingError::BadLength(7))
        );
        assert_eq!(
            decode(Value::Blob(Vec::new()), Some("real")),
            Err(StoredEmbeddingError::Empty)
        );
        assert_eq!(
            decode(Value::Blob(nan), Some("real")),
            Err(StoredEmbeddingError::NonFinite)
        );
        assert_eq!(
            decode(Value::Blob(vec![0u8; 12]), Some("real")),
            Err(StoredEmbeddingError::DimensionMismatch {
                got: 3,
                expected: 2
            })
        );
        assert_eq!(
            decode(Value::Blob(valid.clone()), None),
            Err(StoredEmbeddingError::MissingSource)
        );
        assert_eq!(
            decode(Value::Blob(valid.clone()), Some("tensor")),
            Err(StoredEmbeddingError::UnknownSource("tensor".into()))
        );
        assert_eq!(
            decode(Value::Null, Some("real")),
            Err(StoredEmbeddingError::MissingBlob)
        );
        assert_eq!(
            decode(Value::Text("0.1 0.2".into()), Some("real")),
            Err(StoredEmbeddingError::NotABlob("TEXT"))
        );
        for why in [
            StoredEmbeddingError::BadLength(7),
            StoredEmbeddingError::NonFinite,
            StoredEmbeddingError::MissingSource,
        ] {
            assert!(!why.to_string().is_empty());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // F05c — the generic `upsert` (CLI / HTTP writer: no vector in its
    // signature) must not drop a stored embedding
    // -----------------------------------------------------------------------

    /// `(embedding, embedding_source)` exactly as stored.
    fn stored_vector(db_path: &std::path::Path, id: &str) -> (Option<Vec<u8>>, Option<String>) {
        let db = Connection::open(db_path).expect("open db");
        db.query_row(
            "SELECT embedding, embedding_source FROM documents WHERE id=?1",
            params![id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("row")
    }

    /// What the CLI / HTTP path calls: the trait method, no vector.
    fn legacy_writer_upsert(
        store: &SqliteMetadataStore,
        id: &str,
        text: &str,
        deleted: bool,
        vector_idx: i64,
    ) {
        let mut rec = record(id, text, vector_idx);
        rec.deleted = deleted;
        rec.updated_at = 2.0;
        rec.metadata = Some(serde_json::json!({ "text": text, "updated_by": "legacy-writer" }));
        store.upsert(&rec).expect("generic upsert");
    }

    #[test]
    fn f05c_generic_upsert_keeps_vector_and_provenance_when_the_text_is_unchanged() {
        let dir = unique_dir("f05c_keep");
        let db_path = dir.join("meta.db");
        let store = SqliteMetadataStore::new(db_path.clone()).unwrap();
        let vector = [0.25f32, -0.5, 1.0, 0.125];
        let blob = encode_stored_embedding(&vector).unwrap();
        for (id, provenance, label) in [
            ("keep_real", EmbeddingProvenance::Real, "real"),
            ("keep_simcos", EmbeddingProvenance::SimCos, "simcos"),
            ("keep_provided", EmbeddingProvenance::Provided, "provided"),
        ] {
            store
                .upsert_with_embedding(&record(id, "the text stays", 0), &vector, provenance)
                .unwrap();
            assert_eq!(
                stored_vector(&db_path, id).0.as_deref(),
                Some(blob.as_slice()),
                "{id}: precondition"
            );

            // Metadata-only update by a writer that does not know vectors.
            legacy_writer_upsert(&store, id, "the text stays", false, 7);

            let after = stored_vector(&db_path, id);
            assert_eq!(
                after.0.as_deref(),
                Some(blob.as_slice()),
                "{id}: the generic upsert dropped or changed the stored vector"
            );
            assert_eq!(after.1.as_deref(), Some(label), "{id}: provenance lost");
            // ...while the document columns ARE replaced, as before.
            let rec = store.get(id).unwrap().expect("row");
            assert_eq!(rec.vector_idx, 7, "{id}");
            assert_eq!(rec.updated_at, 2.0, "{id}");
            assert_eq!(
                rec.metadata.as_ref().and_then(|m| m["updated_by"].as_str()),
                Some("legacy-writer"),
                "{id}"
            );
            assert_eq!(rec.document.as_deref(), Some("the text stays"), "{id}");
        }
        let db = Connection::open(&db_path).unwrap();
        assert_eq!(row_count(&db), 3, "an update must not duplicate rows");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_generic_upsert_retires_a_vector_that_would_describe_another_text() {
        let dir = unique_dir("f05c_text");
        let db_path = dir.join("meta.db");
        let store = SqliteMetadataStore::new(db_path.clone()).unwrap();
        store
            .upsert_with_embedding(
                &record("t1", "old text", 0),
                &[1.0, 0.0],
                EmbeddingProvenance::Real,
            )
            .unwrap();
        legacy_writer_upsert(&store, "t1", "a DIFFERENT text", false, 1);
        assert_eq!(
            stored_vector(&db_path, "t1"),
            (None, None),
            "a vector of the old text must not be served for the new one"
        );
        assert_eq!(
            store.get("t1").unwrap().unwrap().document.as_deref(),
            Some("a DIFFERENT text")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_generic_upsert_never_inherits_a_vector_into_a_new_revived_or_deleted_row() {
        let dir = unique_dir("f05c_inherit");
        let db_path = dir.join("meta.db");
        let store = SqliteMetadataStore::new(db_path.clone()).unwrap();

        // A row that never had a vector.
        legacy_writer_upsert(&store, "brand_new", "never had a vector", false, 0);
        assert_eq!(stored_vector(&db_path, "brand_new"), (None, None));

        // A soft-deleted row re-stored by the generic path: its vector was
        // retired by `soft_delete` and must not come back.
        store
            .upsert_with_embedding(
                &record("revived", "same text", 0),
                &[1.0, 0.0],
                EmbeddingProvenance::Real,
            )
            .unwrap();
        store.soft_delete("revived").unwrap();
        legacy_writer_upsert(&store, "revived", "same text", false, 2);
        assert_eq!(
            stored_vector(&db_path, "revived"),
            (None, None),
            "a retired vector must not be resurrected with a revived row"
        );

        // An upsert that stops the row from being live retires the vector.
        store
            .upsert_with_embedding(
                &record("flagged", "same text", 0),
                &[1.0, 0.0],
                EmbeddingProvenance::Real,
            )
            .unwrap();
        legacy_writer_upsert(&store, "flagged", "same text", true, 3);
        assert_eq!(
            stored_vector(&db_path, "flagged"),
            (None, None),
            "a row that stops being live must not keep a stored vector behind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
