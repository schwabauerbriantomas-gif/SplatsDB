//! Minimal HTTP API server for Python bindings
//!
//! Endpoints:
//!   POST /store   - Store a memory (text + optional metadata)
//!   POST /search  - Search memories (query text, top-k, `?rescore=gaussian`)
//!   POST /status  - Get store stats
//!   GET  /health  - Health check
//!   POST /recap   - SAC-style handoff recap from an inline action log (LLM-free)
//!
//! Architecture (R9 audit):
//!   - `/search` NEVER takes the global store mutex: it enqueues into an mpsc
//!     channel and a single `batch_worker` task coalesces requests within a
//!     BATCH_WINDOW_MS window (up to MAX_BATCH) and takes the mutex ONCE per
//!     batch (R9 §1 issue 1: the global lock no longer serializes searches).
//!   - CPU path: `find_neighbors_fused` (splats.rs) — HNSW-first sub-linear
//!     search with exact re-rank. The server config enables HNSW (the old
//!     `SplatsDBConfig::default()` shipped it disabled → every query was a
//!     linear O(N·D) scan).
//!   - GPU path (when built with `--features cuda` and a device is present):
//!     one batched search per coalesced batch over a cached flat snapshot of
//!     the dataset, refreshed only when the store's `data_version` changes
//!     (R9 §2). The snapshot (version, n, dim) additionally keys a
//!     process-level VRAM-resident `GpuIndex` (`gpu::cached_dataset_index`)
//!     so the dataset uploads to the GPU once per version, not per batch.
//!   - GPU fast path (the startup exact index, `serve --gpu`): answers only
//!     while the store is still the dataset that was uploaded
//!     (`GpuIndexKey::is_current`); any mutation (`/store`, WAL replay,
//!     reload, compact) bumps `data_version` and the query falls through to
//!     the batch worker (A03-F05).
//!   - A `/store` that carries no `embedding` vector asks the embed service
//!     for it — the same service and model `/recall` embeds queries with
//!     (`SPLATSDB_RECALL_EMBED_URL`, default :8860, `embeddinggemma-2`),
//!     without a prompt — and answers `embedding_source: "real"`. With no
//!     reachable service it answers 503 and stores nothing: the HTTP server
//!     has no text-hash fallback (fa1e528 parity).
//!
//! Security:
//!   - API key authentication via SPLATSDB_API_KEY env var (optional): the
//!     variable present but EMPTY or not UTF-8 aborts the startup (never
//!     "auth off"); a non-loopback bind without a key is refused unless the
//!     operator opts in with SPLATSDB_ALLOW_UNAUTHENTICATED=1
//!   - CORS restricted to localhost by default
//!   - Input size limits enforced

use crate::recap::{
    build_recap, RecapRequest, RecapResponse, MAX_ACTION_TEXT_LEN, MAX_TOTAL_ACTIONS_LEN,
};
use axum::{
    extract::{Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use tower_http::cors::CorsLayer;

use crate::config::SplatsDBConfig;
use crate::splats::SplatStore;
use crate::storage::persistence::{SplatsDBPersistence, StorageBackend};
use crate::storage::shard_guard::{
    fingerprint, read_wal_base, write_wal_base, DataDirLock, WalBase, WAL_BASE_FILE,
};
use crate::storage::wal::{WalDurability, WriteAheadLog};

// ─── Security constants ──────────────────────────────────
const MAX_TEXT_LEN: usize = 100_000; // 100KB max text input
const MAX_EMBEDDING_DIM: usize = 8192; // Max embedding dimension
const MAX_TOP_K: usize = 1000;

/// True iff every component is finite (rejects NaN and ±inf).
///
/// serde_json happily parses `1e400` into `inf` (JSON has no NaN literal but
/// float overflow decodes to inf); a single non-finite vector accepted into
/// the index poisons every later similarity computation, so every embedding
/// entry point validates this before touching the store.
fn all_finite(e: &[f32]) -> bool {
    e.iter().all(|v| v.is_finite())
}

// ─── GPU fast path dataset key (A03-F05) ──────────────────
/// Identity of the dataset uploaded to the startup GPU index (`serve --gpu`):
/// the store's `data_version` when it was built and the number of rows
/// uploaded.
///
/// `data_version` bumps on every mutation of the store (add/load/compact —
/// hence `/store`, the WAL replay of the boot and any reload), so the fast
/// path can answer only while the store is EXACTLY the dataset that was
/// uploaded. A frozen snapshot never answers for a mutated store.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct GpuIndexKey {
    version: u64,
    n: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl GpuIndexKey {
    /// True iff `version` / `n` describe the dataset this index holds.
    pub(crate) fn is_current(&self, version: u64, n: usize) -> bool {
        self.version == version && self.n == n
    }
}

/// Serve-level holder of the exact GPU index (feature `cuda`), F06a.
///
/// The resident index is a snapshot of ONE dataset version. `/store` (WAL
/// replay, compact) bumps `data_version`; the first request that no longer
/// matches rebuilds the index ONCE for the version current at that moment
/// and swaps it in — concurrent stale requests wait on `rebuild_lock` and
/// reuse the fresh index (a burst of mutations costs one upload, not one per
/// query or per store). Requests whose version matches never touch the GPU
/// state beyond a read lock. An explicit GPU failure (no CUDA, failed upload)
/// disables the fast path for the process: those requests take the batch
/// worker, as before. Before F06a the startup key went stale on the first
/// mutation and every later `/search` took the batch worker (`backend:
/// null`, 2 ms coalescing window: after-mutation p50 was 2.0-2.3× the base).
// The exact-index latch below is the only AtomicBool user in this module and
// it only exists with the cuda feature; without it the import would be dead.
#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicBool;

#[cfg(feature = "cuda")]
pub struct GpuExactIndex {
    current: parking_lot::RwLock<Option<Arc<GpuFastPath>>>,
    rebuild_lock: parking_lot::Mutex<()>,
    /// Set by an explicit GPU failure: no more rebuild attempts.
    disabled: AtomicBool,
    /// Dataset versions uploaded after the startup one (observability/tests).
    rebuilds: AtomicUsize,
}

#[cfg(feature = "cuda")]
impl GpuExactIndex {
    /// Startup upload of the current dataset, read straight from the store's
    /// buffer (no host copy). Returns `(holder, corpus_n)`; the holder is
    /// `None` only when CUDA is not available at all.
    fn startup(store: &SplatStore) -> (Option<Arc<Self>>, usize) {
        let n = store.n_active();
        let holder = Self {
            current: parking_lot::RwLock::new(None),
            rebuild_lock: parking_lot::Mutex::new(()),
            disabled: AtomicBool::new(false),
            rebuilds: AtomicUsize::new(0),
        };
        let Some(mut index) = crate::gpu::cuda_kernel::GpuIndex::new() else {
            eprintln!("[splatsdb] serve: GPU no disponible (--gpu ignorado); búsqueda en CPU");
            return (None, n);
        };
        match store.mu_view() {
            Some(view) => {
                let dim = view.ncols();
                let uploaded = view
                    .as_slice()
                    .map(|flat| index.upload_dataset(flat, n, dim))
                    == Some(true);
                if uploaded {
                    let version = store.data_version();
                    eprintln!(
                        "[splatsdb] serve: GPU exact index listo ({n} vectores, dim {dim}, \
                         data_version {version})"
                    );
                    *holder.current.write() = Some(Arc::new(GpuFastPath {
                        key: GpuIndexKey { version, n },
                        index,
                    }));
                } else {
                    eprintln!("[splatsdb] serve: upload GPU falló; búsqueda en CPU");
                    holder.disabled.store(true, Ordering::Relaxed);
                }
            }
            None => eprintln!("[splatsdb] serve: store vacío, GPU sin dataset"),
        }
        (Some(Arc::new(holder)), n)
    }

    /// The resident index when it is exactly the dataset `(version, n)` this
    /// request read — a read lock and an Arc clone, nothing else.
    fn current_index(&self, version: u64, n: usize) -> Option<Arc<GpuFastPath>> {
        self.current
            .read()
            .as_ref()
            .filter(|f| f.key.is_current(version, n))
            .cloned()
    }

    /// [`Self::current_index`], rebuilding the index once when stale.
    fn index_for_request(
        &self,
        store: &Arc<Mutex<SplatStore>>,
        version: u64,
        n: usize,
    ) -> Option<Arc<GpuFastPath>> {
        if let Some(fast) = self.current_index(version, n) {
            return Some(fast);
        }
        // One rebuild at a time: the others re-check and reuse its result.
        let _serial = self.rebuild_lock.lock();
        if let Some(fast) = self.current_index(version, n) {
            return Some(fast);
        }
        if self.disabled.load(Ordering::Relaxed) {
            return None;
        }
        let Some(mut index) = crate::gpu::cuda_kernel::GpuIndex::new() else {
            self.fail("CUDA no disponible al reconstruir el índice");
            return None;
        };
        // Upload straight from the store's buffer — no host copy (each copy of
        // a 10000×768 dataset is ~30 MB of RSS). The store lock is held only
        // for the duration of the H2D copy, never during a kernel launch.
        let (key, uploaded) = {
            let s = store.blocking_lock();
            let (version, n) = (s.data_version(), s.n_active());
            if let Some(fast) = self.current_index(version, n) {
                return Some(fast);
            }
            let uploaded = match s.mu_view() {
                Some(view) => match view.as_slice() {
                    Some(flat) => index.upload_dataset(flat, n, view.ncols()),
                    None => false,
                },
                None => false,
            };
            (GpuIndexKey { version, n }, uploaded)
        };
        if !uploaded {
            self.fail("upload del dataset falló al reconstruir el índice");
            return None;
        }
        let fast = Arc::new(GpuFastPath { key, index });
        *self.current.write() = Some(fast.clone());
        self.rebuilds.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "[splatsdb] serve: GPU exact index reconstruido ({} vectores, data_version {})",
            key.n, key.version
        );
        Some(fast)
    }

    /// Dataset versions uploaded after the startup one.
    pub fn rebuilds(&self) -> usize {
        self.rebuilds.load(Ordering::Relaxed)
    }

    /// Explicit GPU failure: drop the resident index and stop trying.
    fn fail(&self, why: &str) {
        self.disabled.store(true, Ordering::Relaxed);
        *self.current.write() = None;
        eprintln!("[splatsdb] serve: GPU fast path desactivada ({why}); batch worker");
    }
}

/// Exact GPU index built once at startup plus the dataset key it was built
/// from ([`GpuIndexKey`]).
#[cfg(feature = "cuda")]
pub struct GpuFastPath {
    key: GpuIndexKey,
    index: crate::gpu::cuda_kernel::GpuIndex,
}

/// One exact top-k row from the startup GPU index, sorted by distance, or
/// `None` when the kernel declined or violated its contract (row count ≠ 1,
/// idx/dist length mismatch, out-of-range index). The caller then falls
/// through to the batch path (GPU cached-by-version, else CPU) instead of
/// returning a truncated or fabricated row.
#[cfg(feature = "cuda")]
fn gpu_fast_path_results(
    fast: &GpuFastPath,
    embedding: &[f32],
    k: usize,
) -> Option<Vec<SearchResult>> {
    // F02e1: the kernel orders exact ties by scan position, not by id.
    // `contractual_topk` asks for k+1 pairs, proves the k-set against a tie
    // at the boundary (escalating to the kernel's largest window) and
    // returns the canonical (d², id) order — untied answers are the kernel's
    // own k pairs, bit for bit. A tie the kernel cannot resolve, or a
    // malformed row, falls through to the batch worker (host-exact there).
    let fetch = |window: usize| -> Option<Vec<(usize, f32)>> {
        let rows = fast.index.topk_search(embedding, 1, window, "l2")?;
        let [(idxs, dists)] = rows.as_slice() else {
            eprintln!(
                "[splatsdb] GPU topk devolvió {} filas para 1 query; query cae a CPU",
                rows.len()
            );
            return None;
        };
        if idxs.len() != dists.len() || idxs.iter().any(|&i| i >= fast.key.n) {
            eprintln!(
                "[splatsdb] GPU topk devolvió una fila inválida ({} índices / {} distancias sobre \
                 {} vectores); query cae a CPU",
                idxs.len(),
                dists.len(),
                fast.key.n
            );
            return None;
        }
        Some(
            idxs.iter()
                .zip(dists.iter())
                .map(|(&i, &d)| (i, d))
                .collect(),
        )
    };
    let pairs = match crate::splats::contractual_topk(fast.key.n, k, false, fetch) {
        crate::splats::Contract::Exact(pairs) => pairs,
        crate::splats::Contract::Saturated(_) | crate::splats::Contract::NoFetch => return None,
    };
    Some(
        pairs
            .into_iter()
            .map(|(index, score)| SearchResult {
                index,
                score,
                metadata: None,
                confidence: None,
                topic: None,
            })
            .collect(),
    )
}

// ─── Micro-batching constants (R9 §2) ────────────────────
/// Coalescing window: requests arriving within this window are served by a
/// single batched search instead of one lock acquisition + scan per request.
const BATCH_WINDOW_MS: u64 = 2;
/// Hard cap on queries coalesced into one batch.
const MAX_BATCH: usize = 64;
/// Channel capacity for pending search requests (backpressure beyond this).
const BATCH_CHANNEL_CAP: usize = 256;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<SplatStore>>,
    pub next_id: Arc<AtomicUsize>,
    /// Sender side of the search micro-batching channel. The `/search` hot
    /// path pushes here and never takes the global store mutex; the single
    /// `batch_worker` task takes it once per batch.
    pub batch_tx: mpsc::Sender<BatchRequest>,
    /// Number of batches served by the worker (observability / tests).
    pub batches_served: Arc<AtomicUsize>,
    /// Índice exacto en GPU (kernel PTX l2_topk): subido en el arranque de
    /// serve --gpu y reconstruido UNA vez por versión de dataset tras una
    /// mutación (F06a), con la clave del dataset que subió (`GpuIndexKey`).
    /// topk_search es &self → se comparte sin Mutex. El handler sólo lo usa
    /// mientras la clave siga vigente (A03-F05).
    #[cfg(feature = "cuda")]
    pub gpu: Option<Arc<GpuExactIndex>>,
    /// Campo splat (celdas con radio) para /field-search exacto. None si no
    /// hay field/ persistido en el data-dir.
    pub field: Option<Arc<FieldData>>,
    /// Filas del dataset que subió al fast path GPU en el arranque (la clave
    /// `GpuIndexKey` lo coteja): los splats del shard, nunca los chunks de
    /// field/ (ver `serve_corpus`). Sólo lo lee el fast path (feature cuda).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub corpus_n: usize,
    /// Directorio de datos del server (flag --data-dir). /recall lo usa para
    /// abrir {data_dir}/sidecar.db en mode=ro y loggear fires. None en tests.
    pub data_dir: Option<String>,
    /// Durabilidad: WAL append-only + shard "default" en --data-dir. Un
    /// /store se loggea al WAL y cada COMPACT_EVERY se persiste el shard.
    /// None en tests sin data-dir (comportamiento in-memory clásico).
    pub durability: Option<Arc<Durability>>,
    /// Authentication resolved ONCE at startup from `SPLATSDB_API_KEY`
    /// (A05-F10). The middleware reads it from here and never from the
    /// environment, so an empty/non-UTF-8 key cannot turn into "auth off"
    /// per request.
    pub auth: AuthConfig,
}

/// Durabilidad del `serve` con --data-dir: cada /store aceptado se escribe
/// en un WAL append-only (bajo el lock del store, así el orden del WAL ==
/// orden de índices y el replay reconstruye el dataset idéntico), y cada
/// COMPACT_EVERY stores — y en el shutdown — se persiste el shard
/// "default" completo y se trunca el WAL.
///
/// Durability is interpreted through the session's fixed `WalDurability` mode:
/// - **process-crash** (default): a complete WAL write acknowledges `stored`,
///   `durable:true`, `crash_durable:true`, `power_loss_durable:false`. Group
///   `sync_all` only clears `wal_pending`; it does not change the declared mode.
/// - **power-loss**: `sync_all` must succeed before `stored`; a failed/ambiguous
///   sync returns `outcome:unknown` and never acknowledges `power_loss_durable`.
///
/// Pending entries are therefore "not yet power-loss safe", never "not crash
/// durable" in the default mode.
#[derive(Debug)]
struct LoggedStore {
    _lsn: u64,
    already_logged: bool,
}

#[derive(Debug)]
struct StoreWorkerError {
    status: StatusCode,
    message: String,
    response: Option<Box<StoreResponse>>,
}

pub struct Durability {
    wal: WriteAheadLog,
    persist: SplatsDBPersistence,
    /// Sidecar `wal/store.wal.base` de este data-dir.
    base_path: std::path::PathBuf,
    /// Shard "default" contra el que abrió esta generación del WAL.
    /// `compact` se niega a sobrescribir el shard si en disco ya no es este
    /// (lost update, A03-F12).
    base: std::sync::Mutex<WalBase>,
    /// Entradas "store" del WAL que el replay NO pudo aplicar: su única
    /// copia. Mientras haya, `compact` no trunca el WAL.
    unapplied: std::sync::Mutex<Vec<u64>>,
    last_compact_error: std::sync::Mutex<Option<String>>,
    /// Último LSN asignado a un "store" (fetch_max: monotónico).
    last_lsn: AtomicU64,
    /// Stores aceptados que todavía no pasaron por `sync_all`.
    pending: AtomicU64,
    /// Stores loggeados desde el arranque.
    since_compact: AtomicUsize,
    /// Fixed for the whole session; ACK flags are interpreted through it.
    mode: WalDurability,
    /// Idempotency keys of records that may exist in the WAL. A retry after
    /// an ambiguous result reconciles against these instead of appending twice.
    known_ids: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl Durability {
    /// Stores entre compactaciones (persistir shard + truncar WAL).
    const COMPACT_EVERY: usize = 1000;
    /// Stores between group `sync_all` calls. Default-mode entries are already
    /// process-crash durable while pending; they are not yet power-loss safe.
    const CHECKPOINT_EVERY: u32 = 32;

    /// Abre el WAL de `data_dir`. Devuelve `Err` si el WAL no se puede abrir
    /// ni leer, o si el shard cambió debajo de un WAL con stores sin compactar.
    #[cfg(test)]
    fn open(data_dir: &str) -> anyhow::Result<Self> {
        Self::open_with_mode(data_dir, WalDurability::ProcessCrash)
    }

    fn open_with_mode(data_dir: &str, mode: WalDurability) -> anyhow::Result<Self> {
        let persist = SplatsDBPersistence::with_backend(data_dir, StorageBackend::Sqlite, false)
            .map_err(|e| anyhow::anyhow!("persistence init falló: {e}"))?;
        let shard_path = Self::shard_path(&persist)?;
        let current = fingerprint(&shard_path)?;
        let base_path = std::path::Path::new(data_dir)
            .join("wal")
            .join(WAL_BASE_FILE);
        let wal = WriteAheadLog::new_with_mode(&format!("{data_dir}/wal/store.wal"), 1, mode)?;
        let entries = wal.recover()?;
        let pending = entries.iter().filter(|e| e.operation == "store").count();
        let known_ids = entries
            .iter()
            .filter(|e| e.operation == "store")
            .filter_map(|e| e.data.get("id").and_then(|v| v.as_str()))
            .map(str::to_owned)
            .collect();
        if let Some(recorded) = read_wal_base(&base_path)? {
            if recorded.shard != current && pending > 0 {
                anyhow::bail!(
                    "shard 'default' de {data_dir} cambió en disco mientras el WAL tenía {pending} \
                     store(s) sin compactar: replay por idx no es seguro. Restaurá el shard contra \
                     el que se escribió el WAL, o respaldá y reconciliá wal/store.wal a mano"
                );
            }
        }
        write_wal_base(&base_path, &WalBase { shard: current })?;
        wal.checkpoint()?;
        Ok(Self {
            wal,
            persist,
            base_path,
            base: std::sync::Mutex::new(WalBase { shard: current }),
            unapplied: std::sync::Mutex::new(Vec::new()),
            last_compact_error: std::sync::Mutex::new(None),
            last_lsn: AtomicU64::new(entries.last().map_or(0, |e| e.lsn)),
            pending: AtomicU64::new(0),
            since_compact: AtomicUsize::new(0),
            mode,
            known_ids: std::sync::Mutex::new(known_ids),
        })
    }

    /// `data/vectors/default.bin` del data-dir (mismo layout que
    /// `SplatsDBPersistence::save_vectors`).
    fn shard_path(persist: &SplatsDBPersistence) -> std::io::Result<std::path::PathBuf> {
        Ok(persist
            .storage_path
            .join("data")
            .join("vectors")
            .join("default.bin"))
    }

    fn count_stores(wal: &WriteAheadLog) -> std::io::Result<usize> {
        Ok(wal
            .recover()?
            .iter()
            .filter(|e| e.operation == "store")
            .count())
    }

    fn lock<'a, T>(
        slot: &'a std::sync::Mutex<T>,
        what: &str,
    ) -> std::io::Result<std::sync::MutexGuard<'a, T>> {
        slot.lock()
            .map_err(|_| std::io::Error::other(format!("lock de {what} envenenado")))
    }

    /// Log before the in-memory mutation. A failed write therefore never
    /// leaves an unacknowledged vector in memory. An ambiguous sync/write is
    /// reported as such and its id is retained for idempotent reconciliation.
    fn log_store(
        &self,
        id: &str,
        auto_id: Option<usize>,
        idx: usize,
        vector: &[f32],
    ) -> Result<LoggedStore, StoreWorkerError> {
        {
            let known = self
                .known_ids
                .lock()
                .map_err(|_| self.poisoned_failure(id))?;
            if known.contains(id) {
                drop(known);
                return self.reconcile_retry(id, auto_id, idx, vector);
            }
        }
        let data =
            serde_json::json!({ "id": id, "auto_id": auto_id, "idx": idx, "vector": vector });
        match self.wal.log_operation("store", data) {
            Ok(lsn) => {
                self.known_ids
                    .lock()
                    .map_err(|_| self.poisoned_failure(id))?
                    .insert(id.to_owned());
                self.last_lsn.fetch_max(lsn, Ordering::SeqCst);
                if self.mode == WalDurability::ProcessCrash {
                    self.pending.fetch_add(1, Ordering::SeqCst);
                }
                let n = self.since_compact.fetch_add(1, Ordering::SeqCst) + 1;
                if self.mode == WalDurability::ProcessCrash
                    && (n as u32).is_multiple_of(Self::CHECKPOINT_EVERY)
                {
                    if let Err(e) = self.checkpoint_now() {
                        return Err(self.failure(id, &e.to_string(), true, true));
                    }
                }
                Ok(LoggedStore {
                    _lsn: lsn,
                    already_logged: false,
                })
            }
            Err(e) => {
                if e.entry_reached_kernel || e.may_be_persisted {
                    self.known_ids
                        .lock()
                        .map_err(|_| self.poisoned_failure(id))?
                        .insert(id.to_owned());
                    if self.mode == WalDurability::PowerLoss {
                        self.pending.fetch_add(1, Ordering::SeqCst);
                    }
                }
                Err(self.failure(
                    id,
                    &format!("WAL write failed: {e}"),
                    e.entry_reached_kernel,
                    e.may_be_persisted,
                ))
            }
        }
    }

    /// Reconcile an id returned after an ambiguous failure. Never append a
    /// second record for it; conflicting reuse of an id is rejected.
    fn reconcile_retry(
        &self,
        id: &str,
        auto_id: Option<usize>,
        idx: usize,
        vector: &[f32],
    ) -> Result<LoggedStore, StoreWorkerError> {
        let entries = self
            .wal
            .recover()
            .map_err(|e| self.failure(id, &e.to_string(), false, true))?;
        let entry = entries.iter().rev().find(|e| {
            e.operation == "store" && e.data.get("id").and_then(|v| v.as_str()) == Some(id)
        });
        let Some(entry) = entry else {
            return Err(self.failure(
                id,
                "WAL id disappeared before retry reconciliation",
                false,
                true,
            ));
        };
        let same = entry.data.get("idx").and_then(|v| v.as_u64()) == Some(idx as u64)
            && entry.data.get("auto_id").and_then(|v| v.as_u64()) == auto_id.map(|v| v as u64)
            && entry.data.get("vector") == Some(&serde_json::json!(vector));
        if !same {
            return Err(StoreWorkerError {
                status: StatusCode::CONFLICT,
                message: "id already belongs to a different WAL entry".into(),
                response: Some(Box::new(StoreResponse {
                    embedding_source: None,
                    error: Some("id already belongs to a different WAL entry".into()),
                    id: id.into(),
                    status: "conflict".into(),
                    outcome: "not_stored".into(),
                    may_be_persisted: false,
                    durable: false,
                    crash_durable: false,
                    power_loss_durable: false,
                    wal_pending: self.wal_pending(),
                    durability_mode: self.mode.as_str().into(),
                    persistence: "wal".into(),
                })),
            });
        }
        if self.mode == WalDurability::PowerLoss {
            self.checkpoint_now()
                .map_err(|e| self.failure(id, &e.to_string(), true, true))?;
        }
        Ok(LoggedStore {
            _lsn: entry.lsn,
            already_logged: true,
        })
    }

    /// `sync_all` del WAL: todo lo aceptado hasta acá pasa a ser duradero.
    fn checkpoint_now(&self) -> std::io::Result<()> {
        self.wal.checkpoint()?;
        self.pending.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Mode-level durability. In process-crash mode pending means "not yet
    /// power-loss safe", not "not crash durable".
    fn is_durable(&self) -> bool {
        self.mode == WalDurability::ProcessCrash || self.wal_pending() == 0
    }

    fn crash_durable(&self) -> bool {
        true
    }

    fn power_loss_durable(&self) -> bool {
        self.mode == WalDurability::PowerLoss && self.wal_pending() == 0
    }

    fn wal_pending(&self) -> u64 {
        self.pending.load(Ordering::SeqCst)
    }

    fn success(&self, id: &str) -> StoreResponse {
        StoreResponse {
            embedding_source: None,
            error: None,
            id: id.into(),
            status: "stored".into(),
            outcome: "stored".into(),
            may_be_persisted: true,
            durable: self.is_durable(),
            crash_durable: self.crash_durable(),
            power_loss_durable: self.power_loss_durable(),
            wal_pending: self.wal_pending(),
            durability_mode: self.mode.as_str().into(),
            persistence: "wal".into(),
        }
    }

    fn failure(
        &self,
        id: &str,
        message: &str,
        reached_kernel: bool,
        may_be_persisted: bool,
    ) -> StoreWorkerError {
        StoreWorkerError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            response: Some(Box::new(StoreResponse {
                embedding_source: None,
                error: Some(message.into()),
                id: id.into(),
                status: if may_be_persisted {
                    "unknown"
                } else {
                    "not_stored"
                }
                .into(),
                outcome: if may_be_persisted {
                    "unknown"
                } else {
                    "not_stored"
                }
                .into(),
                may_be_persisted,
                durable: false,
                crash_durable: reached_kernel,
                power_loss_durable: false,
                wal_pending: self.wal_pending(),
                durability_mode: self.mode.as_str().into(),
                persistence: "wal".into(),
            })),
        }
    }

    fn poisoned_failure(&self, id: &str) -> StoreWorkerError {
        self.failure(id, "WAL idempotency state is poisoned", false, true)
    }

    fn stores_since_compact(&self) -> usize {
        self.since_compact.load(Ordering::SeqCst)
    }

    fn last_compact_error(&self) -> Option<String> {
        self.last_compact_error.lock().ok().and_then(|g| g.clone())
    }

    /// Flush + fsync del WAL. Solo tests.
    #[cfg(test)]
    fn flush(&self) -> std::io::Result<()> {
        self.checkpoint_now()
    }

    /// Persiste el shard "default" con el estado ACTUAL del store y trunca
    /// el WAL (todo lo loggeado ya quedó en el shard). Llamar con el lock
    /// del store tomado. Devuelve la cantidad de splats persistidos.
    ///
    /// Se niega (Err, WAL intacto) si el shard en disco ya no es el que esta
    /// generación del WAL vio al arrancar (lost update) o si el WAL todavía
    /// tiene entradas que no se pudieron aplicar.
    fn compact(&self, store: &SplatStore) -> std::io::Result<usize> {
        let outcome = self.compact_checked(store);
        if let Err(e) = &outcome {
            if let Ok(mut slot) = self.last_compact_error.lock() {
                *slot = Some(e.to_string());
            }
        }
        outcome
    }

    fn compact_checked(&self, store: &SplatStore) -> std::io::Result<usize> {
        // (1) Lost update (A03-F12): el shard en disco tiene que seguir
        // siendo el que esta generación del WAL vio al arrancar.
        let shard_path = Self::shard_path(&self.persist)?;
        let current = fingerprint(&shard_path)?;
        let base = *Self::lock(&self.base, "base de WAL")?;
        let pending = Self::count_stores(&self.wal)?;
        if current != base.shard {
            return Err(std::io::Error::other(format!(
                "shard 'default' ({}) cambió en disco desde el arranque de este serve: NO se \
                 sobrescribe con el estado stale en memoria; {pending} store(s) aceptados se \
                 conservan en wal/store.wal",
                shard_path.display()
            )));
        }
        // (2) Entradas del WAL que el replay no pudo aplicar son la única
        // copia de esos datos: no se trunca el WAL mientras existan.
        let unapplied = Self::lock(&self.unapplied, "unapplied")?.clone();
        let n = store.n_active();
        if n > 0 && !unapplied.is_empty() {
            return Err(std::io::Error::other(format!(
                "{} entrada(s) 'store' del WAL (lsn {:?}) no se aplicaron al store: no se trunca \
                 el WAL, se conserva",
                unapplied.len(),
                &unapplied[..unapplied.len().min(8)]
            )));
        }
        //
        // Orden shard→WAL: el shard se escribe con `atomic_write` (fsync del
        // tmp → rename → fsync del directorio) y recién cuando eso terminó se
        // trunca el WAL; si el shard falla, el WAL sigue intacto.
        //
        // Con el store vacío NO trunca: si el WAL todavía tiene entradas
        // "store" son la única copia de datos no recuperados (p.ej. salteados
        // por el replay) y se conservan — devuelve `Err` para que se vea.
        if n > 0 {
            let Some(mu) = store.get_mu() else {
                return Err(std::io::Error::other(
                    "get_mu() devolvió None con n_active > 0",
                ));
            };
            self.persist.save_vectors(&mu, "default")?;
        } else {
            let unrecovered = Self::count_stores(&self.wal)?;
            if unrecovered > 0 {
                return Err(std::io::Error::other(format!(
                    "store vacío (n_active=0) con {unrecovered} entrada(s) 'store' aún en el WAL: \
                     no se persiste ni se trunca, el WAL se conserva"
                )));
            }
        }
        let before = self.last_lsn.load(Ordering::SeqCst).saturating_add(1);
        self.wal.truncate(before)?;
        // (3) Nueva generación: el WAL arranca vacío contra el shard recién
        // escrito. El sidecar se reescribe DESPUÉS del truncate: si el
        // proceso muere entre ambos, el WAL ya está vacío y un sidecar viejo
        // no afirma nada (se refresca en el próximo open).
        let fresh = WalBase {
            shard: fingerprint(&shard_path)?,
        };
        *Self::lock(&self.base, "base de WAL")? = fresh;
        self.last_lsn.store(0, Ordering::SeqCst);
        self.pending.store(0, Ordering::SeqCst);
        self.known_ids.lock().map(|mut ids| ids.clear()).ok();
        if let Ok(mut slot) = self.unapplied.lock() {
            slot.clear();
        }
        if let Err(e) = write_wal_base(&self.base_path, &fresh) {
            eprintln!(
                "[splatsdb] WAL: no se pudo actualizar {}: {e}",
                self.base_path.display()
            );
        }
        Ok(n)
    }
}

/// Al dropear (panic, exit() sin TERM, tests): flush + fsync del WAL, así lo
/// aceptado sin compactar queda duradero en el WAL aunque el compact del
/// shutdown se haya negado (lost update) o fallado.
impl Drop for Durability {
    fn drop(&mut self) {
        let _ = self.wal.checkpoint();
    }
}

/// A single search request queued for batch execution. The reply is delivered
/// on a private oneshot channel.
pub struct BatchRequest {
    query: Vec<f32>,
    k: usize,
    /// `?rescore=gaussian` was requested.
    rescore_gaussian: bool,
    resp: oneshot::Sender<Result<Vec<SearchResult>, String>>,
}

#[derive(Deserialize)]
pub struct StoreRequest {
    pub text: String,
    pub category: Option<String>,
    pub id: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub top_k: Option<usize>,
}

/// Optional query-string parameters for `/search`.
///
/// `?rescore=gaussian` re-ranks the top-k under the exact Gaussian splat
/// score `alpha_i · exp(-kappa_i · ‖q − mu_i‖²)` (higher is better — the
/// response `score` field carries THIS value instead of L2 distance; see
/// `SplatStore::find_neighbors_gaussian`). Any other value is ignored.
#[derive(Deserialize)]
pub struct SearchQuery {
    pub rescore: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StoreResponse {
    pub id: String,
    pub status: String,
    pub outcome: String,
    pub may_be_persisted: bool,
    /// `true` sii, además de estar en el WAL, `sync_all` del WAL ya terminó
    /// después de esta entrada. `false` = aceptado (sobrevive a un crash del
    /// proceso) pero todavía no duradero ante corte de energía, o bien el
    /// server corre efímero sin data-dir.
    pub durable: bool,
    pub crash_durable: bool,
    pub power_loss_durable: bool,
    pub wal_pending: u64,
    pub durability_mode: String,
    /// Diagnostic for a non-stored outcome; absent on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// "wal" (persistente en --data-dir) o "ephemeral" (sólo memoria).
    pub persistence: String,
    /// `"real"` when the request carried no `embedding` vector and the
    /// server asked the embed service for it. Absent when the client
    /// supplied the vector. Same key and label as the MCP server uses for
    /// vectors that came from the embed service.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_source: Option<&'static str>,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    /// "gpu" cuando la query corrió por el kernel PTX exacto (serve --gpu).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct SearchResult {
    pub index: usize,
    pub score: f32,
    pub metadata: Option<String>,
    /// Qué tan cerca del centro de su tópico cayó el resultado:
    /// 1 − d(q,x)/R_topic, clampeado a [0,1] (1 = prototipo del cluster).
    /// Solo con embedding real (los scores de texto-hash no son distancia).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Splat (tópico) al que pertenece el resultado, si hay field/ cargado.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic: Option<u32>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub n_active: usize,
    pub max_splats: usize,
    pub dimension: usize,
    pub has_hnsw: bool,
    pub has_lsh: bool,
    pub has_quantization: bool,
    /// "wal": persistente en --data-dir. "ephemeral": nada sobrevive al
    /// proceso (sólo cuando se pidió explícitamente; `serve` con --data-dir
    /// nunca arranca efímero, A03-F01).
    pub persistence: String,
    /// `true` sii todo lo aceptado hasta ahora pasó por `sync_all`.
    pub durable: bool,
    pub crash_durable: bool,
    pub power_loss_durable: bool,
    /// Entradas aceptadas que todavía no pasaron por `sync_all` (group commit).
    pub wal_pending: u64,
    pub durability_mode: String,
    /// Último error de compact: el shard o el WAL no se pudieron persistir.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_compact_error: Option<String>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    let store = state.store.lock().await;
    let stats = store.get_statistics();
    let (
        persistence,
        durability_mode,
        durable,
        crash_durable,
        power_loss_durable,
        wal_pending,
        last_compact_error,
    ) = match state.durability.as_ref() {
        Some(d) => (
            "wal",
            d.mode.as_str(),
            d.is_durable(),
            d.crash_durable(),
            d.power_loss_durable(),
            d.wal_pending(),
            d.last_compact_error(),
        ),
        None => ("ephemeral", "ephemeral", false, false, false, 0, None),
    };
    Json(StatusResponse {
        n_active: store.n_active(),
        max_splats: store.max_splats(),
        dimension: stats.embedding_dim,
        has_hnsw: store.has_hnsw(),
        has_lsh: store.has_lsh(),
        has_quantization: store.has_quantization(),
        persistence: persistence.into(),
        durable,
        crash_durable,
        power_loss_durable,
        wal_pending,
        durability_mode: durability_mode.into(),
        last_compact_error,
    })
}

async fn store_memory(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    let error =
        |code: StatusCode, message: String| (code, Json(serde_json::json!({ "error": message })));
    if req.text.len() > MAX_TEXT_LEN {
        return Err(error(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Text exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }
    let dim = state.store.lock().await.get_statistics().embedding_dim;
    if let Some(ref emb) = req.embedding {
        if emb.len() != dim {
            return Err(error(
                StatusCode::BAD_REQUEST,
                format!(
                    "Embedding dimension mismatch: expected {dim}, got {}",
                    emb.len()
                ),
            ));
        }
        if emb.len() > MAX_EMBEDDING_DIM {
            return Err(error(
                StatusCode::BAD_REQUEST,
                format!("Embedding dimension exceeds max ({MAX_EMBEDDING_DIM})"),
            ));
        }
        if !all_finite(emb) {
            return Err(error(
                StatusCode::BAD_REQUEST,
                "Embedding contains non-finite values (NaN/inf)".into(),
            ));
        }
    }
    // A request without a vector gets one from the embed service — the same
    // service and model `/recall` embeds queries with, so every vector in
    // the hub lives in one space. Documents are embedded WITHOUT a prompt
    // (the pipeline's measured asymmetry: only queries carry
    // `Retrieval-query`). No service → 503 and nothing stored. There is no
    // text-hash fallback (fa1e528 parity — a hash of the text ranks noise
    // as success).
    let (embedding, embedded_here) = match req.embedding {
        Some(vector) => (vector, false),
        None => {
            let text = req.text.clone();
            let vector =
                tokio::task::spawn_blocking(move || fetch_store_embedding(&text, dim))
                    .await
                    .map_err(|e| {
                        error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("embed task failed: {e}"),
                        )
                    })?
                    .map_err(|e| {
                        error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!(
                                "no embedding service: {e}. Send an `embedding` vector or start the embedding service"
                            ),
                        )
                    })?;
            (vector, true)
        }
    };
    let arr = Array2::from_shape_vec((1, dim), embedding.clone())
        .map_err(|e| error(StatusCode::BAD_REQUEST, format!("Bad embedding shape: {e}")))?;
    let id_num = state.next_id.fetch_add(1, Ordering::SeqCst);
    let user_supplied = req.id.is_some();
    let id = req.id.unwrap_or_else(|| format!("mem_{id_num}"));
    let store = state.store.clone();
    let durability = state.durability.clone();
    let emb_for_wal = embedding.clone();
    let id_for_wal = id.clone();
    let auto_id = (!user_supplied).then_some(id_num);
    let wal_outcome =
        tokio::task::spawn_blocking(move || -> Result<LoggedStore, StoreWorkerError> {
            let mut store = store.blocking_lock();
            if store.n_active() >= store.max_splats() {
                return Err(StoreWorkerError {
                    status: StatusCode::INSUFFICIENT_STORAGE,
                    message: "Store full".into(),
                    response: None,
                });
            }
            let idx = store.n_active();
            let logged = if let Some(d) = durability.as_ref() {
                // WAL first: no unconfirmed mutation is ever applied in memory.
                d.log_store(&id_for_wal, auto_id, idx, &emb_for_wal)?
            } else {
                // The common apply path below owns the only in-memory mutation.
                LoggedStore {
                    _lsn: 0,
                    already_logged: false,
                }
            };
            if !logged.already_logged || store.n_active() == idx {
                if !store.add_splat(&arr) {
                    if let Some(d) = durability.as_ref() {
                        return Err(d.failure(
                            &id_for_wal,
                            "memory apply failed after WAL acceptance",
                            true,
                            true,
                        ));
                    }
                    return Err(StoreWorkerError {
                        status: StatusCode::INSUFFICIENT_STORAGE,
                        message: "Store full".into(),
                        response: None,
                    });
                }
                store.hnsw_sync_incremental();
            }
            if let Some(d) = durability.as_ref() {
                if d.stores_since_compact() > 0
                    && d.stores_since_compact() % Durability::COMPACT_EVERY == 0
                {
                    match d.compact(&store) {
                        Ok(n) => {
                            eprintln!("[splatsdb] compact: shard 'default' persistido ({n} splats)")
                        }
                        Err(e) => eprintln!("[splatsdb] compact falló (el WAL continúa): {e}"),
                    }
                }
            }
            Ok(logged)
        })
        .await
        .map_err(|e| {
            eprintln!("[splatsdb] /store: background task failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
        })?;
    let mut response = match wal_outcome {
        Ok(_) => match state.durability.as_ref() {
            Some(d) => d.success(&id),
            None => StoreResponse {
                embedding_source: None,
                error: None,
                id: id.clone(),
                status: "stored".into(),
                outcome: "stored".into(),
                may_be_persisted: false,
                durable: false,
                crash_durable: false,
                power_loss_durable: false,
                wal_pending: 0,
                durability_mode: "ephemeral".into(),
                persistence: "ephemeral".into(),
            },
        },
        Err(e) => {
            eprintln!("[splatsdb] /store: {}", e.message);
            let body = e
                .response
                .map(|r| serde_json::to_value(r).expect("serializable store response"))
                .unwrap_or_else(|| serde_json::json!({ "error": e.message }));
            return Err((e.status, Json(body)));
        }
    };
    if embedded_here && response.status == "stored" {
        response.embedding_source = Some("real");
    }
    Ok((
        StatusCode::OK,
        Json(serde_json::to_value(response).expect("serializable store response")),
    ))
}

async fn search_memories(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    // Input validation
    if req.query.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Query exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }

    let (dim, n_active, data_version) = {
        let store = state.store.lock().await;
        let dim = store.get_statistics().embedding_dim;
        let n = store.n_active();
        (dim, n, store.data_version())
    }; // lock dropped — the hot path below never re-acquires it
    #[cfg(not(feature = "cuda"))]
    let _ = data_version;

    // Validate user-supplied embedding dimension BEFORE the empty-store
    // early-return so malformed requests (dim, max-dim, NaN/inf) are always
    // rejected with 400.
    if let Some(ref emb) = req.embedding {
        if emb.len() != dim {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Embedding dimension mismatch: expected {dim}, got {}",
                    emb.len()
                ),
            ));
        }
        if emb.len() > MAX_EMBEDDING_DIM {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Embedding dimension exceeds max ({MAX_EMBEDDING_DIM})"),
            ));
        }
        // NaN/inf (e.g. `1e400` parsed as inf) would poison every later
        // similarity computation over the index.
        if !all_finite(emb) {
            return Err((
                StatusCode::BAD_REQUEST,
                "Embedding contains non-finite values (NaN/inf)".into(),
            ));
        }
    }

    if n_active == 0 {
        return Ok(Json(SearchResponse {
            results: vec![],
            backend: None,
        }));
    }

    let k_req = req.top_k.unwrap_or(10).min(n_active).min(MAX_TOP_K);
    if k_req == 0 {
        // top_k=0 explícito → cero resultados (antes la coerción .max(1)
        // devolvía 1 resultado sin señal en la respuesta)
        return Ok(Json(SearchResponse {
            results: vec![],
            backend: None,
        }));
    }
    let k = k_req;

    // (14-nov) embedding obligatorio en /search: el fallback text-hash
    // devolvía L2 contra ruido determinista con 200 OK — un bug de cliente
    // se veía como "no hay memoria relevante" sin error. Los callers
    // internos (hook, sem-search, hermes_recall) siempre lo envían.
    let embedding = match req.embedding {
        Some(e) => e,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                "missing embedding: POST the query vector (dims 768) — the text-hash fallback produced noise ranking and was removed".into(),
            ));
        }
    };

    let rescore_gaussian = matches!(params.rescore.as_deref(), Some("gaussian"));

    // ── GPU exact fast path (serve --gpu): kernel PTX fused l2+topk.
    // Exacto, sin store mutex (índice compartido &self). El rescore gaussiano
    // sigue por el camino CPU (una query sin embedding ya salió con 400).
    //
    // A03-F05: el índice es una foto de UNA versión del dataset, así que sólo
    // puede responder mientras el store siga siendo EXACTAMENTE el dataset que
    // subió (`GpuIndexKey::is_current` sobre la `data_version` que esta
    // request acaba de leer). Antes de este guard, toda búsqueda posterior al
    // primer /store se respondía desde el dataset congelado del arranque.
    //
    // F06a: cuando la clave no calza (un /store, replay o compact movió la
    // versión), el índice se reconstruye UNA vez por versión de dataset —las
    // mutaciones continuas coalescen— y la query vuelve a la ruta GPU. Antes
    // la clave quedaba vieja para siempre y cada query posterior caía al
    // batch worker (backend null, ventana de 2 ms: p50 after-mutation 2× la
    // base). Ante un fallo explícito de GPU (sin CUDA / upload fallido) la
    // ruta rápida se desactiva y la query va al batch worker.
    #[cfg(feature = "cuda")]
    if !rescore_gaussian {
        let fast = match state.gpu.as_ref() {
            // Hit: the resident index is exactly the dataset this request
            // read — no lock on the store, no upload.
            Some(gpu) => match gpu.current_index(data_version, n_active) {
                Some(fast) => Some(fast),
                None => {
                    // Miss: rebuild off the runtime thread; concurrent stale
                    // requests coalesce on the single rebuild.
                    let (gpu, store) = (gpu.clone(), state.store.clone());
                    tokio::task::spawn_blocking(move || {
                        gpu.index_for_request(&store, data_version, n_active)
                    })
                    .await
                    .unwrap_or_else(|e| {
                        eprintln!("[splatsdb] /search: GPU rebuild task failed: {e}");
                        None
                    })
                }
            },
            None => None,
        };
        if let Some(fast) = fast {
            if let Some(mut results) = gpu_fast_path_results(&fast, &embedding, k) {
                enrich_topics(&state, &mut results, Some(true));
                return Ok(Json(SearchResponse {
                    results,
                    backend: Some("gpu".into()),
                }));
            }
        }
    }

    // ── Micro-batched path (R9 §2): enqueue and wait on a private oneshot.
    // The store mutex is taken only by the batch worker, once per batch.
    let (tx, rx) = oneshot::channel();
    state
        .batch_tx
        .send(BatchRequest {
            query: embedding,
            k,
            rescore_gaussian,
            resp: tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "search worker unavailable".into(),
            )
        })?;

    let mut results = rx
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "search worker dropped request".into(),
            )
        })?
        .map_err(|e| {
            // Log the real cause server-side; the client gets a generic 500
            // (the internal String may carry panic payloads from JoinError).
            eprintln!("[splatsdb] /search: batch worker error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "internal error".into())
        })?;
    // scores del batch son d² en AMBOS caminos (cpu_fused_batch desde
    // 87d5d91, gpu_batch_path desde el fix 17-nov): squared=true para que
    // confidence use d=√score. Con false la confidence quedaba inflada
    // (1−d²/R en vez de 1−√d²/R) en el batch (17-nov EGM audit).
    enrich_topics(&state, &mut results, (!rescore_gaussian).then_some(true));

    Ok(Json(SearchResponse {
        results,
        backend: None,
    }))
}

/// Enriquece resultados con tópico (splat) y confianza espacial
/// (1 − d/R del tópico). Solo llamado con scores de distancia L2 reales
/// (no con texto-hash). `squared`: `Some(true)` cuando el score es d² (filas
/// L2 de cualquier ruta); `None` para `rescore=gaussian`, cuyo score es
/// alpha·exp(−kappa·d²) y no es una distancia — esas filas llevan tópico y
/// ninguna confianza.
///
/// Los ids de `/search` son SIEMPRE ids de splat —el espacio de la referencia
/// CPU (`find_neighbors_fused`), del batch worker y de `/store`—, nunca ids
/// de chunk de `field.raw` (ese es el contrato de `/field-search`). Antes el
/// espacio se infería de `corpus_n`, que con field/ era siempre el del corpus
/// de chunks: un id de splat se buscaba en `chunk_cluster` como si fuera un
/// chunk y el tópico (y el radio de la confianza) eran los de OTRO splat.
fn enrich_topics(state: &AppState, results: &mut [SearchResult], squared: Option<bool>) {
    let Some(field) = &state.field else {
        return;
    };
    for r in results.iter_mut() {
        let radius = field.radii.get(r.index).copied().unwrap_or(0.0);
        r.topic = Some(r.index as u32);
        let Some(squared) = squared else {
            continue;
        };
        if radius > 0.0 {
            let dist = if squared {
                r.score.max(0.0).sqrt()
            } else {
                r.score.max(0.0)
            };
            let conf = (1.0 - dist / radius).clamp(0.0, 1.0);
            r.confidence = Some((conf * 1000.0).round() / 1000.0);
        }
    }
}

// ─── Campo splat (celdas con radio): /field-search exacto ────────────────
// Geometría correcta del campo (veredicto bench-latent): la energía
// alpha·exp(−kappa·d²) NO es una cota y destruye el ranking en 768d; la
// desigualdad triangular d(q,x) ≥ d(q,mu_c) − R_c SÍ lo es → sweep exacto.

/// Crudos + splats + membresías CSR + radios. El field se persiste vía el
/// shard "raw" del data-dir (ver `load_field`); el store del `/store` se
/// persiste por chunks con `Durability` (v2.11.0).
pub struct FieldData {
    pub raw: Array2<f32>,
    pub splats: Array2<f32>,
    pub offsets: Vec<u64>,
    pub ids: Vec<u32>,
    pub radii: Vec<f32>,
    /// Inverso del CSR: chunk index → splat (tópico) al que pertenece.
    pub chunk_cluster: Vec<u32>,
}

/// Sanity caps for field/ headers (parsed straight from file bytes, so a
/// corrupt header must not OOM the server on boot).
const MAX_FIELD_N: usize = 100_000_000;
const MAX_FIELD_K: usize = 100_000_000;
const MAX_FIELD_RADII: usize = 100_000_000;

fn load_field(data_dir: &str) -> Option<FieldData> {
    let dir = std::path::Path::new(data_dir).join("field");
    if !dir.join("members.idx").exists() || !dir.join("radii.bin").exists() {
        return None;
    }
    let (n, k, offsets, ids) = read_members_idx(&dir)?;

    let radii = read_radii_bin(&dir)?;

    let persist = crate::storage::persistence::SplatsDBPersistence::with_backend(
        data_dir,
        crate::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .ok()?;
    let raw = persist.load_vectors("raw").ok()??;
    let splats = persist.load_vectors("default").ok()??;
    if raw.nrows() != n || splats.nrows() != k || radii.len() != k {
        eprintln!("[splatsdb] field/ desincronizado; /field-search deshabilitado");
        return None;
    }
    let mut chunk_cluster = vec![0u32; n];
    for c in 0..k {
        let (lo, hi) = (offsets[c] as usize, offsets[c + 1] as usize);
        for &id in &ids[lo..hi] {
            chunk_cluster[id as usize] = c as u32;
        }
    }
    Some(FieldData {
        raw,
        splats,
        offsets,
        radii,
        chunk_cluster,
        ids,
    })
}

/// Parse `field/members.idx`: 16-byte header (n, k) + (k+1)×8 CSR offsets +
/// n×4 member ids, with sanity caps, a size-vs-header truncation check, CSR
/// monotonicity (bounds every later `ids[lo..hi]` slice in
/// field_sweep/field_two_phase) and member-id range validation.
///
/// Extracted from `load_field`; messages and behavior identical.
fn read_members_idx(dir: &std::path::Path) -> Option<(usize, usize, Vec<u64>, Vec<u32>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(dir.join("members.idx")).ok()?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).ok()?;
    let n = u64::from_le_bytes(hdr[..8].try_into().ok()?) as usize;
    let k = u64::from_le_bytes(hdr[8..].try_into().ok()?) as usize;
    // Sanity caps BEFORE any allocation: n*4 / (k+1)*8 bytes are reserved
    // straight from the file header, so a corrupt header must not OOM the
    // server on boot.
    if n > MAX_FIELD_N || k > MAX_FIELD_K {
        eprintln!(
            "[splatsdb] field/ header out of range (n={n}, k={k}); /field-search deshabilitado"
        );
        return None;
    }
    // Header-driven allocations are bounded by the caps above, but a header
    // near the caps would still transiently allocate ~(k+1)*8 + n*4 ≈ 1.2GB
    // before read_exact fails on a truncated file. The file must actually
    // CONTAIN those bytes — check metadata first (DoS via a corrupt boot file).
    let members_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let members_need = 16u64 + (k as u64 + 1) * 8 + n as u64 * 4;
    if members_len < members_need {
        eprintln!(
            "[splatsdb] field/ members.idx truncado ({} bytes, header pide {members_need}); \
             /field-search deshabilitado",
            members_len
        );
        return None;
    }
    let mut ob = vec![0u8; (k + 1) * 8];
    f.read_exact(&mut ob).ok()?;
    let mut offsets = vec![0u64; k + 1];
    for (i, c) in ob.as_chunks::<8>().0.iter().enumerate() {
        offsets[i] = u64::from_le_bytes(*c);
    }
    // CSR sanity: monotone non-decreasing offsets ending exactly at n. This
    // also bounds every later `ids[lo..hi]` slice (field_sweep/field_two_phase).
    if offsets[k] != n as u64 || offsets.windows(2).any(|w| w[0] > w[1]) {
        eprintln!("[splatsdb] field/ CSR offsets inválidos; /field-search deshabilitado");
        return None;
    }
    let mut ib = vec![0u8; n * 4];
    f.read_exact(&mut ib).ok()?;
    let mut ids = vec![0u32; n];
    for (i, c) in ib.as_chunks::<4>().0.iter().enumerate() {
        ids[i] = u32::from_le_bytes(*c);
    }
    // Every membership id must index into the raw corpus: chunk_cluster[id]
    // and field_sweep's raw.row(id) depend on it (raw.nrows() == n is checked
    // below against the persisted vectors).
    if ids.iter().any(|&id| id as usize >= n) {
        eprintln!("[splatsdb] field/ member id fuera de rango; /field-search deshabilitado");
        return None;
    }
    Some((n, k, offsets, ids))
}

/// Parse `field/radii.bin`: 8-byte header (kr) + kr×4 f32 radii, with the
/// same sanity-cap and truncation checks as members.idx.
///
/// Extracted from `load_field`; messages and behavior identical.
fn read_radii_bin(dir: &std::path::Path) -> Option<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(dir.join("radii.bin")).ok()?;
    let mut k8 = [0u8; 8];
    f.read_exact(&mut k8).ok()?;
    let kr = u64::from_le_bytes(k8) as usize;
    // Same sanity cap as n/k: kr*4 bytes are allocated straight from the
    // radii.bin header.
    if kr > MAX_FIELD_RADII {
        eprintln!(
            "[splatsdb] field/ radii header out of range (kr={kr}); /field-search deshabilitado"
        );
        return None;
    }
    let radii_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if radii_len < 8u64 + kr as u64 * 4 {
        eprintln!(
            "[splatsdb] field/ radii.bin truncado ({} bytes, header pide {}); \
             /field-search deshabilitado",
            radii_len,
            8 + kr * 4
        );
        return None;
    }
    let mut rb = vec![0u8; kr * 4];
    f.read_exact(&mut rb).ok()?;
    let mut radii = vec![0f32; kr];
    for (i, c) in rb.as_chunks::<4>().0.iter().enumerate() {
        radii[i] = f32::from_le_bytes(*c);
    }
    Some(radii)
}

fn field_sweep(field: &FieldData, q: &[f32], kq: usize) -> serde_json::Value {
    let t0 = std::time::Instant::now();
    let qv = ndarray::arr1(q);
    let k = field.splats.nrows();
    let n = field.raw.nrows();
    let d_mu: ndarray::Array1<f32> = field.splats.map_axis(ndarray::Axis(1), |r| {
        let d = r.to_owned() - &qv;
        d.dot(&d).max(0.0).sqrt()
    });
    // cota por cluster: un cluster entero es podable si (d(q,mu)−R)² > kth
    let lb: Vec<f32> = (0..k)
        .map(|c| (d_mu[c] - field.radii[c]).max(0.0))
        .collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| d_mu[*a].total_cmp(&d_mu[*b]));

    let mut top: Vec<(f32, u32)> = Vec::with_capacity(kq + 8);
    let mut kth = f32::INFINITY;
    // kth SOLO acota cuando el top-k está completo (cota inválida con top parcial)
    let mut visited = 0usize;
    let mut scanned = 0usize;
    for &c in &order {
        let lb_c = lb[c] * lb[c];
        if !(lb_c < kth || top.len() < kq) {
            continue;
        }
        let (lo, hi) = (field.offsets[c] as usize, field.offsets[c + 1] as usize);
        if hi == lo {
            continue;
        }
        visited += 1;
        scanned += hi - lo;
        for &id in &field.ids[lo..hi] {
            let row = field.raw.row(id as usize);
            let diff = row.to_owned() - &qv;
            let d2 = diff.dot(&diff);
            if top.len() < kq || d2 < kth {
                top.push((d2, id));
                top.sort_by(|a, b| a.0.total_cmp(&b.0));
                if top.len() > kq {
                    top.truncate(kq);
                }
                kth = if top.len() >= kq {
                    top[top.len() - 1].0
                } else {
                    f32::INFINITY
                };
            }
        }
    }
    let pct = |x: f64| (x * 10000.0).round() / 10000.0;
    serde_json::json!({
        "backend": "field_exact_bounds",
        "k": kq,
        "n_vectors": n,
        "n_splats": k,
        "clusters_visited": visited,
        "cluster_frac": pct(visited as f64 / k as f64),
        "vectors_scanned": scanned,
        "scan_frac": pct(scanned as f64 / n as f64),
        "time_ms": t0.elapsed().as_millis() as u64,
        "results": top.iter()
            .map(|(d, i)| serde_json::json!({"index": *i as usize, "score": *d}))
            .collect::<Vec<_>>(),
    })
}

/// Two-phase IVF (aproximado): fase 1 = top-`nprobe` clusters por d(q, mu);
/// fase 2 = re-rank exacto L2² (rayon) de los miembros vía CSR. El recall
/// depende de nprobe — el exacto es field_sweep. Fase 1 CPU: a K moderado es
/// sub-ms y evita un segundo índice en VRAM (el del server es sobre los crudos).
fn field_two_phase(
    field: &FieldData,
    q: &[f32],
    kq: usize,
    nprobe_req: usize,
) -> serde_json::Value {
    use rayon::prelude::*;
    let t0 = std::time::Instant::now();
    let qv = ndarray::arr1(q);
    let k = field.splats.nrows();
    let n = field.raw.nrows();
    let nprobe = nprobe_req.min(k).max(1);

    let t1 = std::time::Instant::now();
    let d_mu: ndarray::Array1<f32> = field.splats.map_axis(ndarray::Axis(1), |r| {
        let d = r.to_owned() - &qv;
        d.dot(&d).max(0.0).sqrt()
    });
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| d_mu[*a].total_cmp(&d_mu[*b]));
    let winners = &order[..nprobe];
    let phase1_us = t1.elapsed().as_micros() as u64;

    let t2 = std::time::Instant::now();
    let mut cand: Vec<u32> = Vec::new();
    let mut scanned = 0usize;
    for &c in winners {
        let (lo, hi) = (field.offsets[c] as usize, field.offsets[c + 1] as usize);
        scanned += hi - lo;
        cand.extend_from_slice(&field.ids[lo..hi]);
    }
    let mut top: Vec<(f32, u32)> = cand
        .par_iter()
        .map(|&id| {
            let row = field.raw.row(id as usize);
            let diff = row.to_owned() - &qv;
            (diff.dot(&diff), id)
        })
        .collect();
    top.sort_by(|a, b| a.0.total_cmp(&b.0));
    top.truncate(kq);
    let phase2_us = t2.elapsed().as_micros() as u64;

    let pct = |x: f64| (x * 10000.0).round() / 10000.0;
    serde_json::json!({
        "backend": "field_two_phase_ivf",
        "k": kq,
        "n_vectors": n,
        "n_splats": k,
        "nprobe": nprobe,
        "clusters_probed": winners.len(),
        "vectors_scanned": scanned,
        "scan_frac": pct(scanned as f64 / n as f64),
        "phase1_us": phase1_us,
        "phase2_us": phase2_us,
        "time_ms": t0.elapsed().as_millis() as u64,
        "results": top.iter()
            .map(|(d, i)| serde_json::json!({"index": *i as usize, "score": *d}))
            .collect::<Vec<_>>(),
    })
}

#[derive(Deserialize)]
pub struct FieldSearchRequest {
    pub embedding: Vec<f32>,
    pub top_k: Option<usize>,
    /// "two-phase" = IVF aproximado (recall depende de nprobe);
    /// default = exacto por cotas
    pub mode: Option<String>,
    /// Clusters a sondear en modo two-phase (default 32)
    pub nprobe: Option<usize>,
}

async fn field_search_handler(
    State(state): State<AppState>,
    Json(req): Json<FieldSearchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let Some(field) = &state.field else {
        return Err((
            StatusCode::NOT_FOUND,
            "no hay field/ en este data-dir (correr ingest con splatdb >= 2.6.1)".into(),
        ));
    };
    if req.embedding.len() != field.splats.ncols() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "dim esperada {}, llegada {}",
                field.splats.ncols(),
                req.embedding.len()
            ),
        ));
    }
    // NaN/inf (e.g. `1e400` parsed as inf) breaks the bound-sweep pruning
    // (all lower bounds compare NaN → false) and poisons the ranking.
    if !all_finite(&req.embedding) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Embedding contains non-finite values (NaN/inf)".into(),
        ));
    }
    let k = req.top_k.unwrap_or(10).clamp(1, 1000);
    // field_sweep/field_two_phase are CPU-bound O(n·d) scans; running them
    // inline on the async runtime freezes ALL endpoints (incl. /health) under
    // a few concurrent requests. Push them onto the blocking pool (FieldData
    // travels as Arc, no locks involved).
    let field = Arc::clone(field);
    let embedding = req.embedding;
    let mode_two_phase = req.mode.as_deref() == Some("two-phase");
    let nprobe = mode_two_phase.then(|| req.nprobe.unwrap_or(32).clamp(1, field.splats.nrows()));
    let result = tokio::task::spawn_blocking(move || {
        if let Some(nprobe) = nprobe {
            field_two_phase(&field, &embedding, k, nprobe)
        } else {
            field_sweep(&field, &embedding, k)
        }
    })
    .await;
    match result {
        Ok(v) => Ok(Json(v)),
        Err(e) => {
            eprintln!("[splatsdb] field-search task panicked: {e}");
            Err((StatusCode::INTERNAL_SERVER_ERROR, "internal error".into()))
        }
    }
}

/// POST /recap — SAC-style handoff recap, LLM-free (see `recap` module docs).
///
/// Stateless: the action log comes inline in the request; the store is never
/// touched (no mutex, no persistence). Deterministic output.
async fn recap_handler(
    Json(req): Json<RecapRequest>,
) -> Result<Json<RecapResponse>, (StatusCode, String)> {
    // Input size guards (mirror /store's limits, applied per action + total).
    let mut total = 0usize;
    for a in &req.actions {
        if a.text.len() > MAX_ACTION_TEXT_LEN {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Action text exceeds max length ({MAX_ACTION_TEXT_LEN})"),
            ));
        }
        total += a.text.len();
        if total > MAX_TOTAL_ACTIONS_LEN {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Total action payload exceeds max length ({MAX_TOTAL_ACTIONS_LEN})"),
            ));
        }
    }
    if req.objective.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Objective exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }
    Ok(Json(build_recap(&req)))
}

// ═══════════════════════════════════════════════════════════════════════
// /recall — native Hermes-memory recall endpoint (20-nov-2026)
//
// POST /recall {text, session_id?, top_k?} → the engine performs the FULL
// recall pipeline that used to live in the Hermes Python hook:
//   1. Embed the query via the OpenAI-style embed service
//      (SPLATSDB_RECALL_EMBED_URL, default http://127.0.0.1:8860/v1/embeddings,
//      SPLATSDB_RECALL_EMBED_PROMPT, default "Retrieval-query" — the
//      asymmetric prompt is load-bearing for distance calibration; do NOT
//      "fix" it away).
//   2. Run the internal /search path by calling `search_memories` directly —
//      zero contract drift: validation, GPU fast path and batching shared.
//   3. Join {data_dir}/sidecar.db (mode=ro, rowid=index+1) for kind/sid/text;
//      hits with no sidecar row are dropped (never invent text).
//   4. Apply the hook's calibrated tiers: sessions (kind != repo) d ≤ 0.55 →
//      up to 3 hits, code (repo) d ≤ 0.70 → 1 hit, JSON-leading pre-filter
//      applied BEFORE the tier split (17-nov contract). Env-overridable:
//      SPLATSDB_RECALL_SESS_STRONG/_N, SPLATSDB_RECALL_CODE_STRONG/_N.
//   5. Fail-closed on the hook's guard: backend != "gpu" (CPU fallback)
//      injects nothing; missing embed service → 503, never SimCos
//      (fa1e528 parity — a hash-text embedding ranks noise as success).
//   6. Append the fire to SPLATSDB_RECALL_LOG (default
//      {data_dir}/recalls.jsonl) with the hook's JSONL schema
//      {ts, session_id, qhash, qlen, n_injected, arm, backend, drops, hits}
//      so sac_recall_digest.py keeps working unchanged. qhash is
//      sha1(text)[:12] — bit-exact with the hook's formula.
//
// NOT covered here (stays in the Hermes shim): the FTS5 fallback over
// Hermes state.db — that database belongs to the agent, not the engine.
//
// Tests here cover the pure logic (JSON filter, tier selection) and the
// fail-closed paths with a dead embed URL. The happy path is validated
// E2E against the live hubs by hermes_smoke.py + the A/B replay.
// ═══════════════════════════════════════════════════════════════════════

const RECALL_TOPK_DEFAULT: usize = 8;
const RECALL_MAX_TOPK: usize = 32;
const RECALL_SESS_STRONG_DEFAULT: f32 = 0.55;
/// 3 → 5 (21-nov LongMemEval-S: recall@contexto 53.2%→63.1%, multi-session
/// +13.4 pts, preference +23.4 pts; w5 vs base3 con answer_session_ids GT).
/// Rollback instantáneo por env: SPLATSDB_RECALL_SESS_N=3.
const RECALL_SESS_N_DEFAULT: usize = 5;
const RECALL_CODE_STRONG_DEFAULT: f32 = 0.70;
const RECALL_CODE_N_DEFAULT: usize = 1;
const RECALL_EMBED_URL_DEFAULT: &str = "http://127.0.0.1:8860/v1/embeddings";
const RECALL_EMBED_PROMPT_DEFAULT: &str = "Retrieval-query";
const RECALL_EMBED_MODEL: &str = "embeddinggemma-2";

/// One sidecar-joined hit (or a client-supplied FTS hit riding in the same
/// bucket). repo/path are the chunks-table columns — only code (kind=repo)
/// hits carry them; clients format "[código repo/path]" headers from them.
struct JoinedHit {
    kind: String,
    repo: String,
    path: String,
    sid: String,
    text: String,
    dist: f32,
    meta: serde_json::Value,
}

fn recall_env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn recall_env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn recall_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Deserialize)]
pub struct RecallFtsHit {
    /// Pseudo-distance (hook FTS literal = 0.40, head of the bucket).
    pub dist: f32,
    pub sid: String,
    pub text: String,
    /// Client-built meta ({session_id, role, date}) — passed through.
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[derive(Deserialize)]
pub struct RecallRequest {
    pub text: String,
    pub session_id: Option<String>,
    pub top_k: Option<usize>,
    /// FTS5 literal hits the client fetched from Hermes state.db. They are
    /// merged into the session bucket BEFORE the gate so there is exactly
    /// ONE gate implementation (this endpoint): the hook gated over the
    /// FTS+vector union, and the shim must not re-implement the gate in
    /// Python (that duplication is what this endpoint removes). Empty on
    /// engines without state.db access — the gate then runs vector-only.
    #[serde(default)]
    pub fts: Vec<RecallFtsHit>,
}

#[derive(Serialize)]
pub struct RecallHit {
    pub sid: String,
    pub kind: String,
    pub dist: f32,
    pub text: String,
    /// Chunks-table columns (code hits only): "[código repo/path]" headers.
    #[serde(default)]
    pub repo: String,
    #[serde(default)]
    pub path: String,
    /// Parsed sidecar meta (session_id/date/title/role) — clients build
    /// their "[sesión ...]" headers from it.
    pub meta: serde_json::Value,
}

#[derive(Serialize)]
pub struct RecallResponse {
    pub backend: Option<String>,
    /// 0 = injected, -1 = cut (no tier hit / fail-closed) — hook semantics.
    pub arm: i8,
    pub results: Vec<RecallHit>,
    /// Hits dropped by the JSON-leading pre-filter (sess, code) — raw counts
    /// BEFORE the tier split, like the hook's `drops`.
    pub drops: (usize, usize),
    pub sidecar: bool,
    /// Reason when the hook's fail-closed guard tripped (backend != gpu).
    pub fail_closed: Option<String>,
}

/// JSON-leading text is unreadable thin content (blob dumps): the hook
/// drops it BEFORE applying tiers (17-nov contract) — lstrip first, then
/// first-char test, exactly like the Python `lstrip` + `in {'{','[','"'}`.
fn is_json_leading(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with('{') || t.starts_with('[') || t.starts_with('"')
}

/// sha1(text) hex, first 12 chars — bit-exact with the hook's qhash
/// (`hashlib.sha1(msg.encode()).hexdigest()[:12]`).
fn recall_qhash(text: &str) -> String {
    use sha1::{Digest, Sha1};
    let digest = Sha1::digest(text.as_bytes());
    let mut out = String::with_capacity(40);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out.chars().take(12).collect()
}

/// Embedding of a QUERY for `/recall`: OpenAI-style request to the Hermes
/// embed service (:8860) with the asymmetric Retrieval-query prompt. The
/// vector is NOT re-normalized: the hub's d² bands were calibrated against
/// this service's raw output.
fn fetch_recall_embedding(text: &str, dim: usize) -> Result<Vec<f32>, String> {
    embed_via_service(
        serde_json::json!({
            "input": [text],
            "model": RECALL_EMBED_MODEL,
            "prompt_name": recall_env_str("SPLATSDB_RECALL_EMBED_PROMPT", RECALL_EMBED_PROMPT_DEFAULT),
            "dimensions": dim,
        }),
        dim,
    )
}

/// Embedding of a DOCUMENT for `/store`: same service and model as the
/// queries, and NO prompt — the pipeline embeds documents without one
/// (docs/hermes-memory.md, "Asimetría de prompts"). Both paths read the same
/// `SPLATSDB_RECALL_EMBED_URL`, so documents and queries cannot be pointed
/// at two different services by mistake.
fn fetch_store_embedding(text: &str, dim: usize) -> Result<Vec<f32>, String> {
    embed_via_service(
        serde_json::json!({
            "input": [text],
            "model": RECALL_EMBED_MODEL,
            "dimensions": dim,
        }),
        dim,
    )
}

/// POST one OpenAI-style embed request to the embed service and return
/// `data[0].embedding`. `dim` is the dimension the request asked for; an
/// answer of any other length is an error. Errors name the service; there is
/// no fallback.
fn embed_via_service(request: serde_json::Value, dim: usize) -> Result<Vec<f32>, String> {
    let url = recall_env_str("SPLATSDB_RECALL_EMBED_URL", RECALL_EMBED_URL_DEFAULT);
    let body = request.to_string();
    let response = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(5)))
        .build()
        .new_agent()
        .post(&url)
        .header("Content-Type", "application/json")
        .send(&body)
        .map_err(|e| format!("embed service: {e}"))?;
    let resp: serde_json::Value = response
        .into_body()
        .read_json()
        .map_err(|e| format!("embed json: {e}"))?;
    let arr = resp
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|o| o.get("embedding"))
        .and_then(|e| e.as_array())
        .ok_or_else(|| "embed response missing data[0].embedding".to_string())?;
    let vec: Vec<f32> = arr
        .iter()
        .filter_map(|v: &serde_json::Value| v.as_f64().map(|f| f as f32))
        .collect();
    if vec.len() != dim {
        return Err(format!(
            "embed dim mismatch: expected {dim}, got {}",
            vec.len()
        ));
    }
    if vec.iter().any(|v| !v.is_finite()) {
        return Err("embed contains non-finite values".into());
    }
    Ok(vec)
}

/// Join search indices against the engine's sidecar (chunks.rowid =
/// index+1, read-only). Returns None when sidecar.db is absent/unreadable
/// (deploy without sidecar: search still works, recall carries no text).
/// Hits with no sidecar row are dropped — never invent text.
fn sidecar_join(data_dir: &str, results: &[SearchResult]) -> Option<Vec<JoinedHit>> {
    let path = std::path::Path::new(data_dir).join("sidecar.db");
    let conn = rusqlite::Connection::open_with_flags(
        &path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        let row = conn
            .query_row(
                "SELECT kind, repo, path, meta, text FROM chunks WHERE rowid = ?1",
                rusqlite::params![r.index as i64 + 1],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .ok();
        let Some((kind, repo, path, meta, text)) = row else {
            continue;
        };
        let meta_val = serde_json::from_str::<serde_json::Value>(&meta)
            .ok()
            .unwrap_or(serde_json::Value::Null);
        let sid = meta_val
            .get("session_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        out.push(JoinedHit {
            kind,
            repo,
            path,
            sid,
            text,
            dist: r.score,
            meta: meta_val,
        });
    }
    Some(out)
}

/// Tier selection, hook-equivalent: buckets by kind (repo → code; anything
/// else → session), JSON-leading dropped BEFORE the split, FTS pseudo-hits
/// already merged into the session bucket by the caller, best-first by
/// distance, sessions first in the output.
///
/// GATE SEMANTICS (14-nov calibration, hook byte-code): the threshold
/// applies to the BUCKET'S BEST hit only; ranks 2..n are taken
/// UNCONDITIONALLY once the gate passes. Per-hit thresholds would inject
/// strictly fewer hits than the deployed hook (measured A/B 20-nov:
/// 4/78 queries diverged, e.g. pool 0.547/0.551/0.565 → hook injects 3,
/// per-hit gating injects 1). Returns (hits, sess_json_drops,
/// code_json_drops).
fn select_recall_tiers(
    hits: Vec<JoinedHit>,
    sess_strong: f32,
    sess_n: usize,
    code_strong: f32,
    code_n: usize,
) -> (Vec<RecallHit>, usize, usize) {
    let mut sess: Vec<&JoinedHit> = hits
        .iter()
        .filter(|h| !h.kind.starts_with("repo"))
        .collect();
    let mut code: Vec<&JoinedHit> = hits.iter().filter(|h| h.kind.starts_with("repo")).collect();
    sess.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    code.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    let sess_json = sess.iter().filter(|h| is_json_leading(&h.text)).count();
    let code_json = code.iter().filter(|h| is_json_leading(&h.text)).count();
    // anti-JSON pre-gate BEFORE the gate itself (hook order: filter, then
    // gate on the best surviving hit)
    let sess_ok: Vec<&JoinedHit> = sess
        .into_iter()
        .filter(|h| !is_json_leading(&h.text))
        .collect();
    let code_ok: Vec<&JoinedHit> = code
        .into_iter()
        .filter(|h| !is_json_leading(&h.text))
        .collect();
    let mut results: Vec<RecallHit> = Vec::new();
    if sess_ok
        .first()
        .is_some_and(|first| first.dist <= sess_strong)
    {
        results.extend(sess_ok.iter().take(sess_n).map(|h| RecallHit {
            sid: h.sid.clone(),
            kind: h.kind.clone(),
            dist: h.dist,
            text: h.text.clone(),
            repo: h.repo.clone(),
            path: h.path.clone(),
            meta: h.meta.clone(),
        }));
    }
    if code_ok
        .first()
        .is_some_and(|first| first.dist <= code_strong)
    {
        results.extend(code_ok.iter().take(code_n).map(|h| RecallHit {
            sid: h.sid.clone(),
            kind: h.kind.clone(),
            dist: h.dist,
            text: h.text.clone(),
            repo: h.repo.clone(),
            path: h.path.clone(),
            meta: h.meta.clone(),
        }));
    }
    (results, sess_json, code_json)
}

fn recall_append_log(state: &AppState, fire: &serde_json::Value) {
    let Some(dir) = &state.data_dir else { return };
    let path = match std::env::var("SPLATSDB_RECALL_LOG") {
        Ok(s) if s.is_empty() => return, // explícitamente deshabilitado
        Ok(s) => s,
        Err(_) => std::path::Path::new(dir)
            .join("recalls.jsonl")
            .display()
            .to_string(),
    };
    let line = format!("{fire}\n");
    let write = || -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        f.write_all(line.as_bytes())
    };
    if let Err(e) = write() {
        // Logging is fail-open: a broken log file must not kill the recall.
        eprintln!("[splatsdb] /recall: log append failed: {e}");
    }
}

async fn recall_handler(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, (StatusCode, String)> {
    // ── input validation (before anything else; cheap paths for tests)
    if req.text.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "recall: empty text".into()));
    }
    if req.text.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Text exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }
    let top_k = req
        .top_k
        .unwrap_or_else(|| recall_env_usize("SPLATSDB_RECALL_TOPK", RECALL_TOPK_DEFAULT))
        .clamp(1, RECALL_MAX_TOPK);

    let (dim, n_active) = {
        let store = state.store.lock().await;
        (store.get_statistics().embedding_dim, store.n_active())
    };

    // Empty store: nothing to recall — arm=-1 (hook logs empty fires too),
    // no embed call, no fail-closed reason.
    if n_active == 0 {
        let resp = RecallResponse {
            backend: None,
            arm: -1,
            results: vec![],
            drops: (0, 0),
            sidecar: false,
            fail_closed: None,
        };
        recall_append_log(
            &state,
            &serde_json::json!({
                "ts": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
                "session_id": req.session_id,
                "qhash": recall_qhash(&req.text),
                "qlen": req.text.len(),
                "n_injected": 0,
                "arm": -1,
                "backend": serde_json::Value::Null,
                "drops": {"sess_json": 0, "code_json": 0},
                "hits": [],
            }),
        );
        return Ok(Json(resp));
    }

    // ── 1. Embed (fail-closed: never SimCos). ureq is blocking →
    // spawn_blocking keeps the async runtime free.
    let text = req.text.clone();
    let embedding = tokio::task::spawn_blocking(move || fetch_recall_embedding(&text, dim))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("recall: {e}")))?
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, format!("recall: {e}")))?;

    // ── 2. The internal /search path, called directly — one implementation,
    // zero contract drift (validation, GPU fast path, batching, enrich).
    let search = search_memories(
        State(state.clone()),
        Query(SearchQuery { rescore: None }),
        Json(SearchRequest {
            query: req.text.clone(),
            embedding: Some(embedding),
            top_k: Some(top_k),
        }),
    )
    .await?
    .0;

    // ── 3-4. Sidecar join + tiers + fail-closed guard (backend != gpu)
    let backend = search.backend.clone();
    let mut best_sess: Option<f32> = None;
    let mut resp = RecallResponse {
        backend: backend.clone(),
        arm: -1,
        results: vec![],
        drops: (0, 0),
        sidecar: false,
        fail_closed: None,
    };

    if backend.as_deref() != Some("gpu") {
        // Hook parity: CPU fallback changes the score scale — the 0.55/0.70
        // bands are calibrated on GPU d². Inject nothing, say why.
        resp.fail_closed = Some(format!(
            "backend != gpu ({:?}) — distance bands are GPU-calibrated; injecting nothing",
            backend
        ));
    } else if let Some(dir) = &state.data_dir {
        if let Some(joined) = sidecar_join(dir, &search.results) {
            resp.sidecar = true;
            // client FTS hits merge into the session bucket PRE-GATE (one
            // gate implementation lives here, not in the shim). Drop
            // malformed entries; cap the array (hook fetches ≤ 2).
            let mut joined: Vec<JoinedHit> = joined;
            joined.extend(
                req.fts
                    .iter()
                    .take(10)
                    .filter(|f| f.dist.is_finite() && !f.text.trim().is_empty())
                    .map(|f| JoinedHit {
                        kind: "fts".to_string(),
                        repo: String::new(),
                        path: String::new(),
                        sid: f.sid.clone(),
                        text: f.text.clone(),
                        dist: f.dist,
                        meta: f.meta.clone(),
                    }),
            );
            let best_sess_dist = joined
                .iter()
                .filter(|h| !h.kind.starts_with("repo"))
                .map(|h| h.dist)
                .fold(f32::INFINITY, f32::min);
            best_sess = if best_sess_dist.is_finite() {
                Some(best_sess_dist)
            } else {
                None
            };
            let sess_strong =
                recall_env_f32("SPLATSDB_RECALL_SESS_STRONG", RECALL_SESS_STRONG_DEFAULT);
            let sess_n = recall_env_usize("SPLATSDB_RECALL_SESS_N", RECALL_SESS_N_DEFAULT);
            let code_strong =
                recall_env_f32("SPLATSDB_RECALL_CODE_STRONG", RECALL_CODE_STRONG_DEFAULT);
            let code_n = recall_env_usize("SPLATSDB_RECALL_CODE_N", RECALL_CODE_N_DEFAULT);
            let (results, sess_json, code_json) =
                select_recall_tiers(joined, sess_strong, sess_n, code_strong, code_n);
            resp.results = results;
            resp.drops = (sess_json, code_json);
        }
    }
    if !resp.results.is_empty() {
        resp.arm = 0;
    }

    // ── 6. Fire log, hook schema (sac_recall_digest.py compatible)
    recall_append_log(
        &state,
        &serde_json::json!({
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            "session_id": req.session_id,
            "qhash": recall_qhash(&req.text),
            "qlen": req.text.len(),
            "n_injected": resp.results.len(),
            "arm": resp.arm,
            "backend": backend,
            "drops": {"sess_json": resp.drops.0, "code_json": resp.drops.1},
            "cut_dist": best_sess
                .filter(|_| resp.arm == -1)
                .map(|d| serde_json::json!((d * 1000.0).round() / 1000.0))
                .unwrap_or(serde_json::Value::Null),
            "hits": resp.results.iter()
                .map(|h| serde_json::json!({"sid": h.sid, "dist": h.dist, "kind": h.kind}))
                .collect::<Vec<_>>(),
        }),
    );

    Ok(Json(resp))
}

// ─── Batch worker (R9 §2) ─────────────────────────────────

/// Cached flat copy of the active dataset for the GPU batch path.
/// Refreshed only when the store's `data_version` changes, so the store
/// mutex is held just long enough to copy — never during the search itself.
struct DatasetSnapshot {
    /// Dataset version (`SplatStore::data_version` at snapshot time —
    /// globally unique across stores, bumps on every add/load/compact).
    /// Staleness check for this host-side cache AND key for the
    /// process-level VRAM dataset cache (`gpu::cached_dataset_index`).
    version: u64,
    n: usize,
    dim: usize,
    /// Row-major [n, dim] flat data.
    data: Vec<f32>,
    /// Per-splat amplitudes [n] (gaussian rescore input).
    alpha: Vec<f32>,
    /// Per-splat concentrations [n] (gaussian rescore input).
    kappa: Vec<f32>,
}

/// Coalescing worker: receives search requests, groups everything that
/// arrives within `BATCH_WINDOW_MS` (up to `MAX_BATCH`) and serves it as a
/// single batch — one lock acquisition and one (GPU) or a few (CPU/HNSW)
/// searches instead of one lock + scan per request.
async fn batch_worker(
    mut rx: mpsc::Receiver<BatchRequest>,
    store: Arc<Mutex<SplatStore>>,
    batches_served: Arc<AtomicUsize>,
    gpu_enabled: bool,
) {
    // Host-side dataset snapshot cache for the GPU path.
    let mut snapshot: Option<DatasetSnapshot> = None;

    while let Some(first) = rx.recv().await {
        let mut batch = Vec::with_capacity(MAX_BATCH.min(16));
        batch.push(first);

        // Coalescing window: drain until the deadline or MAX_BATCH.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(BATCH_WINDOW_MS);
        while batch.len() < MAX_BATCH {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(r)) => batch.push(r),
                Ok(None) | Err(_) => break, // channel closed or window elapsed
            }
        }

        batches_served.fetch_add(1, Ordering::Relaxed);
        serve_batch(batch, &store, &mut snapshot, gpu_enabled).await;
    }
}

/// Execute one coalesced batch and deliver each request's slice of the
/// results on its oneshot channel. Any request that arrives on a channel
/// that was already dropped is silently discarded.
async fn serve_batch(
    batch: Vec<BatchRequest>,
    store: &Arc<Mutex<SplatStore>>,
    snapshot: &mut Option<DatasetSnapshot>,
    gpu_enabled: bool,
) {
    let queries: Vec<Vec<f32>> = batch.iter().map(|b| b.query.clone()).collect();
    let ks: Vec<usize> = batch.iter().map(|b| b.k).collect();
    let rescore: Vec<bool> = batch.iter().map(|b| b.rescore_gaussian).collect();
    // The operator's --gpu opt-in governs the batch path too (17-nov EGM
    // audit): with only is_cuda_available() here, `serve` WITHOUT --gpu on a
    // CUDA machine still uploaded the dataset to VRAM and served through the
    // one route whose score scale differed from the fast path.
    let use_gpu = gpu_enabled && crate::gpu::is_cuda_available();

    let store = store.clone();
    // The snapshot cache must survive the 'static spawn_blocking boundary:
    // take it in, get it back with the results.
    let mut snap = snapshot.take();

    let outcome = tokio::task::spawn_blocking(move || {
        let result = if use_gpu {
            gpu_batch_path(&store, &mut snap, &queries, &ks, &rescore)
        } else {
            cpu_fused_batch(&store, &queries, &ks, &rescore)
        };
        (result, snap)
    })
    .await;

    match outcome {
        Ok((Ok(per_query), snap_back)) => {
            *snapshot = snap_back;
            for (b, r) in batch.into_iter().zip(per_query) {
                let _ = b.resp.send(Ok(r));
            }
        }
        Ok((Err(e), snap_back)) => {
            *snapshot = snap_back;
            for b in batch {
                let _ = b.resp.send(Err(e.clone()));
            }
        }
        Err(join_err) => {
            // spawn_blocking panicked — worker state (snapshot) is lost, which
            // is safe: it is only a cache. Log the JoinError server-side and
            // reply with a generic message (panic payloads stay internal).
            eprintln!("[splatsdb] batch worker task failed: {join_err}");
            for b in batch {
                let _ = b.resp.send(Err("internal error".into()));
            }
        }
    }
}

/// CPU path: single lock acquisition for the whole batch, then
/// `find_neighbors_fused` per query — HNSW-first sub-linear search with
/// exact re-rank (falls back to linear scan internally when no approximate
/// backend is available).
fn cpu_fused_batch(
    store: &Arc<Mutex<SplatStore>>,
    queries: &[Vec<f32>],
    ks: &[usize],
    rescore: &[bool],
) -> Result<Vec<Vec<SearchResult>>, String> {
    // The ONE lock acquisition per batch (R9 §1 issue 1).
    let mut store = store.blocking_lock();

    let per_query = queries
        .iter()
        .zip(ks.iter())
        .zip(rescore.iter())
        .map(|((q, &k), &want_rescore)| {
            let view = ndarray::ArrayView1::from(&q[..]);
            // ?rescore=gaussian → exact Gaussian re-rank of the top-k (B2);
            // otherwise the fused HNSW path returns L2 distances.
            let neighbors = if want_rescore {
                store.find_neighbors_gaussian(&view, k)
            } else {
                store.find_neighbors_fused(&view, k)
            };
            neighbors
                .into_iter()
                .map(|n| SearchResult {
                    index: n.index,
                    // rescore=gaussian → `score` is the Gaussian splat score
                    // (higher is better); otherwise it is the SQUARED exact
                    // L2 distance (lower is better) — misma escala que el
                    // path GPU (d²), para que las bandas calibradas del
                    // cliente (0.55/0.70) sobrevivan un fallback a CPU.
                    score: if want_rescore {
                        n.score
                    } else {
                        n.distance * n.distance
                    },
                    metadata: None,
                    confidence: None,
                    topic: None,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Feed access frequencies for tiering (R9 §5.3): O(Q·k), near-free, and
    // we already hold the lock.
    let touched: Vec<usize> = per_query
        .iter()
        .flat_map(|r| r.iter().map(|s| s.index))
        .collect();
    store.touch(&touched);

    Ok(per_query)
}

/// Outputs of the two GPU kernel wrappers: gaussian score top-k per query
/// and (indices, squared-L2 distances) rows per plain query.
type GpuKernelOutputs = (Vec<Vec<(usize, f32)>>, Vec<(Vec<usize>, Vec<f32>)>);

/// Refresh the CPU-side dataset snapshot when the store's `data_version`
/// moved (add/load/compact) or there is no snapshot yet; drops it when the
/// store is empty. `data_version` drives both the staleness check and the
/// VRAM-cache key; `n_active` sizes the arrays.
///
/// Extracted from `gpu_batch_path` (repowise XL queue, split step); behavior
/// identical. The store lock is taken briefly and released before any GPU
/// work — the search itself never holds the store mutex.
fn refresh_dataset_snapshot(
    store: &Arc<Mutex<SplatStore>>,
    snapshot: &mut Option<DatasetSnapshot>,
) {
    let s = store.blocking_lock();
    let version = s.data_version();
    let n = s.n_active();
    let stale = match snapshot.as_ref() {
        Some(snap) => snap.version != version,
        None => true,
    };
    if n == 0 {
        *snapshot = None;
    } else if stale {
        // F06a: ONE owned copy, taken from a view of the active rows, and the
        // previous snapshot is dropped BEFORE the new one is allocated.
        // Before, `get_mu()` copied the dataset and the collect copied it
        // again, on top of the old snapshot: up to 3× ~30 MB resident at
        // 10000×768.
        if let Some(view) = s.mu_view() {
            let dim = view.ncols();
            *snapshot = None;
            let data = view.iter().copied().collect::<Vec<f32>>();
            // get_alpha/get_kappa solo devuelven None con n_active==0 (caso
            // ya descartado arriba): los defaults son inalcanzables hoy.
            // Alineados a la init real del store (kappa: Array1::ones) para
            // que, si el contrato cambia, el snapshot no diverja en 10× del
            // kernel gaussiano del path CPU (17-nov EGM audit).
            let alpha = s
                .get_alpha()
                .map(|a| a.iter().copied().collect::<Vec<f32>>())
                .unwrap_or_else(|| vec![1.0f32; n]);
            let kappa = s
                .get_kappa()
                .map(|k| k.iter().copied().collect::<Vec<f32>>())
                .unwrap_or_else(|| vec![1.0f32; n]);
            *snapshot = Some(DatasetSnapshot {
                version,
                n,
                dim,
                data,
                alpha,
                kappa,
            });
        }
    }
}

/// Split the batch into gaussian-rescored and plain-L2 queries, flatten each
/// group, and run the VRAM-resident GPU kernels once for the whole batch.
/// Gaussian-rescored queries go through `gpu::gaussian_topk_scores` (B1
/// wrapper: GPU kernel + exact CPU fallback, returns score-sorted top-k
/// directly); plain L2 queries through `gpu::batch_search` (top-k kernel,
/// squared-L2 → sqrt).
///
/// One VRAM-resident index serves the whole batch (and every later batch with
/// the same dataset version): the `*_cached` wrappers key the process-level
/// GpuIndex on (snapshot version, n, dim), so mu+alpha+kappa upload once per
/// dataset version instead of once per batch. On a cache decline (small
/// dataset / no CUDA / kernel failure) they fall back to the per-call
/// wrappers — same results as before.
///
/// Extracted from `gpu_batch_path` (repowise XL queue, split step); behavior
/// identical.
fn split_and_run_gpu_kernels(
    snap: &DatasetSnapshot,
    queries: &[Vec<f32>],
    ks: &[usize],
    rescore: &[bool],
) -> Result<GpuKernelOutputs, String> {
    let gauss_flat: Vec<f32> = queries
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| r)
        .flat_map(|(q, _)| q.iter().copied())
        .collect();
    let gauss_ks: Vec<usize> = ks
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| r)
        .map(|(&k, _)| k)
        .collect();
    let n_gauss = gauss_ks.len();

    let l2_flat: Vec<f32> = queries
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| !r)
        .flat_map(|(q, _)| q.iter().copied())
        .collect();
    let l2_ks: Vec<usize> = ks
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| !r)
        .map(|(&k, _)| k)
        .collect();
    let n_l2 = l2_ks.len();

    let data_view = ndarray::ArrayView2::from_shape((snap.n, snap.dim), &snap.data)
        .map_err(|e| format!("snapshot reshape failed: {e}"))?;

    let gauss_out = if n_gauss > 0 {
        // F02e1: one pair beyond the largest k, so every query can prove
        // its k-set against a tie at the boundary (see `canonicalize_gpu_rows`).
        let kmax_g = gauss_ks.iter().copied().max().unwrap_or(1).min(snap.n);
        crate::gpu::gaussian_topk_scores_cached(
            snap.version,
            &gauss_flat,
            &data_view,
            &snap.alpha,
            &snap.kappa,
            n_gauss,
            gpu_window(kmax_g, snap.n),
        )
        .ok_or_else(|| "gaussian_topk_scores failed".to_string())?
    } else {
        Vec::new()
    };

    let l2_out = if n_l2 > 0 {
        let kmax_l = l2_ks.iter().copied().max().unwrap_or(1).min(snap.n);
        crate::gpu::batch_search_cached(
            snap.version,
            &l2_flat,
            n_l2,
            &data_view,
            gpu_window(kmax_l, snap.n),
            "l2",
        )
    } else {
        Vec::new()
    };
    Ok((gauss_out, l2_out))
}

/// GPU path: refresh the flat snapshot only when the dataset changed, then a
/// single `gpu::batch_search` for the whole batch (GPU top-k kernel with CPU
/// fallback inside). The store mutex is never held during the search.
fn gpu_batch_path(
    store: &Arc<Mutex<SplatStore>>,
    snapshot: &mut Option<DatasetSnapshot>,
    queries: &[Vec<f32>],
    ks: &[usize],
    rescore: &[bool],
) -> Result<Vec<Vec<SearchResult>>, String> {
    // ── Snapshot refresh under a brief lock ──
    refresh_dataset_snapshot(store, snapshot);

    let snap = snapshot
        .as_ref()
        .ok_or_else(|| "store is empty".to_string())?;

    let (mut gauss_out, mut l2_out) = split_and_run_gpu_kernels(snap, queries, ks, rescore)?;

    // ── F02e1: contractual (score, id) rows — ties at the k-th score are
    // proven against the kernel's window and resolved by id.
    canonicalize_gpu_rows(snap, queries, ks, rescore, &mut gauss_out, &mut l2_out);

    // ── Reassemble rows in original request order (shape-validated).
    let per_query = reassemble_gpu_rows(snap.n, ks, rescore, gauss_out, l2_out)?;

    // Feed access frequencies for tiering under a second brief lock.
    let touched: Vec<usize> = per_query
        .iter()
        .flat_map(|r| r.iter().map(|s| s.index))
        .collect();
    store.blocking_lock().touch(&touched);

    Ok(per_query)
}

/// Kernel K requested for a batch whose largest k is `kmax` (F02e1): `kmax+1`
/// — the extra pair lets every query prove its k-set against a tie at the
/// boundary — within what the kernel takes; `kmax` itself when it cannot
/// take one more (then its host fallback answers) or the dataset is smaller.
fn gpu_window(kmax: usize, n: usize) -> usize {
    if kmax < crate::splats::GPU_TOPK_MAX_K {
        (kmax + 1).min(n)
    } else {
        kmax
    }
}

/// Host-exact L2 top-k over a flat row-major dataset: `(id, d²)` in
/// contractual order (d² ascending, then id). The same f32 arithmetic as
/// `SplatStore::find_neighbors`.
fn host_exact_l2(data: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> Vec<(usize, f32)> {
    if q.len() != dim || data.len() < n.saturating_mul(dim) {
        return Vec::new();
    }
    let mut all: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let row = &data[i * dim..(i + 1) * dim];
            let d2: f32 = row
                .iter()
                .zip(q)
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            (i, d2)
        })
        .collect();
    crate::splats::sort_score_then_id(&mut all, false);
    all.truncate(k);
    all
}

/// F02e1: contractual rows on the GPU batch route.
///
/// The kernels answered `gpu_window(kmax)` pairs per query in their own
/// order. Each query's row is reduced to its contractual top-k with
/// [`crate::splats::contractual_topk`]: canonical (score, id) order, and a
/// tie at the k-th score proven against the extra pair — escalated, for that
/// query alone, to the kernel's largest window. A tie the kernel cannot
/// resolve is computed on the host from the same snapshot (`host_exact_l2` /
/// `gpu::cpu_gaussian_topk`): exact and canonical, O(n·dim), only for queries
/// whose k-th score ties with everything up to the window's end. Rows of a
/// malformed shape are left untouched for [`reassemble_gpu_rows`] to reject.
fn canonicalize_gpu_rows(
    snap: &DatasetSnapshot,
    queries: &[Vec<f32>],
    ks: &[usize],
    rescore: &[bool],
    gauss_out: &mut [Vec<(usize, f32)>],
    l2_out: &mut [(Vec<usize>, Vec<f32>)],
) {
    if ks.len() != rescore.len() || queries.len() < ks.len() {
        return;
    }
    let Ok(view) = ndarray::ArrayView2::from_shape((snap.n, snap.dim), &snap.data) else {
        return;
    };
    let (mut g, mut l) = (0usize, 0usize);
    for (i, (&k, &gaussian)) in ks.iter().zip(rescore.iter()).enumerate() {
        let q: &[f32] = &queries[i];
        if gaussian {
            let Some(slot) = gauss_out.get_mut(g) else {
                break;
            };
            g += 1;
            let fetched = slot.clone();
            let fetch = |window: usize| -> Option<Vec<(usize, f32)>> {
                if window <= fetched.len() {
                    return Some(fetched[..window].to_vec());
                }
                crate::gpu::gaussian_topk_scores_cached(
                    snap.version,
                    q,
                    &view,
                    &snap.alpha,
                    &snap.kappa,
                    1,
                    window,
                )?
                .into_iter()
                .next()
            };
            *slot = match crate::splats::contractual_topk(snap.n, k, true, fetch) {
                crate::splats::Contract::Exact(row) => row,
                _ => crate::gpu::cpu_gaussian_topk(
                    q,
                    &snap.data,
                    &snap.alpha,
                    &snap.kappa,
                    snap.n,
                    snap.dim,
                    1,
                    k,
                )
                .pop()
                .unwrap_or_default(),
            };
        } else {
            let Some(slot) = l2_out.get_mut(l) else {
                break;
            };
            l += 1;
            if slot.0.len() != slot.1.len() {
                continue;
            }
            let fetched: Vec<(usize, f32)> = slot
                .0
                .iter()
                .zip(&slot.1)
                .map(|(&ix, &d)| (ix, d))
                .collect();
            let fetch = |window: usize| -> Option<Vec<(usize, f32)>> {
                if window <= fetched.len() {
                    return Some(fetched[..window].to_vec());
                }
                let (idxs, dists) =
                    crate::gpu::batch_search_cached(snap.version, q, 1, &view, window, "l2")
                        .pop()?;
                Some(idxs.into_iter().zip(dists).collect())
            };
            let row = match crate::splats::contractual_topk(snap.n, k, false, fetch) {
                crate::splats::Contract::Exact(row) => row,
                _ => host_exact_l2(&snap.data, snap.n, snap.dim, q, k),
            };
            *slot = (
                row.iter().map(|p| p.0).collect(),
                row.iter().map(|p| p.1).collect(),
            );
        }
    }
}

/// Validate the GPU kernel outputs against the batch they were requested
/// for and reassemble one row per query, in request order.
///
/// `gpu::batch_search_cached` answers an empty `Vec` when it declines the
/// shape, and a kernel may return fewer — or shorter — rows than requested.
/// Indexing `l2_out[l]` on such an answer panicked inside the batch worker
/// (JoinError → 500 for the whole batch; an abort under `panic=abort`), and
/// `zip` over mismatched idx/dist arrays silently truncated a row. Any shape
/// violation is a controlled `Err`: never 200 with truncated or fabricated
/// results. Valid rows keep exactly the previous mapping.
fn reassemble_gpu_rows(
    n: usize,
    ks: &[usize],
    rescore: &[bool],
    gauss_out: Vec<Vec<(usize, f32)>>,
    l2_out: Vec<(Vec<usize>, Vec<f32>)>,
) -> Result<Vec<Vec<SearchResult>>, String> {
    let n_gauss = rescore.iter().filter(|&&r| r).count();
    let n_l2 = rescore.len() - n_gauss;
    if ks.len() != rescore.len() {
        return Err(format!(
            "gpu batch: {} k values for {} queries",
            ks.len(),
            rescore.len()
        ));
    }
    if gauss_out.len() != n_gauss {
        return Err(format!(
            "gpu batch: gaussian kernel returned {} rows for {n_gauss} queries",
            gauss_out.len()
        ));
    }
    if l2_out.len() != n_l2 {
        return Err(format!(
            "gpu batch: l2 kernel returned {} rows for {n_l2} queries",
            l2_out.len()
        ));
    }
    for (q, (idxs, dists)) in l2_out.iter().enumerate() {
        if idxs.len() != dists.len() {
            return Err(format!(
                "gpu batch: l2 row {q} has {} indices for {} distances",
                idxs.len(),
                dists.len()
            ));
        }
    }

    let mut per_query = Vec::with_capacity(ks.len());
    let mut g = 0usize; // cursor into gauss_out
    let mut l = 0usize; // cursor into l2_out
    for (&k, &want_rescore) in ks.iter().zip(rescore.iter()) {
        let row: Vec<SearchResult> = if want_rescore {
            let src = &gauss_out[g];
            g += 1;
            src.iter()
                .filter(|&&(ix, _)| ix < n)
                .take(k)
                .map(|&(ix, score)| SearchResult {
                    index: ix,
                    // Gaussian score: higher is better.
                    score,
                    metadata: None,
                    confidence: None,
                    topic: None,
                })
                .collect()
        } else {
            let (idxs, dists) = &l2_out[l];
            l += 1;
            idxs.iter()
                .zip(dists.iter())
                .filter(|(&ix, _)| ix < n)
                .take(k)
                .map(|(&ix, &d)| SearchResult {
                    index: ix,
                    // Emit raw squared L2 (d²) — same scale as the fast GPU
                    // path and cpu_fused_batch (87d5d91), so the client's
                    // calibrated bands (0.55/0.70) survive a batch-worker
                    // fallback. The old sqrt here described the pre-87d5d91
                    // CPU path and made the scale depend on which route
                    // served the query (17-nov EGM audit).
                    score: d,
                    metadata: None,
                    confidence: None,
                    topic: None,
                })
                .collect()
        };
        per_query.push(row);
    }
    Ok(per_query)
}

// ─── API Key Authentication Middleware ────────────────────

/// Authentication of a `serve` process, resolved ONCE at startup from
/// `SPLATSDB_API_KEY` (A05-F10) and carried in [`AppState::auth`]. The
/// middleware never re-reads the environment, so a request can never be
/// judged against a different key than the one the server booted with.
#[derive(Clone, PartialEq, Eq)]
pub enum AuthConfig {
    /// `SPLATSDB_API_KEY` is absent: loopback dev mode, requests pass
    /// through (a non-loopback bind in this mode needs the explicit opt-in,
    /// see [`check_bind_policy`]).
    Unauthenticated,
    /// A non-empty, non-blank UTF-8 key: every request except `GET /health`
    /// and CORS preflight must carry `Authorization: Bearer <key>`.
    Required(String),
}

/// Never print the key: `Debug` is needed by tests/assertions only.
impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthConfig::Unauthenticated => f.write_str("Unauthenticated"),
            AuthConfig::Required(_) => f.write_str("Required(<redacted>)"),
        }
    }
}

/// Environment variable holding the API key.
pub const API_KEY_ENV: &str = "SPLATSDB_API_KEY";

/// Explicit operator opt-in that allows a non-loopback bind WITHOUT an API
/// key. Only the exact value `1` opts in — `true`, `yes`, `0`, a typo or a
/// padded value do NOT (fail-closed).
pub const ALLOW_UNAUTHENTICATED_ENV: &str = "SPLATSDB_ALLOW_UNAUTHENTICATED";

/// Resolve `SPLATSDB_API_KEY` from the raw environment value (read with
/// `var_os`, so a non-UTF-8 value is observable instead of collapsing into
/// "absent").
///
/// Fail-closed (A05-F10): the variable PRESENT but empty, blank or not
/// UTF-8 is a misconfiguration → `Err`, and the caller must abort startup.
/// Before this, `env::var` mapped non-UTF-8 to "no key" (authentication
/// silently off) and an empty key authenticated every `Authorization:
/// Bearer ` header (`"" == ""`).
pub(crate) fn resolve_api_key(value: Option<std::ffi::OsString>) -> Result<AuthConfig, String> {
    let Some(raw) = value else {
        return Ok(AuthConfig::Unauthenticated);
    };
    match raw.into_string() {
        Ok(key) if key.trim().is_empty() => Err(format!(
            "{API_KEY_ENV} is set but EMPTY/blank: an empty key authenticates every \
             `Authorization: Bearer ` request (authentication effectively disabled). Unset it \
             to serve without authentication on loopback, or set a non-empty key."
        )),
        Ok(key) => Ok(AuthConfig::Required(key)),
        Err(_) => Err(format!(
            "{API_KEY_ENV} is set but is not valid UTF-8: it cannot be compared with a Bearer \
             token, and treating it as absent would silently disable authentication. Set a \
             UTF-8 key or unset the variable."
        )),
    }
}

/// Whether the operator opted in explicitly (value exactly `1`).
pub(crate) fn unauthenticated_opt_in(value: Option<std::ffi::OsString>) -> bool {
    value.as_deref() == Some(std::ffi::OsStr::new("1"))
}

/// Bind policy (A05-F10), fail-closed: a bind address that is NOT loopback
/// without an API key is refused unless the operator opted in explicitly
/// (`SPLATSDB_ALLOW_UNAUTHENTICATED=1`). Loopback without a key keeps
/// serving (localhost dev mode) and any bind WITH a key is allowed.
pub(crate) fn check_bind_policy(addr: &str, auth: &AuthConfig, opt_in: bool) -> anyhow::Result<()> {
    if is_loopback_addr(addr) || matches!(auth, AuthConfig::Required(_)) || opt_in {
        return Ok(());
    }
    anyhow::bail!(
        "refusing to serve on {addr:?} without authentication: {API_KEY_ENV} is not set, so \
         anyone who can reach this address could read and write the entire store. Bind to \
         loopback (127.0.0.1 / ::1 / localhost), set {API_KEY_ENV}, or opt in explicitly with \
         {ALLOW_UNAUTHENTICATED_ENV}=1. Nothing was bound."
    );
}

async fn api_key_auth(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<axum::response::Response, StatusCode> {
    // Skip auth for health endpoint
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    // Skip auth for CORS preflight: OPTIONS carries no credentials and must
    // be answered by the CORS layer (with CORS headers), otherwise browsers
    // get a 401 before ever sending the real authenticated request.
    if request.method() == axum::http::Method::OPTIONS {
        return Ok(next.run(request).await);
    }

    // The key was resolved once at startup — fail-closed on empty/non-UTF-8
    // (A05-F10). An absent key is the ONLY way to pass without a credential;
    // a wrong, empty or malformed bearer NEVER authenticates.
    match &state.auth {
        AuthConfig::Unauthenticated => Ok(next.run(request).await),
        AuthConfig::Required(expected) => match bearer_token(&headers) {
            Some(provided) if constant_time_eq(provided.as_bytes(), expected.as_bytes()) => {
                Ok(next.run(request).await)
            }
            _ => Err(StatusCode::UNAUTHORIZED),
        },
    }
}

/// Token of an `Authorization: Bearer <token>` header, parsed strictly:
/// nothing is trimmed or normalised, so a malformed header can never be
/// massaged into a match.
///
/// `None` for a missing/non-ASCII header, any other scheme (also a
/// lowercase `bearer`), a missing token (`Bearer `) and a token with
/// surrounding or inner whitespace (`Bearer  k`, `Bearer k `, `Bearer k x`).
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    let token = value.strip_prefix("Bearer ")?;
    (!token.is_empty() && !token.bytes().any(|b| b.is_ascii_whitespace())).then_some(token)
}

/// Constant-time byte comparison to prevent timing attacks on API keys.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Load vectors from the `default` shard of `data_dir` into `store`, with the
/// persisted energy state (F02e1).
///
/// Returns the number of vectors loaded. Mirrors the CLI's
/// `load_or_create_store` behaviour (src/cli/helpers.rs), keeping `serve`
/// consistent with `status`/`search` over the same data-dir.
///
/// Energy state — `energy/landscape.bin`, the alpha/kappa per splat that
/// `ingest`, `save` and `verbatim store` write next to the shard. Before
/// F02e1 `serve` loaded only mu (`add_splat` → init alpha/kappa = 1.0), so
/// after every restart the splats answered with fabricated values and
/// `?rescore=gaussian` ranked with them. Policies, in order:
/// * absent → mu only (a data-dir written by a route without energy state);
/// * corrupt — truncated, hostile header, counts that do not match,
///   non-finite or negative values → `Err`, fail-closed like an unreadable
///   shard (A03-F06 / D4): nothing is modified on disk;
/// * foreign or stale — its mu is not a prefix of the shard's rows →
///   ignored with a LOUD warning, the server behaves exactly as before;
/// * otherwise the alpha/kappa of the `m` leading rows the landscape shares
///   with the shard are loaded. A prefix is the normal case after a `serve`
///   session: its compact rewrites the shard with the rows `/store`
///   appended and leaves the landscape alone; those rows keep the init
///   alpha/kappa they had in memory.
fn load_default_shard(store: &mut SplatStore, data_dir: &str) -> anyhow::Result<usize> {
    use crate::storage::persistence::{SplatsDBPersistence, StorageBackend};

    let persist = SplatsDBPersistence::with_backend(data_dir, StorageBackend::Sqlite, false)
        .map_err(|e| anyhow::anyhow!("persistence init failed: {e}"))?;
    match persist.load_vectors("default")? {
        Some(vectors) => load_shard_with_energy(store, data_dir, &vectors),
        None => Ok(0),
    }
}

/// [`load_default_shard`] with the shard already read: adds the vectors, then
/// applies the energy-state policy.
fn load_shard_with_energy(
    store: &mut SplatStore,
    data_dir: &str,
    vectors: &ndarray::Array2<f32>,
) -> anyhow::Result<usize> {
    if !store.add_splat(vectors) {
        anyhow::bail!(
            "store capacity exceeded or dimension mismatch while loading {} vectors \
             (max_splats={}) — check --dim against the shard's dim",
            vectors.nrows(),
            store.max_splats()
        );
    }
    let n = vectors.nrows();
    let path = crate::loaders::EnergyLandscape::path_in(std::path::Path::new(data_dir));
    if !path.exists() {
        eprintln!(
            "[splatsdb] serve: {data_dir} has no energy state ({}): alpha/kappa stay at the init values",
            path.display()
        );
        return Ok(n);
    }
    let landscape = crate::loaders::load_energy_landscape(path.to_str().unwrap_or_default())
        .map_err(anyhow::Error::msg)?;
    let m = landscape.matching_prefix(vectors);
    if m == 0 {
        eprintln!(
            "[splatsdb] serve: WARNING {} does not describe the 'default' shard ({}x{} vs {}x{}): \
             it is IGNORED and alpha/kappa stay at the init values — ?rescore=gaussian \
             rescores with them",
            path.display(),
            landscape.mu.nrows(),
            landscape.mu.ncols(),
            n,
            vectors.ncols()
        );
        return Ok(n);
    }
    let loaded = store
        .load_energy(&landscape.alpha[..m], &landscape.kappa[..m])
        .map_err(anyhow::Error::msg)?;
    eprintln!(
        "[splatsdb] serve: energy state cargado (alpha/kappa de {loaded}/{n} splats desde {}): \
         alpha={} kappa={}",
        path.display(),
        preview(&landscape.alpha[..loaded]),
        preview(&landscape.kappa[..loaded])
    );
    Ok(n)
}

/// First values of a loaded vector for the startup log (4 decimals, at most 8).
fn preview(v: &[f32]) -> String {
    let head: Vec<String> = v.iter().take(8).map(|x| format!("{x:.4}")).collect();
    format!(
        "[{}{}]",
        head.join(", "),
        if v.len() > 8 { ", …" } else { "" }
    )
}

/// Loopback bind check for the bind policy ([`check_bind_policy`]):
/// `localhost`, `::1` and any `127.x.x.x` address are private; everything
/// else (including `0.0.0.0`/`::`) is treated as externally reachable.
fn is_loopback_addr(addr: &str) -> bool {
    let a = addr.trim().to_ascii_lowercase();
    // Bracketed IPv6 literal (`[::1]`) → parse the inner literal.
    let literal = a
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(&a);
    match literal.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        // Only the literal name `localhost` counts. Any other hostname —
        // even one that resolves to 127.0.0.1, or `127.0.0.1.example.com`,
        // which passed the old `starts_with("127.")` prefix test — is
        // treated as externally reachable (fail-closed).
        Err(_) => a == "localhost",
    }
}

/// Auto-load the on-disk inputs for `serve`: the `default` shard from
/// `data_dir` (if present) into `store`, so `serve` boots with the vectors
/// indexed by the CLI instead of an empty store and honours the CLI --dim
/// flag instead of SplatsDBConfig::default() (issue #3); then the `field/`
/// corpus for /field-search and for the topic/confidence of /search —
/// /search itself always searches the shard's splats (F02d: `serve_corpus`).
///
/// Extracted from `run_server` (repowise plan `refac4_829dc22e97478fcffcab`);
/// messages and behavior identical. Returns the field corpus, if any.
fn load_serve_inputs(data_dir: &str, store: &mut SplatStore) -> anyhow::Result<Option<FieldData>> {
    match load_default_shard(store, data_dir) {
        Ok(n) if n > 0 => eprintln!(
            "[splatsdb] serve: loaded {n} vectors (dim {}) from {data_dir}",
            store.get_statistics().embedding_dim
        ),
        Ok(_) => eprintln!("[splatsdb] serve: no existing data in {data_dir}, starting empty"),
        // Fail-closed (A03-F06): seguir con un store vacío haría que el
        // compact del shutdown reemplazara el shard existente por un
        // subconjunto (sólo lo que traje el WAL) o por un default.bin vacío.
        Err(e) => anyhow::bail!(
            "serve: fail-closed, el shard 'default' de {data_dir} no se puede cargar \
             (nada se modificó en disco): {e}"
        ),
    }

    let field_data = load_field(data_dir);
    if field_data.is_some() {
        eprintln!("[splatsdb] serve: campo splat cargado (/field-search disponible)");
    }
    Ok(field_data)
}

pub async fn run_server(
    addr: &str,
    port: u16,
    config: SplatsDBConfig,
    data_dir: &str,
    gpu_enabled: bool,
) -> anyhow::Result<()> {
    run_server_with_mode(
        addr,
        port,
        config,
        data_dir,
        gpu_enabled,
        WalDurability::ProcessCrash,
    )
    .await
}

pub async fn run_server_with_mode(
    addr: &str,
    port: u16,
    config: SplatsDBConfig,
    data_dir: &str,
    gpu_enabled: bool,
    wal_mode: WalDurability,
) -> anyhow::Result<()> {
    // Auth is resolved BEFORE anything is created on disk or bound (A05-F10):
    // a misconfigured key (empty / non-UTF-8) or an unauthenticated
    // non-loopback bind without the explicit opt-in aborts the startup with
    // a diagnosis and no listener.
    let auth = resolve_api_key(std::env::var_os(API_KEY_ENV)).map_err(anyhow::Error::msg)?;
    check_bind_policy(
        addr,
        &auth,
        unauthenticated_opt_in(std::env::var_os(ALLOW_UNAUTHENTICATED_ENV)),
    )?;
    let config = server_config(config);
    let hnsw_on = config.enable_hnsw;
    let mut store = SplatStore::new(config);

    // Fail-closed (A03-F12): lock exclusivo del data-dir por toda la vida del
    // serve, para que dos `serve` no reescriban el shard uno encima del otro.
    // Es un flock del kernel: muere con el proceso, así que un lockfile viejo
    // nunca bloquea.
    std::fs::create_dir_all(data_dir)?;
    let _data_dir_lock = DataDirLock::acquire(std::path::Path::new(data_dir))?;
    let field_data = load_serve_inputs(data_dir, &mut store)?;
    #[cfg(not(feature = "cuda"))]
    let corpus_n = store.n_active();

    // ── Durabilidad: el shard "default" ya lo cargó load_serve_inputs. Se
    // abre el WAL, se re-aplica lo no compactado y queda listo para
    // persistir por chunks (COMPACT_EVERY) y en el shutdown.
    //
    // Fail-closed (A03-F01): con --data-dir un WAL que no se puede abrir ni
    // replayear ABORTA el arranque. Arrancar "sin durabilidad" devolvería
    // 200 `stored` sobre un store que se pierde al salir.
    let (durability, next_id_seed) = match Durability::open_with_mode(data_dir, wal_mode) {
        Ok(d) => match replay_wal(&mut store, &d) {
            Ok((n, max_auto)) => {
                if n > 0 {
                    eprintln!("[splatsdb] WAL replay: {n} stores re-aplicados desde el WAL");
                }
                (Some(Arc::new(d)), max_auto)
            }
            Err(e) => anyhow::bail!(
                "serve: fail-closed, el WAL de {data_dir} no se pudo replayear (nada se acepta): {e}"
            ),
        },
        Err(e) => anyhow::bail!(
            "serve: fail-closed, la durabilidad de {data_dir} no está disponible; no se arranca un \
             server efímero sobre un data-dir: {e}"
        ),
    };

    // ── Índice exacto en GPU (--gpu, feature cuda): kernel PTX l2+topk sobre
    // el corpus vigente DESPUÉS del replay del WAL. Fallback silencioso a CPU
    // si no hay GPU. La clave (data_version, n) es la del dataset subido: el
    // handler sólo lo usa mientras siga vigente (A03-F05).
    #[cfg(feature = "cuda")]
    let (gpu_index, corpus_n) = if gpu_enabled {
        // The corpus the fast path serves is the store's mu (splats/μ — never
        // the field/ chunks, F02d), uploaded straight from the store buffer.
        GpuExactIndex::startup(&store)
    } else {
        (None, store.n_active())
    };

    let store = Arc::new(Mutex::new(store));
    let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
    let batches_served = Arc::new(AtomicUsize::new(0));
    tokio::spawn(batch_worker(
        batch_rx,
        store.clone(),
        batches_served.clone(),
        gpu_enabled,
    ));

    let state = AppState {
        store: store.clone(),
        next_id: Arc::new(AtomicUsize::new(next_id_seed)),
        batch_tx,
        batches_served,
        #[cfg(feature = "cuda")]
        gpu: gpu_index,
        field: field_data.map(Arc::new),
        corpus_n,
        data_dir: Some(data_dir.to_string()),
        durability: durability.clone(),
        auth: auth.clone(),
    };

    let app = build_router(state);

    let bind_addr = format!("{addr}:{port}");
    // Loud warning: bound beyond loopback with no API key → everything (store,
    // search, field-search) is open to anyone who can reach this address.
    // The policy above already refused this configuration unless the
    // operator opted in explicitly; the opt-in stays loud.
    if !is_loopback_addr(addr) && auth == AuthConfig::Unauthenticated {
        eprintln!(
            "[splatsdb] ⚠  WARNING: serving on {bind_addr} with NO SPLATSDB_API_KEY set \
             (explicit opt-in: {ALLOW_UNAUTHENTICATED_ENV}=1).
\
             ⚠  Authentication is DISABLED — anyone who can reach this address can read
\
             ⚠  and write the entire store. Set SPLATSDB_API_KEY to require Bearer auth."
        );
    }
    println!(
        "SplatsDB server listening on {bind_addr} \
         (hnsw={}, batching: window={BATCH_WINDOW_MS}ms max={MAX_BATCH}, gpu={})",
        hnsw_on,
        crate::gpu::is_cuda_available()
    );
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    // Shutdown graceful: en SIGINT/SIGTERM persiste el shard "default" y
    // trunca el WAL antes de soltar el listener — un restart re-levanta el
    // dataset entero desde disco.
    let shutdown_store = store.clone();
    let shutdown_durability = durability.clone();
    let shutdown_result: Arc<std::sync::Mutex<Option<Result<usize, String>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let shutdown_slot = shutdown_result.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            wait_shutdown_signal().await;
            let outcome = if let Some(d) = shutdown_durability.as_ref() {
                let store = shutdown_store.lock().await;
                Some(d.compact(&store).map_err(|e| e.to_string()))
            } else {
                None
            };
            if let Ok(mut slot) = shutdown_slot.lock() {
                *slot = outcome;
            }
        })
        .await?;
    match shutdown_result.lock().ok().and_then(|g| g.clone()) {
        Some(Ok(n)) => {
            eprintln!("[splatsdb] shutdown: shard 'default' persistido ({n} splats), WAL truncado")
        }
        Some(Err(e)) => {
            // Lo aceptado sigue en el WAL (fsync abajo): no se pierde, pero el
            // shard NO se actualizó → el proceso no termina con exit 0.
            if let Some(d) = durability.as_ref() {
                let _ = d.checkpoint_now();
            }
            anyhow::bail!(
                "shutdown: compact falló, el shard 'default' NO se actualizó (los stores aceptados \
                 quedan en wal/store.wal): {e}"
            );
        }
        None => {}
    }
    Ok(())
}

/// Bloquea hasta SIGINT o SIGTERM (shutdown graceful del serve).
async fn wait_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                tokio::signal::ctrl_c().await.ok();
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.ok();
    }
}

/// Re-aplica las entradas "store" del WAL que el shard "default" aún no
/// contiene (idx >= splats ya cargados). Devuelve (aplicados, siguiente
/// auto_id) para sembrar el contador `next_id` del server.
fn replay_wal(store: &mut SplatStore, d: &Durability) -> anyhow::Result<(usize, usize)> {
    let entries = d.wal.recover()?;
    let dim = store.get_statistics().embedding_dim;
    let shard_n = store.n_active();
    let mut applied = 0usize;
    let mut max_auto = 0usize;
    let mut unapplied = Vec::new();
    let mut seen_idx = std::collections::HashMap::new();
    for e in &entries {
        if e.operation != "store" {
            continue;
        }
        let idx = e
            .data
            .get("idx")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);
        if idx < shard_n {
            continue; // ya está dentro del shard persistido
        }
        if let Some(previous) = seen_idx.insert(idx, e) {
            if previous.data.get("id") != e.data.get("id")
                || previous.data.get("vector") != e.data.get("vector")
            {
                unapplied.push(e.lsn);
            } else {
                eprintln!("[splatsdb] WAL: duplicate idx {idx} (ambiguous retry) skipped");
            }
            continue;
        }
        let Some(vals) = e.data.get("vector").and_then(|v| v.as_array()) else {
            eprintln!("[splatsdb] WAL: entrada {} sin vector, salteada", e.lsn);
            unapplied.push(e.lsn);
            continue;
        };
        let vec: Vec<f32> = vals
            .iter()
            .filter_map(|x| x.as_f64())
            .map(|x| x as f32)
            .collect();
        if vec.len() != dim {
            eprintln!(
                "[splatsdb] WAL: entrada {} con dim {} != {dim}, salteada",
                e.lsn,
                vec.len()
            );
            unapplied.push(e.lsn);
            continue;
        }
        let Ok(arr) = Array2::from_shape_vec((1, dim), vec) else {
            unapplied.push(e.lsn);
            continue;
        };
        if store.add_splat(&arr) {
            store.hnsw_sync_incremental();
            applied += 1;
        } else {
            eprintln!(
                "[splatsdb] WAL: entrada {} no entra al store (capacidad), salteada",
                e.lsn
            );
            unapplied.push(e.lsn);
        }
        if let Some(a) = e.data.get("auto_id").and_then(|v| v.as_u64()) {
            max_auto = max_auto.max(a as usize + 1);
        }
    }
    // Las entradas no aplicadas son la única copia de esos datos: `compact`
    // no trunca el WAL mientras existan (extiende A03-F09 a n_active > 0).
    if let Ok(mut slot) = d.unapplied.lock() {
        slot.extend(unapplied);
    }
    Ok((applied, max_auto))
}

/// Server preset applied on top of the CLI-provided config: HNSW ENABLED with
/// a reasonable over-fetch.
///
/// The server previously used `SplatsDBConfig::default()`, which ships HNSW
/// disabled (config/mod.rs: `enable_hnsw: false`) — every `/search` degraded
/// to a linear O(N·D) scan even through `find_neighbors_fused` (R9 §5.3).
/// The CLI config (dim, data_dir plumbing from issue #3) is preserved; only
/// the HNSW tuning knobs are overridden here.
fn server_config(mut base: SplatsDBConfig) -> SplatsDBConfig {
    base.enable_hnsw = true;
    base.hnsw_m = 32;
    base.hnsw_ef_construction = 200;
    base.hnsw_ef_search = 100;
    // HNSW candidate over-fetch factor before exact re-rank.
    base.over_fetch = 5;
    base
}

/// Build the application router (shared by `run_server` and tests).
fn build_router(state: AppState) -> Router {
    // CORS: restrict to localhost by default
    let cors = CorsLayer::new()
        .allow_origin([
            "http://localhost".parse().unwrap(),
            "http://127.0.0.1".parse().unwrap(),
        ])
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ]);

    Router::new()
        .route("/health", get(health))
        .route("/status", post(status))
        .route("/store", post(store_memory))
        .route("/search", post(search_memories))
        .route("/field-search", post(field_search_handler))
        .route("/recap", post(recap_handler))
        .route("/recall", post(recall_handler))
        // Layer order matters: `.layer()` wraps everything added BEFORE it,
        // so the LAST layer is the OUTERMOST. cors must be outermost so a
        // preflight OPTIONS is answered (with CORS headers) without going
        // through auth; auth then guards the actual requests inside.
        .layer(middleware::from_fn_with_state(state.clone(), api_key_auth))
        .layer(cors)
        .with_state(state)
}

// ─── Tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use serde_json::{json, Value};
    use tower::ServiceExt; // oneshot

    fn test_config() -> SplatsDBConfig {
        SplatsDBConfig {
            max_splats: 500,
            latent_dim: 32,
            enable_hnsw: true,
            hnsw_m: 8,
            hnsw_ef_construction: 50,
            hnsw_ef_search: 32,
            over_fetch: 5,
            ..SplatsDBConfig::default()
        }
    }

    /// App instance with a fresh store, a live batch worker and a readable
    /// batches-served counter for verifying coalescing.
    async fn test_app() -> (Router, Arc<AtomicUsize>) {
        let store = Arc::new(Mutex::new(SplatStore::new(test_config())));
        let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let batches_served = Arc::new(AtomicUsize::new(0));
        tokio::spawn(batch_worker(
            batch_rx,
            store.clone(),
            batches_served.clone(),
            false,
        ));
        let state = AppState {
            store,
            next_id: Arc::new(AtomicUsize::new(0)),
            batch_tx,
            batches_served: batches_served.clone(),
            #[cfg(feature = "cuda")]
            gpu: None,
            field: None,
            corpus_n: 0,
            data_dir: None,
            durability: None,
            auth: AuthConfig::Unauthenticated,
        };
        (build_router(state), batches_served)
    }

    async fn send(
        app: Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let body = match body {
            Some(v) => Body::from(v.to_string()),
            None => Body::empty(),
        };
        let req = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(body)
            .unwrap();
        let resp = app.oneshot(req).await.expect("oneshot request failed");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body failed");
        // Error responses (e.g. 400/413) carry a plain-text body — fall back
        // to a string Value so callers can still inspect status + message.
        let json: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
        };
        (status, json)
    }

    fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    /// App con durabilidad sobre `dir`, espejando el boot de `run_server`:
    /// shard "default" + replay del WAL + next_id sembrado con el máximo
    /// auto_id del WAL. Devuelve también la Durability para forzar flush.
    async fn test_app_with_durability(dir: &std::path::Path) -> (Router, Arc<Durability>) {
        let config = test_config();
        let mut store = SplatStore::new(config.clone());
        let d = Arc::new(Durability::open(dir.to_str().unwrap()).expect("durability init"));
        let _ = load_default_shard(&mut store, dir.to_str().unwrap());
        let (replayed, max_auto) = replay_wal(&mut store, &d).expect("wal replay");
        let _ = replayed;
        let store = Arc::new(Mutex::new(store));
        let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let batches_served = Arc::new(AtomicUsize::new(0));
        tokio::spawn(batch_worker(
            batch_rx,
            store.clone(),
            batches_served.clone(),
            false,
        ));
        let state = AppState {
            store,
            next_id: Arc::new(AtomicUsize::new(max_auto)),
            batch_tx,
            batches_served,
            #[cfg(feature = "cuda")]
            gpu: None,
            field: None,
            corpus_n: 0,
            data_dir: Some(dir.to_string_lossy().into_owned()),
            durability: Some(d.clone()),
            auth: AuthConfig::Unauthenticated,
        };
        (build_router(state), d)
    }

    fn temp_wal_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("splatdb_wal_test_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    // ── procedencia del vector: sin vector en el request, el server le pide
    //    uno al servicio de embeddings; nunca un hash del texto ──

    /// `SPLATSDB_RECALL_EMBED_URL` is process-wide: every test that sets it
    /// takes this lock, so two of them never observe each other's value.
    static EMBED_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn embed_env_lock() -> &'static tokio::sync::Mutex<()> {
        &EMBED_ENV_LOCK
    }

    /// Embed requests the mock service received, in order.
    type EmbedRequests = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

    /// Minimal OpenAI-style embed service for tests: answers every request
    /// with the SAME vector — `unit_vec(dimensions, 7)`, using the dimension
    /// that came in the request — and records the request bodies so the test
    /// can assert exactly what the server asked for.
    fn spawn_mock_embed_service() -> (String, EmbedRequests) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind mock embed service");
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests: EmbedRequests = Default::default();
        let seen = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let head_end = loop {
                    let Ok(n) = stream.read(&mut chunk) else {
                        break None;
                    };
                    if n == 0 {
                        break None;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(p + 4);
                    }
                };
                let Some(head_end) = head_end else {
                    continue;
                };
                let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
                let len: usize = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                while buf.len() < head_end + len {
                    let Ok(n) = stream.read(&mut chunk) else {
                        break;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                seen.lock().unwrap().push(body.clone());
                let dim = serde_json::from_str::<Value>(&body)
                    .ok()
                    .and_then(|v| v["dimensions"].as_u64())
                    .unwrap_or(0) as usize;
                let mut vector = vec![0.0f32; dim];
                if dim > 0 {
                    vector[7 % dim] = 1.0;
                }
                let reply = json!({ "data": [{ "embedding": vector }] }).to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n{}",
                    reply.len(),
                    reply
                );
            }
        });
        (base, requests)
    }

    #[tokio::test]
    async fn store_without_embedding_uses_the_embed_service() {
        let _env = embed_env_lock().lock().await;
        let (base, requests) = spawn_mock_embed_service();
        std::env::set_var("SPLATSDB_RECALL_EMBED_URL", format!("{base}/v1/embeddings"));
        // Las dos rutas de almacenamiento: con WAL (un server con --data-dir)
        // y efímera.
        let (app_wal, d) = test_app_with_durability(&temp_wal_dir("server_embedded")).await;
        let (app_mem, _) = test_app().await;
        for (name, app) in [("wal", app_wal), ("ephemeral", app_mem.clone())] {
            let (status, body) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": "prefiere modo oscuro" })),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{name}: {body}");
            assert_eq!(body["status"], "stored", "{name}: {body}");
            assert_eq!(body["embedding_source"], "real", "{name}: {body}");
            assert_eq!(body.get("warning"), None, "{name}: {body}");
            // Lo guardado es el vector del servicio, no un hash del texto: una
            // búsqueda con ese vector exacto da distancia ~0.
            let (status, body) = send(
                app.clone(),
                "POST",
                "/search",
                Some(json!({ "query": "x", "embedding": unit_vec(32, 7), "top_k": 1 })),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{name}: {body}");
            let score = body["results"][0]["score"]
                .as_f64()
                .unwrap_or_else(|| panic!("score is not a number: {body}"));
            assert!(
                score < 1e-4,
                "{name}: the stored vector must be the service's, got distance {score}"
            );
        }
        d.flush().expect("wal flush");

        // Qué le pidió al servicio: el texto, el modelo, la dimensión del
        // store, y SIN prompt (los documentos no llevan prompt).
        let sent = requests.lock().unwrap().clone();
        assert_eq!(
            sent.len(),
            2,
            "one embed request per stored document: {sent:?}"
        );
        for raw in &sent {
            let req: Value = serde_json::from_str(raw).expect("embed request is JSON");
            assert_eq!(req["input"], json!(["prefiere modo oscuro"]), "{req}");
            assert_eq!(req["model"], "embeddinggemma-2", "{req}");
            assert_eq!(req["dimensions"], 32, "the store's dimension: {req}");
            assert_eq!(
                req.get("prompt_name"),
                None,
                "documents carry no prompt: {req}"
            );
        }

        // Con un vector del cliente la respuesta no gana claves y el servicio
        // no es consultado: esa ruta queda igual que antes.
        let (status, body) = send(
            app_mem,
            "POST",
            "/store",
            Some(json!({ "text": "vector del cliente", "embedding": unit_vec(32, 0) })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "stored", "{body}");
        assert_eq!(body.get("embedding_source"), None, "{body}");
        assert_eq!(
            requests.lock().unwrap().len(),
            2,
            "a supplied vector must not call the embed service"
        );
    }

    #[tokio::test]
    async fn store_without_embedding_is_503_when_the_embed_service_is_down() {
        let _env = embed_env_lock().lock().await;
        // URL muerta (puerto 1 → conexión rechazada de inmediato). Sin
        // fallback: 503 y no se guarda nada, en las dos rutas.
        std::env::set_var(
            "SPLATSDB_RECALL_EMBED_URL",
            "http://127.0.0.1:1/v1/embeddings",
        );
        let (app_wal, d) = test_app_with_durability(&temp_wal_dir("no_embed_service")).await;
        let (app_mem, _) = test_app().await;
        for (name, app) in [("wal", app_wal), ("ephemeral", app_mem)] {
            let (status, body) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": "prefiere modo oscuro" })),
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{name}: {body}");
            let err = body["error"].as_str().unwrap_or_default();
            assert!(err.starts_with("no embedding service:"), "{name}: {body}");
            assert_ne!(body["status"], "stored", "{name}: {body}");
            let (_, status_body) = send(app, "POST", "/status", Some(json!({}))).await;
            assert_eq!(
                status_body["n_active"], 0,
                "{name}: a refused store must not reach the store"
            );
        }
        d.flush().expect("wal flush");
    }

    // ── durabilidad: WAL + shard "default" ─────────────────────

    #[tokio::test]
    async fn store_survives_restart_via_wal() {
        let dir = temp_wal_dir("restart");
        {
            let (app, d) = test_app_with_durability(&dir).await;
            for axis in 0..3 {
                let (status, body) = send(
                    app.clone(),
                    "POST",
                    "/store",
                    Some(json!({ "text": format!("vec {axis}"), "embedding": unit_vec(32, axis) })),
                )
                .await;
                assert_eq!(status, StatusCode::OK, "store {axis}: {body}");
                assert_eq!(body["status"], "stored");
                assert_eq!(body["id"], format!("mem_{axis}"));
            }
            // Simular crash: flush explícito (kill -9 pierde lo bufferado,
            // acá forceamos el fsync de los 3 stores) y drop SIN compact.
            d.flush().expect("wal flush");
        }
        // Reinicio: shard vacío + WAL con 3 entradas → dataset idéntico.
        let (app, _) = test_app_with_durability(&dir).await;
        let (status, body) = send(app.clone(), "POST", "/status", Some(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["n_active"], 3, "status tras restart: {body}");
        // El orden de índices sobrevive: el vec del eje 0 es el índice 0.
        let (status, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "vec 0", "embedding": unit_vec(32, 0), "top_k": 1 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["results"][0]["index"], 0,
            "orden de índices tras replay: {body}"
        );
        // El contador next_id continúa (no reusa mem_0..2).
        let (status, body) = send(
            app.clone(),
            "POST",
            "/store",
            Some(json!({ "text": "vec 3", "embedding": unit_vec(32, 3) })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], "mem_3", "next_id sin continuidad: {body}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn compact_persists_shard_and_wal_replay_completes_it() {
        let dir = temp_wal_dir("compact");
        let config = test_config();
        {
            let d = Arc::new(Durability::open(dir.to_str().unwrap()).expect("durability"));
            let mut store = SplatStore::new(config.clone());
            // 2 stores + compact: quedan en el shard "default".
            for axis in 0..2 {
                let arr = Array2::from_shape_vec((1, 32), unit_vec(32, axis)).unwrap();
                assert!(store.add_splat(&arr));
                store.hnsw_sync_incremental();
                let idx = store.n_active() - 1;
                d.log_store(&format!("mem_{axis}"), Some(axis), idx, &unit_vec(32, axis))
                    .expect("wal log");
            }
            let n = d.compact(&store).expect("compact");
            assert_eq!(n, 2);
            // 2 stores post-compact: solo WAL (crash sin shutdown).
            for axis in 2..4 {
                let arr = Array2::from_shape_vec((1, 32), unit_vec(32, axis)).unwrap();
                assert!(store.add_splat(&arr));
                store.hnsw_sync_incremental();
                let idx = store.n_active() - 1;
                d.log_store(&format!("mem_{axis}"), Some(axis), idx, &unit_vec(32, axis))
                    .expect("wal log");
            }
            d.flush().expect("wal flush");
            // drop: el store en memoria muere, disco tiene shard(2)+WAL(2).
        }
        // Reapertura: shard + replay → dataset completo con índices exactos.
        let mut store = SplatStore::new(config.clone());
        load_default_shard(&mut store, dir.to_str().unwrap()).expect("shard load");
        let d = Arc::new(Durability::open(dir.to_str().unwrap()).expect("durability"));
        let (applied, max_auto) = replay_wal(&mut store, &d).expect("replay");
        assert_eq!(
            applied, 2,
            "el replay debe re-aplicar SOLO los post-compact"
        );
        assert_eq!(store.n_active(), 4);
        assert_eq!(max_auto, 4, "siguiente auto_id tras mem_3");
        // Exactitud de índices: el índice 3 es el vec del eje 3.
        let query_vec = unit_vec(32, 3);
        let hits = store.find_neighbors(&ndarray::ArrayView1::from(&query_vec), 1);
        assert_eq!(hits[0].index, 3);
        assert!(hits[0].distance < 1e-3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A03-F09 — compact() con el store vacío NO debe truncar un WAL que
    /// todavía contiene stores no recuperados (p.ej. un replay que salteó
    /// todo por --dim incorrecto): es la única copia duradera.
    #[test]
    fn compact_with_empty_store_keeps_unrecovered_wal_entries() {
        let dir = temp_wal_dir("compact-empty");
        let d = Durability::open(dir.to_str().unwrap()).expect("durability");
        for axis in 0..5 {
            d.log_store(
                &format!("mem_{axis}"),
                Some(axis),
                axis,
                &unit_vec(32, axis),
            )
            .expect("wal log");
        }
        d.flush().expect("wal flush");
        let mut store = SplatStore::new(test_config());
        assert_eq!(store.n_active(), 0);
        let _ = d.compact(&store);

        let store_entries = d
            .wal
            .recover()
            .expect("recover")
            .iter()
            .filter(|e| e.operation == "store")
            .count();
        assert_eq!(
            store_entries, 5,
            "compact truncó el WAL sin haber persistido nada"
        );
        // The entries are still replayable on the next start.
        let (applied, _) = replay_wal(&mut store, &d).expect("replay");
        assert_eq!(applied, 5);
        assert_eq!(store.n_active(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A03-F09 — the other side of the guard: an empty store with nothing
    /// at stake in the WAL keeps the old, harmless behaviour (Ok(0)).
    #[test]
    fn compact_with_empty_store_and_no_store_entries_is_a_noop() {
        let dir = temp_wal_dir("compact-empty-noop");
        let d = Durability::open(dir.to_str().unwrap()).expect("durability");
        d.flush().expect("wal checkpoint");
        let store = SplatStore::new(test_config());
        let n = d.compact(&store).expect("compact without stores");
        assert_eq!(n, 0);
        assert!(!dir.join("data/vectors/default.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A03-F02 — traced at integration time with the LD_PRELOAD fsync
    /// tracer: compact() must fsync the shard temp file, rename, fsync
    /// `data/vectors`, and only THEN rewrite the WAL and fsync `wal/`.
    #[test]
    fn fsync_probe_compact_persists_shard_before_touching_the_wal() {
        let dir = temp_wal_dir("fsync-probe");
        let config = test_config();
        let d = Durability::open(dir.to_str().unwrap()).expect("durability");
        let mut store = SplatStore::new(config);
        for axis in 0..2 {
            let arr = Array2::from_shape_vec((1, 32), unit_vec(32, axis)).unwrap();
            assert!(store.add_splat(&arr));
            store.hnsw_sync_incremental();
            let idx = store.n_active() - 1;
            d.log_store(&format!("mem_{axis}"), Some(axis), idx, &unit_vec(32, axis))
                .expect("wal log");
        }
        assert_eq!(d.compact(&store).expect("compact"), 2);
        assert!(dir.join("data/vectors/default.bin").is_file());
        assert_eq!(d.wal.recover().expect("recover").len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── recap: SAC handoff builder ─────────────────────────────

    #[tokio::test]
    async fn test_recap_endpoint() {
        let (app, _) = test_app().await;
        let actions: Vec<serde_json::Value> = (1..=5u64)
            .map(|i| {
                json!({
                    "seq": i,
                    "kind": if i == 2 { "error" } else { "command" },
                    "text": if i == 2 {
                        "Traceback: connection refused".to_string()
                    } else {
                        format!("pytest tests/test_{i}.py")
                    }
                })
            })
            .collect();
        let (status, body) = send(
            app.clone(),
            "POST",
            "/recap",
            Some(json!({
                "objective": "Fix the netrc auth guard",
                "actions": actions,
                "max_words": 100
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "recap failed: {body}");
        let recap = body["recap"].as_str().expect("recap string");
        assert!(recap.contains("OBJECTIVE:"), "recap must carry objective");
        assert!(
            recap.contains("DEAD ENDS"),
            "error action must land in dead ends"
        );
        assert_eq!(body["actions_in"], 5);
        assert!(body["words"].as_u64().unwrap() <= 100, "budget respected");
        assert!(body["elapsed_ms"].as_f64().unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn test_recap_endpoint_empty_and_deterministic() {
        let (app, _) = test_app().await;
        let (status, body) = send(
            app.clone(),
            "POST",
            "/recap",
            Some(json!({ "objective": "solo objetivo" })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["actions_in"], 0);

        // Same payload twice → byte-identical recap.
        let payload = json!({
            "objective": "Fix leak",
            "actions": [
                {"seq": 1, "kind": "command", "text": "pytest -x"},
                {"seq": 2, "kind": "observation", "text": "1 passed in 0.1s"}
            ]
        });
        let (s1, b1) = send(app.clone(), "POST", "/recap", Some(payload.clone())).await;
        let (s2, b2) = send(app, "POST", "/recap", Some(payload)).await;
        assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
        assert_eq!(b1["recap"], b2["recap"]);
    }

    // ── health ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_endpoint() {
        let (app, _) = test_app().await;
        let (status, body) = send(app, "GET", "/health", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert!(body["version"].is_string());
    }

    // ── store + search roundtrip ────────────────────────────

    #[tokio::test]
    async fn test_store_and_search_roundtrip() {
        let (app, _) = test_app().await;

        // Store 3 orthogonal unit vectors (dim 32).
        for axis in 0..3 {
            let (status, body) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("vec {axis}"), "embedding": unit_vec(32, axis) })),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "store axis {axis} failed: {body}");
            assert_eq!(body["status"], "stored");
            assert!(body["id"].is_string());
        }

        // Status reflects the 3 stored splats.
        let (status, body) = send(app.clone(), "POST", "/status", Some(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["n_active"], 3);
        assert_eq!(body["dimension"], 32);
        assert_eq!(body["has_hnsw"], true);

        // Search with the exact embedding of splat 1 → nearest is index 1.
        let (status, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "exact", "embedding": unit_vec(32, 1), "top_k": 2 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let results = body["results"].as_array().expect("results array");
        assert_eq!(results.len(), 2, "top_k=2 must return 2 results");
        assert_eq!(results[0]["index"], 1, "nearest neighbor must be splat 1");
        let score = results[0]["score"].as_f64().expect("score is number");
        assert!(score < 1e-4, "exact match distance must be ~0, got {score}");
        assert!(results[0]["metadata"].is_null());

        // (fa1e528) Text-only query → 400: el fallback text-hash producía
        // ranking de ruido con 200 OK; el contrato exige embedding.
        let (status, body) = send(
            app,
            "POST",
            "/search",
            Some(json!({ "query": "hello world", "top_k": 3 })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.to_string().contains("missing embedding"),
            "text-only search must be rejected with the reason, got: {body}"
        );
    }

    #[tokio::test]
    async fn test_search_empty_store() {
        let (app, _) = test_app().await;
        let (status, body) =
            send(app, "POST", "/search", Some(json!({ "query": "anything" }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["results"].as_array().unwrap().len(),
            0,
            "empty store must return empty results"
        );
    }

    // ── ?rescore=gaussian accepted ──────────────────────────

    #[tokio::test]
    async fn test_search_rescore_gaussian_param_accepted() {
        let (app, _) = test_app().await;
        send(
            app.clone(),
            "POST",
            "/store",
            Some(json!({ "text": "a", "embedding": unit_vec(32, 0) })),
        )
        .await;

        let (status, body) = send(
            app.clone(),
            "POST",
            "/search?rescore=gaussian",
            Some(json!({ "query": "a", "embedding": unit_vec(32, 0), "top_k": 1 })),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "?rescore=gaussian must be accepted (200), got {status}"
        );
        let results = body["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["index"], 0);

        // Unknown rescore values are tolerated (ignored).
        let (status, _) = send(
            app,
            "POST",
            "/search?rescore=bogus",
            Some(json!({ "query": "a", "embedding": unit_vec(32, 0) })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    // ── ?rescore=gaussian semantics ─────────────────────────

    #[tokio::test]
    async fn test_rescore_gaussian_changes_score_semantics() {
        let (app, _) = test_app().await;

        // Store 5 splats: dense axes 0..5 (dim 32).
        for axis in 0..5 {
            let (status, _) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("g{axis}"), "embedding": unit_vec(32, axis) })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
        }

        let query = {
            // Blend of axes → non-trivial distance profile.
            let mut v = vec![0.0f32; 32];
            v[0] = 0.9;
            v[1] = 0.43589;
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        };

        // Plain L2 search: scores are distances, ascending.
        let (status, plain) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "blend", "embedding": query.clone(), "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let plain_scores: Vec<f64> = plain["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["score"].as_f64().unwrap())
            .collect();
        assert_eq!(plain_scores.len(), 5);
        assert!(
            plain_scores.windows(2).all(|w| w[0] <= w[1]),
            "L2 scores must be ascending distances: {plain_scores:?}"
        );

        // Gaussian rescore: scores are alpha·exp(-kappa·d²), descending, in (0,1].
        let (status, rescored) = send(
            app,
            "POST",
            "/search?rescore=gaussian",
            Some(json!({ "query": "blend", "embedding": query.clone(), "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let rescored_scores: Vec<f64> = rescored["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["score"].as_f64().unwrap())
            .collect();
        assert_eq!(rescored_scores.len(), 5);
        assert!(
            rescored_scores.windows(2).all(|w| w[0] >= w[1]),
            "gaussian scores must be descending: {rescored_scores:?}"
        );
        assert!(
            rescored_scores.iter().all(|s| *s > 0.0 && *s <= 1.000001),
            "gaussian scores must be in (0,1] (alpha=1): {rescored_scores:?}"
        );

        // Same winner (uniform kappa ⇒ same argmax as min-distance).
        assert_eq!(
            plain["results"][0]["index"], rescored["results"][0]["index"],
            "uniform alpha/kappa ⇒ gaussian argmax must equal L2 argmin"
        );
    }

    // ── GPU-path logic (runs on CPU fallback when no CUDA device) ──

    /// Runs `gpu_batch_path` exactly like `serve_batch` does: inside
    /// `spawn_blocking` (it takes a blocking store lock), round-tripping the
    /// snapshot cache.
    async fn run_gpu_batch(
        store: &Arc<Mutex<SplatStore>>,
        snapshot: &mut Option<DatasetSnapshot>,
        queries: &[Vec<f32>],
        ks: &[usize],
        rescore: &[bool],
    ) -> Result<Vec<Vec<SearchResult>>, String> {
        let store2 = store.clone();
        let queries = queries.to_vec();
        let ks = ks.to_vec();
        let rescore = rescore.to_vec();
        let mut snap = snapshot.take();
        let (result, snap_back) = tokio::task::spawn_blocking(move || {
            let r = gpu_batch_path(&store2, &mut snap, &queries, &ks, &rescore);
            (r, snap)
        })
        .await
        .expect("spawn_blocking panicked");
        *snapshot = snap_back;
        result
    }

    #[tokio::test]
    async fn test_gpu_batch_path_snapshot_and_rescore_split() {
        let store = Arc::new(Mutex::new(SplatStore::new(test_config())));
        // Seed 6 splats directly.
        {
            let mut s = store.lock().await;
            let data = Array2::from_shape_fn((6, 32), |(i, j)| if i % 32 == j { 1.0 } else { 0.0 });
            assert!(s.add_splat(&data));
        }

        let mut snapshot = None;
        let q0: Vec<f32> = unit_vec(32, 0);
        let q1: Vec<f32> = unit_vec(32, 3);

        // Mixed batch: one plain L2 query (k=2), one gaussian-rescored (k=2).
        let queries = vec![q0.clone(), q1.clone()];
        let ks = vec![2usize, 2];
        let rescore = vec![false, true];

        let out = run_gpu_batch(&store, &mut snapshot, &queries, &ks, &rescore)
            .await
            .expect("gpu_batch_path must succeed via CPU fallback");

        assert_eq!(out.len(), 2);
        // Row 0: plain L2 — nearest is splat 0 at distance ~0.
        assert_eq!(out[0][0].index, 0);
        assert!(
            out[0][0].score < 1e-4,
            "exact match L2 ~0, got {}",
            out[0][0].score
        );
        assert!(out[0][0].score <= out[0][1].score, "L2 ascending");
        // Row 1: gaussian — nearest is splat 3, score = alpha·exp(0) ≈ 1.
        assert_eq!(out[1][0].index, 3);
        assert!(
            (out[1][0].score - 1.0).abs() < 1e-4,
            "gaussian score of exact match ≈ 1 (alpha=1), got {}",
            out[1][0].score
        );
        assert!(out[1][0].score >= out[1][1].score, "gaussian descending");

        // Snapshot cached: same version → not rebuilt (data ptr identity as a
        // cheap proxy is unreliable; instead verify search still correct and
        // version unchanged after a second call).
        let out2 = run_gpu_batch(&store, &mut snapshot, &queries, &ks, &rescore)
            .await
            .expect("second call must succeed with cached snapshot");
        assert_eq!(out2[0][0].index, 0);
        assert_eq!(out2[1][0].index, 3);
        // Snapshot version tracks the store's data_version (not n_active):
        // unchanged after the second call → snapshot was not rebuilt.
        assert_eq!(
            snapshot.as_ref().unwrap().version,
            store.lock().await.data_version()
        );

        // Store grows → snapshot invalidated and refreshed.
        {
            let mut s = store.lock().await;
            assert!(s.add_splat(&Array2::from_shape_vec((1, 32), unit_vec(32, 5)).unwrap()));
        }
        let out3 = run_gpu_batch(&store, &mut snapshot, &queries, &ks, &rescore)
            .await
            .expect("third call must refresh snapshot");
        assert_eq!(
            snapshot.as_ref().unwrap().version,
            store.lock().await.data_version()
        );
        assert_eq!(out3[0][0].index, 0);
    }

    // ── validation ──────────────────────────────────────────

    #[tokio::test]
    async fn test_embedding_dim_mismatch_rejected() {
        let (app, _) = test_app().await;
        let (status, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "x", "embedding": unit_vec(31, 0) })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.to_string().contains("Embedding dimension mismatch"),
            "error message should mention the mismatch, got: {body}"
        );

        // /store rejects mismatched dims too.
        let (status, _) = send(
            app,
            "POST",
            "/store",
            Some(json!({ "text": "x", "embedding": unit_vec(31, 0) })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // ── micro-batching actually coalesces ───────────────────

    #[tokio::test]
    async fn test_concurrent_searches_coalesce_into_batches() {
        let (app, batches) = test_app().await;

        // Seed the store with 20 splats.
        for axis in 0..20 {
            let mut emb = vec![0.0f32; 32];
            emb[axis % 32] = 1.0;
            if axis >= 32 {
                emb[axis - 32] = 0.5;
            }
            let (status, _) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("v{axis}"), "embedding": emb })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
        }

        let before = batches.load(Ordering::Relaxed);

        // 8 concurrent searches on a current-thread runtime: all sends are
        // enqueued before the worker is polled, so they must land in fewer
        // than 8 batches (in practice exactly 1).
        let mut futs: Vec<BoxFut> = Vec::new();
        for i in 0..8 {
            let app_i = app.clone();
            futs.push(Box::pin(async move {
                send(
                    app_i,
                    "POST",
                    "/search",
                    Some(json!({ "query": format!("q{i}"), "embedding": unit_vec(32, i % 3), "top_k": 4 })),
                )
                .await
            }));
        }
        let replies = futures_join_all(futs).await; // see helper below

        for (i, (status, body)) in replies.iter().enumerate() {
            assert_eq!(*status, StatusCode::OK, "concurrent search {i} failed");
            assert_eq!(body["results"].as_array().unwrap().len(), 4);
        }

        let served = batches.load(Ordering::Relaxed) - before;
        assert!(
            served < 8,
            "8 concurrent searches must coalesce into fewer than 8 batches (got {served})",
        );
        eprintln!("[test] 8 concurrent searches served in {served} batch(es)");
    }

    /// Poor-man's join_all without pulling in the `futures` crate: spawn each
    /// future as a tokio task and collect. On a current-thread runtime this
    /// still enqueues all sends before the worker runs.
    type BoxFut = std::pin::Pin<Box<dyn std::future::Future<Output = (StatusCode, Value)> + Send>>;
    async fn futures_join_all(futs: Vec<BoxFut>) -> Vec<(StatusCode, Value)> {
        let mut handles = Vec::with_capacity(futs.len());
        for f in futs {
            handles.push(tokio::spawn(f));
        }
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.await.expect("task panicked"));
        }
        out
    }

    // ── security: non-finite embeddings (NaN/inf) are rejected ──

    /// Sends a raw JSON string body and returns the status. json!/Value
    /// cannot carry non-finite floats (serde_json serializes them as null),
    /// but the wire text `1e400` decodes to f32::INFINITY inside the
    /// Vec<f32> extractor — so these tests must write the body literally.
    async fn send_raw(app: Router, method: &str, uri: &str, body: String) -> StatusCode {
        let req = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        app.oneshot(req)
            .await
            .expect("oneshot request failed")
            .status()
    }

    #[tokio::test]
    async fn test_store_rejects_non_finite_embedding() {
        let (app, _) = test_app().await;
        let body = format!(
            "{{\"text\":\"x\",\"embedding\":[1e400{}]}}",
            ",0.0".repeat(31)
        );
        let status = send_raw(app, "POST", "/store", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_rejects_non_finite_embedding() {
        let (app, _) = test_app().await;
        let body = format!(
            "{{\"query\":\"x\",\"embedding\":[0.0,1e400{}],\"top_k\":1}}",
            ",0.0".repeat(30)
        );
        let status = send_raw(app, "POST", "/search", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_field_search_rejects_non_finite_embedding() {
        // Minimal FieldData: 2 chunks (dim 2), 1 splat owning both.
        let field = FieldData {
            raw: Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            splats: Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap(),
            offsets: vec![0, 2],
            ids: vec![0, 1],
            radii: vec![1.0],
            chunk_cluster: vec![0, 0],
        };
        let store = Arc::new(Mutex::new(SplatStore::new(test_config())));
        let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let batches_served = Arc::new(AtomicUsize::new(0));
        tokio::spawn(batch_worker(
            batch_rx,
            store.clone(),
            batches_served.clone(),
            false,
        ));
        let state = AppState {
            store,
            next_id: Arc::new(AtomicUsize::new(0)),
            batch_tx,
            batches_served,
            #[cfg(feature = "cuda")]
            gpu: None,
            field: Some(Arc::new(field)),
            corpus_n: 2,
            data_dir: None,
            durability: None,
            auth: AuthConfig::Unauthenticated,
        };
        let app = build_router(state);

        // inf component → 400, never a poisoned sweep.
        let status = send_raw(
            app.clone(),
            "POST",
            "/field-search",
            "{\"embedding\":[1e400,0.0],\"top_k\":1}".to_string(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Finite embedding still works.
        let (status, body) = send(
            app,
            "POST",
            "/field-search",
            Some(json!({ "embedding": [1.0, 0.0], "top_k": 1 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "body: {body}");
        assert!(body["results"].is_array());
    }

    // ── security: CORS preflight must be answered by the CORS layer ──

    #[tokio::test]
    async fn test_cors_preflight_answered_by_cors_layer() {
        let (app, _) = test_app().await;
        let req = HttpRequest::builder()
            .method("OPTIONS")
            .uri("/search")
            .header("Origin", "http://localhost")
            .header("Access-Control-Request-Method", "POST")
            .header("content-type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.expect("oneshot failed");
        assert_ne!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "preflight must not be blocked by auth"
        );
        assert_ne!(
            resp.status(),
            StatusCode::METHOD_NOT_ALLOWED,
            "preflight must be short-circuited by the CORS layer, not reach the router"
        );
        assert!(
            resp.status().is_success(),
            "preflight status: {}",
            resp.status()
        );
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://localhost"),
            "CORS layer must answer the preflight with allow-origin"
        );
    }

    // ── load_field rejects corrupt/oversized headers before allocating ──

    #[test]
    fn test_load_field_rejects_corrupt_headers() {
        let base = std::env::temp_dir().join(format!("splatsdb_field_test_{}", std::process::id()));
        let write_field = |n: u64, k: u64, offsets: &[u64], ids: &[u32], kr: u64| {
            let dir = base.join("field");
            std::fs::create_dir_all(&dir).unwrap();
            let mut members = Vec::new();
            members.extend_from_slice(&n.to_le_bytes());
            members.extend_from_slice(&k.to_le_bytes());
            for o in offsets {
                members.extend_from_slice(&o.to_le_bytes());
            }
            for id in ids {
                members.extend_from_slice(&id.to_le_bytes());
            }
            std::fs::write(dir.join("members.idx"), &members).unwrap();
            let mut radii = Vec::new();
            radii.extend_from_slice(&kr.to_le_bytes());
            radii.extend_from_slice(&1.0f32.to_le_bytes());
            std::fs::write(dir.join("radii.bin"), &radii).unwrap();
        };

        // n over cap → rejected before allocating n*4 bytes.
        write_field(100_000_001, 1, &[0, 1], &[0], 1);
        assert!(load_field(base.to_str().unwrap()).is_none());
        // k over cap → rejected before allocating (k+1)*8 bytes.
        write_field(2, 100_000_001, &[0, 2], &[0, 1], 1);
        assert!(load_field(base.to_str().unwrap()).is_none());
        // offsets[k] != n.
        write_field(2, 1, &[0, 1], &[0, 1], 1);
        assert!(load_field(base.to_str().unwrap()).is_none());
        // non-monotone offsets.
        write_field(2, 2, &[2, 0, 2], &[0, 1], 2);
        assert!(load_field(base.to_str().unwrap()).is_none());
        // member id ≥ n.
        write_field(2, 1, &[0, 2], &[0, 5], 1);
        assert!(load_field(base.to_str().unwrap()).is_none());
        // radii count over cap.
        write_field(2, 1, &[0, 2], &[0, 1], 100_000_001);
        assert!(load_field(base.to_str().unwrap()).is_none());

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_is_loopback_addr() {
        for a in [
            "127.0.0.1",
            "127.0.0.0",
            "127.255.255.255",
            "localhost",
            "LOCALHOST",
            " localhost ",
            "::1",
            "[::1]",
        ] {
            assert!(is_loopback_addr(a), "{a:?} must be loopback");
        }
        for a in ["0.0.0.0", "192.168.1.10", "::", "10.0.0.1", "", "127"] {
            assert!(!is_loopback_addr(a), "{a:?} must NOT be loopback");
        }
    }

    // ─── GPU dataset lifecycle (A03-F05) ──────────────────────────────────

    #[test]
    fn gpu_index_key_is_current_only_for_the_exact_dataset_version() {
        let key = GpuIndexKey { version: 7, n: 16 };
        assert!(key.is_current(7, 16));
        assert!(
            !key.is_current(8, 16),
            "a /store (or replay/compact) bumps data_version: stale"
        );
        assert!(
            !key.is_current(7, 17),
            "same version but another row count: stale"
        );
    }

    /// F06a: after a `/store` the exact GPU index is rebuilt for the new
    /// dataset version and `/search` is served by the GPU fast path again.
    /// Before: the startup key went stale on the first mutation and every
    /// later query took the batch worker (`backend: null`, ~2 ms window).
    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn f06a_gpu_fast_path_rebuilds_once_per_dataset_version_after_a_store() {
        if !crate::gpu::is_cuda_available() {
            eprintln!("[f06a] CUDA not available at runtime: GPU rebuild test skipped");
            return;
        }
        let dim = 32;
        let n0 = 400;
        let mut store = SplatStore::new(test_config());
        let data: Vec<f32> = (0..n0 * dim)
            .map(|i| ((i * 7919) % 1009) as f32 / 1009.0)
            .collect();
        store.add_splat(&Array2::from_shape_vec((n0, dim), data.clone()).unwrap());
        let store = Arc::new(Mutex::new(store));
        let (gpu, n) = GpuExactIndex::startup(&*store.lock().await);
        let gpu = gpu.expect("CUDA index at startup");
        assert_eq!(n, n0);
        assert_eq!(gpu.rebuilds(), 0, "the startup upload is not a rebuild");
        let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let batches_served = Arc::new(AtomicUsize::new(0));
        tokio::spawn(batch_worker(
            batch_rx,
            store.clone(),
            batches_served.clone(),
            true,
        ));
        let state = AppState {
            store,
            next_id: Arc::new(AtomicUsize::new(0)),
            batch_tx,
            batches_served,
            #[cfg(feature = "cuda")]
            gpu: Some(gpu.clone()),
            field: None,
            corpus_n: n,
            data_dir: None,
            durability: None,
            auth: AuthConfig::Unauthenticated,
        };
        let app = build_router(state);
        let old_row: Vec<f32> = data[7 * dim..8 * dim].to_vec();
        let new_row: Vec<f32> = unit_vec(dim, 3).into_iter().map(|v| v * 60.0).collect();

        let (code, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "warm", "embedding": old_row, "top_k": 5 })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{body}");
        assert_eq!(
            body["backend"], "gpu",
            "warm query on the GPU fast path: {body}"
        );

        let (code, body) = send(
            app.clone(),
            "POST",
            "/store",
            Some(json!({ "text": "new", "embedding": new_row })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "/store: {body}");

        // After the mutation: back on the GPU, new vector is its own top-1.
        for i in 0..8 {
            let (code, body) = send(
                app.clone(),
                "POST",
                "/search",
                Some(json!({ "query": "after", "embedding": new_row, "top_k": 5 })),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "{body}");
            assert_eq!(
                body["backend"], "gpu",
                "after /store, query #{i} must be served by the GPU fast path: {body}"
            );
            assert_eq!(
                body["results"][0]["index"].as_u64(),
                Some(n0 as u64),
                "{body}"
            );
            assert_eq!(body["results"][0]["score"].as_f64(), Some(0.0), "{body}");
        }
        // Old rows are still exact on the rebuilt index.
        let (code, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "old", "embedding": old_row, "top_k": 5 })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{body}");
        assert_eq!(body["backend"], "gpu", "{body}");
        assert_eq!(body["results"][0]["index"].as_u64(), Some(7), "{body}");
        assert_eq!(body["results"][0]["score"].as_f64(), Some(0.0), "{body}");
        // 9 queries after ONE mutation -> exactly one rebuild.
        assert_eq!(
            gpu.rebuilds(),
            1,
            "one rebuild per dataset version, not per query"
        );

        // A second mutation -> one more rebuild; again none per query.
        let mut again = unit_vec(dim, 9);
        again[0] = 0.25;
        let (code, body) = send(
            app.clone(),
            "POST",
            "/store",
            Some(json!({ "text": "again", "embedding": again.clone() })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "/store #2: {body}");
        for _ in 0..5 {
            let (code, body) = send(
                app.clone(),
                "POST",
                "/search",
                Some(json!({ "query": "after2", "embedding": again.clone(), "top_k": 5 })),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "{body}");
            assert_eq!(body["backend"], "gpu", "{body}");
            assert_eq!(
                body["results"][0]["index"].as_u64(),
                Some(n0 as u64 + 1),
                "{body}"
            );
            assert_eq!(body["results"][0]["score"].as_f64(), Some(0.0), "{body}");
        }
        assert_eq!(
            gpu.rebuilds(),
            2,
            "one rebuild per dataset version, not per query"
        );
    }

    /// Mutate then search, never restarting: every stored vector must be the
    /// top-1 for its own embedding right away (the batch path refreshes its
    /// snapshot by `data_version`; CPU builds take it for every request).
    #[tokio::test]
    async fn search_sees_every_stored_vector_without_a_restart() {
        let (app, _) = test_app().await;
        let (code, body) = send(app.clone(), "POST", "/status", Some(json!({}))).await;
        assert_eq!(code, StatusCode::OK, "{body}");
        let n0 = body["n_active"].as_u64().unwrap();

        for i in 0..5usize {
            let axis = (i * 7) % 32;
            let emb: Vec<f32> = unit_vec(32, axis).into_iter().map(|v| v * 40.0).collect();
            let (code, body) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("live-{i}"), "embedding": emb })),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "/store #{i}: {body}");
            let (code, body) = send(
                app.clone(),
                "POST",
                "/search",
                Some(json!({ "query": "live", "embedding": emb, "top_k": 1 })),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "{body}");
            assert_eq!(
                body["results"][0]["index"].as_u64(),
                Some(n0 + i as u64),
                "store #{i} is invisible to /search (stale snapshot): {body}"
            );
        }
    }

    // ─── GPU batch shape contract (F03a/F02b borders) ─────────────────────

    fn row_key(row: &[SearchResult]) -> Vec<(usize, f32)> {
        row.iter().map(|r| (r.index, r.score)).collect()
    }

    fn err_of(r: Result<Vec<Vec<SearchResult>>, String>) -> String {
        match r {
            Ok(rows) => panic!("expected a shape error, got {} rows", rows.len()),
            Err(e) => e,
        }
    }

    #[test]
    fn gpu_batch_reassembly_returns_every_query_in_request_order() {
        let rows = reassemble_gpu_rows(
            4,
            &[2, 1, 3],
            &[false, true, false],
            vec![vec![(1, 0.9), (0, 0.5)]],
            vec![
                (vec![2, 0], vec![0.1, 0.4]),
                (vec![3, 9, 1], vec![0.0, 0.0, 0.2]),
            ],
        )
        .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(row_key(&rows[0]), vec![(2, 0.1), (0, 0.4)]); // take(k=2)
        assert_eq!(row_key(&rows[1]), vec![(1, 0.9)]); // take(k=1)
        assert_eq!(row_key(&rows[2]), vec![(3, 0.0), (1, 0.2)]); // index 9 >= n dropped
    }

    #[test]
    fn gpu_batch_reassembly_rejects_a_declined_l2_kernel_instead_of_panicking() {
        // batch_search_cached answered Vec::new() for 2 l2 queries.
        let err = err_of(reassemble_gpu_rows(
            4,
            &[2, 2],
            &[false, false],
            vec![],
            vec![],
        ));
        assert_eq!(err, "gpu batch: l2 kernel returned 0 rows for 2 queries");
        // ... and a partial answer (1 row for 2 queries).
        let err = err_of(reassemble_gpu_rows(
            4,
            &[2, 2],
            &[false, false],
            vec![],
            vec![(vec![1], vec![0.5])],
        ));
        assert_eq!(err, "gpu batch: l2 kernel returned 1 rows for 2 queries");
    }

    #[test]
    fn gpu_batch_reassembly_rejects_a_declined_gaussian_kernel() {
        let err = err_of(reassemble_gpu_rows(
            4,
            &[1, 1],
            &[false, true],
            vec![],
            vec![(vec![1], vec![0.5])],
        ));
        assert_eq!(
            err,
            "gpu batch: gaussian kernel returned 0 rows for 1 queries"
        );
    }

    #[test]
    fn gpu_batch_reassembly_rejects_mismatched_idx_and_dist_lengths() {
        let err = err_of(reassemble_gpu_rows(
            4,
            &[2],
            &[false],
            vec![],
            vec![(vec![1, 2], vec![0.5])],
        ));
        assert_eq!(err, "gpu batch: l2 row 0 has 2 indices for 1 distances");
    }

    #[test]
    fn gpu_batch_reassembly_rejects_k_and_query_counts_that_disagree() {
        let err = err_of(reassemble_gpu_rows(
            4,
            &[2],
            &[false, false],
            vec![],
            vec![],
        ));
        assert_eq!(err, "gpu batch: 1 k values for 2 queries");
    }

    // ─── Auth (A05-F10) ───────────────────────────────────────────────────

    #[test]
    fn api_key_absent_is_unauthenticated_and_a_real_key_is_required() {
        assert_eq!(resolve_api_key(None), Ok(AuthConfig::Unauthenticated));
        assert_eq!(
            resolve_api_key(Some("k".into())),
            Ok(AuthConfig::Required("k".into()))
        );
        // Debug never leaks the key
        assert_eq!(
            format!("{:?}", AuthConfig::Required("secret".into())),
            "Required(<redacted>)"
        );
    }

    #[test]
    fn api_key_present_but_empty_or_blank_is_an_error_not_auth_off() {
        for blank in ["", " ", "\t", "  \n"] {
            let err = resolve_api_key(Some(blank.into())).expect_err("blank key must be rejected");
            assert!(err.contains("SPLATSDB_API_KEY"), "{err}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn api_key_present_but_not_utf8_is_an_error_not_auth_off() {
        use std::os::unix::ffi::OsStringExt;
        let err = resolve_api_key(Some(std::ffi::OsString::from_vec(vec![0xC3, 0x28, 0x41])))
            .expect_err("non-UTF-8 key must be rejected");
        assert!(err.contains("UTF-8"), "{err}");
    }

    #[test]
    fn bind_policy_is_fail_closed_off_loopback_without_a_key() {
        let unauth = AuthConfig::Unauthenticated;
        let key = AuthConfig::Required("k".into());
        // loopback without a key: dev mode preserved
        for a in ["127.0.0.1", "localhost", "::1", "[::1]"] {
            assert!(check_bind_policy(a, &unauth, false).is_ok(), "{a}");
        }
        // non-loopback without a key and without opt-in: refused
        for a in ["0.0.0.0", "::", "192.168.1.10", "127.0.0.1.example.com"] {
            let err = check_bind_policy(a, &unauth, false).expect_err(a);
            assert!(
                err.to_string().contains("SPLATSDB_ALLOW_UNAUTHENTICATED"),
                "{err}"
            );
        }
        // explicit opt-in, or a key: allowed
        assert!(check_bind_policy("0.0.0.0", &unauth, true).is_ok());
        assert!(check_bind_policy("0.0.0.0", &key, false).is_ok());
    }

    #[test]
    fn unauthenticated_opt_in_requires_exactly_one() {
        assert!(unauthenticated_opt_in(Some("1".into())));
        for v in ["0", "true", "yes", " 1", "1 ", ""] {
            assert!(
                !unauthenticated_opt_in(Some(v.into())),
                "{v:?} must NOT opt in"
            );
        }
        assert!(!unauthenticated_opt_in(None));
    }

    #[test]
    fn loopback_check_parses_addresses_instead_of_matching_a_prefix() {
        assert!(is_loopback_addr("127.0.0.1"));
        assert!(is_loopback_addr("[::1]"));
        // a hostname that merely starts with "127." is not loopback
        assert!(!is_loopback_addr("127.0.0.1.example.com"));
        assert!(!is_loopback_addr("127.0.0.1:8080"));
    }

    fn bearer_headers(value: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(v) = value {
            h.insert(
                axum::http::header::AUTHORIZATION,
                axum::http::HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn bearer_token_is_parsed_strictly() {
        assert_eq!(bearer_token(&bearer_headers(Some("Bearer k"))), Some("k"));
        for bad in [
            None,
            Some("Bearer"),
            Some("Bearer "),
            Some("Bearer  "),
            Some("Bearer  k"),
            Some("Bearer k "),
            Some("Bearer k x"),
            Some("bearer k"),
            Some("BEARER k"),
            Some("Basic k"),
            Some("Bearerk"),
        ] {
            assert_eq!(
                bearer_token(&bearer_headers(bad)),
                None,
                "{bad:?} must not yield a token"
            );
        }
    }

    #[test]
    fn constant_time_eq_compares_bytes() {
        assert!(constant_time_eq(b"k", b"k"));
        assert!(!constant_time_eq(b"", b"k"));
        assert!(!constant_time_eq(b"K", b"k"));
        assert!(!constant_time_eq(b"kk", b"k"));
    }

    /// App whose auth is `auth` (everything else as in `test_app`).
    async fn test_app_with_auth(auth: AuthConfig) -> Router {
        let store = Arc::new(Mutex::new(SplatStore::new(test_config())));
        let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let batches_served = Arc::new(AtomicUsize::new(0));
        tokio::spawn(batch_worker(
            batch_rx,
            store.clone(),
            batches_served.clone(),
            false,
        ));
        let state = AppState {
            store,
            next_id: Arc::new(AtomicUsize::new(0)),
            batch_tx,
            batches_served,
            #[cfg(feature = "cuda")]
            gpu: None,
            field: None,
            corpus_n: 0,
            data_dir: None,
            durability: None,
            auth,
        };
        build_router(state)
    }

    async fn send_auth(
        app: &Router,
        method: &str,
        uri: &str,
        body: Value,
        authorization: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut req = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        if let Some(v) = authorization {
            req.headers_mut().insert(
                axum::http::header::AUTHORIZATION,
                axum::http::HeaderValue::from_str(v).unwrap(),
            );
        }
        let resp = app
            .clone()
            .oneshot(req)
            .await
            .expect("oneshot request failed");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body failed");
        let json: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
        };
        (status, json)
    }

    #[tokio::test]
    async fn with_a_key_configured_wrong_or_malformed_authorization_never_authenticates() {
        let app = test_app_with_auth(AuthConfig::Required("k".into())).await;
        for bad in [
            None,
            Some("Bearer "),
            Some("Bearer  "),
            Some("Bearer  k"),
            Some("Bearer k "),
            Some("bearer k"),
            Some("Bearer K"),
            Some("Bearer wrong"),
            Some("Bearer k extra"),
            Some("Basic k"),
        ] {
            let (status, body) = send_auth(&app, "POST", "/status", json!({}), bad).await;
            assert_eq!(
                status,
                StatusCode::UNAUTHORIZED,
                "authorization {bad:?} must be rejected ({body})"
            );
        }
        let (status, _) = send_auth(&app, "POST", "/status", json!({}), Some("Bearer k")).await;
        assert_eq!(status, StatusCode::OK);

        // A rejected /store does not mutate the store.
        let (status, _) = send_auth(
            &app,
            "POST",
            "/store",
            json!({"text": "x"}),
            Some("Bearer nope"),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, body) = send_auth(&app, "POST", "/status", json!({}), Some("Bearer k")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["n_active"], 0, "{body}");

        // /health stays open (liveness probe).
        let resp = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn without_a_key_loopback_dev_mode_passes_through() {
        let app = test_app_with_auth(AuthConfig::Unauthenticated).await;
        let (status, _) = send_auth(&app, "POST", "/status", json!({}), None).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) =
            send_auth(&app, "POST", "/status", json!({}), Some("Bearer anything")).await;
        assert_eq!(status, StatusCode::OK);
    }

    // ─── /recall ──────────────────────────────────────────────────────────

    #[test]
    fn recall_json_leading_detection() {
        assert!(is_json_leading("{\"a\": 1}"));
        assert!(is_json_leading("  [1, 2, 3]"));
        assert!(is_json_leading("\"quoted string\""));
        assert!(!is_json_leading("El deploy falló otra vez"));
        assert!(!is_json_leading(""));
        assert!(!is_json_leading("   "));
    }

    #[test]
    fn recall_tier_selection_matches_hook() {
        let mk = |kind: &str, sid: &str, text: &str, d: f32| JoinedHit {
            kind: kind.to_string(),
            repo: String::new(),
            path: String::new(),
            sid: sid.to_string(),
            text: text.to_string(),
            dist: d,
            meta: serde_json::json!({"session_id": sid}),
        };
        let hits = vec![
            mk("hermes/session", "s1", "texto sesión 1", 0.31),
            mk("hermes/session", "s2", "{\"json\": true}", 0.32), // JSON → pre-gate drop
            mk("hermes/session", "s3", "texto sesión 3", 0.54),
            mk("hermes/session", "s4", "fuera de tier", 0.61), // > 0.55 → no entra
            mk("repo", "c1", "[1, 2, 3]", 0.65),               // JSON code → drop
            mk("repo", "c2", "src/main.rs: ruta valida", 0.68),
            mk("repo", "c3", "src/other.rs lejos", 0.75), // > 0.70 → no entra
        ];
        let (kept, sess_json, code_json) = select_recall_tiers(hits, 0.55, 3, 0.70, 1);
        // bucket-best gate (hook semantics): sess best 0.31 ≤ 0.55 → take 3
        // UNCONDITIONALLY (s4 at 0.61 rides along); code best 0.68 ≤ 0.70 → 1
        assert_eq!(kept.len(), 4);
        assert_eq!(sess_json, 1);
        assert_eq!(code_json, 1);
        // sessions first (best-first by dist), then the single code hit
        assert_eq!(kept[0].sid, "s1");
        assert_eq!(kept[1].sid, "s3");
        assert_eq!(kept[2].sid, "s4");
        assert_eq!(kept[3].sid, "c2");
        assert_eq!(kept[3].kind, "repo");
    }

    #[test]
    fn recall_gate_is_bucket_best_not_per_hit() {
        // Regression (A/B replay 20-nov): pool 0.547/0.551/0.565 — the hook
        // gates on the BEST hit and injects all 3; per-hit thresholds would
        // inject only the first. Encode the deployed behavior.
        let mk = |sid: &str, d: f32| JoinedHit {
            kind: "hermes/session".to_string(),
            repo: String::new(),
            path: String::new(),
            sid: sid.to_string(),
            text: format!("texto {sid}"),
            dist: d,
            meta: serde_json::json!({"session_id": sid}),
        };
        let hits = vec![mk("a", 0.547), mk("b", 0.551), mk("c", 0.565)];
        let (kept, _, _) = select_recall_tiers(hits, 0.55, 3, 0.70, 1);
        assert_eq!(kept.len(), 3, "ranks 2..n ride along once the gate passes");
        // and the cut case: best ABOVE the band → nothing rides along
        let hits = vec![mk("a", 0.56), mk("b", 0.57), mk("c", 0.58)];
        let (kept, _, _) = select_recall_tiers(hits, 0.55, 3, 0.70, 1);
        assert!(kept.is_empty(), "gate on best hit cuts the whole bucket");
    }

    #[test]
    fn recall_tier_limits_enforced() {
        let mk = |i: usize| JoinedHit {
            kind: "hermes/session".to_string(),
            repo: String::new(),
            path: String::new(),
            sid: format!("s{i}"),
            text: format!("texto {i}"),
            dist: 0.40 + i as f32 * 0.01,
            meta: serde_json::json!({"session_id": format!("s{i}")}),
        };
        // 6 candidates all ≤ 0.55 → only the 3 closest survive (sess_n=3)
        let hits: Vec<_> = (0..6).map(mk).collect();
        let (kept, sess_json, _) = select_recall_tiers(hits, 0.55, 3, 0.70, 1);
        assert_eq!(kept.len(), 3);
        assert_eq!(sess_json, 0);
        assert_eq!(kept[0].sid, "s0");
        assert_eq!(kept[2].sid, "s2");
    }

    #[test]
    fn recall_sess_window_five_measured_default() {
        // 21-nov LongMemEval-S (500 Q, judge GLM): recall@contexto 53.2%→63.1%
        // con ventana 3→5 (multi-session +13.4 pts, preference +23.4 pts).
        // El default del gate es decisión MEDIDA, no estética; rollback por
        // env SPLATSDB_RECALL_SESS_N=3 (el handler lo respeta).
        assert_eq!(RECALL_SESS_N_DEFAULT, 5);
        let mk = |i: usize| JoinedHit {
            kind: "hermes/session".to_string(),
            repo: String::new(),
            path: String::new(),
            sid: format!("s{i}"),
            text: format!("texto {i}"),
            dist: 0.30 + i as f32 * 0.01,
            meta: serde_json::json!({"session_id": format!("s{i}")}),
        };
        let hits: Vec<_> = (0..7).map(mk).collect();
        let sess_n = recall_env_usize("SPLATSDB_RECALL_SESS_N", RECALL_SESS_N_DEFAULT);
        let (kept, _, _) = select_recall_tiers(hits, 0.55, sess_n, 0.70, 1);
        assert_eq!(kept.len(), 5, "ventana 5 con todos los hits ≤ 0.55");
        assert_eq!(kept[4].sid, "s4", "best-first: los 5 más cercanos");
    }

    #[test]
    fn recall_embed_fail_closed_no_simcos() {
        // Dead URL (port 1 → immediate connection refused): the embed unit
        // returns Err and there is NO SimCos fallback in this path
        // (fa1e528 parity — a hash-text embedding ranks noise as success).
        let _env = embed_env_lock().blocking_lock();
        std::env::set_var(
            "SPLATSDB_RECALL_EMBED_URL",
            "http://127.0.0.1:1/v1/embeddings",
        );
        let r = fetch_recall_embedding("texto de prueba", 768);
        assert!(r.is_err(), "dead embed URL must fail");
        let err = r.unwrap_err().to_lowercase();
        assert!(!err.contains("simcos"));
    }

    #[tokio::test]
    async fn recall_fail_closed_without_embed_service() {
        // Store populated so the handler actually reaches the embed step
        // (empty stores short-circuit before embedding). Store dim is 32
        // (test_app default — same as the /store unit_vec(32, _) tests).
        // Dead embed URL → 503 with a reason, never a silent success /
        // SimCos fallback (fa1e528 parity in /recall).
        let _env = embed_env_lock().lock().await;
        std::env::set_var(
            "SPLATSDB_RECALL_EMBED_URL",
            "http://127.0.0.1:1/v1/embeddings",
        );
        let (app, _batches) = test_app().await;
        let (s, body) = send(
            app.clone(),
            "POST",
            "/store",
            Some(json!({ "text": "anchor", "embedding": unit_vec(32, 0) })),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "store anchor failed: {body}");
        let (status, body) = send(
            app,
            "POST",
            "/recall",
            Some(json!({ "text": "¿cómo quedó lo del túnel?" })),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        let body_str = body.to_string().to_lowercase();
        assert!(!body_str.contains("simcos"));
    }

    #[tokio::test]
    async fn recall_empty_text_is_400() {
        let (app, _batches) = test_app().await;
        let (status, _) = send(
            app,
            "POST",
            "/recall",
            Some(serde_json::json!({ "text": "   " })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn recall_oversize_text_is_413() {
        let (app, _batches) = test_app().await;
        let big = "x".repeat(MAX_TEXT_LEN + 1);
        let (status, _) = send(
            app,
            "POST",
            "/recall",
            Some(serde_json::json!({ "text": big })),
        )
        .await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    }
    // ── F02b: ciclo de vida storage ───────────────────────────

    /// Data-dir scratch único (TMPDIR).
    fn f02b_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "splatdb-f02b-unit-{}-{}-{}",
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
    fn f02b_compact_refuses_to_overwrite_a_shard_changed_outside_the_server() {
        let dir = f02b_dir("guard");
        let persist = SplatsDBPersistence::new(dir.to_str().unwrap(), false).unwrap();
        let original = Array2::from_shape_fn((3, 32), |(i, j)| (i * 32 + j) as f32);
        persist.save_vectors(&original, "default").unwrap();
        // mismo orden que run_server: shard -> WAL -> replay
        let mut store = SplatStore::new(test_config());
        load_default_shard(&mut store, dir.to_str().unwrap()).unwrap();
        let d = Durability::open(dir.to_str().unwrap()).unwrap();
        replay_wal(&mut store, &d).unwrap();

        // mutación externa válida del shard mientras el server corre
        let external = Array2::from_shape_fn((5, 32), |(i, j)| 100.0 + (i * 32 + j) as f32);
        persist.save_vectors(&external, "default").unwrap();
        let shard = dir.join("data").join("vectors").join("default.bin");
        let external_bytes = std::fs::read(&shard).unwrap();

        // un store aceptado y el compact del shutdown
        d.log_store("mem_0", Some(0), 3, &[0.5f32; 32]).unwrap();
        let err = d.compact(&store).expect_err("compact must refuse");
        assert!(err.to_string().contains("cambió en disco"), "{err}");
        assert_eq!(
            std::fs::read(&shard).unwrap(),
            external_bytes,
            "the external shard must survive byte for byte"
        );
        assert_eq!(
            d.wal
                .recover()
                .unwrap()
                .iter()
                .filter(|e| e.operation == "store")
                .count(),
            1,
            "the accepted store must stay in the WAL"
        );

        // reapertura sobre ese data-dir: fail-closed, no se adivina por idx
        let err = Durability::open(dir.to_str().unwrap())
            .err()
            .expect("reopen must fail closed");
        assert!(err.to_string().contains("cambió en disco"), "{err}");
        assert_eq!(std::fs::read(&shard).unwrap(), external_bytes);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn f02b_compact_keeps_the_wal_while_entries_could_not_be_applied() {
        let dir = f02b_dir("unapplied");
        let d = Durability::open(dir.to_str().unwrap()).unwrap();
        // una entrada cuya dimensión no coincide con el store no se puede aplicar
        d.log_store("wide", None, 0, &[0.25f32; 64]).unwrap();
        let mut store = SplatStore::new(test_config());
        let (applied, _) = replay_wal(&mut store, &d).unwrap();
        assert_eq!(applied, 0);
        // una entrada aplicable posterior y un store no vacío
        d.log_store("ok", None, 0, &[0.75f32; 32]).unwrap();
        assert!(store.add_splat(&Array2::from_shape_fn((1, 32), |(_, j)| j as f32)));
        let err = d
            .compact(&store)
            .expect_err("compact must not truncate the WAL");
        assert!(err.to_string().contains("no se aplicaron"), "{err}");
        let ops: Vec<String> = d
            .wal
            .recover()
            .unwrap()
            .iter()
            .map(|e| e.operation.clone())
            .collect();
        assert_eq!(
            ops.iter().filter(|o| *o == "store").count(),
            2,
            "WAL untouched, operations: {ops:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── F02b: contrato durable explícito de /store y /status ───

    async fn f02b_call(
        app: &Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, String) {
        let req = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.map(|b| b.to_string()).unwrap_or_default()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Router con un WAL real sobre `dir` (sin batch worker: /store y
    /// /status no lo usan).
    fn app_with_wal(dir: &std::path::Path) -> (Router, Arc<Durability>) {
        let d = Arc::new(Durability::open(dir.to_str().unwrap()).expect("durability open"));
        let mut store = SplatStore::new(test_config());
        let _ = load_default_shard(&mut store, dir.to_str().unwrap());
        replay_wal(&mut store, &d).expect("wal replay");
        let (batch_tx, _batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let app = build_router(AppState {
            store: Arc::new(Mutex::new(store)),
            next_id: Arc::new(AtomicUsize::new(0)),
            batch_tx,
            batches_served: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "cuda")]
            gpu: None,
            field: None,
            corpus_n: 0,
            data_dir: Some(dir.to_str().unwrap().to_string()),
            durability: Some(d.clone()),
            auth: AuthConfig::Unauthenticated,
        });
        (app, d)
    }

    #[tokio::test]
    async fn f02b_store_reports_process_crash_mode_and_group_commit() {
        let dir = f02b_dir("durable-flag");
        let (app, _d) = app_with_wal(&dir);
        let (status, body) = f02b_call(&app, "POST", "/status", None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["persistence"], "wal", "{body}");
        assert_eq!(v["durable"], true, "nothing accepted yet: {body}");
        // group commit: las primeras 31 están aceptadas en el WAL pero sin fsync
        for i in 1..32 {
            let (status, body) = f02b_call(
                &app,
                "POST",
                "/store",
                Some(json!({"text": format!("doc {i}"), "embedding": unit_vec(32, i % 32)})),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let v: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["status"], "stored", "{body}");
            assert_eq!(v["durable"], true, "store #{i} reached the WAL: {body}");
            assert_eq!(v["crash_durable"], true, "{body}");
            assert_eq!(v["power_loss_durable"], false, "{body}");
            assert_eq!(v["durability_mode"], "process-crash", "{body}");
        }
        let (_, body) = f02b_call(&app, "POST", "/status", None).await;
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["durable"], true, "{body}");
        assert_eq!(v["crash_durable"], true, "{body}");
        assert_eq!(v["power_loss_durable"], false, "{body}");
        assert_eq!(v["wal_pending"], 31, "{body}");
        // la 32ª dispara el checkpoint: sync_all terminó ANTES de este 200
        let (status, body) = f02b_call(
            &app,
            "POST",
            "/store",
            Some(json!({"text": "doc 32", "embedding": unit_vec(32, 0)})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["durable"], true, "{body}");
        let (_, body) = f02b_call(&app, "POST", "/status", None).await;
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["durable"], true, "{body}");
        assert_eq!(v["wal_pending"], 0, "{body}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn f02b_store_is_503_when_the_wal_write_fails() {
        let dir = f02b_dir("wal-fail");
        let (app, d) = app_with_wal(&dir);
        let (status, body) = f02b_call(
            &app,
            "POST",
            "/store",
            Some(json!({"text": "ok", "embedding": unit_vec(32, 1)})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // fallo real en el fd: la próxima entrada se corta a los 0 bytes y se
        // reporta como write fallido (mismo inyector que los tests del WAL).
        d.wal.debug_fail_next_entry_after(0);
        let (status, body) = f02b_call(
            &app,
            "POST",
            "/store",
            Some(json!({"text": "fails", "embedding": unit_vec(32, 2)})),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert!(body.contains("WAL write failed"), "{body}");
        assert!(!body.contains("\"stored\""), "{body}");
        // el WAL sigue terminando en la última entrada completa y acepta la siguiente
        let (status, body) = f02b_call(
            &app,
            "POST",
            "/store",
            Some(json!({"text": "after", "embedding": unit_vec(32, 3)})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            d.wal
                .recover()
                .unwrap()
                .iter()
                .filter(|e| e.operation == "store")
                .count(),
            2,
            "only the two accepted entries"
        );
        let (_, body) = f02b_call(&app, "POST", "/status", None).await;
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["persistence"], "wal", "{body}");
        assert_eq!(v["durable"], true, "{body}");
        assert_eq!(v["crash_durable"], true, "{body}");
        assert_eq!(v["power_loss_durable"], false, "{body}");
        assert_eq!(v["wal_pending"], 2, "{body}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn f02b_ephemeral_mode_is_visible_and_never_durable() {
        // `durability: None` es el modo efímero explícito (tests / lib);
        // `serve` con --data-dir nunca llega acá (arranque fail-closed).
        let (app, _batches) = test_app().await;
        let (status, body) = f02b_call(
            &app,
            "POST",
            "/store",
            Some(json!({"text": "ephemeral", "embedding": unit_vec(32, 4)})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "stored", "{body}");
        assert_eq!(v["persistence"], "ephemeral", "{body}");
        assert_eq!(v["durable"], false, "{body}");
        let (_, body) = f02b_call(&app, "POST", "/status", None).await;
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["persistence"], "ephemeral", "{body}");
        assert_eq!(v["durable"], false, "{body}");
        assert_eq!(v["wal_pending"], 0, "{body}");
    }
}

#[cfg(test)]
mod f02e1_energy_reload_tests {
    use super::*;
    use ndarray::Array2;

    fn dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir()
            .join("F02e1-unit")
            .join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn new_store() -> SplatStore {
        SplatStore::new(SplatsDBConfig {
            latent_dim: 4,
            max_splats: 16,
            ..Default::default()
        })
    }

    fn mu_of(rows: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, 4), |(i, j)| (i * 4 + j) as f32 - 2.0)
    }

    /// The layout `persistence::save_energy_state` writes — through the real
    /// persistence layer.
    fn write_energy(data_dir: &std::path::Path, mu: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
        let alpha: Vec<f32> = (0..mu.nrows()).map(|i| 0.5 + 0.1 * i as f32).collect();
        let kappa: Vec<f32> = (0..mu.nrows()).map(|i| 1.0 + 2.0 * i as f32).collect();
        let persist = crate::storage::persistence::SplatsDBPersistence::with_backend(
            data_dir.to_str().unwrap(),
            crate::storage::persistence::StorageBackend::Sqlite,
            false,
        )
        .unwrap();
        persist.save_energy_state(mu, &alpha, &kappa).unwrap();
        (alpha, kappa)
    }

    fn assert_init(store: &SplatStore, n: usize) {
        let (init_alpha, init_kappa) = (
            SplatsDBConfig::default().init_alpha as f32,
            SplatsDBConfig::default().init_kappa as f32,
        );
        assert!(store
            .get_alpha()
            .unwrap()
            .iter()
            .take(n)
            .all(|a| *a == init_alpha));
        assert!(store
            .get_kappa()
            .unwrap()
            .iter()
            .take(n)
            .all(|k| *k == init_kappa));
    }

    #[test]
    fn serve_loads_the_energy_state_with_the_shard() {
        let d = dir("full");
        let mu = mu_of(3);
        let (alpha, kappa) = write_energy(&d, &mu);
        let mut store = new_store();
        assert_eq!(
            load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu).unwrap(),
            3
        );
        assert_eq!(store.get_alpha().unwrap().as_slice().unwrap(), &alpha[..]);
        assert_eq!(store.get_kappa().unwrap().as_slice().unwrap(), &kappa[..]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn serve_loads_a_prefix_and_keeps_the_init_for_appended_rows() {
        let d = dir("prefix");
        let mu = mu_of(5);
        let (alpha, kappa) = write_energy(&d, &mu.slice(ndarray::s![..3, ..]).to_owned());
        let mut store = new_store();
        assert_eq!(
            load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu).unwrap(),
            5
        );
        assert_eq!(
            &store.get_alpha().unwrap().as_slice().unwrap()[..3],
            &alpha[..]
        );
        assert_eq!(
            &store.get_kappa().unwrap().as_slice().unwrap()[..3],
            &kappa[..]
        );
        let (init_alpha, init_kappa) = (
            SplatsDBConfig::default().init_alpha as f32,
            SplatsDBConfig::default().init_kappa as f32,
        );
        assert!(store
            .get_alpha()
            .unwrap()
            .iter()
            .skip(3)
            .all(|a| *a == init_alpha));
        assert!(store
            .get_kappa()
            .unwrap()
            .iter()
            .skip(3)
            .all(|k| *k == init_kappa));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn serve_ignores_a_foreign_energy_state() {
        let d = dir("foreign");
        let mu = mu_of(3);
        write_energy(&d, &mu_of(3).mapv(|x| x + 100.0));
        let mut store = new_store();
        assert_eq!(
            load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu).unwrap(),
            3
        );
        assert_init(&store, 3);
        // a landscape of another dimension
        let d2 = dir("foreign-dim");
        write_energy(&d2, &Array2::zeros((3, 5)));
        let mut store = new_store();
        assert_eq!(
            load_shard_with_energy(&mut store, d2.to_str().unwrap(), &mu).unwrap(),
            3
        );
        assert_init(&store, 3);
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }

    #[test]
    fn serve_fails_closed_on_a_corrupt_energy_state() {
        let d = dir("corrupt");
        let mu = mu_of(3);
        write_energy(&d, &mu);
        let path = crate::loaders::EnergyLandscape::path_in(&d);
        let mut store = new_store();
        let bytes = std::fs::read(&path).unwrap();
        // truncated
        std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
        let err = load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu)
            .unwrap_err()
            .to_string();
        assert!(err.contains("bytes"), "{err}");
        // trailing garbage
        let mut trailing = bytes.clone();
        trailing.push(0);
        std::fs::write(&path, &trailing).unwrap();
        let mut store = new_store();
        assert!(load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu).is_err());
        // hostile header (rows = u64::MAX)
        let mut bomb = bytes.clone();
        bomb[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &bomb).unwrap();
        let mut store = new_store();
        let err = load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Allocation too large"), "{err}");
        // NaN alpha
        let mut nan = bytes.clone();
        let o = 16 + 3 * 4 * 4 + 8;
        nan[o..o + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&path, &nan).unwrap();
        let mut store = new_store();
        let err = load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu)
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-finite"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn serve_without_energy_state_loads_mu_only() {
        let d = dir("absent");
        let mu = mu_of(3);
        let mut store = new_store();
        assert_eq!(
            load_shard_with_energy(&mut store, d.to_str().unwrap(), &mu).unwrap(),
            3
        );
        assert_init(&store, 3);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn tie_snapshot() -> (DatasetSnapshot, Vec<f32>) {
        let n = 300;
        let row = |i: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; 32];
            match i {
                0 => v[3] = 1.0,
                1 => v[0] = 2.0,
                2 => v[0] = -2.0,
                3 => v[1] = 2.0,
                256 => v[1] = -2.0,
                257 => v[2] = 2.0,
                258 => v[2] = -2.0,
                _ => v[7 + i % 24] = 4.0 + 0.03125 * i as f32,
            }
            v
        };
        let mut alpha = vec![1.0f32; n];
        alpha[257] = 1000.0;
        (
            DatasetSnapshot {
                version: 0,
                n,
                dim: 32,
                data: (0..n).flat_map(row).collect(),
                alpha,
                kappa: vec![1.0f32; n],
            },
            {
                let mut q = vec![0.0f32; 32];
                q[31] = 0.5;
                q
            },
        )
    }

    /// The kernel's thread-major order over the tie corpus.
    fn kernel_row(scores: &[f32], descending: bool, w: usize) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        all.sort_by(|a, b| {
            let s = a.1.total_cmp(&b.1);
            let s = if descending { s.reverse() } else { s };
            s.then((a.0 % 256).cmp(&(b.0 % 256))).then(a.0.cmp(&b.0))
        });
        all.truncate(w);
        all
    }

    #[test]
    fn gpu_batch_rows_become_contractual() {
        let (snap, q) = tie_snapshot();
        let scores: Vec<f32> = (0..snap.n)
            .map(|i| {
                let r = &snap.data[i * 32..(i + 1) * 32];
                r.iter()
                    .zip(&q)
                    .map(|(a, b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum()
            })
            .collect();
        // what the kernels return for the batch window (k=5 → 6 pairs)
        let mut gauss_out = vec![kernel_row(
            &scores
                .iter()
                .zip(&snap.alpha)
                .map(|(d, a)| a * (-d).exp())
                .collect::<Vec<_>>(),
            true,
            6,
        )];
        let mut l2_out = vec![(
            kernel_row(&scores, false, 6)
                .iter()
                .map(|p| p.0)
                .collect::<Vec<_>>(),
            kernel_row(&scores, false, 6)
                .iter()
                .map(|p| p.1)
                .collect::<Vec<_>>(),
        )];
        canonicalize_gpu_rows(
            &snap,
            &[q.clone(), q.clone()],
            &[5, 5],
            &[true, false],
            &mut gauss_out,
            &mut l2_out,
        );
        assert_eq!(
            gauss_out[0].iter().map(|p| p.0).collect::<Vec<_>>(),
            vec![257, 0, 1, 2, 3]
        );
        assert_eq!(l2_out[0].0.clone(), vec![0, 1, 2, 3, 256]);
        assert!((l2_out[0].1[1] - 4.25).abs() < 1e-5, "{:?}", l2_out[0].1);
        // malformed shapes are left for reassemble_gpu_rows to reject
        let mut bad = vec![(vec![0usize, 1], vec![1.0f32])];
        canonicalize_gpu_rows(
            &snap,
            std::slice::from_ref(&q),
            &[5],
            &[false],
            &mut [],
            &mut bad,
        );
        assert_eq!(bad[0].0.len(), 2);
        assert_eq!(bad[0].1.len(), 1);
        // host fallbacks are contractual on their own
        let exact = host_exact_l2(&snap.data, snap.n, 32, &q, 5);
        assert_eq!(
            exact.iter().map(|p| p.0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 256]
        );
        assert_eq!(gpu_window(5, 300), 6);
        assert_eq!(gpu_window(5, 6), 6);
        assert_eq!(gpu_window(5, 4), 4);
        assert_eq!(gpu_window(32, 300), 32);
        assert_eq!(gpu_window(40, 300), 40);
    }
}
