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
//!   - `simple_hash_embedding` hashes the text ONCE and derives the D
//!     dimensions from a SplitMix64 PRNG (O(1) hashes instead of O(D)).
//!
//! Security:
//!   - API key authentication via SPLATSDB_API_KEY env var (optional)
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use tower_http::cors::CorsLayer;

use crate::config::SplatsDBConfig;
use crate::splats::SplatStore;

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
    /// Índice exacto en GPU (kernel PTX l2_topk). topk_search es &self → se
    /// comparte sin Mutex. Presente solo con serve --gpu y feature cuda.
    #[cfg(feature = "cuda")]
    pub gpu: Option<Arc<crate::gpu::cuda_kernel::GpuIndex>>,
    /// Campo splat (celdas con radio) para /field-search exacto. None si no
    /// hay field/ persistido en el data-dir.
    pub field: Option<Arc<FieldData>>,
    /// Tamaño del corpus que atiende /search: field.raw si hay field/ (los
    /// ids de resultados son chunks), o el shard cargado (los ids son splats).
    pub corpus_n: usize,
    /// Directorio de datos del server (flag --data-dir). /recall lo usa para
    /// abrir {data_dir}/sidecar.db en mode=ro y loggear fires. None en tests.
    pub data_dir: Option<String>,
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

#[derive(Serialize)]
pub struct StoreResponse {
    pub id: String,
    pub status: String,
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
    Json(StatusResponse {
        n_active: store.n_active(),
        max_splats: store.max_splats(),
        dimension: stats.embedding_dim,
        has_hnsw: store.has_hnsw(),
        has_lsh: store.has_lsh(),
        has_quantization: store.has_quantization(),
    })
}

async fn store_memory(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> Result<(StatusCode, Json<StoreResponse>), (StatusCode, String)> {
    // Input validation
    if req.text.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Text exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }

    let dim = {
        let store = state.store.lock().await;
        store.get_statistics().embedding_dim
    }; // lock dropped here — embedding computation doesn't need the mutex

    // Validate user-supplied embedding dimension
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
        // NaN/inf (e.g. `1e400` parsed as inf) would poison the index.
        if !all_finite(emb) {
            return Err((
                StatusCode::BAD_REQUEST,
                "Embedding contains non-finite values (NaN/inf)".into(),
            ));
        }
    }

    let embedding = req
        .embedding
        .unwrap_or_else(|| simple_hash_embedding(&req.text, dim));

    let arr = Array2::from_shape_vec((1, dim), embedding)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Bad embedding shape: {e}")))?;

    let id_num = state.next_id.fetch_add(1, Ordering::SeqCst);
    let id = req.id.unwrap_or_else(|| format!("mem_{id_num}"));

    // Wrap CPU-intensive work in spawn_blocking to avoid blocking the tokio runtime
    let store = state.store.clone();
    let added = tokio::task::spawn_blocking(move || {
        let mut store = store.blocking_lock();
        let added = store.add_splat(&arr);
        if added {
            // Incrementally insert the new vector into HNSW (no full rebuild)
            store.hnsw_sync_incremental();
        }
        added
    })
    .await
    .map_err(|e| {
        // Log the real cause server-side; the client gets a generic 500
        // (JoinError strings can carry panic payloads).
        eprintln!("[splatsdb] /store: background task failed: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, "internal error".into())
    })?;

    if !added {
        return Err((StatusCode::INSUFFICIENT_STORAGE, "Store full".into()));
    }

    Ok((
        StatusCode::OK,
        Json(StoreResponse {
            id,
            status: "stored".into(),
        }),
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

    let (dim, n_active) = {
        let store = state.store.lock().await;
        let dim = store.get_statistics().embedding_dim;
        let n = store.n_active();
        (dim, n)
    }; // lock dropped — the hot path below never re-acquires it

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
    // Exacto, sin store mutex (índice compartido &self). Rescore gaussiano y
    // query de texto (hash embedding) siguen por el camino CPU.
    #[cfg(feature = "cuda")]
    if let Some(gpu) = &state.gpu {
        if !rescore_gaussian {
            if let Some(mut per_query) = gpu.topk_search(&embedding, 1, k, "l2") {
                if let Some((idxs, dists)) = per_query.pop() {
                    let mut pairs: Vec<(usize, f32)> = idxs.into_iter().zip(dists).collect();
                    pairs
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    let mut results = pairs
                        .into_iter()
                        .map(|(i, d)| SearchResult {
                            index: i,
                            score: d,
                            metadata: None,
                            confidence: None,
                            topic: None,
                        })
                        .collect::<Vec<_>>();
                    enrich_topics(&state, &mut results, true);
                    return Ok(Json(SearchResponse {
                        results,
                        backend: Some("gpu".into()),
                    }));
                }
            }
            eprintln!("[splatsdb] GPU topk devolvió None; query cae a CPU");
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
    enrich_topics(&state, &mut results, true);

    Ok(Json(SearchResponse {
        results,
        backend: None,
    }))
}

/// Enriquece resultados con tópico (splat) y confianza espacial
/// (1 − d/R del tópico). Solo llamado con scores de distancia L2 reales
/// (no con gaussian rescore ni texto-hash). `squared` = score es d² (GPU).
fn enrich_topics(state: &AppState, results: &mut [SearchResult], squared: bool) {
    let Some(field) = &state.field else {
        return;
    };
    // Si el corpus activo ES field.raw, los ids del índice son chunks:
    // topic = splat al que pertenece el chunk. Si el corpus son los splats
    // (μ), el id ya es un tópico y la confianza mide qué tan cerca del
    // centro de ese tópico cayó la query.
    let ids_are_chunks = field.raw.nrows() == state.corpus_n;
    for r in results.iter_mut() {
        let (c, radius) = if ids_are_chunks && r.index < field.chunk_cluster.len() {
            let c = field.chunk_cluster[r.index] as usize;
            (c, field.radii[c])
        } else {
            (r.index, field.radii.get(r.index).copied().unwrap_or(0.0))
        };
        r.topic = Some(c as u32);
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

/// Crudos + splats + membresías CSR + radios, persistidos por
/// `field_cmds::persist_field` tras cada ingest.
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
    order.sort_by(|a, b| {
        d_mu[*a]
            .partial_cmp(&d_mu[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
                top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
    order.sort_by(|a, b| {
        d_mu[*a]
            .partial_cmp(&d_mu[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
    top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
const RECALL_SESS_N_DEFAULT: usize = 3;
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

/// OpenAI-style embed request to the Hermes embed service (:8860) with the
/// asymmetric Retrieval-query prompt. The vector is NOT re-normalized: the
/// hub's d² bands were calibrated against this service's raw output.
fn fetch_recall_embedding(text: &str, dim: usize) -> Result<Vec<f32>, String> {
    let url = recall_env_str("SPLATSDB_RECALL_EMBED_URL", RECALL_EMBED_URL_DEFAULT);
    let body = serde_json::json!({
        "input": [text],
        "model": RECALL_EMBED_MODEL,
        "prompt_name": recall_env_str("SPLATSDB_RECALL_EMBED_PROMPT", RECALL_EMBED_PROMPT_DEFAULT),
        "dimensions": dim,
    })
    .to_string();
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
    sess.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    code.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
        if let Some(mu) = s.get_mu() {
            let dim = mu.ncols();
            let data = mu.iter().copied().collect::<Vec<f32>>();
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
        let kmax_g = gauss_ks.iter().copied().max().unwrap_or(1).min(snap.n);
        crate::gpu::gaussian_topk_scores_cached(
            snap.version,
            &gauss_flat,
            &data_view,
            &snap.alpha,
            &snap.kappa,
            n_gauss,
            kmax_g,
        )
        .ok_or_else(|| "gaussian_topk_scores failed".to_string())?
    } else {
        Vec::new()
    };

    let l2_out = if n_l2 > 0 {
        let kmax_l = l2_ks.iter().copied().max().unwrap_or(1).min(snap.n);
        crate::gpu::batch_search_cached(snap.version, &l2_flat, n_l2, &data_view, kmax_l, "l2")
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

    let n_queries = queries.len();

    let (gauss_out, l2_out) = split_and_run_gpu_kernels(snap, queries, ks, rescore)?;

    // ── Reassemble rows in original request order.
    let mut per_query = Vec::with_capacity(n_queries);
    let mut g = 0usize; // cursor into gauss_out
    let mut l = 0usize; // cursor into l2_out
    for (i, (&k, &want_rescore)) in ks.iter().zip(rescore.iter()).enumerate() {
        let row: Vec<SearchResult> = if want_rescore {
            let src = &gauss_out[g];
            g += 1;
            src.iter()
                .filter(|&&(ix, _)| ix < snap.n)
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
                .filter(|(&ix, _)| ix < snap.n)
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
        let _ = i; // index kept for clarity in debug
    }

    // Feed access frequencies for tiering under a second brief lock.
    let touched: Vec<usize> = per_query
        .iter()
        .flat_map(|r| r.iter().map(|s| s.index))
        .collect();
    store.blocking_lock().touch(&touched);

    Ok(per_query)
}

// ─── Embedding ────────────────────────────────────────────

/// Deterministic hash-based pseudo-embedding for text.
/// NOT a real embedding - placeholder for structure.
/// Replace with actual model (DINOv2, BGE, etc.) in production.
///
/// O(1) hashes: the text is hashed ONCE and the D dimensions are derived
/// from a SplitMix64 PRNG seeded with that hash (previously the text was
/// re-hashed D times — O(D) hasher passes — R9 §5.1).
fn simple_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    // Odd, non-zero seed so SplitMix64 output is well distributed from step 0.
    let mut state = hasher.finish() | 1;

    let mut result = Vec::with_capacity(dim);
    for _ in 0..dim {
        // SplitMix64 finalizer step
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        result.push((z as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    // Normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in result.iter_mut() {
            *x /= norm;
        }
    }
    result
}

// ─── API Key Authentication Middleware ────────────────────
async fn api_key_auth(
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

    // If no SPLATSDB_API_KEY env var set, skip auth (localhost dev mode)
    let Ok(expected_key) = std::env::var("SPLATSDB_API_KEY") else {
        return Ok(next.run(request).await);
    };

    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(key) if constant_time_eq(key, &expected_key) => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Constant-time string comparison to prevent timing attacks on API keys.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.as_bytes().iter().zip(b.as_bytes().iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Load vectors from the `default` shard of `data_dir` into `store`.
///
/// Returns the number of vectors loaded. Mirrors the CLI's
/// `load_or_create_store` behaviour (src/cli/helpers.rs), keeping `serve`
/// consistent with `status`/`search` over the same data-dir.
fn load_default_shard(store: &mut SplatStore, data_dir: &str) -> anyhow::Result<usize> {
    let persist = crate::storage::persistence::SplatsDBPersistence::with_backend(
        data_dir,
        crate::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .map_err(|e| anyhow::anyhow!("persistence init failed: {e}"))?;
    match persist.load_vectors("default")? {
        Some(vectors) => {
            if !store.add_splat(&vectors) {
                anyhow::bail!(
                    "store capacity exceeded or dimension mismatch while loading {} vectors \
                     (max_splats={}) — check --dim against the shard's dim",
                    vectors.nrows(),
                    store.max_splats()
                );
            }
            Ok(vectors.nrows())
        }
        None => Ok(0),
    }
}

/// Loopback bind check for the no-API-key warning: `localhost`, `::1` and
/// any `127.x.x.x` address are private; everything else (including
/// `0.0.0.0`/`::`) is treated as externally reachable.
fn is_loopback_addr(addr: &str) -> bool {
    let a = addr.trim().to_ascii_lowercase();
    matches!(a.as_str(), "localhost" | "::1" | "[::1]") || a.starts_with("127.")
}

/// Auto-load the on-disk inputs for `serve`: the `default` shard from
/// `data_dir` (if present) into `store`, so `serve` boots with the vectors
/// indexed by the CLI instead of an empty store and honours the CLI --dim
/// flag instead of SplatsDBConfig::default() (issue #3); then the `field/`
/// corpus for /search — si hay field/, el índice busca el corpus completo
/// (field.raw — exacto sobre TODOS los chunks); si no, el shard cargado
/// (splats/μ — en stores planos son los propios vectores).
///
/// Extracted from `run_server` (repowise plan `refac4_829dc22e97478fcffcab`);
/// messages and behavior identical. Returns the field corpus, if any.
fn load_serve_inputs(data_dir: &str, store: &mut SplatStore) -> Option<FieldData> {
    match load_default_shard(store, data_dir) {
        Ok(n) if n > 0 => eprintln!(
            "[splatsdb] serve: loaded {n} vectors (dim {}) from {data_dir}",
            store.get_statistics().embedding_dim
        ),
        Ok(_) => eprintln!("[splatsdb] serve: no existing data in {data_dir}, starting empty"),
        Err(e) => eprintln!("[splatsdb] serve: failed to load {data_dir}: {e}"),
    }

    let field_data = load_field(data_dir);
    if field_data.is_some() {
        eprintln!("[splatsdb] serve: campo splat cargado (/field-search disponible)");
    }
    field_data
}

pub async fn run_server(
    addr: &str,
    port: u16,
    config: SplatsDBConfig,
    data_dir: &str,
    gpu_enabled: bool,
) -> anyhow::Result<()> {
    let config = server_config(config);
    let hnsw_on = config.enable_hnsw;
    let mut store = SplatStore::new(config);

    let field_data = load_serve_inputs(data_dir, &mut store);
    let corpus: Option<(Vec<f32>, usize, usize)> = if let Some(f) = &field_data {
        let (n, d) = f.raw.dim();
        let slice = f
            .raw
            .as_slice()
            .map(<[f32]>::to_vec)
            .unwrap_or_else(|| f.raw.iter().copied().collect());
        Some((slice, n, d))
    } else {
        store.get_mu().and_then(|mu| {
            if mu.nrows() == 0 || mu.ncols() == 0 {
                return None;
            }
            let (n, d) = (mu.nrows(), mu.ncols());
            let slice = mu
                .as_slice()
                .map(<[f32]>::to_vec)
                .unwrap_or_else(|| mu.iter().copied().collect());
            Some((slice, n, d))
        })
    };
    let corpus_n = corpus.as_ref().map(|c| c.1).unwrap_or(0);

    // ── Índice exacto en GPU (--gpu, feature cuda): kernel PTX l2+topk sobre
    // el corpus. Fallback silencioso a CPU si no hay GPU.
    #[cfg(feature = "cuda")]
    let gpu_index: Option<Arc<crate::gpu::cuda_kernel::GpuIndex>> = if gpu_enabled {
        match crate::gpu::cuda_kernel::GpuIndex::new() {
            Some(mut g) => match corpus {
                Some((slice, n, d)) => {
                    if g.upload_dataset(&slice, n, d) {
                        eprintln!(
                            "[splatsdb] serve: GPU exact index listo ({n} vectores, dim {d})"
                        );
                        Some(Arc::new(g))
                    } else {
                        eprintln!("[splatsdb] serve: upload GPU falló; búsqueda en CPU");
                        None
                    }
                }
                None => {
                    eprintln!("[splatsdb] serve: store vacío, GPU sin dataset");
                    None
                }
            },
            None => {
                eprintln!("[splatsdb] serve: GPU no disponible (--gpu ignorado); búsqueda en CPU");
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    drop(corpus);

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
        store,
        next_id: Arc::new(AtomicUsize::new(0)),
        batch_tx,
        batches_served,
        #[cfg(feature = "cuda")]
        gpu: gpu_index,
        field: field_data.map(Arc::new),
        corpus_n,
        data_dir: Some(data_dir.to_string()),
    };

    let app = build_router(state);

    let bind_addr = format!("{addr}:{port}");
    // Loud warning: bound beyond loopback with no API key → everything (store,
    // search, field-search) is open to anyone who can reach this address.
    if !is_loopback_addr(addr) && std::env::var("SPLATSDB_API_KEY").is_err() {
        eprintln!(
            "[splatsdb] ⚠  WARNING: serving on {bind_addr} with NO SPLATSDB_API_KEY set.\n\
             ⚠  Authentication is DISABLED — anyone who can reach this address can read\n\
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
    Ok(axum::serve(listener, app).await?)
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
        .layer(middleware::from_fn(api_key_auth))
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

    // ── hash embedding ──────────────────────────────────────

    #[test]
    fn test_hash_embedding_properties() {
        let dim = 64;
        let emb = simple_hash_embedding("hello splats", dim);
        assert_eq!(emb.len(), dim);
        // Deterministic
        assert_eq!(emb, simple_hash_embedding("hello splats", dim));
        // Unit norm
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
        // Bounded
        assert!(emb.iter().all(|x| x.abs() <= 1.0));
        // Different text → different embedding
        let other = simple_hash_embedding("hello splats!", dim);
        assert_ne!(emb, other);
        // O(1)-hash construction must still cover the full dim
        assert!(emb.iter().filter(|x| **x != 0.0).count() > dim / 2);
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
    fn recall_embed_fail_closed_no_simcos() {
        // Dead URL (port 1 → immediate connection refused): the embed unit
        // returns Err and there is NO SimCos fallback in this path
        // (fa1e528 parity — a hash-text embedding ranks noise as success).
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
}
