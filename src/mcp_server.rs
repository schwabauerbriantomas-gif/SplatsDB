//! MCP (Model Context Protocol) Server — stdio transport
//!
//! Exposes SplatsDB Vector Search as an MCP server for AI agent integration.
//! Uses JSON-RPC 2.0 over stdin/stdout (stdio transport).
//! All logs go to stderr to keep the protocol channel clean.
//!
//! v2.5: Graph Splat tools — knowledge graph over Gaussian splats.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::config::SplatsDBConfig;
use crate::splats::SplatStore;
use crate::storage::metadata_store::{DocumentRecord, MetadataStore};
use crate::storage::sqlite_store::{EmbeddingProvenance, SqliteMetadataStore};

// ============================================================================
// Embedding client — real model via HTTP, SimCos fallback
// ============================================================================

const EMBEDDING_SERVICE_URL_DEFAULT: &str = "http://127.0.0.1:8788/embed";
static EMBED_SERVICE_AVAILABLE: AtomicBool = AtomicBool::new(true);
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Explicit opt-in for the SimCos text-hash fallback (`SPLATSDB_ALLOW_SIMCOS=1`),
/// read once at startup. OFF by default: without it every text-embedding
/// entry point fails closed when the real service is unavailable.
static SIMCOS_ALLOWED: AtomicBool = AtomicBool::new(false);
static SIMCOS_WARNED: AtomicBool = AtomicBool::new(false);
const MAX_DOC_TEXT_CACHE: usize = 10_000;
const MAX_WARM_START_DOCS: usize = 50_000;

/// True iff every component is finite (rejects NaN and ±inf). JSON has no
/// NaN literal but `1e400` parses to +inf; one poisoned vector corrupts
/// every similarity computation over the index, so every embedding entry
/// point (and the remote embedding service response) validates this.
fn all_finite(e: &[f32]) -> bool {
    e.iter().all(|v| v.is_finite())
}

fn embedding_service_url() -> String {
    std::env::var("SPLATSDB_EMBED_URL")
        .unwrap_or_else(|_| EMBEDDING_SERVICE_URL_DEFAULT.to_string())
}

#[cfg(unix)]
extern "C" fn handle_sigint(_: i32) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
}

/// Embedding from the service only: None when the service is unavailable or
/// fails. No SimCos fallback — /search is fail-closed (fa1e528 parity: a
/// hash-text embedding produces noise ranking reported with a success status).
fn get_embedding_strict(text: &str, dim: usize) -> Option<Vec<f32>> {
    if EMBED_SERVICE_AVAILABLE.load(Ordering::Relaxed) {
        fetch_real_embedding(text, dim)
    } else {
        None
    }
}

/// Explicit opt-in parser for the SimCos fallback (`SPLATSDB_ALLOW_SIMCOS`).
/// Pure (no env access) so it is directly testable.
fn simcos_opt_in(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Where an embedding came from. Always reported to the client, which never
/// has to guess whether ranking is semantic or a text hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingSource {
    /// The real embedding service answered.
    Real,
    /// SimCos text-hash fallback — only ever under explicit opt-in.
    SimCos,
    /// Supplied by the client in the request.
    Provided,
}

impl EmbeddingSource {
    fn label(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::SimCos => "simcos",
            Self::Provided => "provided",
        }
    }
}

/// How the provenance of an indexed vector is reported per search result:
/// `real`/`simcos`/`provided` when the vector was produced in this session,
/// `stored_real`/`stored_simcos`/`stored_provided` when the warm start
/// loaded it from SQLite (A03-F10, persistence half). The label travels
/// with the vector: a SimCos vector is never reported as real.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VectorProvenance {
    Live(EmbeddingSource),
    Stored(EmbeddingProvenance),
}

impl VectorProvenance {
    fn label(self) -> String {
        match self {
            Self::Live(src) => src.label().to_string(),
            Self::Stored(src) => format!("stored_{}", src.label()),
        }
    }
}

/// Provenance label persisted next to a vector in SQLite.
fn provenance_of(src: EmbeddingSource) -> EmbeddingProvenance {
    match src {
        EmbeddingSource::Real => EmbeddingProvenance::Real,
        EmbeddingSource::SimCos => EmbeddingProvenance::SimCos,
        EmbeddingSource::Provided => EmbeddingProvenance::Provided,
    }
}

/// Accounting of one warm start (logged and exported by `splatsdb_status`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WarmStartStats {
    /// Live documents found in SQLite.
    documents: usize,
    /// Vectors loaded straight from SQLite — no embedding was computed.
    vectors_loaded: usize,
    stored_real: usize,
    stored_simcos: usize,
    stored_provided: usize,
    /// Documents WITHOUT a stored vector (legacy rows) that were embedded
    /// and indexed in this warm start.
    reembedded: usize,
    /// Documents without a stored vector that could not be embedded.
    unindexed: usize,
    /// Rows whose stored vector was refused (left untouched on disk).
    quarantined: usize,
}

/// Everything `run_mcp_server` builds its state from after a warm start.
struct WarmStartOutcome {
    vector_to_doc: HashMap<usize, String>,
    doc_texts: DocTextCache,
    next_id: usize,
    spatial: WarmStartSpatial,
    stats: WarmStartStats,
    /// Docs whose indexed vector came from SQLite, with the provenance they
    /// were originally stored with.
    sources: HashMap<String, EmbeddingProvenance>,
}

const EMBEDDING_UNAVAILABLE: &str = "embedding unavailable: embed service unreachable or returned an unusable vector (dimension mismatch / non-finite values) and the SimCos fallback is not authorized (start the embed server, send a pre-computed 'embedding', or opt in explicitly with SPLATSDB_ALLOW_SIMCOS=1)";

/// Embedding for `text`, never silent (A03-F10): the real service first;
/// SimCos only when explicitly authorized via `SPLATSDB_ALLOW_SIMCOS`, and
/// then labelled as such. Anything else is a controlled error.
fn get_embedding_labeled(text: &str, dim: usize) -> Result<(Vec<f32>, EmbeddingSource), String> {
    if let Some(vec) = get_embedding_strict(text, dim) {
        return Ok((vec, EmbeddingSource::Real));
    }
    if SIMCOS_ALLOWED.load(Ordering::Relaxed) {
        if !SIMCOS_WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[mcp] WARNING: embed service unavailable; using the AUTHORIZED SimCos fallback (SPLATSDB_ALLOW_SIMCOS): ranking is text-hash, not semantic — every response is labelled embedding=simcos"
            );
        }
        return Ok((simcos_embedding(text, dim), EmbeddingSource::SimCos));
    }
    Err(EMBEDDING_UNAVAILABLE.to_string())
}

/// Optional client-provided embedding (`embedding` array argument).
fn parse_embedding_param(params: &Value) -> Option<Vec<f32>> {
    params["embedding"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
    })
}

fn fetch_real_embedding(text: &str, target_dim: usize) -> Option<Vec<f32>> {
    use std::time::Duration;
    let body = serde_json::json!({ "texts": [text] }).to_string();
    let response = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .new_agent()
        .post(&embedding_service_url())
        .header("Content-Type", "application/json")
        .send(&body)
        .ok()?;
    let resp: serde_json::Value = response.into_body().read_json().ok()?;
    let embeddings = resp.get("embeddings")?.as_array()?.first()?.as_array()?;
    let mut vec: Vec<f32> = embeddings
        .iter()
        .filter_map(|v: &Value| v.as_f64().map(|f| f as f32))
        .collect();
    // A service answering with another dimension is a configuration error:
    // zero-padding or truncating would index a different vector space as if
    // it were valid (A03-F13, same contract as HTTP /recall).
    if vec.len() != target_dim {
        eprintln!(
            "[mcp] embedding service returned {} dims, expected {target_dim}; rejecting",
            vec.len()
        );
        return None;
    }
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    // A hostile/broken embedding service returning NaN/inf must not poison
    // the index: reject the vector (any SimCos fallback is opt-in + labelled).
    if !all_finite(&vec) {
        eprintln!("[mcp] embedding service returned non-finite values; rejecting");
        return None;
    }
    Some(vec)
}

// ============================================================================
// Types
// ============================================================================

struct McpState {
    store: SplatStore,
    doc_store: SqliteMetadataStore,
    /// False when `doc_store` fell back to `:memory:` (A03-F13): writes
    /// succeed but persist nothing, and the client must see that.
    doc_store_persistent: bool,
    /// Maps vector index → doc_id for retrieval
    vector_to_doc: HashMap<usize, String>,
    /// Inverse index, doc_id → its ONLY live vector (A03-F11).
    doc_to_vec: HashMap<String, usize>,
    /// Vector indices whose doc was deleted or re-stored. The splat stays
    /// in the store (the ANN index has no delete), so every search path
    /// drops them and refills top-k from the next candidates.
    tombstones: std::collections::HashSet<usize>,
    /// Maps doc_id → text content (in-memory cache with eviction)
    doc_texts: DocTextCache,
    next_id: usize,
    /// Knowledge graph store (Gaussian splat graph)
    graph: crate::graph_splat::GaussianGraphStore,
    /// Spatial memory: maps (wing, room, hall) → vector indices for pre-filter search.
    /// Updated when docs are stored/loaded with spatial metadata.
    spatial_index: crate::spatial::SpatialIndex,
    /// Provenance of every live doc's indexed vector (A03-F10, persistence
    /// half): how the vector that sits in the ANN was produced, and whether
    /// the warm start loaded it from SQLite (`Stored`) or this session made
    /// it (`Live`). Reported per search result; a SimCos vector is never
    /// promoted to `real`.
    doc_sources: HashMap<String, VectorProvenance>,
    /// Accounting of the warm start that built this state (`splatsdb_status`).
    warm_start: WarmStartStats,
}

/// LRU cache of doc_id → text with O(1) insert / O(1) remove via lazy
/// deletion: removal leaves a tombstone in the order queue; eviction skips
/// entries already gone. Each queue entry carries the generation at which it
/// was pushed, so a re-inserted id is never killed by its own older
/// tombstone reaching the front (remove → re-add of the same id puts two
/// entries in the queue; only the one whose generation matches the live
/// entry evicts).
///
/// Note on LRU semantics: this preserves the historic behavior — eviction
/// order is *insertion* order (FIFO), not access order; `get` does not
/// refresh recency. Making it true LRU would change which docs survive
/// under load.
struct DocTextCache {
    map: HashMap<String, String>,
    /// Live id → generation of its queue entry (absent once removed).
    seq: HashMap<String, u64>,
    /// Insertion order; may contain stale (tombstoned) entries.
    order: std::collections::VecDeque<(String, u64)>,
    next_seq: u64,
    /// Max live entries. Production default: `MAX_DOC_TEXT_CACHE`; smaller
    /// in tests to exercise eviction.
    cap: usize,
}

impl Default for DocTextCache {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            seq: HashMap::new(),
            order: std::collections::VecDeque::new(),
            next_seq: 0,
            cap: MAX_DOC_TEXT_CACHE,
        }
    }
}

impl DocTextCache {
    fn len(&self) -> usize {
        self.map.len()
    }
    /// Insert/replace a text, evicting oldest-inserted entries over the cap.
    ///
    /// Replacement (id already cached) refreshes the text in place without
    /// touching the queue — same as the previous HashMap-based behavior —
    /// which also keeps the queue bounded on hot keys that are re-stored
    /// often.
    fn insert(&mut self, id: String, text: String) {
        if let Some(slot) = self.map.get_mut(&id) {
            *slot = text;
            return;
        }
        self.next_seq += 1;
        let gen = self.next_seq;
        self.seq.insert(id.clone(), gen);
        self.order.push_back((id.clone(), gen));
        self.map.insert(id, text);
        while self.map.len() > self.cap {
            match self.order.pop_front() {
                Some((old, g)) => {
                    // Evict only if this queue entry is still the live one
                    // (a tombstone or a superseded generation is skipped).
                    if self.seq.get(&old) == Some(&g) {
                        self.seq.remove(&old);
                        self.map.remove(&old);
                    }
                }
                None => break,
            }
        }
    }

    fn get(&self, id: &str) -> Option<&String> {
        self.map.get(id)
    }

    /// O(1) lazy removal: the id's queue entry becomes a tombstone and is
    /// skipped when it reaches the front during eviction.
    fn remove(&mut self, id: &str) {
        self.map.remove(id);
        self.seq.remove(id);
    }

    #[cfg(test)]
    fn live_order(&self) -> Vec<String> {
        self.order
            .iter()
            .filter(|(k, g)| self.seq.get(k) == Some(g))
            .map(|(k, _)| k.clone())
            .collect()
    }
}

impl McpState {
    /// Assemble the live state around an (optionally persistent) doc store.
    fn new(store: SplatStore, doc_store: SqliteMetadataStore, doc_store_persistent: bool) -> Self {
        Self {
            store,
            doc_store,
            doc_store_persistent,
            vector_to_doc: HashMap::new(),
            doc_to_vec: HashMap::new(),
            tombstones: std::collections::HashSet::new(),
            doc_texts: DocTextCache::default(),
            next_id: 0,
            graph: crate::graph_splat::GaussianGraphStore::new(),
            spatial_index: crate::spatial::SpatialIndex::new(),
            doc_sources: HashMap::new(),
            warm_start: WarmStartStats::default(),
        }
    }

    /// Register `vector_idx` as THE live vector of `id`. Any previous
    /// mapping of `id` is retired first, so a doc never has two live
    /// vectors (re-store used to leave the old one matching).
    fn map_vector(&mut self, vector_idx: usize, id: &str) {
        self.unmap_doc(id);
        self.vector_to_doc.insert(vector_idx, id.to_string());
        self.doc_to_vec.insert(id.to_string(), vector_idx);
    }

    /// Retire the mapping of `id`, tombstoning its vector. Returns whether
    /// `id` had a live vector.
    fn unmap_doc(&mut self, id: &str) -> bool {
        let Some(idx) = self.doc_to_vec.remove(id) else {
            return false;
        };
        self.vector_to_doc.remove(&idx);
        self.tombstones.insert(idx);
        self.doc_sources.remove(id);
        true
    }

    /// A vector is searchable iff a live doc maps to it.
    fn is_live(&self, idx: usize) -> bool {
        !self.tombstones.contains(&idx) && self.vector_to_doc.contains_key(&idx)
    }

    /// Live (searchable) vectors.
    fn live_vectors(&self) -> usize {
        self.vector_to_doc.len()
    }

    /// k to request from the ANN so that `top_k` LIVE results remain after
    /// dropping the dead vectors (deleted / re-stored) it still returns.
    fn fetch_k(&self, top_k: usize, n_active: usize) -> usize {
        let dead = n_active.saturating_sub(self.live_vectors());
        top_k.saturating_add(dead).min(n_active)
    }

    /// Text of a doc for a search result: the text cache first, else the
    /// persisted `document` column. The cache is capped (FIFO, 10K) while
    /// warm start loads up to 50K docs — a live doc evicted from the cache
    /// must never come back as `"text": ""` (A03-F14).
    fn doc_text(&self, id: &str) -> String {
        if let Some(t) = self.doc_texts.get(id).filter(|t| !t.is_empty()) {
            return t.clone();
        }
        self.doc_store
            .get(id)
            .ok()
            .flatten()
            .and_then(|r| r.document)
            .filter(|t| !t.is_empty())
            .unwrap_or_default()
    }

    /// (text, metadata) of a doc for a search result, from ONE SQLite read
    /// (cached statement) — no extra roundtrip for the text fallback.
    fn result_text_and_metadata(&self, id: &str) -> (String, Value) {
        let cached = self.doc_texts.get(id).filter(|t| !t.is_empty()).cloned();
        match self.doc_store.get(id) {
            Ok(Some(record)) => {
                let text = cached
                    .or(record.document)
                    .filter(|t| !t.is_empty())
                    .unwrap_or_default();
                (text, record.metadata.unwrap_or(json!(null)))
            }
            _ => (cached.unwrap_or_default(), json!(null)),
        }
    }

    /// Explicit provenance of a result's indexed vector.
    fn vector_label(&self, id: &str) -> String {
        self.doc_sources
            .get(id)
            .map_or_else(|| "unknown".to_string(), |p| p.label())
    }

    /// Persistence / provenance block of `splatsdb_status`.
    fn vector_persistence_json(&self) -> Value {
        json!({
            "schema_ready": self.doc_store.embedding_schema_ready(),
            "stored_vectors": self.doc_store.stored_vector_count().unwrap_or(0),
            "warm_start": {
                "documents": self.warm_start.documents,
                "vectors_loaded": self.warm_start.vectors_loaded,
                "stored_real": self.warm_start.stored_real,
                "stored_simcos": self.warm_start.stored_simcos,
                "stored_provided": self.warm_start.stored_provided,
                "reembedded": self.warm_start.reembedded,
                "unindexed": self.warm_start.unindexed,
                "quarantined": self.warm_start.quarantined,
            }
        })
    }

    /// Live state built from a warm start (production path and tests).
    fn from_warm_start(
        store: SplatStore,
        doc_store: SqliteMetadataStore,
        doc_store_persistent: bool,
        outcome: &mut WarmStartOutcome,
    ) -> Self {
        let mut state = Self::new(store, doc_store, doc_store_persistent);
        state.vector_to_doc = std::mem::take(&mut outcome.vector_to_doc);
        state.doc_to_vec = state
            .vector_to_doc
            .iter()
            .map(|(idx, id)| (id.clone(), *idx))
            .collect();
        state.doc_texts = std::mem::take(&mut outcome.doc_texts);
        state.next_id = outcome.next_id;
        state.warm_start = outcome.stats;
        state.doc_sources = std::mem::take(&mut outcome.sources)
            .into_iter()
            .map(|(id, src)| (id, VectorProvenance::Stored(src)))
            .collect();
        state
    }

    fn cache_doc_text(&mut self, id: String, text: String) {
        self.doc_texts.insert(id, text);
    }

    /// Remove text from cache (O(1) lazy deletion).
    fn uncache_doc_text(&mut self, id: &str) {
        self.doc_texts.remove(id);
    }
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    // id is ALWAYS serialized (null when undetectable): JSON-RPC 2.0 requires
    // the id member present on error responses (17-nov EGM audit).
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

// ============================================================================
// Tool definitions
// ============================================================================

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "splatsdb_store",
            "description": "Store a memory in the SplatsDB vector search engine. Returns the memory ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text content to store" },
                    "category": { "type": "string", "description": "Optional category tag" },
                    "id": { "type": "string", "description": "Optional custom ID (auto-generated if omitted)" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed embedding vector" },
                    "wing": { "type": "string", "description": "Spatial: top-level scope (project, persona, domain)" },
                    "room": { "type": "string", "description": "Spatial: semantic grouping within wing" },
                    "hall": { "type": "string", "description": "Spatial: memory type filter (fact, decision, event, error)" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatsdb_search",
            "description": "Search for similar memories in the SplatsDB vector store. Returns ranked results with similarity scores.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "top_k": { "type": "number", "description": "Number of results to return (default: 10)" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed query embedding" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_status",
            "description": "Get the current status of the SplatsDB vector store (number of memories, dimensions, active indexes).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "splatsdb_doc_add",
            "description": "Add a document with metadata to the SplatsDB store. Persists to SQLite.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID" },
                    "text": { "type": "string", "description": "Document text content" },
                    "metadata": { "type": "string", "description": "Optional JSON metadata string" }
                },
                "required": ["id", "text"]
            }
        }),
        json!({
            "name": "splatsdb_doc_get",
            "description": "Retrieve a document by ID from the SplatsDB store (SQLite-backed).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID to retrieve" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "splatsdb_doc_del",
            "description": "Delete a document from the SplatsDB store (SQLite-backed, soft delete).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID to delete" }
                },
                "required": ["id"]
            }
        }),
        // ── Graph Splat tools ──────────────────────────────────────────────
        json!({
            "name": "splatsdb_graph_add_doc",
            "description": "Add a document node to the knowledge graph. Auto-embeds the text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Document text content" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatsdb_graph_add_entity",
            "description": "Add an entity node to the knowledge graph. Auto-embeds the name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Entity name" },
                    "entity_type": { "type": "string", "description": "Entity type (e.g. person, organization, location, concept)" }
                },
                "required": ["name", "entity_type"]
            }
        }),
        json!({
            "name": "splatsdb_graph_add_relation",
            "description": "Add a directed edge between two nodes in the knowledge graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": { "type": "number", "description": "Source node ID" },
                    "target_id": { "type": "number", "description": "Target node ID" },
                    "relation_type": { "type": "string", "description": "Relation type (e.g. MENTIONS, RELATED_TO)" },
                    "weight": { "type": "number", "description": "Edge weight (default: 1.0)" }
                },
                "required": ["source_id", "target_id", "relation_type"]
            }
        }),
        json!({
            "name": "splatsdb_graph_traverse",
            "description": "BFS traversal from a start node in the knowledge graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "number", "description": "Start node ID" },
                    "max_depth": { "type": "number", "description": "Maximum traversal depth (default: 3)" }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "splatsdb_graph_search",
            "description": "Hybrid search over the knowledge graph (vector similarity + graph context boost). Auto-embeds query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "k": { "type": "number", "description": "Number of results (default: 10)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_graph_search_entities",
            "description": "Search entity nodes by embedding similarity. Auto-embeds query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "k": { "type": "number", "description": "Number of results (default: 10)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_graph_stats",
            "description": "Get knowledge graph statistics (node counts, edge counts, etc.).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        // ── Spatial Memory tools ──
        serde_json::json!({
            "name": "splatsdb_spatial_search",
            "description": "Search memories with spatial filters. Organize memory like physical space: Wings (projects), Rooms (topics), Halls (types). Filters reduce the search space before vector distance computation for higher recall with less noise.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed query embedding" },
                    "wing": {
                        "type": "string",
                        "description": "Filter by wing (project/persona/domain). E.g. 'project-x', 'personal'"
                    },
                    "room": {
                        "type": "string",
                        "description": "Filter by room (semantic cluster/topic). E.g. 'auth', 'billing', 'migration'"
                    },
                    "hall": {
                        "type": "string",
                        "description": "Filter by hall (memory type). E.g. 'decision', 'fact', 'error', 'event'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10)"
                    }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "splatsdb_spatial_info",
            "description": "Show the spatial memory structure: all Wings (projects), Rooms (topics), Halls (memory types), and Tunnels (cross-wing connections). Use this to understand the organization before querying with splatsdb_spatial_search.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        // ── Verbatim Storage tools ──────────────────────────────────────
        json!({
            "name": "splatsdb_verbatim_search",
            "description": "Search with confidence scoring to prevent hallucination. Returns results with HIGH/MEDIUM/LOW confidence based on vector distance, plus original verbatim text for verification.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed query embedding" },
                    "query": { "type": "string", "description": "Search query text" },
                    "top_k": { "type": "number", "description": "Number of results (default: 10)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_compress",
            "description": "Compress text using AAAK (semantic + binary compression). Returns semantically compressed text that LLMs can read natively, plus binary compressed data for storage.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to compress" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatsdb_decompress",
            "description": "Decompress AAAK binary data back to LLM-readable semantic text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data": { "type": "string", "description": "Hex-encoded compressed data" }
                },
                "required": ["data"]
            }
        }),
    ]
}

// ============================================================================
// Embedding: SimCos — similarity-consistent n-gram hashing
// ============================================================================

/// Produce a similarity-consistent embedding from text.
///
/// Unlike simple hash (where "hello world" and "hello earth" have 0 similarity),
/// SimCos uses character n-gram overlap to produce embeddings where:
/// - Similar texts have high cosine similarity
/// - Different texts have low similarity
/// - It's fast and requires no model loading
///
/// For production use, pass pre-computed embeddings via the `embedding` parameter.
fn simcos_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Generate character trigrams from the text
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text_lower.chars().collect();
    let mut trigram_set: Vec<String> = Vec::new();

    // Character trigrams
    if chars.len() >= 3 {
        for i in 0..chars.len() - 2 {
            let tg: String = chars[i..i + 3].iter().collect();
            trigram_set.push(tg);
        }
    } else {
        // Short text: use the whole thing
        trigram_set.push(text_lower.clone());
    }

    // Word unigrams and bigrams
    let words: Vec<&str> = text_lower.split_whitespace().collect();
    for w in &words {
        trigram_set.push(format!("w:{}", w));
    }
    if words.len() >= 2 {
        for i in 0..words.len() - 1 {
            trigram_set.push(format!("wb:{}:{}", words[i], words[i + 1]));
        }
    }

    // Hash trigrams into the embedding vector using random projection
    let mut result = vec![0.0f32; dim];
    for tg in &trigram_set {
        // Each trigram contributes to multiple dimensions
        for band in 0..3 {
            let mut hasher = DefaultHasher::new();
            tg.hash(&mut hasher);
            band.hash(&mut hasher);
            let hash_val = hasher.finish();

            // Map hash to dimension index
            let idx = (hash_val as usize) % dim;
            // Use sign from a second hash
            let mut hasher2 = DefaultHasher::new();
            tg.hash(&mut hasher2);
            band.hash(&mut hasher2);
            (idx as u64).hash(&mut hasher2);
            let sign_hash = hasher2.finish();

            let sign = if sign_hash.is_multiple_of(2) {
                1.0f32
            } else {
                -1.0f32
            };
            result[idx] += sign;
        }
    }

    // L2 normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in result.iter_mut() {
            *x /= norm;
        }
    }

    result
}

// ============================================================================
// Tool handlers
// ============================================================================

/// Validated parameters for `splatsdb_store`.
struct StoreParams<'a> {
    text: &'a str,
    category: Option<&'a str>,
    embedding_opt: Option<Vec<f32>>,
}

/// Validate the `splatsdb_store` request fields BEFORE touching the store:
/// text required (≤100k bytes), id/category optional but non-empty and
/// ≤256 bytes, embedding parsed as f32 (dimension checked later against the
/// store, once `dim` is known).
///
/// Extracted from `handle_store`; behavior identical, including the check
/// order (text → id → category).
fn validate_store_params(params: &Value) -> Result<StoreParams<'_>, String> {
    let text = params["text"].as_str().ok_or("missing 'text' field")?;
    const MAX_TEXT_LEN: usize = 100_000;
    if text.len() > MAX_TEXT_LEN {
        return Err(format!("Text exceeds max length ({MAX_TEXT_LEN})"));
    }
    let category = params["category"].as_str();
    // Same bounds as handle_doc_add: non-empty, ≤256 bytes — validated BEFORE
    // the vector is added so a rejected request never mutates the store.
    const MAX_ID_LEN: usize = 256;
    if let Some(id) = params["id"].as_str() {
        if id.is_empty() || id.len() > MAX_ID_LEN {
            return Err("invalid document ID".into());
        }
        // mem_<n> is the auto-id namespace: warm_start_reload derives
        // next_id from persisted ids, so an explicit mem_18446744073709551615
        // would wrap the counter on the next store and alias over persisted
        // docs via INSERT OR REPLACE (17-nov EGM audit). Reject it.
        if id.len() > 4 && id.starts_with("mem_") && id[4..].bytes().all(|b| b.is_ascii_digit()) {
            return Err("'mem_<n>' IDs are reserved for auto-generated documents".into());
        }
    }
    if let Some(cat) = category {
        if cat.is_empty() || cat.len() > MAX_ID_LEN {
            return Err("invalid category".into());
        }
    }
    let embedding_opt = params["embedding"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
    });
    Ok(StoreParams {
        text,
        category,
        embedding_opt,
    })
}

fn handle_store(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let p = validate_store_params(params)?;
    let text = p.text;
    let category = p.category;
    let embedding_opt = p.embedding_opt;

    // Read dimension under a brief lock, then compute embedding WITHOUT holding
    // the mutex — get_embedding() may make an HTTP call (5s timeout) and would
    // block all concurrent operations if held inside the lock.
    let dim = {
        let s = state.lock().map_err(|e| {
            eprintln!("[mcp] mutex poisoned: {}", e);
            "internal server error".to_string()
        })?;
        s.store.get_statistics().embedding_dim
    }; // lock dropped here

    // Provided embedding, or a real one — or SimCos under explicit opt-in.
    // Never a silent fallback: an unauthorized fallback is a controlled error.
    let (embedding, embedding_src) = match embedding_opt {
        Some(e) => (e, EmbeddingSource::Provided),
        None => get_embedding_labeled(text, dim)?,
    };

    if embedding.len() != dim {
        return Err(format!(
            "Embedding dimension mismatch: expected {dim}, got {}",
            embedding.len()
        ));
    }
    // NaN/inf (user input like `1e400`, or a bad embedding service) must
    // never reach add_splat — it poisons the whole index.
    if !all_finite(&embedding) {
        return Err("Embedding contains non-finite values (NaN/inf)".into());
    }

    let arr = Array2::from_shape_vec((1, dim), embedding.clone())
        .map_err(|e| format!("bad embedding shape: {}", e))?;

    // Re-acquire lock only for state mutation
    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let vector_idx = s.store.n_active();
    let added = s.store.add_splat(&arr);
    if !added {
        return Err("store is full".into());
    }
    s.store.hnsw_sync_incremental();

    // Counter advance must fail cleanly instead of wrapping into persisted
    // mem_<n> ids (17-nov EGM audit: a wrapped counter regenerates mem_1..mem_N
    // via INSERT OR REPLACE and corrupts existing docs silently).
    s.next_id = s
        .next_id
        .checked_add(1)
        .ok_or_else(|| "document id counter exhausted".to_string())?;
    let id = params["id"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("mem_{}", s.next_id));

    // Store text and mapping
    s.map_vector(vector_idx, &id);
    s.cache_doc_text(id.clone(), text.to_string());
    s.doc_sources
        .insert(id.clone(), VectorProvenance::Live(embedding_src));

    // Persist to SQLite
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let mut metadata_json = serde_json::Map::new();
    metadata_json.insert("text".into(), json!(text));
    if let Some(cat) = category {
        metadata_json.insert("category".into(), json!(cat));
    }

    let record = DocumentRecord {
        id: id.clone(),
        shard_idx: 0,
        vector_idx: vector_idx as i64,
        metadata: Some(serde_json::Value::Object(metadata_json)),
        document: Some(text.to_string()),
        deleted: false,
        created_at: now,
        updated_at: now,
    };

    // Persistence failure must be visible to the client (17-nov EGM audit):
    // a swallowed upsert error made "stored" a lie after a restart wiped the
    // doc from warm start.
    // Document, metadata, vector and its provenance in ONE SQLite
    // statement (F05b): a failed write leaves neither behind.
    let persisted =
        match s
            .doc_store
            .upsert_with_embedding(&record, &embedding, provenance_of(embedding_src))
        {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[mcp] warning: SQLite upsert failed: {}", e);
                false
            }
        };

    // Register spatial metadata if present
    let wing = params["wing"].as_str();
    let room = params["room"].as_str();
    let hall = params["hall"].as_str();
    if wing.is_some() || room.is_some() || hall.is_some() {
        s.spatial_index.register_doc(&id, wing, room, hall);
    } else {
        // Re-store without spatial keys: the previous registration
        // described the OLD content — drop it.
        s.spatial_index.remove_doc(&id);
    }

    eprintln!(
        "[mcp] store: id={}, vec_idx={}, text_len={}",
        id,
        vector_idx,
        text.len()
    );
    // `persisted` must be false when the store itself is in memory (A03-F13):
    // a successful upsert into `:memory:` survives nothing.
    let persisted = persisted && s.doc_store_persistent;
    Ok(json!({
        "id": id,
        "status": "stored",
        "persisted": persisted,
        "ephemeral": !s.doc_store_persistent,
        "embedding": embedding_src.label()
    }))
}

fn handle_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query text exceeds maximum length of 10000 characters".into());
    }
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    if top_k > 1000 {
        return Err("top_k exceeds maximum value of 1000".into());
    }

    // Read dimension and active count under a brief lock, then compute embedding
    // WITHOUT holding the mutex — get_embedding() may make an HTTP call.
    let (dim, n_active) = {
        let s = state.lock().map_err(|e| {
            eprintln!("[mcp] mutex poisoned: {}", e);
            "internal server error".to_string()
        })?;
        let dim = s.store.get_statistics().embedding_dim;
        let n = s.store.n_active();
        (dim, n)
    }; // lock dropped here

    if n_active == 0 {
        return Ok(json!({ "results": [] }));
    }

    let embedding_opt = params["embedding"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
    });
    let (embedding, embedding_src) = match embedding_opt {
        Some(e) => (e, EmbeddingSource::Provided),
        // fa1e528 parity (HTTP /search answers 400 without embedding): no
        // SILENT text-hash fallback — SimCos noise reported with a success
        // status is garbage ranking. It is allowed only under explicit
        // opt-in and is then labelled `embedding: simcos` (A03-F10).
        None => get_embedding_labeled(query, dim)?,
    };

    if embedding.len() != dim {
        return Err(format!(
            "Embedding dimension mismatch: expected {dim}, got {}",
            embedding.len()
        ));
    }
    // NaN/inf query (user input like `1e400`, or a bad embedding service)
    // yields NaN distances for every result — reject it.
    if !all_finite(&embedding) {
        return Err("Embedding contains non-finite values (NaN/inf)".into());
    }

    let query_vec = Array1::from_vec(embedding);

    // Re-acquire lock for the actual search
    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    // Over-fetch by the number of dead vectors (deleted / re-stored),
    // drop them, and keep `top_k` live results.
    let k = s.fetch_k(top_k, s.store.n_active());
    let neighbors = s.store.find_neighbors_fast(&query_vec.view(), k);

    let results: Vec<Value> = neighbors
        .into_iter()
        .filter(|n| s.is_live(n.index))
        .take(top_k)
        .map(|n| {
            // is_live() guarantees a mapping: anonymous `vec_N` hits can no
            // longer be produced.
            let doc_id = s.vector_to_doc.get(&n.index).cloned().unwrap_or_default();
            let (text, metadata) = s.result_text_and_metadata(&doc_id);

            json!({
                "id": doc_id,
                "index": n.index,
                "score": n.distance,
                "text": text,
                "metadata": metadata,
                "embedding_source": s.vector_label(&doc_id),
            })
        })
        .collect();

    eprintln!(
        "[mcp] search: query_len={}, results={}",
        query.len(),
        results.len()
    );
    // backend marker: scores are HRM2 engine distances (approximate path for
    // large stores), NOT the exact d² the HTTP /search emits — clients must
    // not reuse HTTP band calibrations against them (17-nov EGM audit).
    Ok(json!({ "results": results, "backend": "hrm2", "embedding": embedding_src.label() }))
}

fn handle_status(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let stats = s.store.get_statistics();

    let doc_count = s.doc_store.count(false).unwrap_or(0);

    Ok(json!({
        "n_active": s.store.n_active(),
        "live_vectors": s.live_vectors(),
        "tombstoned_vectors": s.tombstones.len(),
        "max_splats": s.store.max_splats(),
        "dimension": stats.embedding_dim,
        "doc_count": doc_count,
        "persisted": s.doc_store_persistent,
        "ephemeral": !s.doc_store_persistent,
        "embedding_fallback": if SIMCOS_ALLOWED.load(Ordering::Relaxed) { "simcos" } else { "none" },
        "vector_persistence": s.vector_persistence_json(),
        "has_hnsw": s.store.has_hnsw(),
        "has_lsh": s.store.has_lsh(),
        "has_quantization": s.store.has_quantization(),
    }))
}

fn handle_doc_add(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;
    if id.len() > 256 || id.is_empty() {
        return Err("invalid document ID".into());
    }
    let text = params["text"].as_str().ok_or("missing 'text'")?;
    let metadata = params["metadata"].as_str();
    const MAX_TEXT_LEN: usize = 100_000;
    if text.len() > MAX_TEXT_LEN {
        return Err(format!("Text exceeds max length ({MAX_TEXT_LEN})"));
    }

    // Read dimension under a brief lock, then compute embedding WITHOUT holding
    // the mutex — get_embedding() may make an HTTP call (5s timeout).
    let dim = {
        let s = state.lock().map_err(|e| {
            eprintln!("[mcp] mutex poisoned: {}", e);
            "internal server error".to_string()
        })?;
        s.store.get_statistics().embedding_dim
    }; // lock dropped here

    let (embedding, embedding_src) = get_embedding_labeled(text, dim)?;
    let arr = Array2::from_shape_vec((1, dim), embedding.clone())
        .map_err(|e| format!("embedding error: {}", e))?;

    // Re-acquire lock for state mutation
    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let vector_idx = s.store.n_active();
    if !s.store.add_splat(&arr) {
        return Err("add_splat failed".into());
    }
    s.store.hnsw_sync_incremental();

    // Update mappings
    s.map_vector(vector_idx, id);
    s.cache_doc_text(id.to_string(), text.to_string());
    s.doc_sources
        .insert(id.to_string(), VectorProvenance::Live(embedding_src));

    // Persist to SQLite
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let meta_val: Option<serde_json::Value> = metadata
        .and_then(|m| {
            if m.len() > 1_000_000 {
                return None;
            } // 1MB max metadata
            serde_json::from_str(m).ok()
        })
        .or_else(|| Some(json!({})));

    let mut full_meta = meta_val.unwrap_or(json!({}));
    if let Some(obj) = full_meta.as_object_mut() {
        obj.insert("text".into(), json!(text));
    }

    let record = DocumentRecord {
        id: id.to_string(),
        shard_idx: 0,
        vector_idx: vector_idx as i64,
        metadata: Some(full_meta),
        document: Some(text.to_string()),
        deleted: false,
        created_at: now,
        updated_at: now,
    };

    if let Err(e) =
        s.doc_store
            .upsert_with_embedding(&record, &embedding, provenance_of(embedding_src))
    {
        // Internal detail stays server-side; the client gets a generic error.
        eprintln!("[mcp] doc_add SQLite error for id={id}: {e}");
        // No persisted row → the new vector must not stay searchable as an
        // anonymous hit (A03-F11).
        s.unmap_doc(id);
        s.uncache_doc_text(id);
        return Err("internal error".to_string());
    }

    eprintln!(
        "[mcp] doc_add: id={}, vec_idx={}, text_len={}, has_meta={}",
        id,
        vector_idx,
        text.len(),
        metadata.is_some()
    );
    Ok(json!({ "ok": true, "id": id, "embedding": embedding_src.label() }))
}

fn handle_doc_get(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    match s.doc_store.get(id) {
        Ok(Some(record)) => {
            let text = record.document.unwrap_or_default();
            Ok(json!({
                "id": record.id,
                "text": text,
                "metadata": record.metadata,
                "vector_idx": record.vector_idx,
                "created_at": record.created_at,
            }))
        }
        Ok(None) => Err(format!("document not found: {}", id)),
        Err(e) => {
            // Internal detail stays server-side; the client gets a generic error.
            eprintln!("[mcp] doc_get SQLite error for id={id}: {e}");
            Err("internal error".into())
        }
    }
}

fn handle_doc_del(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    // Soft delete in SQLite
    match s.doc_store.soft_delete(id) {
        Ok(true) => {
            // Remove from in-memory caches
            s.uncache_doc_text(id);
            // The splat stays in the ANN index (no delete): tombstone its
            // vector so every search path filters it and refills top-k,
            // and drop it from the spatial index too (A03-F11).
            s.unmap_doc(id);
            s.spatial_index.remove_doc(id);
            eprintln!("[mcp] doc_del: id={} (soft deleted)", id);
            Ok(json!({ "ok": true, "id": id }))
        }
        Ok(false) => Err(format!("document not found: {}", id)),
        Err(e) => {
            // Internal detail stays server-side; the client gets a generic error.
            eprintln!("[mcp] doc_del SQLite error for id={id}: {e}");
            Err("internal error".into())
        }
    }
}

// ============================================================================
// Graph Splat tool handlers
// ============================================================================

fn handle_graph_add_doc(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let text = params["text"].as_str().ok_or("missing 'text' field")?;
    const MAX_TEXT_LEN: usize = 100_000;
    if text.len() > MAX_TEXT_LEN {
        return Err(format!("Text exceeds max length ({MAX_TEXT_LEN})"));
    }

    let dim = {
        let s = state.lock().map_err(|e| {
            eprintln!("[mcp] mutex poisoned: {}", e);
            "internal server error".to_string()
        })?;
        s.store.get_statistics().embedding_dim
    }; // lock dropped here

    let (embedding, embedding_src) = get_embedding_labeled(text, dim)?;

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let node_id = s
        .graph
        .add_document(text, &embedding)
        .map_err(|e| e.to_string())?;

    eprintln!(
        "[mcp] graph_add_doc: node_id={}, text_len={}",
        node_id,
        text.len()
    );
    Ok(json!({ "node_id": node_id, "node_type": "document", "embedding": embedding_src.label() }))
}

fn handle_graph_add_entity(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let name = params["name"].as_str().ok_or("missing 'name' field")?;
    let entity_type = params["entity_type"]
        .as_str()
        .ok_or("missing 'entity_type' field")?;
    const MAX_NAME_LEN: usize = 10_000;
    const MAX_TYPE_LEN: usize = 256;
    if name.len() > MAX_NAME_LEN {
        return Err(format!("Name exceeds max length ({MAX_NAME_LEN})"));
    }
    if entity_type.len() > MAX_TYPE_LEN {
        return Err(format!("entity_type exceeds max length ({MAX_TYPE_LEN})"));
    }

    let dim = {
        let s = state.lock().map_err(|e| {
            eprintln!("[mcp] mutex poisoned: {}", e);
            "internal server error".to_string()
        })?;
        s.store.get_statistics().embedding_dim
    }; // lock dropped here

    let (embedding, embedding_src) = get_embedding_labeled(name, dim)?;

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let node_id = s
        .graph
        .add_entity(name, &embedding, entity_type)
        .map_err(|e| e.to_string())?;

    eprintln!(
        "[mcp] graph_add_entity: node_id={}, name_len={}, type='{}'",
        node_id,
        name.len(),
        entity_type
    );
    Ok(
        json!({ "node_id": node_id, "name": name, "entity_type": entity_type, "embedding": embedding_src.label() }),
    )
}

fn handle_graph_add_relation(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let source_id = params["source_id"].as_u64().ok_or("missing 'source_id'")? as usize;
    let target_id = params["target_id"].as_u64().ok_or("missing 'target_id'")? as usize;
    let relation_type = params["relation_type"]
        .as_str()
        .ok_or("missing 'relation_type'")?;
    const MAX_RELATION_TYPE_LEN: usize = 256;
    if relation_type.len() > MAX_RELATION_TYPE_LEN {
        return Err(format!(
            "relation_type exceeds max length ({MAX_RELATION_TYPE_LEN})"
        ));
    }
    let weight = params["weight"].as_f64().unwrap_or(1.0);

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    s.graph
        .add_relation(source_id, target_id, relation_type, weight)
        .map_err(|e| e.to_string())?;

    eprintln!(
        "[mcp] graph_add_relation: {} --{}--> {} (w={})",
        source_id, relation_type, target_id, weight
    );
    Ok(json!({
        "ok": true,
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": relation_type,
        "weight": weight
    }))
}

fn handle_graph_traverse(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let start_id = params["start_id"].as_u64().ok_or("missing 'start_id'")? as usize;
    let max_depth = params["max_depth"].as_u64().unwrap_or(3).min(20) as usize;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    if s.graph.get_node(start_id).is_none() {
        return Err(format!("node not found: {}", start_id));
    }

    let visited = s.graph.traverse(start_id, max_depth);

    let nodes: Vec<Value> = visited
        .iter()
        .filter_map(|&id| {
            s.graph.get_node(id).map(|n| {
                json!({
                    "id": n.id,
                    "node_type": format!("{:?}", n.node_type).to_lowercase(),
                    "content": n.content,
                })
            })
        })
        .collect();

    eprintln!(
        "[mcp] graph_traverse: start={}, depth={}, found={}",
        start_id,
        max_depth,
        nodes.len()
    );
    Ok(json!({ "nodes": nodes, "count": nodes.len() }))
}

fn handle_graph_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query too long".into());
    }
    let k = params["k"].as_u64().unwrap_or(10).min(1000) as usize;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    drop(s); // release lock before potentially blocking HTTP call

    let (embedding, embedding_src) = get_embedding_labeled(query, dim)?;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let results = s.graph.hybrid_search(&embedding, k);

    let out: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "node_id": r.node_id,
                "content": r.content,
                "score": r.score,
                "node_type": format!("{:?}", r.node_type).to_lowercase(),
            })
        })
        .collect();

    eprintln!(
        "[mcp] graph_search: query_len={}, results={}",
        query.len(),
        out.len()
    );
    Ok(json!({ "results": out, "embedding": embedding_src.label() }))
}

fn handle_graph_search_entities(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query too long".into());
    }
    let k = params["k"].as_u64().unwrap_or(10).min(1000) as usize;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    drop(s); // release lock before potentially blocking HTTP call

    let (embedding, embedding_src) = get_embedding_labeled(query, dim)?;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let results = s.graph.search_entities(&embedding, k);

    let out: Vec<Value> = results
        .iter()
        .map(|n| {
            json!({
                "id": n.id,
                "content": n.content,
                "node_type": format!("{:?}", n.node_type).to_lowercase(),
            })
        })
        .collect();

    eprintln!(
        "[mcp] graph_search_entities: query_len={}, results={}",
        query.len(),
        out.len()
    );
    Ok(json!({ "results": out, "embedding": embedding_src.label() }))
}

fn handle_graph_stats(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let stats = s.graph.get_stats();

    Ok(json!({
        "total_nodes": stats.total_nodes,
        "total_edges": stats.total_edges,
        "documents": stats.documents,
        "entities": stats.entities,
        "concepts": stats.concepts,
    }))
}

// ============================================================================
// Spatial Memory handlers
// ============================================================================

/// Spatial search: uses SpatialIndex for O(|candidates|×dim) pre-filter search.
///
/// Pipeline:
///   1. SpatialIndex.filter() → candidate doc IDs (in-memory, no SQLite)
///   2. doc_id → vector_idx mapping (in-memory HashMap)
///   3. SplatStore.find_neighbors_filtered() → distance only against candidates
///
/// This avoids the old post-filter approach that computed distances against ALL N vectors.
fn handle_spatial_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query text exceeds maximum length of 10000 characters".into());
    }
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    if top_k > 1000 {
        return Err("top_k exceeds maximum value of 1000".into());
    }

    let wing_filter = params["wing"].as_str().map(|s| s.to_string());
    let room_filter = params["room"].as_str().map(|s| s.to_string());
    let hall_filter = params["hall"].as_str().map(|s| s.to_string());

    let has_filter = wing_filter.is_some() || room_filter.is_some() || hall_filter.is_some();

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let dim = s.store.get_statistics().embedding_dim;
    let n_active = s.store.n_active();
    if n_active == 0 {
        return Ok(json!({ "results": [], "total": 0, "filter": "no documents" }));
    }

    // doc_id → vector_idx comes from the persistent inverse index
    // (`McpState::doc_to_vec`), which only ever holds live mappings.
    // Get candidate vector indices via SpatialIndex (pre-filter)
    let candidate_vec_indices: Vec<usize> = if has_filter {
        let spatial_filter = crate::spatial::SpatialFilter {
            wing: wing_filter.clone(),
            room: room_filter.clone(),
            hall: hall_filter.clone(),
        };
        let candidate_ids = s.spatial_index.filter(&spatial_filter);
        candidate_ids
            .iter()
            .filter_map(|id| s.doc_to_vec.get(id).copied())
            .collect()
    } else {
        // No spatial filter — search all vectors
        (0..n_active).collect()
    };

    if candidate_vec_indices.is_empty() {
        let filter_desc = describe_filter(&wing_filter, &room_filter, &hall_filter);
        return Ok(json!({
            "results": [],
            "total": 0,
            "filter": filter_desc,
            "candidates_scanned": 0,
        }));
    }

    drop(s); // release lock before potentially blocking HTTP call

    // Embedding for the query: client-provided when present, else the
    // service — or SimCos only under explicit opt-in (may make an HTTP call).
    let (embedding, embedding_src) = match parse_embedding_param(params) {
        Some(e) => (e, EmbeddingSource::Provided),
        None => get_embedding_labeled(query, dim)?,
    };
    if embedding.len() != dim {
        return Err(format!(
            "Embedding dimension mismatch: expected {dim}, got {}",
            embedding.len()
        ));
    }
    let query_vec = Array1::from_vec(embedding);

    // Re-acquire lock for the actual search
    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    // Pre-filter search: only compute distances against candidates
    let neighbors = if has_filter {
        s.store
            .find_neighbors_filtered(&query_vec.view(), &candidate_vec_indices, top_k)
    } else {
        let k = s.fetch_k(top_k, s.store.n_active());
        s.store.find_neighbors_fast(&query_vec.view(), k)
    };

    let mut out = Vec::new();
    for n in neighbors {
        if !s.is_live(n.index) {
            continue;
        }
        let doc_id = s.vector_to_doc.get(&n.index).cloned().unwrap_or_default();
        let (text, metadata) = s.result_text_and_metadata(&doc_id);
        out.push(json!({
            "id": doc_id,
            "index": n.index,
            "score": n.distance,
            "text": text,
            "metadata": metadata,
            "embedding_source": s.vector_label(&doc_id),
        }));
        if out.len() >= top_k {
            break;
        }
    }

    let filter_desc = describe_filter(&wing_filter, &room_filter, &hall_filter);
    eprintln!(
        "[mcp] spatial_search: query_len={}, filter={}, candidates={}, results={}",
        query.len(),
        filter_desc,
        candidate_vec_indices.len(),
        out.len()
    );

    Ok(json!({
        "results": out,
        "total": out.len(),
        "embedding": embedding_src.label(),
        "filter": filter_desc,
        "candidates_scanned": candidate_vec_indices.len(),
    }))
}

fn describe_filter(wing: &Option<String>, room: &Option<String>, hall: &Option<String>) -> String {
    match (wing, room, hall) {
        (Some(w), Some(r), Some(h)) => format!("wing={w}, room={r}, hall={h}"),
        (Some(w), Some(r), None) => format!("wing={w}, room={r}"),
        (Some(w), None, Some(h)) => format!("wing={w}, hall={h}"),
        (None, Some(r), Some(h)) => format!("room={r}, hall={h}"),
        (Some(w), None, None) => format!("wing={w}"),
        (None, Some(r), None) => format!("room={r}"),
        (None, None, Some(h)) => format!("hall={h}"),
        _ => "no spatial filter".into(),
    }
}

/// Show spatial memory structure from metadata and in-memory SpatialIndex.
fn handle_spatial_info(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let all_ids = s.doc_store.list_ids(false).map_err(|e| e.to_string())?;

    // Scan metadata to build spatial structure
    let mut wings: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    let mut halls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let total_with_spatial = scan_spatial_metadata(&all_ids, &s, &mut wings, &mut halls);

    // Detect tunnels: rooms shared across wings
    let mut room_wings: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (wing, rooms) in &wings {
        for room in rooms {
            room_wings
                .entry(room.clone())
                .or_default()
                .push(wing.clone());
        }
    }
    let tunnels: Vec<Value> = room_wings
        .iter()
        .filter(|(_, ws)| ws.len() >= 2)
        .map(|(room, ws)| {
            json!({
                "room": room,
                "wings": ws,
            })
        })
        .collect();

    let wings_json: Value = wings
        .iter()
        .map(|(w, rooms)| {
            let rooms_vec: Vec<&String> = rooms.iter().collect();
            (w.clone(), json!(rooms_vec))
        })
        .collect();

    Ok(json!({
        "total_documents": all_ids.len(),
        "documents_with_spatial_metadata": total_with_spatial,
        "wings": wings_json,
        "halls": halls.iter().collect::<Vec<_>>(),
        "tunnels": tunnels,
        "tunnel_count": tunnels.len(),
    }))
}

/// Scan SQLite document metadata and build the spatial structure: every doc
/// carrying wing/room/hall keys counts toward `total_with_spatial`, wing+room
/// pairs fill the wings→rooms map and halls collect their keys.
///
/// Extracted from `handle_spatial_info` (repowise plan
/// `refac4_6f2457a170b6b105922b`) — deviation from the plan signature: wings
/// and halls are filled here, so they pass by `&mut` (the plan's `&` would
/// not compile). Behavior identical; the guard stays held by the caller.
fn scan_spatial_metadata(
    all_ids: &[String],
    s: &McpState,
    wings: &mut std::collections::HashMap<String, std::collections::HashSet<String>>,
    halls: &mut std::collections::HashSet<String>,
) -> usize {
    let mut total_with_spatial = 0usize;

    for id in all_ids {
        let Ok(Some(rec)) = s.doc_store.get(id) else {
            continue;
        };
        let Some(ref meta) = rec.metadata else {
            continue;
        };
        let has_wing = meta.get("wing").and_then(|v| v.as_str()).is_some();
        let has_room = meta.get("room").and_then(|v| v.as_str()).is_some();
        let has_hall = meta.get("hall").and_then(|v| v.as_str());

        if has_wing || has_room || has_hall.is_some() {
            total_with_spatial += 1;
        }

        if let (Some(w), Some(r)) = (
            meta.get("wing").and_then(|v| v.as_str()),
            meta.get("room").and_then(|v| v.as_str()),
        ) {
            wings
                .entry(w.to_string())
                .or_default()
                .insert(r.to_string());
        }

        if let Some(h) = has_hall {
            halls.insert(h.to_string());
        }
    }
    total_with_spatial
}

// ============================================================================
// Verbatim Storage + AAAK Compression handlers
// ============================================================================

fn handle_verbatim_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query too long".into());
    }
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    let top_k = top_k.min(1000);

    // Read dimension and active count under a brief lock, then compute embedding
    // WITHOUT holding the mutex — get_embedding() may make an HTTP call (5s timeout).
    let (dim, n_active) = {
        let s = state.lock().map_err(|e| {
            eprintln!("[mcp] mutex poisoned: {}", e);
            "internal server error".to_string()
        })?;
        let dim = s.store.get_statistics().embedding_dim;
        let n = s.store.n_active();
        (dim, n)
    }; // lock dropped here

    if n_active == 0 {
        return Ok(json!({ "results": [], "message": "No documents in store" }));
    }

    let (embedding, embedding_src) = match parse_embedding_param(params) {
        Some(e) => (e, EmbeddingSource::Provided),
        None => get_embedding_labeled(query, dim)?,
    };
    let query_vec = Array1::from_vec(embedding);

    // Re-acquire lock for the actual search and result assembly
    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    // Over-fetch by the number of dead vectors (deleted / re-stored),
    // drop them, and keep `top_k` live results.
    let k = s.fetch_k(top_k, s.store.n_active());
    let neighbors = s.store.find_neighbors_fast(&query_vec.view(), k);

    let results: Vec<Value> = neighbors
        .into_iter()
        .filter(|n| s.is_live(n.index))
        .take(top_k)
        .map(|n| {
            let doc_id = s.vector_to_doc.get(&n.index).cloned().unwrap_or_default();
            let text = s.doc_text(&doc_id);

            // Confidence classification
            let (confidence, emoji, explanation) = if n.distance < 0.3 {
                ("HIGH", "🟢", "Very reliable match")
            } else if n.distance < 0.6 {
                ("MEDIUM", "🟡", "Good match — verify context")
            } else {
                ("LOW", "🔴", "Weak match — verify against source")
            };

            let similarity = (1.0 / (1.0 + n.distance as f64)).min(1.0);

            json!({
                "id": doc_id,
                "index": n.index,
                "distance": (n.distance * 10000.0).round() / 10000.0,
                "similarity": (similarity * 10000.0).round() / 10000.0,
                "confidence": confidence,
                "confidence_emoji": emoji,
                "explanation": explanation,
                "text": text,
                "warning": if confidence == "LOW" {
                    Some("LOW CONFIDENCE — may be hallucinated, verify against original source")
                } else {
                    None
                },
            })
        })
        .collect();

    let high_count = results
        .iter()
        .filter(|r| r["confidence"].as_str() == Some("HIGH"))
        .count();
    let medium_count = results
        .iter()
        .filter(|r| r["confidence"].as_str() == Some("MEDIUM"))
        .count();
    let low_count = results
        .iter()
        .filter(|r| r["confidence"].as_str() == Some("LOW"))
        .count();

    Ok(json!({
        "query": query,
        "results": results,
        "embedding": embedding_src.label(),
        "summary": {
            "total": results.len(),
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "reliable": high_count + medium_count,
        },
        "disclaimer": "Always verify LOW confidence results against original source text to prevent hallucination",
    }))
}

fn handle_compress(params: &Value) -> Result<Value, String> {
    let text = params["text"].as_str().ok_or("missing 'text' field")?;
    if text.len() > 100_000 {
        return Err("text too long".into());
    }
    let result = crate::text_compression::compress(text);

    let hex_data: String = result
        .binary_data
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    Ok(json!({
        "semantic_text": result.semantic_text,
        "original_size": result.original_size,
        "semantic_size": result.semantic_size,
        "binary_size": result.binary_size,
        "semantic_ratio": format!("{:.2}×", result.semantic_ratio),
        "total_ratio": format!("{:.2}×", result.compression_ratio),
        "data": hex_data,
    }))
}

fn handle_decompress(params: &Value) -> Result<Value, String> {
    let hex_data = params["data"].as_str().ok_or("missing 'data' field")?;
    const MAX_DECOMPRESS_HEX_LEN: usize = 20_000_000; // 10MB binary = 20M hex chars
    if hex_data.len() > MAX_DECOMPRESS_HEX_LEN {
        return Err("data too large".into());
    }
    let binary_data = hex_to_bytes(hex_data)?;

    let text = crate::text_compression::decompress(&binary_data)
        .map_err(|e| format!("Decompression error: {}", e))?;

    Ok(json!({
        "text": text,
        "binary_size": binary_data.len(),
        "text_size": text.len(),
    }))
}

/// Strict hex decode for `handle_decompress`.
///
/// The previous implementation (`(0..len).step_by(2).filter_map(from_str_radix)`)
/// PANICKED on odd-length input (slice out of bounds) and on multibyte UTF-8
/// (indexing at non-char-boundaries), killing the whole MCP process — there
/// is no catch_unwind in the dispatcher — and silently DROPPED non-hex pairs.
/// This version rejects all of those with a clean Err. ASCII whitespace
/// between pairs (hex-dump style) is tolerated.
fn hex_to_bytes(hex_data: &str) -> Result<Vec<u8>, String> {
    let compact: Vec<u8> = hex_data
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if !compact.iter().all(|b| b.is_ascii_hexdigit()) {
        return Err("data contains non-hex characters".into());
    }
    if !compact.len().is_multiple_of(2) {
        return Err("data must contain an even number of hex digits".into());
    }
    let (pairs, _remainder) = compact.as_chunks::<2>();
    Ok(pairs
        .iter()
        .map(|p| hex_digit(p[0]) << 4 | hex_digit(p[1]))
        .collect())
}

/// Hex digit → value. Only called on bytes already validated with
/// `is_ascii_hexdigit`, so the fallback arm is unreachable in practice.
fn hex_digit(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        _ => b - b'A' + 10, // b'A'..=b'F'
    }
}

// ============================================================================
// JSON-RPC dispatcher
// ============================================================================

fn dispatch_request(state: &Mutex<McpState>, req: &JsonRpcRequest) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "splatsdb",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        },

        "notifications/initialized" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: None,
            result: None,
            error: None,
        },

        // MCP liveness probe answers an empty result object (was -32601).
        "ping" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: Some(json!({})),
            error: None,
        },

        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: Some(json!({ "tools": tool_definitions() })),
            error: None,
        },

        "tools/call" => {
            let params = req.params.as_ref();
            let tool_name = params.and_then(|p| p["name"].as_str()).unwrap_or("");

            let arguments = params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or(json!({}));

            let result = match tool_name {
                "splatsdb_store" => handle_store(state, &arguments),
                "splatsdb_search" => handle_search(state, &arguments),
                "splatsdb_status" => handle_status(state),
                "splatsdb_doc_add" => handle_doc_add(state, &arguments),
                "splatsdb_doc_get" => handle_doc_get(state, &arguments),
                "splatsdb_doc_del" => handle_doc_del(state, &arguments),
                "splatsdb_graph_add_doc" => handle_graph_add_doc(state, &arguments),
                "splatsdb_graph_add_entity" => handle_graph_add_entity(state, &arguments),
                "splatsdb_graph_add_relation" => handle_graph_add_relation(state, &arguments),
                "splatsdb_graph_traverse" => handle_graph_traverse(state, &arguments),
                "splatsdb_graph_search" => handle_graph_search(state, &arguments),
                "splatsdb_graph_search_entities" => handle_graph_search_entities(state, &arguments),
                "splatsdb_graph_stats" => handle_graph_stats(state),
                "splatsdb_spatial_search" => handle_spatial_search(state, &arguments),
                "splatsdb_spatial_info" => handle_spatial_info(state),
                "splatsdb_verbatim_search" => handle_verbatim_search(state, &arguments),
                "splatsdb_compress" => handle_compress(&arguments),
                "splatsdb_decompress" => handle_decompress(&arguments),
                // An unknown tool is a PROTOCOL error (invalid params),
                // not a tool execution error wrapped in `isError` (A03-F15).
                _ => {
                    return JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: req.id.clone(),
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: format!("Unknown tool: {}", tool_name),
                        }),
                    };
                }
            };

            match result {
                Ok(val) => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id.clone(),
                    result: Some(json!({
                        "content": [{ "type": "text", "text": val.to_string() }],
                        "isError": false
                    })),
                    error: None,
                },
                Err(msg) => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id.clone(),
                    result: Some(json!({
                        "content": [{ "type": "text", "text": format!("Error: {}", msg) }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        }

        _ => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
            }),
        },
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Maximum bytes for a single JSON-RPC line on stdin (10MB) — larger lines
/// are rejected and drained (see `read_request_line`).
const MAX_JSONRPC_LINE_LEN: usize = 10_000_000;

pub fn run_mcp_server() {
    eprintln!(
        "[splatsdb] MCP server v{} starting (stdio transport)...",
        env!("CARGO_PKG_VERSION")
    );

    SIMCOS_ALLOWED.store(
        simcos_opt_in(&std::env::var("SPLATSDB_ALLOW_SIMCOS").unwrap_or_default()),
        Ordering::Relaxed,
    );

    let config = SplatsDBConfig::mcp(None);
    let mut store = SplatStore::new(config);

    // Initialize SQLite doc store
    let (doc_store, doc_store_persistent) = init_doc_store();

    // Warm start: reload documents from SQLite into vector store.
    // SPLATDB_SPATIAL_WARMSTART=0|off|false skips the spatial registration
    // phase (and its metadata extraction) — useful for large stores where
    // spatial search is not used. Default: enabled.
    let spatial_warmstart_enabled = spatial_warmstart_enabled_from_env();
    let warm_start_t0 = std::time::Instant::now();

    let mut outcome = warm_start_reload(
        &doc_store,
        &mut store,
        spatial_warmstart_enabled,
        warm_start_t0,
    );

    let state = Mutex::new(McpState::from_warm_start(
        store,
        doc_store,
        doc_store_persistent,
        &mut outcome,
    ));

    register_warm_start_spatial(
        &state,
        &outcome.spatial,
        spatial_warmstart_enabled,
        warm_start_t0,
    );

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut reader = stdin.lock();

    // Set up signal handler for graceful shutdown
    #[cfg(unix)]
    {
        unsafe {
            libc::signal(libc::SIGINT, handle_sigint as *const () as usize);
        }
    }

    eprintln!("[splatsdb] MCP server ready. Waiting for JSON-RPC on stdin...");
    eprintln!(
        "[splatsdb] Env vars: SPLATSDB_EMBED_URL={}, SPLATSDB_DOC_PATH={}, SPLATSDB_ALLOW_SIMCOS={}",
        std::env::var("SPLATSDB_EMBED_URL").unwrap_or_else(|_| "(default)".into()),
        std::env::var("SPLATSDB_DOC_PATH").unwrap_or_else(|_| "(default)".into()),
        if SIMCOS_ALLOWED.load(Ordering::Relaxed) { "1 (SimCos fallback AUTHORIZED)" } else { "0" }
    );

    // Bounded line reader (fix: `reader.lines()` buffers the ENTIRE line in
    // memory before the 10MB check below could reject it — one huge line
    // could OOM the server). read_until under a take() cap bounds the buffer
    // at MAX+1 bytes; the remainder of an oversized line is drained in 8KB
    // chunks and skipped, never buffered whole.
    let mut raw_line: Vec<u8> = Vec::with_capacity(4096);
    // Bytes that followed an oversized line's '\n' inside a drained chunk:
    // they are the START of the next (valid) request and must NOT be thrown
    // away, or the JSON-RPC stream desynchronizes and the next message is
    // corrupted. They are seeded into the next line read instead.
    let mut pending: Vec<u8> = Vec::new();
    loop {
        if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
            eprintln!("[splatsdb] Shutdown requested, exiting main loop.");
            break;
        }

        let line = match read_request_line(&mut reader, &mut raw_line, &mut pending) {
            LineRead::Eof => break,
            LineRead::Skip => {
                // The oversized line was drained inside read_request_line.
                // Reply anyway: the client's request (undetectable → id:null)
                // must not hang forever waiting for a response (17-nov).
                write_rpc_error(
                    &mut stdout,
                    -32600,
                    &format!(
                        "Invalid Request: message exceeds {} bytes",
                        MAX_JSONRPC_LINE_LEN
                    ),
                );
                continue;
            }
            LineRead::Line(line) => line,
        };

        {
            if line.is_empty() {
                continue;
            }
            if line.len() > MAX_JSONRPC_LINE_LEN {
                eprintln!(
                    "[splatsdb] rejecting oversized JSON-RPC message ({} bytes)",
                    line.len()
                );
                write_rpc_error(
                    &mut stdout,
                    -32600,
                    &format!(
                        "Invalid Request: message exceeds {} bytes",
                        MAX_JSONRPC_LINE_LEN
                    ),
                );
                continue;
            }

            let req: JsonRpcRequest = match parse_jsonrpc_request(&line, &mut stdout) {
                Ok(r) => r,
                Err(()) => continue,
            };

            let is_notification = req.id.is_none() && req.method != "initialize";

            // Defense-in-depth: a panic inside any tool handler must not kill
            // the whole MCP stdio process — the client would lose the stream.
            // `state` is a `&Mutex` (not UnwindSafe, by poisoning design) and
            // `req` is not guaranteed UnwindSafe either → AssertUnwindSafe.
            // dispatch_request returns JsonRpcResponse directly (handlers map
            // their Result into error responses internally), so we only need
            // to map the Err(panic-payload) arm. If a panic poisoned the
            // mutex, later handlers already degrade via their
            // `lock().map_err(...)` arms instead of panicking again.
            let resp = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dispatch_request(&state, &req)
            })) {
                Ok(resp) => resp,
                Err(payload) => {
                    let detail = panic_payload_detail(&payload);
                    // Panic detail goes to server stderr only, never to the
                    // client (same convention as api_server.rs).
                    eprintln!(
                        "[splatsdb] PANIC in dispatch_request (method={}): {}",
                        req.method, detail
                    );
                    JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: req.id.clone(),
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32603,
                            message: "Internal error".into(),
                        }),
                    }
                }
            };

            if !is_notification && resp.id.is_some() {
                write_response(&resp, &mut stdout);
            }
        }
    }

    eprintln!("[splatsdb] MCP server shutting down (stdin closed.");
}

/// (doc_id, wing, room, hall) pairs collected during warm start for spatial
/// registration afterwards.
type WarmStartSpatial = Vec<(String, Option<String>, Option<String>, Option<String>)>;

/// Register the spatial metadata collected during warm start into the live
/// index and report the outcome on stderr.
///
/// Extracted from `run_mcp_server` (repowise plan `refac4_c7b69cacc17990c0271a`).
/// Behavior is identical to the former scoped block, including the lock scope:
/// the state mutex is acquired and released entirely inside this helper, never
/// held across the request loop that follows.
fn register_warm_start_spatial(
    state: &Mutex<McpState>,
    warm_start_spatial: &WarmStartSpatial,
    spatial_warmstart_enabled: bool,
    warm_start_t0: std::time::Instant,
) {
    // Never `unwrap()` a production lock (A05-F12): a poisoned mutex must
    // degrade to a logged, controlled failure — not kill the stdio server.
    let mut s = match state.lock() {
        Ok(guard) => guard,
        Err(e) => {
            eprintln!("[splatsdb] ERROR: state mutex poisoned, skipping spatial warm-start registration: {e}");
            return;
        }
    };
    for (doc_id, wing, room, hall) in warm_start_spatial {
        s.spatial_index
            .register_doc(doc_id, wing.as_deref(), room.as_deref(), hall.as_deref());
    }
    if !warm_start_spatial.is_empty() {
        eprintln!(
            "[splatsdb] Warm start: registered {} spatial metadata entries in {:.0} ms (SPLATDB_SPATIAL_WARMSTART={})",
            warm_start_spatial.len(),
            warm_start_t0.elapsed().as_secs_f64() * 1000.0,
            if spatial_warmstart_enabled { "on" } else { "off" }
        );
    } else if !spatial_warmstart_enabled {
        eprintln!(
            "[splatsdb] Warm start: spatial registration skipped (SPLATDB_SPATIAL_WARMSTART=0)"
        );
    }
}

/// Initialize the SQLite doc store from `SPLATSDB_DOC_PATH` (default
/// `$HOME/.hermes/splatsdb_docs.db`), degrading to an in-memory database if
/// the on-disk one cannot be opened (docs won't persist across restarts).
///
/// Extracted from `run_mcp_server`; messages and behavior identical,
/// including the FATAL exit when even `:memory:` fails.
fn init_doc_store() -> (SqliteMetadataStore, bool /* persistent */) {
    let db_path = std::env::var("SPLATSDB_DOC_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            std::path::PathBuf::from(format!("{home}/.hermes/splatsdb_docs.db"))
        });
    match SqliteMetadataStore::new(db_path.clone()) {
        Ok(ds) => {
            eprintln!("[splatsdb] SQLite doc store: {:?}", db_path);
            (ds, true)
        }
        Err(e) => {
            eprintln!(
                "[splatsdb] WARNING: SQLite init failed ({}), docs won't persist — running on an in-memory store (ephemeral:true, persisted:false)",
                e
            );
            let ephemeral = SqliteMetadataStore::new(std::path::PathBuf::from(":memory:"))
                .unwrap_or_else(|e| {
                    eprintln!("[splatsdb] FATAL: in-memory SQLite failed: {}", e);
                    std::process::exit(1);
                });
            (ephemeral, false)
        }
    }
}

/// Warm start: reload persisted documents from SQLite into the vector store.
///
/// F05b (A03-F10, persistence half): a document that has a VALID stored
/// vector is indexed from that vector — no embed service, no re-embedding —
/// and keeps the provenance it was stored with (`stored_real` /
/// `stored_simcos` / `stored_provided`; SimCos is never promoted).
///
/// - No stored vector (legacy rows of the pre-vector schema): embedded only
///   when the real service answers, or SimCos under explicit opt-in (the
///   A03-F10 strictness is unchanged); the vector is then backfilled so the
///   next restart needs no service. Without either the document stays
///   readable but unindexed — counted and reported, never deleted.
/// - Stored vector that fails validation (length ≠ dim·4, non-finite,
///   another dimension, missing/unknown label): QUARANTINED — not indexed,
///   not resized, not overwritten, reported with a per-document diagnostic.
///
/// Caps at `MAX_WARM_START_DOCS`, honours the shutdown flag mid-loop, skips
/// empty texts, extracts spatial metadata only when
/// `spatial_warmstart_enabled`, advances `next_id` past any `mem_N` id, syncs
/// HNSW once at the end and reports the accounting on stderr.
fn warm_start_reload(
    doc_store: &SqliteMetadataStore,
    store: &mut SplatStore,
    spatial_warmstart_enabled: bool,
    warm_start_t0: std::time::Instant,
) -> WarmStartOutcome {
    let mut outcome = WarmStartOutcome {
        vector_to_doc: HashMap::new(),
        doc_texts: DocTextCache::default(),
        next_id: 0,
        spatial: Vec::new(),
        stats: WarmStartStats::default(),
        sources: HashMap::new(),
    };
    let stats = &mut outcome.stats;
    let mut quarantined: Vec<String> = Vec::new();
    let dim = store.get_statistics().embedding_dim;
    let total = doc_store.count(false).unwrap_or(0);
    stats.documents = total;
    match doc_store.list_with_embeddings(MAX_WARM_START_DOCS, dim) {
        Ok(docs) => {
            if total > MAX_WARM_START_DOCS {
                eprintln!(
                    "[splatsdb] Warm start: {} docs found, capping at {} for performance",
                    total, MAX_WARM_START_DOCS
                );
            }
            for (i, stored) in docs.iter().enumerate() {
                if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
                    eprintln!("[splatsdb] Warm start interrupted by shutdown signal");
                    break;
                }
                let doc_id = stored.record.id.as_str();
                let text = stored.record.document.clone().unwrap_or_default();
                if text.is_empty() {
                    continue;
                }

                // Extract spatial metadata from the JSON metadata field
                // (skipped entirely when spatial warm-start is disabled)
                let meta = stored.record.metadata.as_ref();
                let (meta_wing, meta_room, meta_hall) = if spatial_warmstart_enabled {
                    let key = |k: &str| {
                        meta.and_then(|m| m.get(k))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    };
                    (key("wing"), key("room"), key("hall"))
                } else {
                    (None, None, None)
                };

                // Where the indexed vector comes from (see the doc comment).
                let (embedding, source) = match (&stored.embedding, &stored.embedding_error) {
                    (Some(kept), _) => {
                        stats.vectors_loaded += 1;
                        match kept.source {
                            EmbeddingProvenance::Real => stats.stored_real += 1,
                            EmbeddingProvenance::SimCos => stats.stored_simcos += 1,
                            EmbeddingProvenance::Provided => stats.stored_provided += 1,
                        }
                        (kept.vector.clone(), kept.source)
                    }
                    (None, Some(why)) => {
                        stats.quarantined += 1;
                        quarantined.push(format!("{doc_id}: {why}"));
                        continue;
                    }
                    (None, None) => match get_embedding_labeled(&text, dim) {
                        Ok((vector, src)) => {
                            // Backfill the legacy row so the next restart
                            // needs no service. A write that cannot happen is
                            // reported, not hidden.
                            if let Err(e) =
                                doc_store.set_embedding(doc_id, &vector, provenance_of(src))
                            {
                                eprintln!(
                                    "[splatsdb] WARNING: Warm start: could not persist the embedding of legacy doc {doc_id}: {e}"
                                );
                            }
                            stats.reembedded += 1;
                            (vector, provenance_of(src))
                        }
                        Err(_) => {
                            stats.unindexed += 1;
                            continue;
                        }
                    },
                };
                if let Ok(arr) = Array2::from_shape_vec((1, dim), embedding) {
                    let vector_idx = store.n_active();
                    if store.add_splat(&arr) {
                        outcome.vector_to_doc.insert(vector_idx, doc_id.to_string());
                        outcome.doc_texts.insert(doc_id.to_string(), text);
                        outcome.sources.insert(doc_id.to_string(), source);
                        outcome.next_id = outcome.next_id.max(
                            doc_id
                                .strip_prefix("mem_")
                                .and_then(|s| s.parse::<usize>().ok())
                                .unwrap_or(0),
                        );
                        // Save spatial metadata for later registration
                        if meta_wing.is_some() || meta_room.is_some() || meta_hall.is_some() {
                            outcome.spatial.push((
                                doc_id.to_string(),
                                meta_wing.clone(),
                                meta_room.clone(),
                                meta_hall.clone(),
                            ));
                        }
                    }
                }
                if i > 0 && i % 100 == 0 {
                    eprintln!(
                        "[splatsdb] Warm start: {}/{} docs loaded",
                        i,
                        total.min(MAX_WARM_START_DOCS)
                    );
                }
            }
            if !docs.is_empty() {
                store.hnsw_sync_incremental();
            }
            if total > 0 {
                let indexed = outcome.doc_texts.len();
                eprintln!(
                    "[splatsdb] Warm start: reloaded {indexed}/{total} documents from SQLite in {:.0} ms (stored vectors loaded: {} [real={} simcos={} provided={}], re-embedded: {}, not indexed: {}, quarantined: {})",
                    warm_start_t0.elapsed().as_secs_f64() * 1000.0,
                    stats.vectors_loaded,
                    stats.stored_real,
                    stats.stored_simcos,
                    stats.stored_provided,
                    stats.reembedded,
                    stats.unindexed,
                    stats.quarantined
                );
            }
        }
        Err(e) => eprintln!("[splatsdb] Warm start: could not list docs: {e}"),
    }
    if stats.unindexed > 0 {
        eprintln!(
            "[splatsdb] WARNING: Warm start: {} documents were NOT indexed (no stored vector — legacy rows — and no usable embedding now: embed service down and the SimCos fallback is not authorized). They stay in SQLite but are invisible to search until a restart with the embed service up, with SPLATSDB_ALLOW_SIMCOS=1, or until they are re-stored.",
            stats.unindexed
        );
    }
    if stats.quarantined > 0 {
        eprintln!(
            "[splatsdb] WARNING: Warm start: {} stored vectors QUARANTINED (refused, not indexed; the blobs were left untouched on disk — re-store the document to replace it): {}",
            stats.quarantined,
            quarantined
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    if stats.stored_simcos > 0 {
        eprintln!(
            "[splatsdb] WARNING: Warm start: {} stored vectors are SimCos text-hash (labelled stored_simcos, never promoted): ranking for those documents is not semantic — re-store them once the embed service is up",
            stats.stored_simcos
        );
    }
    outcome
}

/// Outcome of one bounded line read from the JSON-RPC stdin stream.
enum LineRead {
    /// A complete (possibly empty after trimming) request line.
    Line(String),
    /// Oversized line rejected and drained — skip to the next read.
    Skip,
    /// EOF or stdin error — leave the main loop.
    Eof,
}

/// Bounded line reader (fix: `reader.lines()` buffers the ENTIRE line in
/// memory before the 10MB check could reject it — one huge line could OOM
/// the server). `read_until` under a `take()` cap bounds the buffer at
/// MAX+1 bytes; the remainder of an oversized line is drained in 8KB chunks
/// and skipped, never buffered whole.
///
/// Extracted from `run_mcp_server`; behavior identical, including the
/// `pending` seeding: bytes that followed an oversized line's '\n' inside a
/// drained chunk are the START of the next (valid) request and must NOT be
/// thrown away, or the JSON-RPC stream desynchronizes.
fn read_request_line(
    reader: &mut impl std::io::BufRead,
    raw_line: &mut Vec<u8>,
    pending: &mut Vec<u8>,
) -> LineRead {
    raw_line.clear();
    let read = if pending.is_empty() {
        let mut capped = reader.take(MAX_JSONRPC_LINE_LEN as u64 + 1);
        capped.read_until(b'\n', raw_line)
    } else if let Some(pos) = pending.iter().position(|&b| b == b'\n') {
        // Pending holds ≥1 COMPLETE line (pipelined requests that followed the
        // oversized one): return the first now and keep the rest queued.
        // Returning it without consuming fresh stdin keeps the JSON-RPC stream
        // in sync — merging it with the next read_until would concatenate
        // distinct requests into one unparseable line, or hang waiting for a
        // newline the client never sends (17-nov EGM audit).
        let line_bytes: Vec<u8> = pending[..=pos].to_vec();
        let rest: Vec<u8> = pending[(pos + 1)..].to_vec();
        pending.clear();
        pending.extend_from_slice(&rest);
        raw_line.extend_from_slice(&line_bytes);
        return LineRead::Line(String::from_utf8_lossy(raw_line).trim().to_string());
    } else {
        // Pending is a partial tail (no newline yet): seed it and keep reading
        // the current line under the same cap (identical semantics to a
        // fresh read: newline detection and oversize check still apply).
        raw_line.extend_from_slice(pending);
        pending.clear();
        let seeded = raw_line.len();
        let mut capped = reader.take(MAX_JSONRPC_LINE_LEN as u64 + 1);
        match capped.read_until(b'\n', raw_line) {
            Ok(n) => Ok(seeded + n),
            Err(e) => Err(e),
        }
    };
    match read {
        Ok(0) => LineRead::Eof, // EOF — stdin closed
        Ok(_) => {
            let ends_nl = raw_line.last() == Some(&b'\n');
            // Hit the MAX+1 cap with no newline → oversized line: reject
            // and drain the rest in bounded chunks.
            if !ends_nl && raw_line.len() > MAX_JSONRPC_LINE_LEN {
                eprintln!(
                    "[splatsdb] rejecting oversized JSON-RPC message (>{} bytes)",
                    MAX_JSONRPC_LINE_LEN
                );
                let mut discard = [0u8; 8192];
                loop {
                    match reader.read(&mut discard) {
                        Ok(0) => break, // EOF
                        Ok(m) => {
                            if let Some(pos) = discard[..m].iter().position(|&b| b == b'\n') {
                                // Keep whatever follows the newline: the
                                // next request already arrived.
                                pending.extend_from_slice(&discard[pos + 1..m]);
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                LineRead::Skip
            } else {
                LineRead::Line(String::from_utf8_lossy(raw_line).trim().to_string())
            }
        }
        Err(e) => {
            eprintln!("[splatsdb] stdin error: {}", e);
            LineRead::Eof
        }
    }
}

/// Parse and validate one JSON-RPC request line. On a parse error or a bad
/// jsonrpc version, writes the corresponding error response to stdout and
/// returns `Err(())` (the caller skips the line). On success logs the method
/// and returns the request.
/// Write a JSON-RPC error response with `id: null` (request undetectable).
fn write_rpc_error(stdout: &mut io::Stdout, code: i32, message: &str) {
    let resp = JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id: None,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.to_string(),
        }),
    };
    let out = serde_json::to_string(&resp).unwrap_or_else(|_| {
        r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid Request"}}"#
            .to_string()
    });
    writeln!(stdout, "{}", out).ok();
    stdout.flush().ok();
}

/// Parse one JSON-RPC line.
///
/// Extracted from `run_mcp_server`; messages, error codes and behavior
/// identical.
fn parse_jsonrpc_request(line: &str, stdout: &mut io::Stdout) -> Result<JsonRpcRequest, ()> {
    let req: JsonRpcRequest = match serde_json::from_str::<JsonRpcRequest>(line) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[splatsdb] parse error: {}", e);
            let msg = e.to_string();
            // JSON-RPC 2.0 spec: syntactically invalid JSON → -32700 Parse
            // error; valid JSON with an invalid request schema (missing
            // field) → -32600 Invalid Request (17-nov EGM audit).
            let (code, prefix) = if msg.starts_with("missing field") {
                (-32600, "Invalid Request")
            } else {
                (-32700, "Parse error")
            };
            let resp = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code,
                    message: format!("{prefix}: {msg}"),
                }),
            };
            let out = serde_json::to_string(&resp).unwrap_or_else(|_| {
                r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}"#
                    .to_string()
            });
            writeln!(stdout, "{}", out).ok();
            stdout.flush().ok();
            return Err(());
        }
    };

    // Validate jsonrpc version
    if req.jsonrpc != "2.0" {
        eprintln!("[splatsdb] invalid jsonrpc version: {}", req.jsonrpc);
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: format!(
                    "Invalid jsonrpc version: expected '2.0', got '{}'",
                    req.jsonrpc
                ),
            }),
        };
        let out = serde_json::to_string(&resp).unwrap_or_else(|_| {
            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid request"}}"#
                .to_string()
        });
        writeln!(stdout, "{}", out).ok();
        stdout.flush().ok();
        return Err(());
    }
    eprintln!("[splatsdb] <- method={}", req.method);
    Ok(req)
}

/// Serialize a JSON-RPC response and write it as one stdout line, with a
/// 200-char preview on stderr. On serialization failure writes the
/// canonical internal-error response instead.
///
/// Extracted from `run_mcp_server`; behavior identical.
fn write_response(resp: &JsonRpcResponse, stdout: &mut io::Stdout) {
    let out = match serde_json::to_string(resp) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[splatsdb] serialization error: {}", e);
            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#
                .to_string()
        }
    };
    let preview: String = out.chars().take(200).collect();
    eprintln!("[splatsdb] -> {}...", preview);
    writeln!(stdout, "{}", out).ok();
    stdout.flush().ok();
}

/// Best-effort string rendering of a panic payload caught by
/// `catch_unwind` around `dispatch_request`.
///
/// Extracted from `run_mcp_server`; behavior identical.
fn panic_payload_detail(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// SPLATDB_SPATIAL_WARMSTART parser: absent/anything-but-falsy → true
/// (default ON); "0", "off", "false", "no" (case-insensitive, trimmed) →
/// false.
fn spatial_warmstart_enabled_from_env() -> bool {
    match std::env::var("SPLATDB_SPATIAL_WARMSTART") {
        Ok(v) => !spatial_warmstart_is_falsy(&v),
        Err(_) => true,
    }
}

/// Pure parser — separated from the env read so it is directly testable
/// (no `std::env::set_var` in tests: racy under parallel test threads).
fn spatial_warmstart_is_falsy(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "0" | "off" | "false" | "no"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LRU behavior with a 4-key cache: insertion-order (FIFO) eviction,
    /// tombstone skipping, and the generation guard — a re-inserted key must
    /// survive when its own older tombstone reaches the eviction front.
    #[test]
    fn doc_text_cache_eviction_and_tombstones() {
        let mut c = DocTextCache {
            cap: 4,
            ..Default::default()
        };

        // Fill to cap: FIFO order, no eviction yet.
        for (k, v) in [("k1", "1"), ("k2", "2"), ("k3", "3"), ("k4", "4")] {
            c.insert(k.into(), v.into());
        }
        assert_eq!(c.live_order(), vec!["k1", "k2", "k3", "k4"]);
        assert_eq!(c.len(), 4);

        // O(1) remove: entry leaves the map, order queue keeps a tombstone.
        c.remove("k2");
        assert_eq!(c.len(), 3);
        assert!(c.get("k2").is_none());
        assert_eq!(c.live_order(), vec!["k1", "k3", "k4"]);

        // Re-insert k2: two queue entries for k2 (old + new gen); only the
        // newest is live.
        c.insert("k2".into(), "2b".into());
        assert_eq!(c.get("k2"), Some(&"2b".to_string()));
        assert_eq!(c.live_order(), vec!["k1", "k3", "k4", "k2"]);

        // Replacement of a live key: text updated, eviction order untouched.
        c.insert("k4".into(), "4b".into());
        assert_eq!(c.get("k4"), Some(&"4b".to_string()));
        assert_eq!(c.live_order(), vec!["k1", "k3", "k4", "k2"]);
        assert_eq!(c.len(), 4);

        // THE critical scenario: remove k1 (tombstone for gen 1 sits at the
        // front of the queue) and re-insert it immediately (new gen at the
        // back). Queue front is now: [k1-tombstone, k2-stale, k3, k4, k2, k1].
        c.remove("k1");
        assert_eq!(c.live_order(), vec!["k3", "k4", "k2"]);
        c.insert("k1".into(), "1b".into());
        assert_eq!(c.get("k1"), Some(&"1b".to_string()));
        assert_eq!(c.live_order(), vec!["k3", "k4", "k2", "k1"]);

        // Churn: each insert pops stale entries from the front (skipping
        // them) until it evicts the oldest LIVE entry. n0 must evict k3 —
        // and must NOT evict k1, whose own tombstone was popped and skipped.
        c.insert("n0".into(), "v0".into());
        assert_eq!(c.len(), 4);
        assert!(
            c.get("k1").is_some(),
            "re-inserted k1 must survive its own tombstone"
        );
        assert!(
            c.get("k3").is_none(),
            "oldest live entry k3 must be evicted"
        );
        assert_eq!(c.live_order(), vec!["k4", "k2", "k1", "n0"]);

        c.insert("n1".into(), "v1".into());
        assert_eq!(c.live_order(), vec!["k2", "k1", "n0", "n1"]);

        c.insert("n2".into(), "v2".into());
        assert_eq!(c.live_order(), vec!["k1", "n0", "n1", "n2"]);

        // k1 eventually evicted too — it was inserted before n0..n2 (FIFO),
        // and its queue entry is the live generation this time.
        c.insert("n3".into(), "v3".into());
        assert!(c.get("k1").is_none());
        assert_eq!(c.live_order(), vec!["n0", "n1", "n2", "n3"]);

        // Sustained churn: cap invariant holds forever.
        for i in 4..40 {
            c.insert(format!("n{i:02}"), format!("v{i}"));
            assert!(c.len() <= c.cap, "cap exceeded at i={i}: {}", c.len());
        }
        assert_eq!(c.len(), c.cap);
        assert_eq!(c.live_order(), vec!["n36", "n37", "n38", "n39"]);
    }

    /// P4: truth table of the SPLATDB_SPATIAL_WARMSTART parser (pure helper,
    /// no env mutation). Default (unset) → ON; falsy spellings → OFF.
    #[test]
    fn spatial_warmstart_flag_truth_table() {
        for v in ["0", "off", "false", "no", "OFF", "False", " 0 ", "NO"] {
            assert!(
                spatial_warmstart_is_falsy(v),
                "{v:?} must parse as disabled"
            );
        }
        for v in [
            "1",
            "on",
            "true",
            "yes",
            "",
            "anything-else",
            "TRUE",
            "  on  ",
        ] {
            assert!(
                !spatial_warmstart_is_falsy(v),
                "{v:?} must parse as enabled"
            );
        }
    }

    /// Fix: hex decode must never panic (odd length / multibyte UTF-8 slice
    /// out of bounds killed the whole MCP process) and must not silently
    /// drop non-hex pairs. Whitespace between pairs is tolerated.
    #[test]
    fn hex_decode_strict_no_panic() {
        // odd length → Err
        assert!(hex_to_bytes("abc").is_err());
        // non-hex pair → Err (previously silently dropped by filter_map)
        assert!(hex_to_bytes("zz").is_err());
        assert!(hex_to_bytes("0g").is_err());
        // whitespace between pairs tolerated
        assert_eq!(hex_to_bytes("ab cd").unwrap(), vec![0xab, 0xcd]);
        assert_eq!(hex_to_bytes("abcd").unwrap(), vec![0xab, 0xcd]);
        // multibyte UTF-8 → clean Err, not a panic
        assert!(hex_to_bytes("aé").is_err());
        assert!(hex_to_bytes("aéc").is_err());
        // case + roundtrip sanity
        assert_eq!(hex_to_bytes("00Ff10").unwrap(), vec![0x00, 0xff, 0x10]);
        assert!(hex_to_bytes("").unwrap().is_empty());
    }

    /// Fix: embeddings with NaN/inf components are rejected at every entry
    /// point (JSON `1e400` decodes to f32::INFINITY).
    #[test]
    fn all_finite_rejects_nan_and_inf() {
        assert!(all_finite(&[]));
        assert!(all_finite(&[0.0, -1.5, 3.25]));
        assert!(!all_finite(&[0.0, f32::INFINITY]));
        assert!(!all_finite(&[f32::NEG_INFINITY]));
        assert!(!all_finite(&[f32::NAN]));
        assert!(!all_finite(&[0.0, 1.0, f32::NAN, 2.0]));
    }

    /// A03-F10: the opt-in parser accepts only explicit affirmations.
    #[test]
    fn f10_simcos_opt_in_truth_table() {
        for v in ["1", "true", "yes", "on", "TRUE", " yes ", "On"] {
            assert!(simcos_opt_in(v), "{v:?} must parse as opt-in");
        }
        for v in ["", "0", "false", "no", "off", "2", "anything-else"] {
            assert!(!simcos_opt_in(v), "{v:?} must NOT parse as opt-in");
        }
    }

    /// A03-F10: without the service, the resolver errors unless SimCos was
    /// explicitly authorized — and then labels it. No silent path exists.
    #[test]
    fn f10_resolver_never_falls_back_silently() {
        let (svc, opt) = (
            EMBED_SERVICE_AVAILABLE.load(Ordering::Relaxed),
            SIMCOS_ALLOWED.load(Ordering::Relaxed),
        );
        EMBED_SERVICE_AVAILABLE.store(false, Ordering::Relaxed);
        SIMCOS_ALLOWED.store(false, Ordering::Relaxed);

        let err = get_embedding_labeled("hello world", 16)
            .expect_err("no service + no opt-in must fail closed");
        assert!(err.contains("embedding unavailable"), "{err}");
        assert!(err.contains("SPLATSDB_ALLOW_SIMCOS"), "{err}");

        SIMCOS_ALLOWED.store(true, Ordering::Relaxed);
        let (vec, src) =
            get_embedding_labeled("hello world", 16).expect("opt-in authorizes SimCos");
        assert_eq!(src, EmbeddingSource::SimCos);
        assert_eq!(src.label(), "simcos");
        assert_eq!(vec.len(), 16);
        assert_eq!(vec, simcos_embedding("hello world", 16));
        assert!(all_finite(&vec));

        EMBED_SERVICE_AVAILABLE.store(svc, Ordering::Relaxed);
        SIMCOS_ALLOWED.store(opt, Ordering::Relaxed);
        assert_eq!(EmbeddingSource::Real.label(), "real");
        assert_eq!(EmbeddingSource::Provided.label(), "provided");
    }

    /// Small, fast `McpState` for handler-level tests (8-dim store,
    /// in-memory docs, non-persistent).
    fn f05a_state() -> Mutex<McpState> {
        let mut config = SplatsDBConfig::mcp(None);
        config.max_splats = 256;
        config.latent_dim = 8;
        let store = SplatStore::new(config);
        let docs = SqliteMetadataStore::new(std::path::PathBuf::from(":memory:"))
            .expect("in-memory doc store");
        Mutex::new(McpState::new(store, docs, false))
    }

    fn f05a_emb(i: f64) -> Value {
        json!([1.0, 0.3 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }

    fn f05a_ids(res: &Value) -> Vec<String> {
        res["results"]
            .as_array()
            .unwrap_or_else(|| panic!("no results array: {res}"))
            .iter()
            .map(|r| r["id"].as_str().unwrap_or("").to_string())
            .collect()
    }

    /// F05b (A03-F10, persistence half): the warm start indexes stored
    /// vectors directly — each with the label it was stored with — instead
    /// of embedding again; a blob of another dimension is quarantined.
    #[test]
    fn p05_warm_start_loads_stored_vectors_with_their_labels() {
        let dir = std::env::temp_dir().join(format!(
            "splatsdb-mcp-p05-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let doc_store = SqliteMetadataStore::new(dir.join("docs.db")).expect("doc store");
        let record = |id: &str, text: &str| DocumentRecord {
            id: id.into(),
            shard_idx: 0,
            vector_idx: 0,
            metadata: Some(json!({ "text": text, "wing": format!("{id}_wing") })),
            document: Some(text.into()),
            deleted: false,
            created_at: 1.0,
            updated_at: 1.0,
        };
        doc_store
            .upsert_with_embedding(
                &record("r", "real text"),
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                EmbeddingProvenance::Real,
            )
            .unwrap();
        doc_store
            .upsert_with_embedding(
                &record("s", "simcos text"),
                &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                EmbeddingProvenance::SimCos,
            )
            .unwrap();
        doc_store
            .upsert_with_embedding(
                &record("p", "provided text"),
                &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                EmbeddingProvenance::Provided,
            )
            .unwrap();
        // Another dimension than the store → quarantined.
        doc_store
            .upsert_with_embedding(
                &record("bad", "bad dim text"),
                &[0.0; 4],
                EmbeddingProvenance::Real,
            )
            .unwrap();
        // A legacy row without any vector (pre-F05b writer).
        let mut legacy = record("legacy", "legacy text");
        legacy.metadata = Some(json!({ "text": "legacy text" }));
        doc_store.upsert(&legacy).unwrap();

        let mut config = SplatsDBConfig::mcp(None);
        config.max_splats = 256;
        config.latent_dim = 8;
        let mut store = SplatStore::new(config);
        let mut outcome =
            warm_start_reload(&doc_store, &mut store, true, std::time::Instant::now());

        let stats = outcome.stats;
        assert_eq!(stats.vectors_loaded, 3);
        assert_eq!(
            (
                stats.stored_real,
                stats.stored_simcos,
                stats.stored_provided
            ),
            (1, 1, 1)
        );
        assert_eq!(
            stats.quarantined, 1,
            "a 4-dim blob in an 8-dim store is quarantined"
        );
        // The legacy row is either embedded or reported unindexed — whatever
        // the process-global service/SimCos flags say while the other lib
        // tests run — but it is never loaded from a vector it does not have.
        assert_eq!(stats.reembedded + stats.unindexed, 1);

        let state = Mutex::new(McpState::from_warm_start(
            store,
            doc_store,
            true,
            &mut outcome,
        ));
        let res = handle_search(
            &state,
            &json!({ "query": "q", "embedding": f05a_emb(1.0), "top_k": 10 }),
        )
        .unwrap();
        let mut labels: Vec<(String, String)> = res["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                (
                    r["id"].as_str().unwrap().to_string(),
                    r["embedding_source"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        labels.sort();
        for (id, label) in [
            ("p", "stored_provided"),
            ("r", "stored_real"),
            ("s", "stored_simcos"),
        ] {
            let wanted = (id.to_string(), label.to_string());
            assert!(labels.contains(&wanted), "{labels:?}");
        }
        assert!(labels
            .iter()
            .all(|(_, l)| l.starts_with("stored_") || l == "simcos" || l == "real"));
        assert_eq!(res["embedding"], json!("provided"));
        let status = handle_status(&state).unwrap();
        assert_eq!(
            status["vector_persistence"]["warm_start"]["vectors_loaded"],
            json!(3)
        );
        assert_eq!(
            status["vector_persistence"]["warm_start"]["quarantined"],
            json!(1)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A03-F11 (handlers): doc_del tombstones the vector and top-k is
    /// refilled; re-store retires the old mapping (exactly one hit, new text).
    #[test]
    fn f11_handlers_tombstone_delete_and_restore() {
        let state = f05a_state();
        for i in 1..=5u32 {
            let id = format!("t{i}");
            let stored = handle_store(
                &state,
                &json!({ "text": format!("text {id} v1"), "id": id, "embedding": f05a_emb(i as f64) }),
            )
            .expect("store");
            assert_eq!(stored["id"], json!(id));
        }
        let search = |top_k: u64, q: Value| {
            handle_search(
                &state,
                &json!({ "query": "q", "embedding": q, "top_k": top_k }),
            )
            .expect("search")
        };
        assert_eq!(f05a_ids(&search(3, f05a_emb(3.0))), vec!["t3", "t2", "t4"]);

        handle_doc_del(&state, &json!({ "id": "t3" })).expect("del");
        {
            let s = state.lock().unwrap();
            assert_eq!(s.store.n_active(), 5, "the splat physically stays");
            assert_eq!(s.tombstones, std::collections::HashSet::from([2]));
            assert!(!s.doc_to_vec.contains_key("t3"));
            assert!(!s.vector_to_doc.contains_key(&2));
            assert_eq!(s.live_vectors(), 4);
            assert!(s.doc_texts.get("t3").is_none());
        }
        // Top-k is refilled past the tombstone: 4 live docs, never a `vec_N`.
        let res = search(5, f05a_emb(3.0));
        let got = f05a_ids(&res);
        assert_eq!(got.len(), 4, "{res}");
        assert!(!got.contains(&"t3".to_string()), "{res}");
        assert!(res["results"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| !r["text"].as_str().unwrap_or("").is_empty()));
        // status exposes the difference between physical and live vectors
        let status = handle_status(&state).unwrap();
        assert_eq!(status["n_active"], json!(5));
        assert_eq!(status["live_vectors"], json!(4));
        assert_eq!(status["tombstoned_vectors"], json!(1));

        // Re-store t1 with a NEW vector and text → exactly one hit, the new text.
        handle_store(
            &state,
            &json!({ "text": "text t1 v2", "id": "t1", "embedding": json!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) }),
        )
        .expect("re-store");
        for q in [
            json!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            f05a_emb(1.0),
        ] {
            let res = search(10, q);
            let got = f05a_ids(&res);
            assert_eq!(
                got.iter().filter(|i| *i == "t1").count(),
                1,
                "duplicate mapping: {res}"
            );
            assert_eq!(got.len(), 4, "{res}");
            let t1 = res["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["id"] == "t1")
                .unwrap();
            assert_eq!(t1["text"], json!("text t1 v2"), "stale text: {res}");
        }
        {
            let s = state.lock().unwrap();
            assert_eq!(s.live_vectors(), 4);
            assert_eq!(s.tombstones.len(), 2);
            assert_eq!(s.doc_to_vec["t1"], 5);
        }

        // Deleting an unknown doc is still a clean "not found".
        let err = handle_doc_del(&state, &json!({ "id": "nope" })).unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    /// A03-F14: with the text cache capped at ONE entry, the doc evicted by
    /// the next store still returns its text from `record.document` in every
    /// search handler (it used to come back as `"text": ""`).
    #[test]
    fn f14_search_text_falls_back_to_record_when_cache_evicted() {
        let state = f05a_state();
        state.lock().unwrap().doc_texts.cap = 1;
        for i in 1..=2u32 {
            let id = format!("e{i}");
            handle_store(
                &state,
                &json!({ "text": format!("persisted text of {id}"), "id": id, "embedding": f05a_emb(i as f64) }),
            )
            .expect("store");
        }
        {
            let s = state.lock().unwrap();
            assert!(
                s.doc_texts.get("e1").is_none(),
                "precondition: e1 evicted from the cache"
            );
            assert_eq!(
                s.doc_store.get("e1").unwrap().unwrap().document.as_deref(),
                Some("persisted text of e1")
            );
            assert_eq!(s.doc_text("e1"), "persisted text of e1");
        }
        // Same signature for the three search handlers (coerced to one fn type).
        for h in [
            handle_search as fn(&Mutex<McpState>, &Value) -> Result<Value, String>,
            handle_spatial_search,
            handle_verbatim_search,
        ] {
            let res = h(
                &state,
                &json!({ "query": "q", "embedding": f05a_emb(1.0), "top_k": 2 }),
            )
            .expect("search");
            let texts: Vec<String> = res["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["text"].as_str().unwrap_or("").to_string())
                .collect();
            assert_eq!(
                texts,
                vec![
                    "persisted text of e1".to_string(),
                    "persisted text of e2".to_string()
                ],
                "{res}"
            );
        }
        // A doc with no row at all (cannot happen through the handlers) still degrades to "".
        assert_eq!(state.lock().unwrap().doc_text("ghost"), "");
    }

    /// A05-F12: a poisoned state mutex never panics the stdio server. Every
    /// tool that takes the lock answers a controlled error, and the
    /// cold-start helper returns instead of unwrapping.
    #[test]
    fn f12_poisoned_state_mutex_yields_controlled_errors() {
        let state = f05a_state();
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = state.lock().unwrap();
            panic!("poison probe");
        }));
        std::panic::set_hook(prev_hook);
        assert!(
            poisoned.is_err(),
            "probe must have panicked while holding the lock"
        );
        assert!(state.is_poisoned());

        // Cold-start path: no panic on the poisoned lock.
        register_warm_start_spatial(
            &state,
            &vec![("d".to_string(), None, None, None)],
            true,
            std::time::Instant::now(),
        );

        let calls: [(&str, Value); 14] = [
            ("splatsdb_store", json!({ "text": "t", "id": "p" })),
            ("splatsdb_search", json!({ "query": "q" })),
            ("splatsdb_status", json!({})),
            ("splatsdb_doc_add", json!({ "id": "p", "text": "t" })),
            ("splatsdb_doc_get", json!({ "id": "p" })),
            ("splatsdb_doc_del", json!({ "id": "p" })),
            ("splatsdb_graph_add_doc", json!({ "text": "t" })),
            (
                "splatsdb_graph_add_entity",
                json!({ "name": "n", "entity_type": "x" }),
            ),
            (
                "splatsdb_graph_add_relation",
                json!({ "source_id": 1, "target_id": 2, "relation_type": "r" }),
            ),
            ("splatsdb_graph_traverse", json!({ "start_id": 1 })),
            ("splatsdb_graph_search", json!({ "query": "q" })),
            ("splatsdb_graph_search_entities", json!({ "query": "q" })),
            ("splatsdb_spatial_search", json!({ "query": "q" })),
            ("splatsdb_spatial_info", json!({})),
        ];
        for (i, (tool, args)) in calls.iter().enumerate() {
            let req: JsonRpcRequest = serde_json::from_value(json!({
                "jsonrpc": "2.0", "id": i, "method": "tools/call",
                "params": { "name": tool, "arguments": args }
            }))
            .unwrap();
            let resp = dispatch_request(&state, &req);
            assert!(
                resp.error.is_none(),
                "{tool}: protocol error instead of a controlled tool error"
            );
            let result = resp.result.expect("tool result");
            assert_eq!(result["isError"], json!(true), "{tool}: {result}");
            let text = result["content"][0]["text"].as_str().unwrap();
            assert!(text.contains("internal server error"), "{tool}: {text}");
        }
        // The verbatim search reaches the lock after its parameter validation.
        let req: JsonRpcRequest = serde_json::from_value(json!({
            "jsonrpc": "2.0", "id": 99, "method": "tools/call",
            "params": { "name": "splatsdb_verbatim_search", "arguments": { "query": "q" } }
        }))
        .unwrap();
        let resp = dispatch_request(&state, &req);
        assert_eq!(resp.result.unwrap()["isError"], json!(true));
    }

    // -- end of F05a inline tests --
}
