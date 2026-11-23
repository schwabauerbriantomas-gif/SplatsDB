//! Graph Splat — graph structures over Gaussian splats.
//! Hybrid store combining vector search with knowledge graph.
//! Ported from splatsdb Python.
//!
//! ## Stack Overflow Fix (2026-04-04)
//! Original code had potential stack overflow from:
//! - Unbounded `Vec` growth in `get_outgoing()` returning owned Vecs
//! - No max result cap on BFS `traverse()`
//! - `add_entity` unused `entity_type` parameter masked with `__`
//! - Missing `Send + Sync` bounds for thread-safe usage

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum nodes returned by traversal to prevent unbounded memory growth.
const MAX_TRAVERSE_RESULTS: usize = 10_000;
/// Maximum embedding dimension we accept.
const MAX_EMBEDDING_DIM: usize = 8192;
/// Filename inside the state dir used by save_to_dir/load_from_dir.
pub const GRAPH_STATE_FILENAME: &str = "graph_state.json";
/// On-disk format version: the only one `load_from_dir` accepts.
const GRAPH_STATE_VERSION: u32 = 1;
/// Prefix of the transient file a save writes before its atomic rename.
const GRAPH_STATE_TMP_PREFIX: &str = "graph_state.json.tmp.";

/// Node types in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Document,
    Entity,
    Concept,
}

/// Directed edge in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source_id: usize,
    pub target_id: usize,
    pub relation_type: String,
    pub weight: f64,
}

/// Node data — kept lightweight (no nested Vecs of edges inline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: usize,
    pub node_type: NodeType,
    pub content: String,
    pub embedding: Vec<f32>,
    pub alpha: f64,
    pub kappa: f64,
}

/// On-disk JSON document for graph persistence (see save_to_dir/load_from_dir).
#[derive(Debug, Serialize, Deserialize)]
struct GraphFile {
    version: u32,
    next_id: usize,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

/// Error type for graph operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// Embedding dimension exceeds maximum allowed.
    EmbeddingTooLarge { dim: usize, max: usize },
    /// Content string exceeds maximum allowed length.
    ContentTooLong { len: usize, max: usize },
    /// Node ID not found in graph.
    NodeNotFound { id: usize },
    /// Result set hit the safety cap.
    ResultCapHit { cap: usize },
    /// Invalid weight value (NaN or infinite).
    InvalidWeight,
    /// I/O or serialization failure during persistence (save/load).
    Io(String),
    /// The state file is not the one this store was loaded from or last
    /// saved as: writing would discard another writer's nodes, or resurrect
    /// a file that was removed on purpose. The file is never touched.
    OutOfSync { path: String, reason: &'static str },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmbeddingTooLarge { dim, max } => {
                write!(f, "embedding dim {dim} exceeds max {max}")
            }
            Self::ContentTooLong { len, max } => {
                write!(f, "content length {len} exceeds max {max}")
            }
            Self::NodeNotFound { id } => write!(f, "node {id} not found"),
            Self::ResultCapHit { cap } => write!(f, "result set hit safety cap of {cap}"),
            Self::InvalidWeight => write!(f, "weight must be finite"),
            Self::Io(context) => write!(f, "graph persistence error: {context}"),
            Self::OutOfSync { path, reason } => {
                write!(
                    f,
                    "graph state {path} {reason}; it was NOT overwritten — reload it before saving"
                )
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Maximum content string length.
const MAX_CONTENT_LEN: usize = 1_000_000;

/// Hybrid store combining vector search with knowledge graph.
/// All heavy data stored in HashMaps (heap) to avoid stack overflow.
pub struct GaussianGraphStore {
    nodes: HashMap<usize, GraphNode>,
    edges: Vec<GraphEdge>,
    /// node_id -> outgoing edge indices
    outgoing: HashMap<usize, Vec<usize>>,
    /// node_id -> incoming edge indices
    incoming: HashMap<usize, Vec<usize>>,
    type_index: HashMap<NodeType, HashSet<usize>>,
    entity_name_index: HashMap<String, usize>,
    next_id: usize,
    /// Fingerprint of the state file this store is in sync with (see
    /// `FileFingerprint`); `synced_set == false` → no file existed when it
    /// was built. Atomic so that `save_to_dir(&self)` — the CLI signature —
    /// can record what it wrote.
    synced_len: AtomicU64,
    synced_hash: AtomicU64,
    synced_set: AtomicBool,
}

impl GaussianGraphStore {
    /// Create an empty graph store.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            type_index: HashMap::new(),
            entity_name_index: HashMap::new(),
            next_id: 1,
            synced_len: AtomicU64::new(0),
            synced_hash: AtomicU64::new(0),
            synced_set: AtomicBool::new(false),
        }
    }

    /// Validate embedding dimensions are within bounds.
    fn validate_embedding(embedding: &[f32]) -> Result<(), GraphError> {
        if embedding.len() > MAX_EMBEDDING_DIM {
            return Err(GraphError::EmbeddingTooLarge {
                dim: embedding.len(),
                max: MAX_EMBEDDING_DIM,
            });
        }
        Ok(())
    }

    /// Validate content string length.
    fn validate_content(content: &str) -> Result<(), GraphError> {
        if content.len() > MAX_CONTENT_LEN {
            return Err(GraphError::ContentTooLong {
                len: content.len(),
                max: MAX_CONTENT_LEN,
            });
        }
        Ok(())
    }

    /// Add a document node.
    ///
    /// # Errors
    /// Returns `GraphError::EmbeddingTooLarge` if embedding dimension exceeds `MAX_EMBEDDING_DIM`.
    /// Returns `GraphError::ContentTooLong` if content exceeds `MAX_CONTENT_LEN`.
    pub fn add_document(&mut self, text: &str, embedding: &[f32]) -> Result<usize, GraphError> {
        Self::validate_embedding(embedding)?;
        Self::validate_content(text)?;

        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            GraphNode {
                id,
                node_type: NodeType::Document,
                content: text.to_string(),
                embedding: embedding.to_vec(),
                alpha: 1.0,
                kappa: 10.0,
            },
        );
        self.type_index
            .entry(NodeType::Document)
            .or_default()
            .insert(id);
        Ok(id)
    }

    /// Add an entity node (with name deduplication).
    ///
    /// # Errors
    /// Returns `GraphError::EmbeddingTooLarge` if embedding dimension exceeds `MAX_EMBEDDING_DIM`.
    /// Returns `GraphError::ContentTooLong` if name exceeds `MAX_CONTENT_LEN`.
    pub fn add_entity(
        &mut self,
        name: &str,
        embedding: &[f32],
        entity_type: &str,
    ) -> Result<usize, GraphError> {
        Self::validate_embedding(embedding)?;
        Self::validate_content(name)?;

        let key = name.to_lowercase();
        if let Some(&id) = self.entity_name_index.get(&key) {
            return Ok(id);
        }

        let id = self.next_id;
        self.next_id += 1;
        let mut node = GraphNode {
            id,
            node_type: NodeType::Entity,
            content: name.to_string(),
            embedding: embedding.to_vec(),
            alpha: 0.8,
            kappa: 15.0,
        };

        // Use entity_type to influence node parameters
        match entity_type {
            "concept" => {
                node.alpha = 0.6;
                node.kappa = 20.0;
            }
            "person" | "organization" | "location" => {
                node.alpha = 0.9;
                node.kappa = 12.0;
            }
            _ => {}
        }

        self.nodes.insert(id, node);
        self.type_index
            .entry(NodeType::Entity)
            .or_default()
            .insert(id);
        self.entity_name_index.insert(key, id);
        Ok(id)
    }

    /// Add a relation edge between two nodes.
    ///
    /// # Errors
    /// Returns `GraphError::NodeNotFound` if source or target doesn't exist.
    /// Returns `GraphError::InvalidWeight` if weight is NaN or infinite.
    pub fn add_relation(
        &mut self,
        source_id: usize,
        target_id: usize,
        relation_type: &str,
        weight: f64,
    ) -> Result<(), GraphError> {
        if !weight.is_finite() {
            return Err(GraphError::InvalidWeight);
        }
        if !self.nodes.contains_key(&source_id) {
            return Err(GraphError::NodeNotFound { id: source_id });
        }
        if !self.nodes.contains_key(&target_id) {
            return Err(GraphError::NodeNotFound { id: target_id });
        }

        let edge_idx = self.edges.len();
        self.edges.push(GraphEdge {
            source_id,
            target_id,
            relation_type: relation_type.into(),
            weight,
        });
        self.outgoing.entry(source_id).or_default().push(edge_idx);
        self.incoming.entry(target_id).or_default().push(edge_idx);
        Ok(())
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get outgoing edge count for a node (avoids allocating a Vec).
    pub fn outgoing_count(&self, id: usize) -> usize {
        self.outgoing.get(&id).map_or(0, |v| v.len())
    }

    /// Get outgoing edges for a node. Returns borrowed references to avoid
    /// excessive allocation on large graphs.
    pub fn get_outgoing(&self, id: usize) -> Vec<&GraphEdge> {
        self.outgoing
            .get(&id)
            .map(|indices| indices.iter().filter_map(|&i| self.edges.get(i)).collect())
            .unwrap_or_default()
    }

    /// Get neighbors of a node. Caps result at `MAX_TRAVERSE_RESULTS`.
    pub fn get_neighbors(&self, id: usize) -> Vec<usize> {
        let mut neighbors = HashSet::new();
        if let Some(indices) = self.outgoing.get(&id) {
            for &i in indices {
                if let Some(e) = self.edges.get(i) {
                    neighbors.insert(e.target_id);
                }
            }
        }
        if let Some(indices) = self.incoming.get(&id) {
            for &i in indices {
                if let Some(e) = self.edges.get(i) {
                    neighbors.insert(e.source_id);
                }
            }
        }
        neighbors.into_iter().take(MAX_TRAVERSE_RESULTS).collect()
    }

    /// Search entities by embedding similarity.
    ///
    /// Returns the top-k entities most similar to `query_emb`.
    /// Returns empty Vec if no entities exist.
    pub fn search_entities(&self, query_emb: &[f32], k: usize) -> Vec<&GraphNode> {
        let entity_ids = self
            .type_index
            .get(&NodeType::Entity)
            .map(|s| s.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        if entity_ids.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = entity_ids
            .iter()
            .filter_map(|&id| {
                self.nodes
                    .get(&id)
                    .map(|n| (id, cosine_sim(query_emb, &n.embedding)))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .filter_map(|(id, _)| self.nodes.get(&id))
            .collect()
    }

    /// BFS traversal from a start node.
    ///
    /// Returns nodes reachable within `max_depth` hops, capped at
    /// `MAX_TRAVERSE_RESULTS` to prevent unbounded memory growth on
    /// highly connected graphs.
    pub fn traverse(&self, start_id: usize, max_depth: usize) -> Vec<usize> {
        if !self.nodes.contains_key(&start_id) {
            return Vec::new();
        }
        let mut visited = HashSet::new();
        visited.insert(start_id);
        let mut queue = VecDeque::new();
        queue.push_back((start_id, 0usize));
        let mut result = vec![start_id];

        while let Some((current_id, depth)) = queue.pop_front() {
            // Hard cap to prevent stack overflow / OOM
            if result.len() >= MAX_TRAVERSE_RESULTS {
                break;
            }
            if depth >= max_depth {
                continue;
            }
            if let Some(indices) = self.outgoing.get(&current_id) {
                for &i in indices {
                    if result.len() >= MAX_TRAVERSE_RESULTS {
                        break;
                    }
                    if let Some(e) = self.edges.get(i) {
                        if !visited.contains(&e.target_id) {
                            visited.insert(e.target_id);
                            result.push(e.target_id);
                            queue.push_back((e.target_id, depth + 1));
                        }
                    }
                }
            }
        }
        result
    }

    /// Hybrid search: vector similarity + graph context boost.
    pub fn hybrid_search(&self, query_emb: &[f32], k: usize) -> Vec<HybridResult> {
        let doc_ids = self
            .type_index
            .get(&NodeType::Document)
            .map(|s| s.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        if doc_ids.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f64)> = doc_ids
            .iter()
            .filter_map(|&id| {
                self.nodes.get(&id).map(|node| {
                    let sim = cosine_sim(query_emb, &node.embedding) as f64;
                    let graph_boost = (self.outgoing_count(id) as f64 * 0.05).min(0.2);
                    (id, sim + graph_boost)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .filter_map(|(id, score)| {
                self.nodes.get(&id).map(|n| HybridResult {
                    node_id: id,
                    content: n.content.clone(),
                    score,
                    node_type: n.node_type,
                })
            })
            .collect()
    }

    /// Graph statistics.
    pub fn get_stats(&self) -> GraphStats {
        GraphStats {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            documents: self
                .type_index
                .get(&NodeType::Document)
                .map(|s| s.len())
                .unwrap_or(0),
            entities: self
                .type_index
                .get(&NodeType::Entity)
                .map(|s| s.len())
                .unwrap_or(0),
            concepts: self
                .type_index
                .get(&NodeType::Concept)
                .map(|s| s.len())
                .unwrap_or(0),
        }
    }

    /// Number of nodes in the graph.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    /// Number of edges in the graph.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Serialize the serializable state. Nodes are sorted by id so that
    /// equal graphs produce equal files.
    fn encode(&self) -> Result<Vec<u8>, GraphError> {
        let mut nodes: Vec<GraphNode> = self.nodes.values().cloned().collect();
        nodes.sort_unstable_by_key(|n| n.id);
        let doc = GraphFile {
            version: GRAPH_STATE_VERSION,
            next_id: self.next_id,
            nodes,
            edges: self.edges.clone(),
        };
        serde_json::to_vec_pretty(&doc)
            .map_err(|e| GraphError::Io(format!("serialize graph: {}", e)))
    }

    /// Fingerprint of the state file this store is in sync with (`None` =
    /// no file existed when it was built).
    fn synced(&self) -> Option<FileFingerprint> {
        self.synced_set
            .load(Ordering::Relaxed)
            .then_some(FileFingerprint {
                len: self.synced_len.load(Ordering::Relaxed),
                hash: self.synced_hash.load(Ordering::Relaxed),
            })
    }

    fn set_synced(&self, fingerprint: Option<FileFingerprint>) {
        match fingerprint {
            Some(fp) => {
                self.synced_len.store(fp.len, Ordering::Relaxed);
                self.synced_hash.store(fp.hash, Ordering::Relaxed);
                self.synced_set.store(true, Ordering::Relaxed);
            }
            None => self.synced_set.store(false, Ordering::Relaxed),
        }
    }

    /// Persist nodes + edges to `<dir>/graph_state.json`, crash-safe (A04-F10).
    ///
    /// * Atomic: unique temp file in the same directory → `write_all` +
    ///   `sync_all` → `rename` over the live file → `fsync` of the
    ///   directory. Every reader — and a process restarting after a SIGKILL
    ///   — sees either the previous complete state or the new one, never a
    ///   partial file. On any failure the temp file is removed and the live
    ///   file is untouched.
    /// * Fail closed on a lost update: the save is refused while the file is
    ///   not the one this store was loaded from or last saved as (another
    ///   process wrote it, or it was removed): a whole-file rewrite would
    ///   silently discard those nodes.
    ///
    /// Only serializable state is saved; the derived indices
    /// (outgoing/incoming/type_index/entity_name_index) are rebuilt on load.
    /// After a successful save the store is in sync with the file it wrote.
    pub fn save_to_dir<P: AsRef<Path>>(&self, dir: P) -> Result<std::path::PathBuf, GraphError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| GraphError::Io(format!("create_dir_all {}: {}", dir.display(), e)))?;
        let final_path = dir.join(GRAPH_STATE_FILENAME);
        let json = self.encode()?;

        let on_disk = match std::fs::read(&final_path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(GraphError::Io(format!(
                    "read {}: {}",
                    final_path.display(),
                    e
                )))
            }
        };
        let stale_reason = match (self.synced(), on_disk.as_deref().map(fingerprint)) {
            (None, None) => None,
            (Some(expected), Some(actual)) if expected == actual => None,
            (None, Some(_)) => Some("exists but was never loaded by this store"),
            (Some(_), None) => Some("was removed since this store loaded or last saved it"),
            (Some(_), Some(_)) => Some("changed since this store loaded or last saved it"),
        };
        if let Some(reason) = stale_reason {
            return Err(GraphError::OutOfSync {
                path: final_path.display().to_string(),
                reason,
            });
        }
        replace_file_atomically(&final_path, &json)
            .map_err(|e| GraphError::Io(format!("write {}: {}", final_path.display(), e)))?;
        self.set_synced(Some(fingerprint(&json)));
        Ok(final_path)
    }

    /// Load a graph previously written by `save_to_dir`, rebuilding every
    /// derived index (type_index, entity_name_index, adjacency). `next_id`
    /// never goes below `max node id + 1`, so loaded ids can't be reused.
    ///
    /// Fails closed: an unreadable, truncated or corrupt file, a foreign
    /// `version`, duplicate node ids and dangling edges are all errors. A
    /// caller must not continue with an empty graph and save over the file.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<GaussianGraphStore, GraphError> {
        let path = dir.as_ref().join(GRAPH_STATE_FILENAME);
        let data = std::fs::read(&path)
            .map_err(|e| GraphError::Io(format!("read {}: {}", path.display(), e)))?;
        let store = Self::decode(&data, &path)?;
        store.set_synced(Some(fingerprint(&data)));
        Ok(store)
    }

    /// `load_from_dir` when a state file exists, an empty store when none
    /// does. Only "absent" maps to an empty graph; everything else is the
    /// `load_from_dir` error.
    pub fn load_or_empty<P: AsRef<Path>>(dir: P) -> Result<GaussianGraphStore, GraphError> {
        let path = dir.as_ref().join(GRAPH_STATE_FILENAME);
        match path.try_exists() {
            Ok(false) => Ok(Self::new()),
            Ok(true) => Self::load_from_dir(dir),
            Err(e) => Err(GraphError::Io(format!("access {}: {}", path.display(), e))),
        }
    }

    fn decode(data: &[u8], path: &Path) -> Result<GaussianGraphStore, GraphError> {
        let doc: GraphFile = serde_json::from_slice(data)
            .map_err(|e| GraphError::Io(format!("parse {}: {}", path.display(), e)))?;
        if doc.version != GRAPH_STATE_VERSION {
            return Err(GraphError::Io(format!(
                "parse {}: unsupported state version {} (this build reads and writes version {GRAPH_STATE_VERSION})",
                path.display(),
                doc.version
            )));
        }

        let mut store = GaussianGraphStore::new();
        for node in doc.nodes {
            let id = node.id;
            if store.nodes.insert(id, node).is_some() {
                return Err(GraphError::Io(format!(
                    "duplicate node id {} in {}",
                    id,
                    path.display()
                )));
            }
        }
        let max_id = store.nodes.keys().copied().max().unwrap_or(0);
        store.next_id = doc.next_id.max(max_id + 1);

        for (&id, node) in &store.nodes {
            store
                .type_index
                .entry(node.node_type)
                .or_default()
                .insert(id);
            if node.node_type == NodeType::Entity {
                store
                    .entity_name_index
                    .insert(node.content.to_lowercase(), id);
            }
        }

        // Adjacency rebuilt in this store's edge-index space; both endpoints
        // must exist (no silent dangling edges in a corrupt file).
        for (edge_idx, edge) in doc.edges.into_iter().enumerate() {
            if !store.nodes.contains_key(&edge.source_id) {
                return Err(GraphError::NodeNotFound { id: edge.source_id });
            }
            if !store.nodes.contains_key(&edge.target_id) {
                return Err(GraphError::NodeNotFound { id: edge.target_id });
            }
            let (s, t) = (edge.source_id, edge.target_id);
            store.edges.push(edge);
            store.outgoing.entry(s).or_default().push(edge_idx);
            store.incoming.entry(t).or_default().push(edge_idx);
        }
        Ok(store)
    }
}

/// Identity of a state file's bytes: length + FNV-1a 64. Enough to tell the
/// file this store wrote from any other content (not a security tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileFingerprint {
    len: u64,
    hash: u64,
}

fn fingerprint(bytes: &[u8]) -> FileFingerprint {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    FileFingerprint {
        len: bytes.len() as u64,
        hash,
    }
}

/// Sequence number that makes temp names unique within a process (threads).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Transient file `replace_file_atomically` writes before the rename.
fn temp_path_for(final_path: &Path) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let name = format!(
        "{GRAPH_STATE_TMP_PREFIX}{}.{nanos}.{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    final_path.with_file_name(name)
}

/// Crash-safe replacement of `final_path` by `bytes` (A04-F10).
///
/// 1. unique temp file in the SAME directory (`graph_state.json.tmp.<pid>.
///    <nanos>.<seq>`): two writers — threads or processes — never share a
///    temp file and a temp file is never the live name;
/// 2. `write_all` + `sync_all` → the payload is on stable storage;
/// 3. `rename` over the live path → readers see the old or the new complete
///    file;
/// 4. `fsync` of the directory → the rename itself is durable (Unix).
///
/// On ANY failure the temp file is removed and the live path is left as it
/// was (after step 3 the complete new file is in place even if step 4
/// fails, which is reported as an error rather than as durable).
///
/// INTEGRATION NOTE: `storage::persistence::atomic_write` is the crate's
/// canonical helper, but on this branch it neither fsyncs the directory nor
/// names its temp file uniquely per call — the storage wave is fixing it
/// there. This copy exists so `fix/mcp-contracts` does not depend on a fix
/// this branch does not carry: DEDUPE at integration by pointing
/// `save_to_dir` at the fixed helper.
fn replace_file_atomically(final_path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp_path = temp_path_for(final_path);
    let written = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp_path)?;
        #[cfg(test)]
        if let Some(limit) = TEST_FAIL_WRITE_AFTER.with(std::cell::Cell::get) {
            file.write_all(&bytes[..limit.min(bytes.len())])?;
            return Err(std::io::Error::other("injected write failure"));
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp_path, final_path)?;
        sync_dir(final_path.parent().unwrap_or_else(|| Path::new(".")))
    })();
    if let Err(err) = written {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    Ok(())
}

/// Durability of the rename itself needs the directory entry on disk.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

/// std cannot open a directory for fsync on Windows; the rename is still
/// atomic there and its durability is left to the OS.
#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Test injection: fail the payload write after this many bytes.
    static TEST_FAIL_WRITE_AFTER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Manual `Clone`: the sync fingerprint lives in atomics so that
/// `save_to_dir(&self)` can record what it wrote, and atomics are not
/// `Clone`. A clone copies the data and is in sync with the same state
/// file as the original (the MCP server mutates a copy and swaps it in).
impl Clone for GaussianGraphStore {
    fn clone(&self) -> Self {
        let mut copy = Self::new();
        copy.nodes = self.nodes.clone();
        copy.edges = self.edges.clone();
        copy.outgoing = self.outgoing.clone();
        copy.incoming = self.incoming.clone();
        copy.type_index = self.type_index.clone();
        copy.entity_name_index = self.entity_name_index.clone();
        copy.next_id = self.next_id;
        copy.set_synced(self.synced());
        copy
    }
}

impl Default for GaussianGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid search result.
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub node_id: usize,
    pub content: String,
    pub score: f64,
    pub node_type: NodeType,
}

/// Graph statistics.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub documents: usize,
    pub entities: usize,
    pub concepts: usize,
}

/// Compute cosine similarity between two vectors.
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let d = na * nb;
    if d < 1e-8 {
        0.0
    } else {
        dot / d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_document_and_entity() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Hello", &[1.0, 0.0]).unwrap();
        let e = g.add_entity("Greeting", &[1.0, 0.0], "concept").unwrap();
        assert_ne!(d, e);
        assert_eq!(g.n_nodes(), 2);
    }

    #[test]
    fn test_entity_dedup() {
        let mut g = GaussianGraphStore::new();
        let a = g.add_entity("Same", &[1.0], "t").unwrap();
        let b = g.add_entity("same", &[1.0], "t").unwrap();
        assert_eq!(a, b);
        assert_eq!(g.n_nodes(), 1);
    }

    #[test]
    fn test_relation() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Doc", &[1.0, 0.0]).unwrap();
        let e = g.add_entity("Ent", &[0.0, 1.0], "t").unwrap();
        assert!(g.add_relation(d, e, "MENTIONS", 0.9).is_ok());
        assert!(g.add_relation(999, e, "X", 1.0).is_err());
        assert_eq!(g.n_edges(), 1);
        assert_eq!(g.get_neighbors(d), vec![e]);
    }

    #[test]
    fn test_invalid_weight() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Doc", &[1.0]).unwrap();
        let e = g.add_entity("Ent", &[0.0], "t").unwrap();
        assert!(g.add_relation(d, e, "X", f64::NAN).is_err());
        assert!(g.add_relation(d, e, "X", f64::INFINITY).is_err());
        assert!(g.add_relation(d, e, "X", f64::NEG_INFINITY).is_err());
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn test_search_entities() {
        let mut g = GaussianGraphStore::new();
        g.add_entity("Cat", &[1.0, 0.0], "animal").unwrap();
        g.add_entity("Dog", &[0.9, 0.1], "animal").unwrap();
        g.add_entity("Car", &[0.0, 1.0], "vehicle").unwrap();
        let r = g.search_entities(&[1.0, 0.0], 2);
        assert!(!r.is_empty());
        assert_eq!(r[0].content, "Cat");
    }

    #[test]
    fn test_traverse() {
        let mut g = GaussianGraphStore::new();
        let n1 = g.add_document("D1", &[1.0, 0.0]).unwrap();
        let n2 = g.add_entity("E1", &[0.0, 1.0], "t").unwrap();
        let n3 = g.add_entity("E2", &[0.5, 0.5], "t").unwrap();
        g.add_relation(n1, n2, "MENTIONS", 1.0).unwrap();
        g.add_relation(n2, n3, "RELATED", 0.8).unwrap();
        let visited = g.traverse(n1, 2);
        assert!(visited.contains(&n2));
        assert!(visited.contains(&n3));
    }

    #[test]
    fn test_traverse_cyclic_graph() {
        // Cycle: A -> B -> C -> A — must not hang or overflow
        let mut g = GaussianGraphStore::new();
        let a = g.add_document("A", &[1.0]).unwrap();
        let b = g.add_entity("B", &[1.0], "t").unwrap();
        let c = g.add_entity("C", &[1.0], "t").unwrap();
        g.add_relation(a, b, "R", 1.0).unwrap();
        g.add_relation(b, c, "R", 1.0).unwrap();
        g.add_relation(c, a, "R", 1.0).unwrap();
        let visited = g.traverse(a, 100);
        // Should visit all 3 exactly once despite cycle
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(visited.contains(&c));
        assert_eq!(visited.len(), 3);
    }

    #[test]
    fn test_traverse_large_graph_capped() {
        // Build a star graph: center -> N leaves
        let mut g = GaussianGraphStore::new();
        let center = g.add_document("center", &[1.0]).unwrap();
        for i in 0..MAX_TRAVERSE_RESULTS + 100 {
            let leaf = g.add_entity(&format!("leaf_{i}"), &[1.0], "t").unwrap();
            g.add_relation(center, leaf, "R", 1.0).unwrap();
        }
        let visited = g.traverse(center, 1);
        assert!(visited.len() <= MAX_TRAVERSE_RESULTS);
    }

    #[test]
    fn test_hybrid_search() {
        let mut g = GaussianGraphStore::new();
        let d = g
            .add_document("Python programming", &[1.0, 0.0, 0.0])
            .unwrap();
        let e = g.add_entity("Python", &[0.9, 0.1, 0.0], "lang").unwrap();
        g.add_relation(d, e, "MENTIONS", 0.9).unwrap();
        let r = g.hybrid_search(&[1.0, 0.0, 0.0], 5);
        assert!(!r.is_empty());
        assert!(r[0].score > 0.0);
    }

    #[test]
    fn test_stats() {
        let mut g = GaussianGraphStore::new();
        g.add_document("D", &[1.0]).unwrap();
        g.add_entity("E", &[0.0], "t").unwrap();
        let s = g.get_stats();
        assert_eq!(s.documents, 1);
        assert_eq!(s.entities, 1);
        assert_eq!(s.total_edges, 0);
    }

    #[test]
    fn test_embedding_too_large() {
        let mut g = GaussianGraphStore::new();
        let huge_emb = vec![0.0f32; MAX_EMBEDDING_DIM + 1];
        let result = g.add_document("test", &huge_emb);
        assert!(matches!(result, Err(GraphError::EmbeddingTooLarge { .. })));
    }

    #[test]
    fn test_content_too_long() {
        let mut g = GaussianGraphStore::new();
        let long_content = "x".repeat(MAX_CONTENT_LEN + 1);
        let result = g.add_document(&long_content, &[1.0]);
        assert!(matches!(result, Err(GraphError::ContentTooLong { .. })));
    }

    #[test]
    fn test_cosine_sim_mismatched_dims() {
        let sim = cosine_sim(&[1.0, 0.0], &[1.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_sim_zero_vectors() {
        let sim = cosine_sim(&[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_sim_identical() {
        let sim = cosine_sim(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entity_type_influences_params() {
        let mut g = GaussianGraphStore::new();
        let concept = g.add_entity("c", &[1.0], "concept").unwrap();
        let person = g.add_entity("p", &[1.0], "person").unwrap();
        let default = g.add_entity("d", &[1.0], "other").unwrap();

        assert_eq!(g.get_node(concept).unwrap().alpha, 0.6);
        assert_eq!(g.get_node(concept).unwrap().kappa, 20.0);
        assert_eq!(g.get_node(person).unwrap().alpha, 0.9);
        assert_eq!(g.get_node(person).unwrap().kappa, 12.0);
        assert_eq!(g.get_node(default).unwrap().alpha, 0.8);
        assert_eq!(g.get_node(default).unwrap().kappa, 15.0);
    }

    #[test]
    fn test_nonexistent_node_traverse() {
        let g = GaussianGraphStore::new();
        let visited = g.traverse(999, 5);
        assert!(visited.is_empty());
    }

    #[test]
    fn test_nonexistent_node_neighbors() {
        let g = GaussianGraphStore::new();
        let neighbors = g.get_neighbors(999);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_save_load_round_trip() {
        let dir = std::env::temp_dir().join("splatdb_graph_roundtrip_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Doc text", &[1.0, 0.0]).unwrap();
        let e = g.add_entity("Ent", &[0.0, 1.0], "person").unwrap();
        g.add_relation(d, e, "MENTIONS", 0.9).unwrap();

        let path = g.save_to_dir(&dir).unwrap();
        assert!(path.exists());

        let g2 = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert_eq!(g2.n_nodes(), 2);
        assert_eq!(g2.n_edges(), 1);
        assert_eq!(g2.get_node(d).unwrap().content, "Doc text");
        // entity_type-derived params survive the round trip
        assert_eq!(g2.get_node(e).unwrap().alpha, 0.9);
        // adjacency rebuilt on load
        assert_eq!(g2.get_neighbors(d), vec![e]);
        let s1 = g.get_stats();
        let s2 = g2.get_stats();
        assert_eq!(s1.total_nodes, s2.total_nodes);
        assert_eq!(s1.total_edges, s2.total_edges);

        // next_id continues after load: no id reuse across processes
        let mut g3 = g2;
        let d2 = g3.add_document("second", &[1.0, 1.0]).unwrap();
        assert!(d2 > d && d2 > e);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_add_relation_after_load() {
        // Regression: before persistence, graph-add-relation ALWAYS failed
        // with NodeNotFound because each process started an empty store.
        let dir = std::env::temp_dir().join("splatdb_graph_relation_test");
        let _ = std::fs::remove_dir_all(&dir);

        let mut g1 = GaussianGraphStore::new();
        let a = g1.add_document("A", &[1.0]).unwrap();
        g1.save_to_dir(&dir).unwrap();

        let mut g2 = GaussianGraphStore::load_from_dir(&dir).unwrap();
        let b = g2.add_entity("B", &[0.0], "t").unwrap();
        assert!(g2.add_relation(a, b, "R", 1.0).is_ok());
        assert_eq!(g2.get_neighbors(a), vec![b]);

        // The edge survives into the next "process" (fresh load)
        g2.save_to_dir(&dir).unwrap();
        let g3 = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert_eq!(g3.n_edges(), 1);
        assert!(g3.get_node(a).is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // F05c (A04-F10) — crash-safe, fail-closed persistence of the graph
    // -----------------------------------------------------------------------

    fn f05c_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "splatsdb-graph-f05c-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn f05c_state(dir: &Path) -> std::path::PathBuf {
        dir.join(GRAPH_STATE_FILENAME)
    }

    /// Transient files a save left behind (none outside a crash).
    fn f05c_tmp_files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("read dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("graph_state.json.tmp."))
            .collect();
        names.sort();
        names
    }

    /// Make `dir` unwritable for this process. False (permissions restored)
    /// when they are ignored — root, or a filesystem without modes.
    #[cfg(unix)]
    fn f05c_make_unwritable(dir: &Path) -> bool {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(dir).expect("metadata").permissions();
        perms.set_mode(0o500);
        std::fs::set_permissions(dir, perms).expect("chmod read-only");
        match std::fs::File::create(dir.join(".write_probe")) {
            Ok(probe) => {
                drop(probe);
                f05c_restore_writable(dir);
                let _ = std::fs::remove_file(dir.join(".write_probe"));
                false
            }
            Err(_) => true,
        }
    }

    #[cfg(unix)]
    fn f05c_restore_writable(dir: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(dir).expect("metadata").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(dir, perms).expect("chmod writable");
    }

    /// A process killed after writing (part of) the temp file but before
    /// the rename leaves an orphan temp file behind. It is never read, never
    /// promoted and does not block the next save.
    #[test]
    fn f05c_crash_between_write_and_rename_keeps_the_previous_state() {
        let dir = f05c_dir("crash");
        let mut g = GaussianGraphStore::new();
        let a = g.add_document("acknowledged", &[1.0, 0.0]).unwrap();
        g.save_to_dir(&dir).unwrap();
        let acknowledged = std::fs::read(f05c_state(&dir)).unwrap();
        let half = acknowledged[..acknowledged.len() / 2].to_vec();

        let orphan = dir.join(format!("{GRAPH_STATE_FILENAME}.tmp.4711.17"));
        std::fs::write(&orphan, &half).unwrap();

        let mut reloaded = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert_eq!(
            reloaded.n_nodes(),
            1,
            "the orphan must not be read as state"
        );
        assert!(reloaded.get_node(a).is_some());

        let b = reloaded
            .add_entity("after the crash", &[0.0, 1.0], "x")
            .unwrap();
        reloaded.save_to_dir(&dir).unwrap();
        let after = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert_eq!(after.n_nodes(), 2);
        assert!(after.get_node(b).is_some());
        assert_eq!(
            std::fs::read(&orphan).unwrap(),
            half,
            "the orphan is never promoted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_foreign_version_is_refused_not_silently_loaded() {
        let dir = f05c_dir("version");
        let foreign = br#"{"version":99,"next_id":5,"nodes":[],"edges":[]}"#;
        std::fs::write(f05c_state(&dir), foreign).unwrap();
        match GaussianGraphStore::load_from_dir(&dir) {
            Ok(g) => panic!(
                "a state file of another format version loaded as an empty graph ({} nodes)",
                g.n_nodes()
            ),
            Err(e) => assert!(e.to_string().contains("version"), "{e}"),
        }
        assert_eq!(std::fs::read(f05c_state(&dir)).unwrap(), foreign.as_slice());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two processes on one directory: the one holding the older state must
    /// not overwrite the file the other just wrote.
    #[test]
    fn f05c_stale_store_does_not_overwrite_the_file_it_did_not_write() {
        let dir = f05c_dir("stale");
        let mut w1 = GaussianGraphStore::new();
        w1.add_document("A", &[1.0]).unwrap();
        w1.save_to_dir(&dir).unwrap();
        // w2 loads, then w1 writes again.
        let mut w2 = GaussianGraphStore::load_from_dir(&dir).unwrap();
        let b = w1.add_document("B", &[1.0]).unwrap();
        w1.save_to_dir(&dir).unwrap();
        let after_w1 = std::fs::read(f05c_state(&dir)).unwrap();

        w2.add_document("C", &[1.0]).unwrap();
        assert!(
            w2.save_to_dir(&dir).is_err(),
            "a store that did not write the current file overwrote it (B was lost)"
        );
        assert_eq!(std::fs::read(f05c_state(&dir)).unwrap(), after_w1);
        let w3 = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert!(w3.get_node(b).is_some(), "B must survive");
        assert!(w3.get_node(3).is_none(), "C was refused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_removed_state_file_is_not_recreated_from_a_stale_memory_copy() {
        let dir = f05c_dir("removed");
        let mut g = GaussianGraphStore::new();
        g.add_document("A", &[1.0]).unwrap();
        g.save_to_dir(&dir).unwrap();
        let mut g = GaussianGraphStore::load_from_dir(&dir).unwrap();
        std::fs::remove_file(f05c_state(&dir)).unwrap();

        g.add_document("B", &[1.0]).unwrap();
        assert!(
            g.save_to_dir(&dir).is_err(),
            "a file removed on purpose was recreated from a stale in-memory copy"
        );
        assert!(!f05c_state(&dir).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A save that cannot write: error, previous file byte-for-byte intact,
    /// no temp file left behind, and the next save (once writable) works.
    #[cfg(unix)]
    #[test]
    fn f05c_failed_write_keeps_the_previous_file_and_leaves_no_temp_file() {
        let dir = f05c_dir("readonly");
        let mut g = GaussianGraphStore::new();
        g.add_document("acknowledged", &[1.0, 0.0]).unwrap();
        g.save_to_dir(&dir).unwrap();
        let before = std::fs::read(f05c_state(&dir)).unwrap();
        g.add_document("not acknowledged", &[0.0, 1.0]).unwrap();

        if !f05c_make_unwritable(&dir) {
            eprintln!("skipped: directory permissions are ignored by this user");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let result = g.save_to_dir(&dir);
        f05c_restore_writable(&dir);

        assert!(
            result.is_err(),
            "a save into an unwritable directory reported success"
        );
        assert_eq!(std::fs::read(f05c_state(&dir)).unwrap(), before);
        assert!(
            f05c_tmp_files(&dir).is_empty(),
            "{:?}",
            f05c_tmp_files(&dir)
        );
        g.save_to_dir(&dir).expect("writable again");
        let reloaded = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert_eq!(reloaded.n_nodes(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_file_format_is_unchanged_and_reproducible() {
        let dir = f05c_dir("format");
        let mut g = GaussianGraphStore::new();
        let c = g.add_document("doc C", &[1.0, 0.0]).unwrap();
        let a = g.add_entity("Entity A", &[0.0, 1.0], "person").unwrap();
        g.add_relation(c, a, "MENTIONS", 0.5).unwrap();
        g.save_to_dir(&dir).unwrap();
        let bytes = std::fs::read(f05c_state(&dir)).unwrap();

        let doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let keys: std::collections::BTreeSet<&str> = doc
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["edges", "next_id", "nodes", "version"]
                .into_iter()
                .collect()
        );
        assert_eq!(doc["version"], serde_json::json!(1));
        let nodes = doc["nodes"].as_array().unwrap();
        let ids: Vec<u64> = nodes.iter().map(|n| n["id"].as_u64().unwrap()).collect();
        assert_eq!(
            ids,
            vec![c as u64, a as u64],
            "nodes are written sorted by id"
        );
        assert_eq!(nodes[0]["node_type"], serde_json::json!("Document"));

        // Equal graph → equal bytes, and a store may save again and again
        // (the MCP server does, once per mutation).
        g.save_to_dir(&dir).unwrap();
        assert_eq!(std::fs::read(f05c_state(&dir)).unwrap(), bytes);
        let reloaded = GaussianGraphStore::load_from_dir(&dir).unwrap();
        assert_eq!(reloaded.get_neighbors(c), vec![a]);
        assert_eq!(
            reloaded.get_node(a).unwrap().alpha,
            g.get_node(a).unwrap().alpha
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_absent_state_loads_empty_and_saves_may_repeat() {
        let dir = f05c_dir("absent");
        let mut g = GaussianGraphStore::load_or_empty(&dir).unwrap();
        assert_eq!(g.n_nodes(), 0);
        g.add_document("first", &[1.0]).unwrap();
        g.save_to_dir(&dir).unwrap();
        let mut g = GaussianGraphStore::load_or_empty(&dir).unwrap();
        assert_eq!(g.n_nodes(), 1);
        g.add_document("second", &[1.0]).unwrap();
        g.save_to_dir(&dir).unwrap();
        assert_eq!(
            GaussianGraphStore::load_from_dir(&dir).unwrap().n_nodes(),
            2
        );
        // A missing directory is "absent" too.
        let deep = dir.join("deep").join("er");
        assert_eq!(
            GaussianGraphStore::load_or_empty(&deep).unwrap().n_nodes(),
            0
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_unreadable_state_is_an_error_for_load_or_empty_too() {
        let dir = f05c_dir("unreadable");
        std::fs::write(f05c_state(&dir), b"{\"version\": 1, \"next_id\"").unwrap();
        match GaussianGraphStore::load_or_empty(&dir) {
            Ok(g) => panic!(
                "a corrupt state loaded as an empty graph ({} nodes)",
                g.n_nodes()
            ),
            Err(e) => assert!(e.to_string().contains(GRAPH_STATE_FILENAME), "{e}"),
        }
        // A directory in place of the file is not "absent" either.
        let _ = std::fs::remove_dir_all(&dir);
        let dir = f05c_dir("dirfile");
        std::fs::create_dir_all(f05c_state(&dir)).unwrap();
        assert!(GaussianGraphStore::load_or_empty(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_out_of_sync_error_names_the_file_and_the_reason() {
        let dir = f05c_dir("oos");
        let mut w = GaussianGraphStore::new();
        w.add_document("A", &[1.0]).unwrap();
        w.save_to_dir(&dir).unwrap();
        let mut stale = GaussianGraphStore::load_from_dir(&dir).unwrap();
        std::fs::write(f05c_state(&dir), b"changed by someone else").unwrap();

        stale.add_document("B", &[1.0]).unwrap();
        match stale.save_to_dir(&dir) {
            Ok(_) => panic!("overwrote a file it did not write"),
            Err(GraphError::OutOfSync { path, reason }) => {
                assert!(path.ends_with(GRAPH_STATE_FILENAME), "{path}");
                assert_eq!(reason, "changed since this store loaded or last saved it");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
        assert_eq!(
            std::fs::read(f05c_state(&dir)).unwrap(),
            b"changed by someone else"
        );

        // A store that never loaded an existing file may not replace it.
        let mut fresh = GaussianGraphStore::new();
        fresh.add_document("X", &[1.0]).unwrap();
        assert!(matches!(
            fresh.save_to_dir(&dir),
            Err(GraphError::OutOfSync {
                reason: "exists but was never loaded by this store",
                ..
            })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_temp_names_are_unique_per_call_and_nothing_is_left_behind() {
        let dir = f05c_dir("tmpnames");
        let live = f05c_state(&dir);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
        let handles: Vec<_> = (0..4u32)
            .map(|writer| {
                let live = live.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..50u32 {
                        replace_file_atomically(
                            &live,
                            format!("writer {writer} payload {i}").as_bytes(),
                        )
                        .unwrap();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let final_text = std::fs::read_to_string(&live).unwrap();
        assert!(final_text.starts_with("writer "), "{final_text}");
        assert!(
            f05c_tmp_files(&dir).is_empty(),
            "{:?}",
            f05c_tmp_files(&dir)
        );
        let names: std::collections::HashSet<_> =
            (0..10_000).map(|_| temp_path_for(&live)).collect();
        assert_eq!(names.len(), 10_000, "temp names repeat within a process");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn f05c_write_failure_midway_removes_the_temp_file_and_keeps_the_previous_state() {
        let dir = f05c_dir("midway");
        let mut g = GaussianGraphStore::new();
        g.add_document("acknowledged", &[1.0, 0.0]).unwrap();
        g.save_to_dir(&dir).unwrap();
        let before = std::fs::read(f05c_state(&dir)).unwrap();
        g.add_document("not acknowledged", &[0.0, 1.0]).unwrap();

        TEST_FAIL_WRITE_AFTER.with(|cell| cell.set(Some(37)));
        let result = g.save_to_dir(&dir);
        TEST_FAIL_WRITE_AFTER.with(|cell| cell.set(None));

        let err = match result {
            Ok(_) => panic!("a failed write reported success"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("injected write failure"), "{err}");
        assert_eq!(
            std::fs::read(f05c_state(&dir)).unwrap(),
            before,
            "the previous state must be intact"
        );
        assert!(
            f05c_tmp_files(&dir).is_empty(),
            "temp file left behind: {:?}",
            f05c_tmp_files(&dir)
        );
        // Still in sync with the file it wrote before: the next save works.
        g.save_to_dir(&dir).unwrap();
        assert_eq!(
            GaussianGraphStore::load_from_dir(&dir).unwrap().n_nodes(),
            2
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
