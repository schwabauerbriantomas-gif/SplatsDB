//! HNSW Index — Hierarchical Navigable Small World graph for ANN search.
//! Native Rust implementation optimized for speed.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::interfaces::{IndexSearchResult, VectorIndex};

const HNSW_MAGIC: u64 = 0x484E5357; // "HNSW"

/// Upper bounds shared by `new()` and `load()` so an index built in memory
/// obeys the same limits as one read from disk. m >= 2 keeps
/// `level_mult = 1/ln(m)` finite and positive (m=1 → +inf, m=0 → -0.0);
/// m_max0 <= MAX_HNSW_M_MAX0 is checked separately on load because there it
/// comes from the file header rather than being derived as 2*m.
const MAX_HNSW_M: usize = 4096;
const MAX_HNSW_M_MAX0: usize = 8192;
/// Ceiling on levels per node, hence on the top layer index a graph can
/// hold (< MAX_HNSW_NODE_LEVELS). search()/insert() walk layers
/// `(1..=max_level)` unconditionally and pay one distance evaluation per
/// layer even when no node owns it, so a header `max_level` the graph
/// does not back is dead work — u64::MAX used to load and hang every
/// later query.
const MAX_HNSW_NODE_LEVELS: usize = 64;

/// Wrapper for (distance, index) that implements Ord for min-heap.
#[derive(Clone, Copy, Debug)]
struct DistIdx(f32, usize);

impl PartialEq for DistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for DistIdx {}
impl PartialOrd for DistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap wrapper (negate distance).
#[derive(Clone, Copy, Debug)]
struct MaxDistIdx(f32, usize);
impl PartialEq for MaxDistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxDistIdx {}
impl PartialOrd for MaxDistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxDistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// A node in the HNSW graph.
struct HNSWNode {
    _level: usize,
    neighbors: Vec<Vec<usize>>,
}

/// HNSW index for approximate nearest neighbor search.
pub struct HNSWIndex {
    _dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    level_mult: f64,
    metric: String,
    rng: ChaCha8Rng,

    vectors: Vec<Vec<f32>>,
    /// Precomputed L2 norms for cosine distance (one per vector)
    norms: Vec<f32>,
    nodes: Vec<HNSWNode>,
    entry_point: usize,
    max_level: usize,
    removed: HashSet<usize>,
}

/// Serialize tombstones in canonical order: HashSet iteration order is not
/// reproducible across processes even for identical graph contents.
fn sorted_removed_ids(removed: &HashSet<usize>) -> Vec<u64> {
    let mut ids: Vec<u64> = removed.iter().map(|&idx| idx as u64).collect();
    ids.sort_unstable();
    ids
}

impl HNSWIndex {
    /// Number of vectors currently in the index (incl. tombstoned).
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True when the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Dimensionality the index was built with.
    pub fn dim(&self) -> usize {
        self._dim
    }

    /// New.
    pub fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        seed: u64,
    ) -> Self {
        // Same m bounds load() enforces: an index built here must survive its
        // own save/load round-trip (load rejects m outside [2, MAX_HNSW_M]).
        // m_max0 = 2*m, so m <= MAX_HNSW_M keeps m_max0 <= MAX_HNSW_M_MAX0.
        assert!(
            (2..=MAX_HNSW_M).contains(&m),
            "m out of valid range [2, {MAX_HNSW_M}]: {m}"
        );
        Self {
            _dim: dim,
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            level_mult: 1.0 / (m as f64).ln(),
            metric: metric.to_string(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            vectors: Vec::new(),
            norms: Vec::new(),
            nodes: Vec::new(),
            entry_point: 0,
            max_level: 0,
            removed: HashSet::new(),
        }
    }

    /// Check if the index has been built (has vectors loaded).
    pub fn is_built(&self) -> bool {
        !self.vectors.is_empty()
    }

    /// Save the HNSW graph to a binary file.
    ///
    /// Atomic and buffered (A01-F09 + A05-F09): the bytes are buffered into a
    /// uniquely named temp file next to `path`, fsynced, then renamed over
    /// `path`, and the directory is fsynced. A crash or I/O error therefore
    /// leaves the previous snapshot untouched instead of a truncated file; on
    /// a reported error the temp file is removed. The byte layout of a valid
    /// file is unchanged.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut f = AtomicWriter::create(path)?;

        let n_vectors = self.vectors.len() as u64;
        let dim = if self.vectors.is_empty() {
            self._dim as u64
        } else {
            self.vectors[0].len() as u64
        };

        // Header
        f.write_all(&HNSW_MAGIC.to_le_bytes())
            .map_err(|e| format!("write magic: {}", e))?;
        f.write_all(&n_vectors.to_le_bytes())
            .map_err(|e| format!("write n_vectors: {}", e))?;
        f.write_all(&dim.to_le_bytes())
            .map_err(|e| format!("write dim: {}", e))?;
        f.write_all(&(self.m as u64).to_le_bytes())
            .map_err(|e| format!("write m: {}", e))?;
        f.write_all(&(self.m_max0 as u64).to_le_bytes())
            .map_err(|e| format!("write m_max0: {}", e))?;
        f.write_all(&(self.ef_construction as u64).to_le_bytes())
            .map_err(|e| format!("write ef_construction: {}", e))?;
        f.write_all(&(self.ef_search as u64).to_le_bytes())
            .map_err(|e| format!("write ef_search: {}", e))?;
        f.write_all(&(self.entry_point as u64).to_le_bytes())
            .map_err(|e| format!("write entry_point: {}", e))?;
        f.write_all(&(self.max_level as u64).to_le_bytes())
            .map_err(|e| format!("write max_level: {}", e))?;

        let removed_vec = sorted_removed_ids(&self.removed);
        f.write_all(&(removed_vec.len() as u64).to_le_bytes())
            .map_err(|e| format!("write n_removed: {}", e))?;
        for &idx in &removed_vec {
            f.write_all(&idx.to_le_bytes())
                .map_err(|e| format!("write removed idx: {}", e))?;
        }

        // Nodes
        for node in &self.nodes {
            let n_levels = node.neighbors.len() as u64;
            f.write_all(&n_levels.to_le_bytes())
                .map_err(|e| format!("write n_levels: {}", e))?;
            for level_neighbors in &node.neighbors {
                let n_neighbors = level_neighbors.len() as u64;
                f.write_all(&n_neighbors.to_le_bytes())
                    .map_err(|e| format!("write n_neighbors: {}", e))?;
                for &nb in level_neighbors {
                    f.write_all(&(nb as u64).to_le_bytes())
                        .map_err(|e| format!("write neighbor: {}", e))?;
                }
            }
        }

        // Raw vectors: flatten all f32 values
        for vec in &self.vectors {
            let bytes: &[u8] = bytemuck::cast_slice(vec);
            f.write_all(bytes)
                .map_err(|e| format!("write vector data: {}", e))?;
        }

        f.finish()?;
        Ok(())
    }

    /// Load an HNSW graph from a binary file.
    pub fn load(
        path: &Path,
        _m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        seed: u64,
    ) -> Result<Self, String> {
        let file =
            std::fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
        // Buffered: the file is a long run of 8-byte fields (A01-F09).
        let mut f = std::io::BufReader::new(file);

        let read_u64 = |f: &mut std::io::BufReader<std::fs::File>| -> Result<u64, String> {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)
                .map_err(|e| format!("read u64: {}", e))?;
            Ok(u64::from_le_bytes(buf))
        };

        let magic = read_u64(&mut f)?;
        if magic != HNSW_MAGIC {
            return Err(format!(
                "Invalid HNSW file magic: expected {:#x}, got {:#x}",
                HNSW_MAGIC, magic
            ));
        }

        let n_vectors_raw = read_u64(&mut f)?;
        let dim_raw = read_u64(&mut f)?;
        let m_stored_raw = read_u64(&mut f)?;
        let m_max0_raw = read_u64(&mut f)?;
        let _ef_construction_stored = read_u64(&mut f)?;
        let _ef_search_stored = read_u64(&mut f)?;
        let entry_point_raw = read_u64(&mut f)?;
        let max_level_raw = read_u64(&mut f)?;

        // Bounds validation for n_vectors, dim, m, m_max0, entry_point, max_level
        const MAX_HNSW_VECTORS: usize = 10_000_000;
        const MAX_HNSW_DIM: usize = 100_000;
        let n_vectors =
            usize::try_from(n_vectors_raw).map_err(|_| "n_vectors too large for platform")?;
        let dim = usize::try_from(dim_raw).map_err(|_| "dim too large for platform")?;
        let m_stored = usize::try_from(m_stored_raw).map_err(|_| "m too large for platform")?;
        let m_max0 = usize::try_from(m_max0_raw).map_err(|_| "m_max0 too large for platform")?;
        let entry_point =
            usize::try_from(entry_point_raw).map_err(|_| "entry_point too large for platform")?;
        let max_level =
            usize::try_from(max_level_raw).map_err(|_| "max_level too large for platform")?;

        if n_vectors > MAX_HNSW_VECTORS {
            return Err(format!(
                "too many vectors: {} (max {})",
                n_vectors, MAX_HNSW_VECTORS
            ));
        }
        if dim > MAX_HNSW_DIM {
            return Err(format!(
                "dimension too large: {} (max {})",
                dim, MAX_HNSW_DIM
            ));
        }
        // dim == 0 is only what save() writes for an EMPTY index. With
        // vectors present it loaded with `vectors` empty against n_vectors
        // `nodes`: is_built() lied and the next insert indexed
        // vectors[b_idx] out of bounds (panic).
        if n_vectors > 0 && dim == 0 {
            return Err(format!(
                "dimension is 0 but the file holds {n_vectors} vectors"
            ));
        }
        // m/m_max0 come from the file header: unbounded values would drive
        // giant Vec::with_capacity allocations (DoS via a ~100-byte hostile
        // index). Bounds are the module-level consts shared with new().
        if !(2..=MAX_HNSW_M).contains(&m_stored) {
            return Err(format!(
                "m out of valid range [2, {MAX_HNSW_M}]: {m_stored}"
            ));
        }
        if m_max0 > MAX_HNSW_M_MAX0 {
            return Err(format!(
                "m_max0 too large (max {MAX_HNSW_M_MAX0}): {m_max0}"
            ));
        }
        // new() derives m_max0 = 2*m and insert() assigns up to m
        // neighbors on layer 0, where only m_max0 fit: a header below m
        // cannot describe a graph this code produces.
        if m_max0 < m_stored {
            return Err(format!(
                "m_max0 {m_max0} below m {m_stored} (expected m_max0 >= m)"
            ));
        }
        // Header cap for max_level: no node holds more than
        // MAX_HNSW_NODE_LEVELS levels, so no such layer index can exist.
        if max_level >= MAX_HNSW_NODE_LEVELS {
            return Err(format!(
                "max_level {max_level} too large (max {}",
                MAX_HNSW_NODE_LEVELS - 1
            ));
        }
        let _total_elements = n_vectors
            .checked_mul(dim)
            .ok_or("overflow in n_vectors*dim")?;

        // Validate entry_point is within bounds
        if n_vectors > 0 && entry_point >= n_vectors {
            return Err(format!(
                "entry_point {} out of range (n_vectors={})",
                entry_point, n_vectors
            ));
        }

        let n_removed_raw = read_u64(&mut f)?;
        let n_removed =
            usize::try_from(n_removed_raw).map_err(|_| "n_removed too large for platform")?;
        if n_removed > n_vectors {
            return Err(format!("n_removed {} > n_vectors {}", n_removed, n_vectors));
        }
        let mut removed = HashSet::new();
        for _ in 0..n_removed {
            let idx_raw = read_u64(&mut f)?;
            let idx =
                usize::try_from(idx_raw).map_err(|_| "removed index too large for platform")?;
            if idx >= n_vectors {
                return Err(format!(
                    "removed index {} out of range (n_vectors={})",
                    idx, n_vectors
                ));
            }
            removed.insert(idx);
        }

        // Nodes with per-node bounds checks
        let max_neighbors_per_level = m_max0.saturating_mul(2);
        let mut total_neighbors: usize = 0;
        let mut nodes = Vec::with_capacity(n_vectors);
        for node_idx in 0..n_vectors {
            let n_levels_raw = read_u64(&mut f)?;
            let n_levels =
                usize::try_from(n_levels_raw).map_err(|_| "n_levels too large for platform")?;
            if n_levels > MAX_HNSW_NODE_LEVELS {
                return Err(format!(
                    "too many levels ({}) for node {}",
                    n_levels, node_idx
                ));
            }
            let mut neighbors = Vec::with_capacity(n_levels);
            for lev in 0..n_levels {
                let n_neighbors_raw = read_u64(&mut f)?;
                let n_neighbors = usize::try_from(n_neighbors_raw)
                    .map_err(|_| "n_neighbors too large for platform")?;
                if n_neighbors > max_neighbors_per_level {
                    return Err(format!(
                        "too many neighbors ({}) for node {} level {} (max {})",
                        n_neighbors, node_idx, lev, max_neighbors_per_level
                    ));
                }
                total_neighbors = total_neighbors
                    .checked_add(n_neighbors)
                    .ok_or("overflow in total neighbor count")?;
                let mut level_neighbors = Vec::with_capacity(n_neighbors);
                for _ in 0..n_neighbors {
                    let nb_raw = read_u64(&mut f)?;
                    let nb = usize::try_from(nb_raw)
                        .map_err(|_| "neighbor index too large for platform")?;
                    if nb >= n_vectors {
                        return Err(format!(
                            "neighbor index {} out of range for node {} (n_vectors={})",
                            nb, node_idx, n_vectors
                        ));
                    }
                    level_neighbors.push(nb);
                }
                neighbors.push(level_neighbors);
            }
            nodes.push(HNSWNode {
                _level: neighbors.len().saturating_sub(1),
                neighbors,
            });
        }

        // The graph must back the header: max_level is the entry point's
        // top layer (insert() keeps `max_level + 1 == entry_point levels`;
        // an empty index has max_level 0). Anything else means header and
        // node section disagree — tampered or corrupt.
        let entry_levels = if n_vectors > 0 {
            nodes[entry_point].neighbors.len()
        } else {
            1
        };
        if max_level + 1 != entry_levels {
            return Err(format!(
                "max_level {max_level} does not match entry_point {entry_point} \
                 ({entry_levels} levels, n_vectors={n_vectors})"
            ));
        }

        // Raw vectors
        let mut vectors = Vec::with_capacity(n_vectors);
        if n_vectors > 0 && dim > 0 {
            let mut buf = vec![0.0f32; dim];
            for _ in 0..n_vectors {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                f.read_exact(bytes)
                    .map_err(|e| format!("read vector: {}", e))?;
                vectors.push(buf.clone());
            }
        }

        // Reconstruct precomputed norms from loaded vectors
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let level_mult = 1.0 / (m_stored as f64).ln();

        Ok(Self {
            _dim: dim,
            m: m_stored,
            m_max0,
            ef_construction,
            ef_search,
            level_mult,
            metric: metric.to_string(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            vectors,
            norms,
            nodes,
            entry_point,
            max_level,
            removed,
        })
    }

    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        (-(r.ln()) * self.level_mult).clamp(0.0, 16.0) as usize
    }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.metric == "cosine" {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // Use precomputed norms when available (stored vector vs query)
            // For stored vectors, norms[idx] is precomputed. For queries, compute on the fly.
            // We can't distinguish here, so we use norms for the second vector if available.
            // Actually, distance() is called symmetrically — we keep the fast path for both.
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = na * nb;
            if denom < 1e-10 {
                return 1.0;
            }
            1.0 - dot / denom
        } else {
            // L2 distance — much cheaper than cosine
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        }
    }

    /// Norm of a query, computed ONCE per search/insert and threaded through
    /// the layer walks (A01-F08a) instead of once per distance evaluation.
    /// Same expression on the same input as the per-call value it replaces,
    /// so cosine distances are bit-identical. 0.0 for L2, where it is unused.
    #[inline]
    fn query_norm(&self, query: &[f32]) -> f32 {
        if self.metric == "cosine" {
            query.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            0.0
        }
    }

    /// Distance with precomputed norm for the stored vector (avoids recomputing).
    /// `query` is the search vector, `q_norm` its norm (see [`Self::query_norm`]),
    /// `b_idx` the stored vector index.
    #[inline]
    fn distance_cached(&self, query: &[f32], q_norm: f32, b_idx: usize) -> f32 {
        let b = &self.vectors[b_idx];
        if self.metric == "cosine" {
            let dot: f32 = query.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // Query norm computed once per search/insert (A01-F08a): the same
            // expression on the same input, same bits.
            let na: f32 = q_norm;
            let nb = self.norms[b_idx];
            let denom = na * nb;
            if denom < 1e-10 {
                return 1.0;
            }
            1.0 - dot / denom
        } else {
            query
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        }
    }

    fn greedy_closest(&self, query: &[f32], q_norm: f32, entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.distance_cached(query, q_norm, current);

        loop {
            let mut changed = false;
            let node = &self.nodes[current];
            if level >= node.neighbors.len() {
                break;
            }
            for &neighbor in &node.neighbors[level] {
                if self.removed.contains(&neighbor) {
                    continue;
                }
                let d = self.distance_cached(query, q_norm, neighbor);
                if d < current_dist {
                    current_dist = d;
                    current = neighbor;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        query: &[f32],
        q_norm: f32,
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited = HashSet::new();
        visited.insert(entry);
        let dist_ep = self.distance_cached(query, q_norm, entry);

        let mut candidates = BinaryHeap::new(); // min-heap via DistIdx
        let mut results = BinaryHeap::new(); // max-heap via MaxDistIdx

        candidates.push(DistIdx(dist_ep, entry));
        results.push(MaxDistIdx(dist_ep, entry));

        while let Some(DistIdx(dist_c, c)) = candidates.pop() {
            let farthest = match results.peek() {
                Some(MaxDistIdx(d, _)) => *d,
                None => break,
            };
            if dist_c > farthest {
                break;
            }

            let node = &self.nodes[c];
            if level >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[level] {
                if visited.contains(&neighbor) || self.removed.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist_n = self.distance_cached(query, q_norm, neighbor);
                let farthest = match results.peek() {
                    Some(MaxDistIdx(d, _)) => *d,
                    None => f32::MAX,
                };

                if dist_n < farthest || results.len() < ef {
                    candidates.push(DistIdx(dist_n, neighbor));
                    results.push(MaxDistIdx(dist_n, neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut r: Vec<(f32, usize)> = results.into_iter().map(|MaxDistIdx(d, i)| (d, i)).collect();
        r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        r
    }

    pub fn insert(&mut self, vector: &[f32]) {
        if self.vectors.len() >= 10_000_000 {
            return;
        }
        let idx = self.vectors.len();
        let level = self.random_level();
        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }
        self.vectors.push(vector.to_vec());
        // Precompute norm for cosine distance caching
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.norms.push(norm);
        self.nodes.push(HNSWNode {
            _level: level,
            neighbors,
        });

        if idx == 0 {
            self.entry_point = 0;
            self.max_level = level;
            return;
        }

        let mut ep = self.entry_point;
        let q_norm = self.query_norm(vector);

        // Phase 1: greedy descent from top to above insertion level
        for lev in (level + 1..=self.max_level).rev() {
            ep = self.greedy_closest(vector, q_norm, ep, lev);
        }

        // Phase 2: insert into each layer
        for lev in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(vector, q_norm, ep, self.ef_construction, lev);
            let m_max = if lev == 0 { self.m_max0 } else { self.m };
            let m = self.m;

            // Select M closest neighbors
            let selected: Vec<usize> = candidates.iter().take(m).map(|(_, idx)| *idx).collect();

            // Set bidirectional connections
            self.nodes[idx].neighbors[lev] = selected.clone();
            for &neighbor_idx in &selected {
                if lev >= self.nodes[neighbor_idx].neighbors.len() {
                    continue;
                }
                if !self.nodes[neighbor_idx].neighbors[lev].contains(&idx) {
                    self.nodes[neighbor_idx].neighbors[lev].push(idx);
                }
                // Prune by re-scoring the FULL neighbor list AFTER the push.
                // The previous version scored the pre-push list, so the
                // brand-new edge to `idx` was silently discarded whenever the
                // neighbor was already at capacity (m_max).
                if self.nodes[neighbor_idx].neighbors[lev].len() > m_max {
                    // Score the full list under shared borrows — no clone of
                    // the neighbor vector (A01-F08b). Same distances, same
                    // stable sort, same order.
                    let n_vec = &self.vectors[neighbor_idx];
                    let mut scored: Vec<(f32, usize)> = self.nodes[neighbor_idx].neighbors[lev]
                        .iter()
                        .map(|&ni| (self.distance(n_vec, &self.vectors[ni]), ni))
                        .collect();
                    scored
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    self.nodes[neighbor_idx].neighbors[lev] =
                        scored.into_iter().take(m_max).map(|(_, i)| i).collect();
                }
            }

            if !candidates.is_empty() {
                ep = candidates[0].1;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = idx;
        }
    }
}

impl VectorIndex for HNSWIndex {
    fn build(&mut self, vectors: Array2<f32>) {
        self.vectors.clear();
        self.norms.clear();
        self.nodes.clear();
        self.removed.clear();
        self.entry_point = 0;
        self.max_level = 0;

        for i in 0..vectors.nrows() {
            let row: Vec<f32> = vectors.row(i).to_vec();
            self.insert(&row);
        }
    }

    fn search(&self, query: ArrayView1<f32>, k: usize) -> IndexSearchResult {
        if self.vectors.is_empty() {
            return IndexSearchResult {
                indices: vec![],
                distances: vec![],
            };
        }

        let q: Vec<f32> = query.to_vec();
        let q_norm = self.query_norm(&q);
        let ef = self.ef_search.max(k);
        let mut ep = self.entry_point;

        // Phase 1: greedy descent from top layer to layer 1
        for level in (1..=self.max_level).rev() {
            ep = self.greedy_closest(&q, q_norm, ep, level);
        }

        // Phase 2: search at layer 0
        let candidates = self.search_layer(&q, q_norm, ep, ef, 0);

        let results: Vec<(f32, usize)> = candidates
            .into_iter()
            .filter(|(_, idx)| !self.removed.contains(idx))
            .take(k)
            .collect();

        IndexSearchResult {
            indices: results.iter().map(|(_, i)| *i).collect(),
            distances: results.iter().map(|(d, _)| *d).collect(),
        }
    }

    fn add(&mut self, vectors: Array2<f32>) {
        for i in 0..vectors.nrows() {
            let row: Vec<f32> = vectors.row(i).to_vec();
            self.insert(&row);
        }
    }

    fn remove(&mut self, indices: &[usize]) {
        for &idx in indices {
            self.removed.insert(idx);
        }
    }

    fn n_items(&self) -> usize {
        self.vectors.len().saturating_sub(self.removed.len())
    }

    fn supports_remove(&self) -> bool {
        true
    }
}

/// Write-then-rename writer for the HNSW snapshot: the minimal local atomic
/// write (dedupe with `storage::atomic_write` at integration — that helper
/// takes one full in-memory buffer and a predictable temp name, neither of
/// which fits a multi-GB snapshot).
///
/// The temp file is created with O_EXCL under a name unique per process and
/// per call, in the target's own directory (same filesystem, so the rename
/// cannot cross devices). [`AtomicWriter::finish`] flushes, fsyncs the file,
/// renames it over the target and fsyncs the directory. A writer dropped
/// without `finish` — an error return, a panic — removes the temp file,
/// leaving the previous target bytes in place.
struct AtomicWriter {
    target: std::path::PathBuf,
    temp: std::path::PathBuf,
    file: Option<std::io::BufWriter<std::fs::File>>,
}

impl AtomicWriter {
    fn create(target: &Path) -> Result<Self, String> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let name = target
            .file_name()
            .ok_or_else(|| format!("invalid target path {}", target.display()))?
            .to_string_lossy();
        let mut last_err = String::from("no unique temp name");
        for _ in 0..16 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let candidate = target.with_file_name(format!(
                "{name}.{}.{nanos:x}.{}.tmp",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(f) => {
                    return Ok(Self {
                        target: target.to_path_buf(),
                        temp: candidate,
                        file: Some(std::io::BufWriter::new(f)),
                    })
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    last_err = e.to_string();
                }
                Err(e) => return Err(format!("create temp for {}: {}", target.display(), e)),
            }
        }
        Err(format!(
            "create temp for {}: {}",
            target.display(),
            last_err
        ))
    }

    fn finish(mut self) -> Result<(), String> {
        let file = self
            .file
            .take()
            .ok_or_else(|| format!("finish {}: writer already closed", self.target.display()))?;
        // into_inner() flushes the buffer; on failure the file inside the
        // error is dropped (closed) and `drop` below removes the temp file.
        let file = file
            .into_inner()
            .map_err(|e| format!("flush {}: {e}", self.temp.display()))?;
        file.sync_all()
            .map_err(|e| format!("fsync {}: {e}", self.temp.display()))?;
        drop(file);
        std::fs::rename(&self.temp, &self.target).map_err(|e| {
            format!(
                "rename {} -> {}: {e}",
                self.temp.display(),
                self.target.display()
            )
        })?;
        // The rename itself is a directory entry: fsync the directory so the
        // new snapshot is the one a crash leaves behind.
        let dir = self
            .target
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let dir_file = std::fs::File::open(dir)
            .map_err(|e| format!("open dir {} after rename: {e}", dir.display()))?;
        dir_file
            .sync_all()
            .map_err(|e| format!("fsync dir {} after rename: {e}", dir.display()))
    }
}

impl Drop for AtomicWriter {
    fn drop(&mut self) {
        // Close the handle first (an OS may pin open files)...
        self.file = None;
        // ...then remove the temp file unless `finish` renamed it away.
        if self.temp.exists() {
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

impl Write for AtomicWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self.file.as_mut() {
            Some(f) => f.write(buf),
            None => Err(std::io::Error::other("atomic writer is closed")),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.file.as_mut() {
            Some(f) => f.flush(),
            None => Err(std::io::Error::other("atomic writer is closed")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn small_random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    #[test]
    fn test_hnsw_creation() {
        let idx = HNSWIndex::new(4, 16, 200, 50, "l2", 42);
        assert_eq!(idx.n_items(), 0);
    }

    #[test]
    fn test_hnsw_build_and_search() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(50, 4, 123);
        idx.build(data.clone());

        assert_eq!(idx.n_items(), 50);

        // Search for an indexed vector — should return itself
        let query = data.row(0);
        let result = idx.search(query, 5);
        assert!(!result.indices.is_empty());
        assert_eq!(result.indices.len(), result.distances.len());
        // Index 0 should be in top results
        assert!(
            result.indices.contains(&0),
            "Expected index 0 in results, got {:?}",
            result.indices
        );
        // Distance to itself should be very small
        let self_pos = result.indices.iter().position(|&i| i == 0).unwrap();
        assert!(result.distances[self_pos] < 0.01);
    }

    #[test]
    fn test_hnsw_empty_search() {
        let idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let query = array![1.0f32, 0.0, 0.0, 0.0];
        let result = idx.search(query.view(), 5);
        assert!(result.indices.is_empty());
        assert!(result.distances.is_empty());
    }

    #[test]
    fn test_hnsw_add_and_remove() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(20, 4, 99);
        idx.build(data);

        assert_eq!(idx.n_items(), 20);
        assert!(idx.supports_remove());

        idx.remove(&[0, 1]);
        assert_eq!(idx.n_items(), 18);
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "cosine", 42);
        let data = small_random_vectors(30, 4, 777);
        idx.build(data.clone());

        let query = data.row(0);
        let result = idx.search(query, 3);
        assert!(!result.indices.is_empty());
        assert!(result.indices.contains(&0));
    }

    #[test]
    fn test_hnsw_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("hnsw_save_load_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_hnsw.bin");

        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(30, 4, 555);
        idx.build(data.clone());
        idx.remove(&[2, 5]);

        idx.save(&path).unwrap();

        let loaded = HNSWIndex::load(&path, 8, 100, 50, "l2", 42).unwrap();
        assert_eq!(loaded.n_items(), 28); // 30 - 2 removed
        assert!(loaded.is_built());

        // Search results should match
        let query = data.row(0);
        let orig_result = idx.search(query, 5);
        let loaded_result = loaded.search(query, 5);
        assert_eq!(orig_result.indices, loaded_result.indices);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_hnsw_is_built() {
        let idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        assert!(!idx.is_built());

        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(5, 4, 999);
        idx.build(data);
        assert!(idx.is_built());
    }

    #[test]
    fn test_insert_prune_keeps_newest_closest_edge() {
        // Regression: when a neighbor is already at capacity (m_max0 = 2*m = 4
        // here), pruning scored the PRE-push list, so the brand-new edge to
        // the freshly inserted node was always discarded. The new closest
        // edge must survive the prune. (m=2: m_max0=4; m=1 is rejected by
        // new() since level_mult = 1/ln(m) is undefined there.)
        let mut idx = HNSWIndex::new(1, 2, 100, 50, "l2", 42);
        idx.insert(&[0.0]); // node 0 (hub)
        idx.insert(&[20.0]); // node 1 → n0: {1}
        idx.insert(&[-6.0]); // node 2 → n0: {1, 2}
        idx.insert(&[6.0]); // node 3 → n0: {1, 2, 3}
        idx.insert(&[0.9]); // node 4 → n0: {1, 2, 3, 4} (at capacity)
        idx.insert(&[0.1]); // node 5 → n0 pushed to 5 > m_max0 → prune
        let n0 = &idx.nodes[0].neighbors[0];
        assert!(
            n0.contains(&5),
            "new closest edge (d=0.1) must survive pruning, got {:?}",
            n0
        );
        assert!(n0.contains(&4), "second closest (d=0.9) must survive");
        assert!(
            !n0.contains(&1),
            "farthest edge (d=20.0) should be pruned, got {:?}",
            n0
        );
    }

    // ── A01-F03: hostile / corrupt hnsw_index.bin ───────────────────────────
    // Header as written by save(): magic, n_vectors, dim, m, m_max0,
    // ef_construction, ef_search, entry_point, max_level — u64 LE each.
    const HDR_DIM: usize = 16;
    const HDR_M_MAX0: usize = 32;
    const HDR_MAX_LEVEL: usize = 64;

    /// Scratch path unique per process and test (tests run in parallel).
    fn temp_file(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("hnsw_hostile_{}_{name}.bin", std::process::id()))
    }

    /// The exact bytes `save()` writes for `idx`.
    fn saved_bytes(idx: &HNSWIndex, name: &str) -> Vec<u8> {
        let path = temp_file(name);
        idx.save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        bytes
    }

    /// `bytes` with one u64 header field overwritten (a tampered header).
    fn with_header_u64(bytes: &[u8], offset: usize, value: u64) -> Vec<u8> {
        let mut tampered = bytes.to_vec();
        tampered[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        tampered
    }

    /// `load()` exactly as `SplatStore::try_load_hnsw` calls it.
    fn load_bytes(name: &str, bytes: &[u8]) -> Result<HNSWIndex, String> {
        let path = temp_file(name);
        std::fs::write(&path, bytes).unwrap();
        let loaded = HNSWIndex::load(&path, 8, 100, 50, "l2", 42);
        std::fs::remove_file(&path).ok();
        loaded
    }

    /// `load()` must fail on these bytes; returns the error message.
    fn load_err(name: &str, bytes: &[u8]) -> String {
        match load_bytes(name, bytes) {
            Err(e) => e,
            Ok(_) => panic!("{name}: load() accepted a hostile file"),
        }
    }

    /// Seeded index whose entry point sits above layer 0, so `max_level`
    /// can be tampered in both directions.
    fn index_with_upper_layer() -> HNSWIndex {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        idx.build(small_random_vectors(2_000, 4, 555));
        assert!(
            idx.max_level >= 1,
            "fixture needs a node above layer 0, got max_level {}",
            idx.max_level
        );
        idx
    }

    /// search()/insert() walk layers (1..=max_level) unconditionally — one
    /// distance evaluation per layer even when no node owns it. A header
    /// `max_level` that the graph does not back must therefore be refused at
    /// load: before the fix max_level = u64::MAX loaded fine and hung every
    /// later query. The untampered file must keep loading byte-identically.
    #[test]
    fn test_load_rejects_unbacked_max_level() {
        let idx = index_with_upper_layer();
        let good = saved_bytes(&idx, "max_level_good");
        let real = u64::from_le_bytes(good[HDR_MAX_LEVEL..HDR_MAX_LEVEL + 8].try_into().unwrap());
        assert!(real >= 1);

        let loaded =
            load_bytes("max_level_untampered", &good).expect("untampered file must keep loading");
        assert_eq!(saved_bytes(&loaded, "max_level_resaved"), good);

        // Beyond the format limit: refused from the header alone.
        for bad in [u64::MAX, 1 << 40, 64] {
            let tampered = with_header_u64(&good, HDR_MAX_LEVEL, bad);
            let err = load_err("max_level_above_cap", &tampered);
            assert!(err.contains("max_level"), "unexpected error: {err}");
        }
        // Within the limit, but not the entry point's top layer — above it
        // (the dead-layer hang) and below it (header/graph disagreement).
        for bad in [63, real + 1, real - 1] {
            let tampered = with_header_u64(&good, HDR_MAX_LEVEL, bad);
            let err = load_err("max_level_unbacked", &tampered);
            assert!(err.contains("max_level"), "unexpected error: {err}");
        }
    }

    /// dim == 0 is only what save() writes for an EMPTY index. With vectors
    /// present it used to load with an empty `vectors` against n_vectors
    /// `nodes`: is_built() lied and the next insert indexed vectors[b_idx]
    /// out of bounds (panic).
    #[test]
    fn test_load_rejects_zero_dim_with_vectors() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        idx.build(small_random_vectors(30, 4, 555));
        let good = saved_bytes(&idx, "dim0_good");
        let tampered = with_header_u64(&good, HDR_DIM, 0);
        let err = load_err("dim0_with_vectors", &tampered);
        assert!(err.contains("dim"), "unexpected error: {err}");

        // Boundary: dim = 0 WITHOUT vectors (an empty index) still loads.
        let empty = saved_bytes(&HNSWIndex::new(0, 8, 100, 50, "l2", 42), "dim0_empty");
        let loaded = load_bytes("dim0_empty_index", &empty).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.dim(), 0);
    }

    /// new() always derives m_max0 = 2*m; insert() assigns up to m
    /// neighbors on layer 0, where only m_max0 fit. A header below m cannot
    /// describe a graph this code produces and used to be accepted.
    /// m_max0 == m is the smallest consistent value and must still load.
    #[test]
    fn test_load_rejects_m_max0_below_m() {
        // 10 vectors: every layer-0 list holds at most 9 entries, so the
        // per-node neighbor bound never masks the header rule under test.
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        idx.build(small_random_vectors(10, 4, 555));
        let good = saved_bytes(&idx, "m_max0_good");
        for bad in [0u64, 7] {
            let tampered = with_header_u64(&good, HDR_M_MAX0, bad);
            let err = load_err("m_max0_below_m", &tampered);
            assert!(err.contains("m_max0"), "unexpected error: {err}");
        }
        let smallest = with_header_u64(&good, HDR_M_MAX0, 8);
        let loaded = load_bytes("m_max0_eq_m", &smallest).unwrap();
        assert_eq!(loaded.n_items(), 10);
    }

    /// The canonical on-disk layout — header, sorted removed ids, node
    /// adjacency, raw f32 vectors — as an independent UNBUFFERED reference.
    /// HashSet iteration order must not make identical graphs byte-different.
    fn reference_bytes(idx: &HNSWIndex) -> Vec<u8> {
        use std::io::Cursor;
        let mut f = Cursor::new(Vec::new());
        let dim = if idx.vectors.is_empty() {
            idx._dim as u64
        } else {
            idx.vectors[0].len() as u64
        };
        for v in [
            HNSW_MAGIC,
            idx.vectors.len() as u64,
            dim,
            idx.m as u64,
            idx.m_max0 as u64,
            idx.ef_construction as u64,
            idx.ef_search as u64,
            idx.entry_point as u64,
            idx.max_level as u64,
        ] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let mut removed: Vec<u64> = idx.removed.iter().map(|&x| x as u64).collect();
        removed.sort_unstable();
        f.write_all(&(removed.len() as u64).to_le_bytes()).unwrap();
        for r in &removed {
            f.write_all(&r.to_le_bytes()).unwrap();
        }
        for node in &idx.nodes {
            f.write_all(&(node.neighbors.len() as u64).to_le_bytes())
                .unwrap();
            for level in &node.neighbors {
                f.write_all(&(level.len() as u64).to_le_bytes()).unwrap();
                for &nb in level {
                    f.write_all(&(nb as u64).to_le_bytes()).unwrap();
                }
            }
        }
        for v in &idx.vectors {
            f.write_all(bytemuck::cast_slice(v.as_slice())).unwrap();
        }
        f.into_inner()
    }

    /// A01-F09 + A05-F09: valid files stay byte-identical to the layout the
    /// unbuffered writer produced, and a buffered load round-trips them
    /// (ids AND distances, both metrics, with removed ids).
    #[test]
    fn test_save_is_byte_identical_and_buffered_load_roundtrips() {
        for (metric, seed) in [("l2", 555u64), ("cosine", 777)] {
            let mut idx = HNSWIndex::new(4, 8, 100, 50, metric, 42);
            idx.build(small_random_vectors(2_000, 16, seed));
            idx.remove(&[3, 17, 999]);
            let bytes = saved_bytes(&idx, &format!("layout_{metric}"));
            assert!(
                bytes == reference_bytes(&idx),
                "{metric}: save() bytes differ from the on-disk layout"
            );
            let path = temp_file(&format!("layout_load_{metric}"));
            std::fs::write(&path, &bytes).unwrap();
            let loaded = HNSWIndex::load(&path, 8, 100, 50, metric, 42).unwrap();
            std::fs::remove_file(&path).ok();
            assert_eq!(loaded.n_items(), idx.n_items());
            for i in [0usize, 1, 500, 1999] {
                let q = ndarray::arr1(&idx.vectors[i]);
                let a = idx.search(q.view(), 10);
                let b = loaded.search(q.view(), 10);
                assert_eq!(
                    (a.indices, a.distances),
                    (b.indices, b.distances),
                    "{metric} query {i}"
                );
            }
        }
    }

    /// In-memory index with `n` zero-neighbor nodes — no graph build, so the
    /// child process of the crash test spends its time writing, not inserting.
    fn bulk_index(n: usize, dim: usize) -> HNSWIndex {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        for i in 0..n {
            idx.vectors.push(vec![(i % 7) as f32; dim]);
            idx.norms.push(0.0);
            idx.nodes.push(HNSWNode {
                _level: 0,
                neighbors: vec![Vec::new()],
            });
        }
        idx
    }

    /// A01-F09 + A05-F09: a save that is killed mid-write (SIGKILL — the
    /// crash case) must leave the previous snapshot byte-identical and
    /// loadable. The writer used to truncate the live file first, so an
    /// interrupted save replaced it with a short, unreadable one.
    #[test]
    fn test_save_killed_midway_preserves_previous_file() {
        let dir = std::env::temp_dir().join(format!("hnsw_atomic_crash_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hnsw_index.bin");
        let mut prev = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        prev.build(small_random_vectors(300, 8, 555));
        prev.save(&path).unwrap();
        let prev_bytes = std::fs::read(&path).unwrap();

        if let Some(target) = std::env::var_os("SPLATSDB_TEST_HNSW_SAVE_CHILD") {
            // Child: write a ~77 MB snapshot over the previous file.
            bulk_index(150_000, 128).save(Path::new(&target)).unwrap();
            return;
        }

        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "hnsw_index::tests::test_save_killed_midway_preserves_previous_file",
                "--nocapture",
            ])
            .env("SPLATSDB_TEST_HNSW_SAVE_CHILD", &path)
            .spawn()
            .unwrap();

        // Wait until the child's save has BEGUN, then kill it mid-write.
        let start = std::time::Instant::now();
        let mut observed = false;
        while start.elapsed() < std::time::Duration::from_secs(120) {
            let target_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let temps = std::fs::read_dir(&dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| e.path() != path)
                        .count()
                })
                .unwrap_or(0);
            if target_len != prev_bytes.len() as u64 || temps > 0 {
                observed = true;
                break;
            }
            if child.try_wait().ok().flatten().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _ = child.kill();
        let _ = child.wait();
        assert!(observed, "the child's save was never observed in progress");
        let after = std::fs::read(&path).unwrap();
        assert!(
            after == prev_bytes,
            "an interrupted save damaged the previous snapshot ({} -> {} bytes)",
            prev_bytes.len(),
            after.len()
        );
        assert!(HNSWIndex::load(&path, 8, 100, 50, "l2", 42).is_ok());
        // A temp file left by the killed writer must not block the next save.
        prev.save(&path).unwrap();
        assert!(std::fs::read(&path).unwrap() == prev_bytes);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A save whose final step fails (here: the target is a directory, so the
    /// rename is refused) reports the error, leaves the target untouched and
    /// removes its temp file.
    #[test]
    fn test_save_error_keeps_target_and_removes_temp_file() {
        let dir = std::env::temp_dir().join(format!("hnsw_atomic_err_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("is_a_dir.bin");
        std::fs::create_dir_all(&target).unwrap();
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        idx.build(small_random_vectors(20, 4, 555));
        assert!(idx.save(&target).is_err());
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert!(
            entries.iter().all(|p| *p == target),
            "temp file left behind: {entries:?}"
        );
        assert!(target.is_dir());
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── A01-F08 hot paths: exact parity guards ──────────────────────────────

    /// Tie-heavy fixture: coordinates on a 3-value grid — many exact
    /// duplicates and equidistant points, so ordering depends on tie-breaks.
    fn tie_heavy_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..n * dim)
            .map(|_| (rng.gen_range(0..3) as f32) * 0.5)
            .collect();
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    /// Rows with non-unit norms on a wide scale (0.05 .. ~19.5), a zero row
    /// every 97th row (cosine falls back to 1.0) and exact duplicate rows
    /// (ties) — `small_random_vectors` alone never leaves [0, 1).
    fn scaled_random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dup: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            let scale = match i % 97 {
                0 => 0.0,
                1 | 2 => 1.0,
                k => 0.05 + (k as f32) * 0.2,
            };
            let v: Vec<f32> = if matches!(i % 97, 1 | 2) {
                dup.clone()
            } else {
                (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()
            };
            for x in v {
                data.push(scale * x);
            }
        }
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    /// `hotpath_fixture(kind, n, seed)`: "tie" -> dim 4 grid, "scaled" -> dim 12.
    fn hotpath_fixture(kind: &str, n: usize, seed: u64) -> Array2<f32> {
        match kind {
            "tie" => tie_heavy_vectors(n, 4, seed),
            _ => scaled_random_vectors(n, 12, seed),
        }
    }

    /// The search path as it was BEFORE A01-F08: query norm recomputed for
    /// every distance, the same heap walks in the same order. Reference for
    /// the parity guard — it ran green against the untouched production path
    /// first (reference faithfulness), then against the optimised one.
    fn reference_search(idx: &HNSWIndex, query: &[f32], k: usize) -> IndexSearchResult {
        let dc = |b_idx: usize| -> f32 {
            let b = &idx.vectors[b_idx];
            if idx.metric == "cosine" {
                let dot: f32 = query.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let na: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb = idx.norms[b_idx];
                let denom = na * nb;
                if denom < 1e-10 {
                    return 1.0;
                }
                1.0 - dot / denom
            } else {
                query
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt()
            }
        };
        if idx.vectors.is_empty() {
            return IndexSearchResult {
                indices: vec![],
                distances: vec![],
            };
        }
        let greedy = |mut current: usize, level: usize| -> usize {
            let mut current_dist = dc(current);
            loop {
                let mut changed = false;
                let node = &idx.nodes[current];
                if level >= node.neighbors.len() {
                    break;
                }
                for &neighbor in &node.neighbors[level] {
                    if idx.removed.contains(&neighbor) {
                        continue;
                    }
                    let d = dc(neighbor);
                    if d < current_dist {
                        current_dist = d;
                        current = neighbor;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            current
        };
        let ef = idx.ef_search.max(k);
        let mut ep = idx.entry_point;
        for level in (1..=idx.max_level).rev() {
            ep = greedy(ep, level);
        }
        let mut visited = HashSet::new();
        visited.insert(ep);
        let dist_ep = dc(ep);
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        candidates.push(DistIdx(dist_ep, ep));
        results.push(MaxDistIdx(dist_ep, ep));
        while let Some(DistIdx(dist_c, c)) = candidates.pop() {
            let farthest = match results.peek() {
                Some(MaxDistIdx(d, _)) => *d,
                None => break,
            };
            if dist_c > farthest {
                break;
            }
            let node = &idx.nodes[c];
            if node.neighbors.is_empty() {
                continue;
            }
            for &neighbor in &node.neighbors[0] {
                if visited.contains(&neighbor) || idx.removed.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);
                let dist_n = dc(neighbor);
                let farthest = match results.peek() {
                    Some(MaxDistIdx(d, _)) => *d,
                    None => f32::MAX,
                };
                if dist_n < farthest || results.len() < ef {
                    candidates.push(DistIdx(dist_n, neighbor));
                    results.push(MaxDistIdx(dist_n, neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut r: Vec<(f32, usize)> = results.into_iter().map(|MaxDistIdx(d, i)| (d, i)).collect();
        r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<(f32, usize)> = r
            .into_iter()
            .filter(|(_, i)| !idx.removed.contains(i))
            .take(k)
            .collect();
        IndexSearchResult {
            indices: results.iter().map(|(_, i)| *i).collect(),
            distances: results.iter().map(|(d, _)| *d).collect(),
        }
    }

    /// A01-F08 guard: the production search path must return exactly the ids
    /// AND distances of `reference_search` (the pre-optimisation path) on
    /// tie-heavy and non-unit-norm fixtures, both metrics, with removed ids,
    /// before and after a save/load round-trip and after an insert + remove
    /// on the reloaded index, for every k. A tie ordered differently or a
    /// distance off by one ulp fails here.
    #[test]
    fn test_search_matches_reference_path_exact() {
        let dim_of = |kind: &str| if kind == "tie" { 4 } else { 12 };
        let mut compared = 0usize;
        for (metric, kind, seed) in [
            ("l2", "tie", 11u64),
            ("cosine", "tie", 12),
            ("l2", "scaled", 13),
            ("cosine", "scaled", 14),
        ] {
            let tag = format!("{metric}/{kind}");
            let mut idx = HNSWIndex::new(dim_of(kind), 8, 100, 50, metric, 42);
            idx.build(hotpath_fixture(kind, 3_000, seed));
            idx.remove(&[5, 77, 800]);
            idx.insert(&vec![0.25; dim_of(kind)]);
            assert!(
                idx.max_level >= 1,
                "{tag}: fixture must walk the upper layers"
            );
            let queries = hotpath_fixture(kind, 40, seed + 100);
            let mut check = |idx: &HNSWIndex, phase: &str| {
                for i in 0..queries.nrows() {
                    let q = queries.row(i).to_vec();
                    for k in [1usize, 5, 10, 50] {
                        let got = idx.search(queries.row(i), k);
                        let want = reference_search(idx, &q, k);
                        assert_eq!(got.indices, want.indices, "{tag} {phase} q{i} k{k}: ids");
                        assert_eq!(
                            got.distances, want.distances,
                            "{tag} {phase} q{i} k{k}: distances"
                        );
                        assert!(!got.indices.is_empty());
                        compared += 1;
                    }
                }
            };
            check(&idx, "in-memory");
            let path = temp_file(&format!("hotpath_{metric}_{kind}"));
            idx.save(&path).unwrap();
            let mut reloaded = HNSWIndex::load(&path, 8, 100, 50, metric, 42).unwrap();
            std::fs::remove_file(&path).ok();
            check(&reloaded, "reloaded");
            reloaded.insert(&vec![-0.5; dim_of(kind)]);
            reloaded.remove(&[1, 2]);
            check(&reloaded, "reloaded+insert+remove");
        }
        assert_eq!(compared, 4 * 3 * 40 * 4);
    }

    /// Golden fixtures: (metric, kind, seed).
    const GOLDEN_FIXTURES: &[(&str, &str, u64)] = &[
        ("l2", "tie", 1),
        ("l2", "tie", 2),
        ("cosine", "tie", 3),
        ("cosine", "tie", 4),
        ("l2", "scaled", 5),
        ("cosine", "scaled", 6),
    ];

    /// FNV-1a 64 over bytes (parity hash of a saved HNSW file / search output).
    fn fnv64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// The saved-file layout (header, removed ids, adjacency, raw vectors)
    /// with ONE difference: the removed ids are sorted. `save()` writes them
    /// in `HashSet` iteration order, which is randomised per process
    /// (RandomState), so the raw bytes of a file with >= 2 removed ids are
    /// not reproducible across processes; every other section is.
    fn canonical_file_bytes(idx: &HNSWIndex) -> Vec<u8> {
        let mut removed: Vec<u64> = idx.removed.iter().map(|&x| x as u64).collect();
        removed.sort_unstable();
        let dim = if idx.vectors.is_empty() {
            idx._dim as u64
        } else {
            idx.vectors[0].len() as u64
        };
        let mut h = Vec::new();
        for v in [
            HNSW_MAGIC,
            idx.vectors.len() as u64,
            dim,
            idx.m as u64,
            idx.m_max0 as u64,
            idx.ef_construction as u64,
            idx.ef_search as u64,
            idx.entry_point as u64,
            idx.max_level as u64,
            removed.len() as u64,
        ] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        for r in &removed {
            h.extend_from_slice(&r.to_le_bytes());
        }
        for node in &idx.nodes {
            h.extend_from_slice(&(node.neighbors.len() as u64).to_le_bytes());
            for level in &node.neighbors {
                h.extend_from_slice(&(level.len() as u64).to_le_bytes());
                for &nb in level {
                    h.extend_from_slice(&(nb as u64).to_le_bytes());
                }
            }
        }
        for v in &idx.vectors {
            h.extend_from_slice(bytemuck::cast_slice(v.as_slice()));
        }
        h
    }

    /// One deterministic fixture -> (name, len, FNV) of the raw saved file
    /// (`/file`), of the canonical file layout (`/canonical-file`) and of
    /// the SEARCH OUTPUT (ids + raw f32 bits of every distance, in order;
    /// 64 queries x k in {1, 5, 10, 50}) at three points: built, reloaded
    /// from disk, and reloaded + 4 inserts + 4 removals (the insert/prune
    /// path after load).
    fn golden_fixture(metric: &str, kind: &str, seed: u64) -> Vec<(String, usize, u64)> {
        let dim = if kind == "tie" { 4 } else { 12 };
        let mut out = Vec::new();
        let mut idx = HNSWIndex::new(dim, 8, 100, 50, metric, 42);
        idx.build(hotpath_fixture(kind, 2_000, seed));
        idx.remove(&[5, 77, 800]);
        idx.insert(&vec![0.5; dim]);
        idx.insert(&vec![0.25; dim]);
        let queries = hotpath_fixture(kind, 64, seed + 100);
        let mut record = |idx: &HNSWIndex, tag: &str| {
            let bytes = saved_bytes(idx, &format!("golden_{metric}_{kind}_{seed}_{tag}"));
            out.push((
                format!("{metric}/{kind}/{seed}/{tag}/file"),
                bytes.len(),
                fnv64(&bytes),
            ));
            let canonical = canonical_file_bytes(idx);
            out.push((
                format!("{metric}/{kind}/{seed}/{tag}/canonical-file"),
                canonical.len(),
                fnv64(&canonical),
            ));
            let mut h = Vec::new();
            for i in 0..queries.nrows() {
                for k in [1usize, 5, 10, 50] {
                    let r = idx.search(queries.row(i), k);
                    for id in &r.indices {
                        h.extend_from_slice(&(*id as u64).to_le_bytes());
                    }
                    for d in &r.distances {
                        h.extend_from_slice(&d.to_bits().to_le_bytes());
                    }
                }
            }
            out.push((
                format!("{metric}/{kind}/{seed}/{tag}/search"),
                h.len(),
                fnv64(&h),
            ));
        };
        record(&idx, "built");
        let path = temp_file(&format!("golden_file_{metric}_{kind}_{seed}"));
        idx.save(&path).unwrap();
        let mut reloaded = HNSWIndex::load(&path, 8, 100, 50, metric, 42).unwrap();
        std::fs::remove_file(&path).ok();
        record(&reloaded, "reloaded");
        for s in [0.125f32, -0.75, 3.5, -8.0] {
            reloaded.insert(&vec![s; dim]);
        }
        reloaded.remove(&[0, 1, 2, 3]);
        record(&reloaded, "after_inserts");
        out
    }

    /// (fixture, bytes, FNV-1a) recorded on the code BEFORE the A01-F08 hot
    /// paths (4b78909; logs/F03c_hnsw_pre.log).
    const GOLDEN: &[(&str, usize, u64)] = &[
        ("l2/tie/1/built/canonical-file", 281248, 0x50a2a702ace9cb63),
        ("l2/tie/1/built/search", 49536, 0xae34808971c598df),
        (
            "l2/tie/1/reloaded/canonical-file",
            281248,
            0x50a2a702ace9cb63,
        ),
        ("l2/tie/1/reloaded/search", 49536, 0xae34808971c598df),
        (
            "l2/tie/1/after_inserts/canonical-file",
            281736,
            0x2d1af6079a4f1c90,
        ),
        ("l2/tie/1/after_inserts/search", 49536, 0x7befad7ececc3d7f),
        ("l2/tie/2/built/canonical-file", 280168, 0xe5d85ed9cf84bbdb),
        ("l2/tie/2/built/search", 50232, 0x1b9e230d22e5ec7b),
        (
            "l2/tie/2/reloaded/canonical-file",
            280168,
            0xe5d85ed9cf84bbdb,
        ),
        ("l2/tie/2/reloaded/search", 50232, 0x1b9e230d22e5ec7b),
        (
            "l2/tie/2/after_inserts/canonical-file",
            280704,
            0x602f427dca2481f9,
        ),
        ("l2/tie/2/after_inserts/search", 50232, 0x4947e36b0d5796f6),
        (
            "cosine/tie/3/built/canonical-file",
            278528,
            0xe8ac5bf23ba1179e,
        ),
        ("cosine/tie/3/built/search", 49872, 0xa198cb9f0e4354b3),
        (
            "cosine/tie/3/reloaded/canonical-file",
            278528,
            0xe8ac5bf23ba1179e,
        ),
        ("cosine/tie/3/reloaded/search", 49872, 0xa198cb9f0e4354b3),
        (
            "cosine/tie/3/after_inserts/canonical-file",
            279000,
            0x03ed5b2f1568e39e,
        ),
        (
            "cosine/tie/3/after_inserts/search",
            49872,
            0x4b16d72f66985215,
        ),
        (
            "cosine/tie/4/built/canonical-file",
            278544,
            0xfd030ab6341fbf70,
        ),
        ("cosine/tie/4/built/search", 50688, 0x985bf0c83168ee60),
        (
            "cosine/tie/4/reloaded/canonical-file",
            278544,
            0xfd030ab6341fbf70,
        ),
        ("cosine/tie/4/reloaded/search", 50688, 0x985bf0c83168ee60),
        (
            "cosine/tie/4/after_inserts/canonical-file",
            279016,
            0x8e276d482d829da8,
        ),
        (
            "cosine/tie/4/after_inserts/search",
            50688,
            0x154c67417bf6eff2,
        ),
        (
            "l2/scaled/5/built/canonical-file",
            344456,
            0xcfcdee2a0d0c89c7,
        ),
        ("l2/scaled/5/built/search", 50688, 0x83dcbeebf20a476a),
        (
            "l2/scaled/5/reloaded/canonical-file",
            344456,
            0xcfcdee2a0d0c89c7,
        ),
        ("l2/scaled/5/reloaded/search", 50688, 0x83dcbeebf20a476a),
        (
            "l2/scaled/5/after_inserts/canonical-file",
            345128,
            0xc896841b9e55fdac,
        ),
        (
            "l2/scaled/5/after_inserts/search",
            50688,
            0x38c5dae6d16676b6,
        ),
        (
            "cosine/scaled/6/built/canonical-file",
            352712,
            0x95e43902b906cf69,
        ),
        ("cosine/scaled/6/built/search", 50688, 0x9b271a5e8627deb4),
        (
            "cosine/scaled/6/reloaded/canonical-file",
            352712,
            0x95e43902b906cf69,
        ),
        ("cosine/scaled/6/reloaded/search", 50688, 0x9b271a5e8627deb4),
        (
            "cosine/scaled/6/after_inserts/canonical-file",
            353440,
            0x0b8435b5516eaf33,
        ),
        (
            "cosine/scaled/6/after_inserts/search",
            50688,
            0xe76fa11027429f68,
        ),
    ];

    /// Golden hashes of the saved HNSW file (graph, removed ids, raw vectors)
    /// and of the search output (ids + distance bits, in order) on the
    /// deterministic fixtures above. One different byte, id, distance or tie
    /// order fails here.
    #[test]
    fn test_hot_path_graph_and_search_match_golden() {
        let mut n = 0usize;
        for &(metric, kind, seed) in GOLDEN_FIXTURES {
            for (name, len, got) in golden_fixture(metric, kind, seed) {
                // Raw file bytes: removed ids in HashSet order -> not pinned.
                if name.ends_with("/file") {
                    continue;
                }
                let (want_len, want_hash) = GOLDEN
                    .iter()
                    .find(|(g, _, _)| *g == name.as_str())
                    .map(|(_, l, h)| (l, h))
                    .unwrap_or_else(|| panic!("fixture {name} not pinned"));
                assert_eq!(len, *want_len, "{name}: length");
                assert_eq!(
                    got, *want_hash,
                    "{name}: hash differs from the pre-hot-path golden"
                );
                n += 1;
            }
        }
        assert_eq!(n, GOLDEN.len(), "every golden entry must be covered");
    }

    #[test]
    fn f03d_removed_ids_serialization_is_canonical() {
        let removed: HashSet<usize> = [800usize, 5, 77, 2, 99].into_iter().collect();
        assert_eq!(sorted_removed_ids(&removed), vec![2u64, 5, 77, 99, 800]);
        assert!(sorted_removed_ids(&HashSet::new()).is_empty());
    }
}
