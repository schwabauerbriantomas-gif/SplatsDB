//! VectorIndex trait and BruteForceIndex implementation.

use ndarray::{Array2, ArrayView1};

/// Result of a vector search operation.
#[derive(Clone, Debug)]
pub struct IndexSearchResult {
    pub indices: Vec<usize>,
    pub distances: Vec<f32>,
}

/// Trait for vector search backends.
pub trait VectorIndex: Send + Sync {
    fn build(&mut self, vectors: Array2<f32>);
    fn search(&self, query: ArrayView1<f32>, k: usize) -> IndexSearchResult;
    fn add(&mut self, vectors: Array2<f32>);
    fn remove(&mut self, indices: &[usize]);
    fn n_items(&self) -> usize;
    fn supports_remove(&self) -> bool;
}

/// Result of index strategy selection.
#[derive(Clone, Debug)]
pub struct IndexSelectionResult {
    pub recommended: String,
    pub n_vectors: usize,
    pub dim: usize,
    pub reason: String,
}

/// Select the best index strategy based on data size.
pub fn select_index_strategy(n_vectors: usize, dim: usize) -> IndexSelectionResult {
    if n_vectors < 15_000 {
        IndexSelectionResult {
            recommended: "bruteforce".into(),
            n_vectors,
            dim,
            reason: format!("{} < 15K -> linear scan sufficient", n_vectors),
        }
    } else if n_vectors < 100_000 {
        IndexSelectionResult {
            recommended: "hrm2".into(),
            n_vectors,
            dim,
            reason: format!("{} in [15K, 100K) -> HRM2 hierarchical", n_vectors),
        }
    } else {
        IndexSelectionResult {
            recommended: "hnsw".into(),
            n_vectors,
            dim,
            reason: format!("{} >= 100K -> HNSW for scalability", n_vectors),
        }
    }
}
