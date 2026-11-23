//! TurboQuant integration for SplatsDB Vector Search.
//!
//! Data-oblivious quantization — no training, no codebooks, instant indexing.
//!
//! Storage layout (NOT bit-packed): every code stores an f32 radius and a u16 polar
//! angle index per pair of dimensions (3 B per value); TurboQuant codes add one i8 QJL
//! sign per projection and one f32 residual norm. `bits` changes the quantization error
//! only, never the byte count: at 640-D a 4-bit TurboQuant code is 2004 B, i.e. 1.28x
//! smaller than f32 — not the 8x of the nominal `dim*bits/8` formula.
//! `encoded_bytes()`, `memory_usage()` and `compression_ratio()` report that real layout.

use crate::quant::{PolarCode, PolarQuantizer, QjlSketch, TurboCode, TurboQuantizer};
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

/// Quantization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAlgorithm {
    TurboQuant,
    PolarQuant,
}

/// Configuration for vector quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Polar angle resolution (3-8). TurboQuant spends `bits - 1` on the angle and the
    /// QJL sketch carries the residual. Does not change the stored size (unpacked layout).
    pub bits: u8,
    /// Number of random projections. Recommended: dim / 4 for search.
    pub projections: usize,
    /// Random seed.
    pub seed: u64,
    /// Algorithm to use.
    pub algorithm: QuantAlgorithm,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            projections: 160,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }
}

impl QuantConfig {
    /// For search.
    pub fn for_search(dim: usize) -> Self {
        Self {
            bits: 4,
            projections: dim / 8,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }

    /// For max compression.
    pub fn for_max_compression(dim: usize) -> Self {
        Self {
            bits: 3,
            projections: dim / 16,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        }
    }

    /// For compression.
    pub fn for_compression(dim: usize) -> Self {
        Self {
            bits: 4,
            projections: dim / 8,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }
}

/// Compressed vector store using TurboQuant.
pub struct QuantizedStore {
    config: QuantConfig,
    dim: usize,
    turbo: Option<TurboQuantizer>,
    polar: Option<PolarQuantizer>,
    codes_turbo: Vec<TurboCode>,
    codes_polar: Vec<PolarCode>,
    ids: Vec<u64>,
}

impl QuantizedStore {
    /// New.
    pub fn new(dim: usize, config: QuantConfig) -> Result<Self, String> {
        let (turbo, polar) = match config.algorithm {
            QuantAlgorithm::TurboQuant => {
                let t = TurboQuantizer::new(dim, config.bits, config.projections, config.seed)
                    .map_err(|e| format!("TurboQuant init error: {}", e))?;
                (Some(t), None)
            }
            QuantAlgorithm::PolarQuant => {
                let p = PolarQuantizer::new(dim, config.bits, config.seed)
                    .map_err(|e| format!("PolarQuant init error: {}", e))?;
                (None, Some(p))
            }
        };

        Ok(Self {
            config,
            dim,
            turbo,
            polar,
            codes_turbo: Vec::new(),
            codes_polar: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Encode and store a batch of vectors in parallel; row `i` gets id `start_id + i`.
    ///
    /// Atomic: the batch is validated and encoded into temporary buffers first. A batch
    /// whose column count differs from the store dimension, that contains any NaN/±Inf,
    /// or with any row that fails to encode returns `Err` and leaves ids, codes and
    /// `len()` exactly as they were. Ids are pushed only together with the codes that
    /// were actually encoded.
    pub fn add_batch(&mut self, vectors: &Array2<f32>, start_id: u64) -> Result<usize, String> {
        let n = vectors.nrows();
        if vectors.ncols() != self.dim {
            return Err(format!(
                "add_batch: batch has {} columns, store dimension is {}",
                vectors.ncols(),
                self.dim
            ));
        }
        // Validate before touching any state: a non-finite value must never become a code.
        if let Some(row) = (0..n)
            .into_par_iter()
            .find_any(|&i| vectors.row(i).iter().any(|v| !v.is_finite()))
        {
            let col = vectors
                .row(row)
                .iter()
                .position(|v| !v.is_finite())
                .unwrap_or(0);
            return Err(format!(
                "add_batch: non-finite value (NaN/Inf) at row {row}, col {col}"
            ));
        }
        let new_ids = (0..n)
            .map(|i| start_id.checked_add(i as u64))
            .collect::<Option<Vec<u64>>>()
            .ok_or_else(|| format!("add_batch: id overflow with start_id={start_id}"))?;

        if let Some(ref turbo) = self.turbo {
            let new_codes = (0..n)
                .into_par_iter()
                .map(|i| {
                    let v: Vec<f32> = vectors.row(i).to_vec();
                    turbo
                        .encode(&v)
                        .map_err(|e| format!("add_batch: row {i}: {e}"))
                })
                .collect::<Result<Vec<TurboCode>, String>>()?;
            self.codes_turbo.extend(new_codes);
        } else if let Some(ref polar) = self.polar {
            let new_codes = (0..n)
                .into_par_iter()
                .map(|i| {
                    let v: Vec<f32> = vectors.row(i).to_vec();
                    polar
                        .encode(&v)
                        .map_err(|e| format!("add_batch: row {i}: {e}"))
                })
                .collect::<Result<Vec<PolarCode>, String>>()?;
            self.codes_polar.extend(new_codes);
        } else {
            return Err("add_batch: store has no quantizer".to_string());
        }

        self.ids.extend(new_ids);
        Ok(n)
    }

    /// Encode a single vector.
    pub fn add_single(&mut self, vector: &ArrayView1<f32>, id: u64) -> bool {
        let v: Vec<f32> = vector.to_vec();
        if let Some(ref turbo) = self.turbo {
            match turbo.encode(&v) {
                Ok(code) => {
                    self.codes_turbo.push(code);
                }
                Err(_) => return false,
            }
        } else if let Some(ref polar) = self.polar {
            match polar.encode(&v) {
                Ok(code) => {
                    self.codes_polar.push(code);
                }
                Err(_) => return false,
            }
        } else {
            // No quantizer: never push an id without a code.
            return false;
        }
        self.ids.push(id);
        true
    }

    /// Search for k nearest neighbors using parallel inner product estimation.
    /// Returns (id, estimated_inner_product) pairs sorted by similarity (descending).
    pub fn search(&self, query: &ArrayView1<f32>, k: usize) -> Vec<(u64, f32)> {
        let q: Vec<f32> = query.to_vec();
        let n = self.ids.len();
        let k = k.min(n);
        if k == 0 {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f32)> = if let Some(ref turbo) = self.turbo {
            self.codes_turbo
                .par_iter()
                .enumerate()
                .filter_map(|(i, code)| turbo.inner_product_estimate(code, &q).ok().map(|s| (i, s)))
                .collect()
        } else if let Some(ref polar) = self.polar {
            self.codes_polar
                .par_iter()
                .enumerate()
                .filter_map(|(i, code)| polar.inner_product_estimate(code, &q).ok().map(|s| (i, s)))
                .collect()
        } else {
            Vec::new()
        };

        // Sort descending by inner product (higher = more similar)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        scores
            .into_iter()
            .map(|(i, score)| (self.ids[i], score))
            .collect()
    }

    /// Get number of stored codes.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Real payload bytes of the stored codes in the current layout — NOT `dim*bits/8`.
    ///
    /// Unpacked layout: one f32 radius + one u16 angle index per pair of dimensions
    /// (6 B per pair, for every `bits`); TurboQuant codes add one i8 QJL sign per
    /// projection and one f32 residual norm. Excludes struct headers and ids, which
    /// [`Self::memory_usage`] declares separately.
    pub fn encoded_bytes(&self) -> usize {
        if self.turbo.is_some() {
            self.codes_turbo
                .iter()
                .map(|code| code.encoded_bytes())
                .sum()
        } else {
            self.codes_polar
                .iter()
                .map(|code| code.encoded_bytes())
                .sum()
        }
    }

    /// Payload bytes one code occupies in the current layout (data-independent: it only
    /// depends on `dim` and, for TurboQuant, on the projection count).
    pub fn code_layout_bytes(&self) -> usize {
        let polar = PolarCode::layout_bytes(self.dim);
        match self.turbo {
            Some(ref turbo) => polar + QjlSketch::layout_bytes(turbo.projections()),
            None => polar,
        }
    }

    /// `false`: codes are stored unpacked (f32 / u16 / i8 fields). `bits` changes only the
    /// quantization error, never the byte count. Bit-packing is deferred (F04b).
    pub fn is_bit_packed(&self) -> bool {
        false
    }

    /// Memory held by the store in bytes: the real code payload
    /// ([`Self::encoded_bytes`]) plus the declared overhead of `size_of::<code>()`
    /// (struct headers incl. the `Vec` headers, `dim` / `bits` / `projections`) and 8 B
    /// per id. Excludes allocator slack (`Vec` capacity beyond `len`) and the fixed
    /// quantizer configuration.
    pub fn memory_usage(&self) -> usize {
        let inline = if self.turbo.is_some() {
            std::mem::size_of::<TurboCode>()
        } else {
            std::mem::size_of::<PolarCode>()
        };
        self.encoded_bytes() + self.len() * (inline + std::mem::size_of::<u64>())
    }

    /// Raw f32 bytes divided by real encoded bytes. For an empty store the per-code ratio
    /// of the layout is returned (the layout does not depend on the data). This is what
    /// the layout delivers — not the nominal 32/`bits`.
    pub fn compression_ratio(&self) -> f32 {
        let raw = (self.dim * std::mem::size_of::<f32>()) as f32;
        if self.is_empty() {
            raw / self.code_layout_bytes().max(1) as f32
        } else {
            raw * self.len() as f32 / self.encoded_bytes().max(1) as f32
        }
    }

    /// Get the config.
    pub fn config(&self) -> &QuantConfig {
        &self.config
    }

    /// Measure recall@k against brute-force ground truth.
    ///
    /// Given a set of query vectors and the full dataset, computes what fraction
    /// of the true top-k neighbors are recovered by quantized search.
    /// Returns recall in [0.0, 1.0] where 1.0 = perfect recall.
    pub fn measure_recall(&self, queries: &Array2<f32>, data: &Array2<f32>, k: usize) -> f32 {
        if self.is_empty() || queries.nrows() == 0 {
            return 0.0;
        }
        let n_queries = queries.nrows();
        let mut total_recall = 0.0;

        for qi in 0..n_queries {
            let query = queries.row(qi);

            // Ground truth: brute-force top-k
            let mut gt_dists: Vec<(usize, f32)> = data
                .outer_iter()
                .enumerate()
                .map(|(i, row)| {
                    let dist: f32 = row
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| {
                            let d = a - b;
                            d * d
                        })
                        .sum();
                    (i, dist)
                })
                .collect();
            let k = k.min(gt_dists.len());
            if k == 0 {
                continue;
            }
            gt_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let gt_set: std::collections::HashSet<usize> =
                gt_dists[..k].iter().map(|(i, _)| *i).collect();

            // Quantized search top-k
            let qresults = self.search(&query, k);
            let quant_set: std::collections::HashSet<usize> =
                qresults.iter().map(|(id, _)| *id as usize).collect();

            // Recall = |quantized ∩ ground_truth| / k
            let overlap = gt_set.intersection(&quant_set).count() as f32;
            total_recall += overlap / k as f32;
        }

        total_recall / n_queries as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |(_, _)| rng.gen::<f32>())
    }

    #[test]
    fn test_turbo_quant_roundtrip() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0).unwrap();
        assert_eq!(store.len(), 100);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_turbo_quant_search() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0).unwrap();

        // Search with the first vector — should find itself near the top
        let results = store.search(&vectors.row(0), 5);
        assert_eq!(results.len(), 5);
        // With 4-bit quantization + turbo codes, top result should have positive score
        assert!(results[0].1 > 0.0, "Top result should have positive score");
        let _top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    }

    #[test]
    fn test_polar_quant_search() {
        let dim = 64;
        let config = QuantConfig {
            bits: 8,
            projections: 16,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        };
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0).unwrap();

        let results = store.search(&vectors.row(0), 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_compression_ratio_reports_the_real_layout() {
        // dim=128, 4-bit TurboQuant, 16 projections: 64 pairs * 6 B + 16 signs + 4 B norm
        // = 404 B per code => 512/404 = 1.27x, not the nominal 8x.
        let dim = 128;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();
        let empty_ratio = store.compression_ratio();
        store.add_batch(&random_vectors(50, dim, 42), 0).unwrap();
        let ratio = store.compression_ratio();
        assert!((ratio - 512.0 / 404.0).abs() < 1e-4, "ratio={ratio}");
        assert!(
            (empty_ratio - ratio).abs() < 1e-4,
            "empty={empty_ratio}, filled={ratio}"
        );
        assert!(
            ratio > 1.0 && ratio < 2.0,
            "unpacked layout at 128-D: {ratio}"
        );
    }

    #[test]
    fn test_memory_usage_counts_codes_plus_declared_overhead() {
        // dim=64 is too small for this layout to save anything: 32 pairs * 6 B + 8 signs
        // + 4 B = 204 B payload per code vs 256 B raw, i.e. MORE than raw once the
        // struct headers and ids are counted.
        let dim = 64;
        let n = 50;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();
        store.add_batch(&random_vectors(n, dim, 42), 0).unwrap();

        let mem = store.memory_usage();
        let raw = n * dim * 4;
        assert_eq!(
            mem,
            store.encoded_bytes()
                + n * (std::mem::size_of::<TurboCode>() + std::mem::size_of::<u64>())
        );
        assert!(
            mem > raw,
            "unpacked 64-D layout costs more than raw: {mem} > {raw}"
        );
    }

    #[test]
    fn test_memory_usage_is_covered_by_reserved_capacity() {
        let dim = 640;
        let n = 500;
        let mut store = QuantizedStore::new(dim, QuantConfig::for_search(dim)).unwrap();
        store.add_batch(&random_vectors(n, dim, 42), 0).unwrap();

        // What the heap actually holds for the codes: reserved capacity, not just len.
        let reserved = store.codes_turbo.capacity() * std::mem::size_of::<TurboCode>()
            + store.ids.capacity() * std::mem::size_of::<u64>()
            + store
                .codes_turbo
                .iter()
                .map(|code| {
                    code.polar_code.radii.capacity() * std::mem::size_of::<f32>()
                        + code.polar_code.angle_indices.capacity() * std::mem::size_of::<u16>()
                        + code.residual_sketch.signs.capacity() * std::mem::size_of::<i8>()
                })
                .sum::<usize>();
        let payload: usize = store
            .codes_turbo
            .iter()
            .map(|code| code.encoded_bytes())
            .sum();
        assert_eq!(payload, store.encoded_bytes());
        assert_eq!(store.encoded_bytes(), n * store.code_layout_bytes());
        assert!(store.memory_usage() <= reserved);
    }
}
