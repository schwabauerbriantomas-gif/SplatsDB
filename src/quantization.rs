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
//!
//! # Ranking objective (D18)
//!
//! A search ranks by [`ScoreMode`], stated explicitly and carried by every
//! [`SearchHit`]. The default, [`ScoreMode::L2Heuristic`], preserves the order of
//! versions <= 2.11.0 (`estimate / ||x||`: a heuristic for L2 candidate generation, not
//! an unbiased estimator); [`ScoreMode::InnerProduct`] ranks by the unbiased inner-product
//! estimate and is selected on purpose. `search` is the default mode;
//! `search_scored` takes the mode and rejects a NaN/±Inf query before ranking; a search
//! whose query context or whose score overflows f32 fails as a whole (`Err`): it never
//! returns a ranking with a hole or a non-finite score (F04e).

pub use crate::quant::qjl::ScoreMode;
use crate::quant::{PolarCode, PolarQuantizer, QjlSketch, QuantError, TurboCode, TurboQuantizer};
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

/// One ranked result of a quantized search. `score` means exactly what `mode` says:
/// under [`ScoreMode::L2Heuristic`] it is the legacy `estimate / ||x||` order value
/// (`sqrt(pi/2) * cos`-like, higher = closer), under [`ScoreMode::InnerProduct`] the
/// unbiased inner-product estimate. Neither is a distance; neither is comparable
/// across modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    pub id: u64,
    pub score: f32,
    pub mode: ScoreMode,
}

/// Objective of the brute-force ground truth of a recall measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundTruth {
    /// The `k` smallest exact squared L2 distances.
    L2,
    /// The `k` largest exact inner products.
    InnerProduct,
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
    /// or with any row that fails to encode (a finite row whose rotated radius or norm
    /// overflows f32 included) returns `Err` and leaves ids, codes and
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

    /// The objective `search` ranks by when no mode is given (D18): the legacy
    /// `estimate / ||x||` order, as an explicit L2 candidate heuristic.
    pub const fn default_score_mode() -> ScoreMode {
        ScoreMode::L2Heuristic
    }

    /// Top-k neighbours of `query` under [`Self::default_score_mode`], as
    /// `(id, score)` pairs sorted by score (descending).
    ///
    /// Compatibility wrapper: a query of the wrong dimension or with a NaN/±Inf entry
    /// yields an empty result (it is rejected before anything is ranked — never NaN
    /// scores). Use [`Self::search_scored`] to get the rejection as an error.
    pub fn search(&self, query: &ArrayView1<f32>, k: usize) -> Vec<(u64, f32)> {
        self.search_scored(query, k, Self::default_score_mode())
            .unwrap_or_default()
            .into_iter()
            .map(|hit| (hit.id, hit.score))
            .collect()
    }

    /// Top-k neighbours of `query` ranked by `mode`, higher score first (ties keep
    /// insertion order).
    ///
    /// The query-dependent work is done once per call — the rotation `R·q` and the
    /// projections `g_i·q` are shared by every code (A02-F02) — and the query is
    /// validated before anything is ranked: a wrong dimension or a NaN/±Inf entry is
    /// an `Err`, not a ranking with NaN scores.
    pub fn search_scored(
        &self,
        query: &ArrayView1<f32>,
        k: usize,
        mode: ScoreMode,
    ) -> Result<Vec<SearchHit>, QuantError> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if let Some(index) = query.iter().position(|v| !v.is_finite()) {
            return Err(QuantError::NonFiniteValue { index });
        }
        let q: Vec<f32> = query.to_vec();
        let n = self.ids.len();
        let k = k.min(n);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Every code of a store was produced by its own quantizer's `encode`, which only
        // emits valid codes (lengths, finite non-negative radii, in-range angle indices,
        // +/-1 signs): those invariants are checked once at ingestion, not per query.
        // What is checked per code here is the one thing valid inputs can still break --
        // a score that overflows -- and it fails the whole search. Until c2c6206 a code
        // that failed to score was dropped from the ranking: a silent hole.
        let scored: Vec<(usize, std::result::Result<f32, QuantError>)> =
            if let Some(ref turbo) = self.turbo {
                let context = turbo.prepare_query(&q)?;
                self.codes_turbo
                    .par_iter()
                    .enumerate()
                    .map(|(i, code)| (i, turbo.score_projected_validated(code, &context, mode)))
                    .collect()
            } else if let Some(ref polar) = self.polar {
                // No QJL stage: both modes rank by the polar reconstruction inner product
                // (the legacy order; nothing was ever divided by a norm here).
                let rotated_query = polar.prepared_query(&q)?;
                self.codes_polar
                    .par_iter()
                    .enumerate()
                    .map(|(i, code)| (i, polar.score_validated(code, &rotated_query)))
                    .collect()
            } else {
                Vec::new()
            };
        // The first code (lowest index) that failed to score is the error of the search.
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(scored.len());
        for (i, result) in scored {
            scores.push((i, result?));
        }

        // Sort descending by score (higher = more similar)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores
            .into_iter()
            .map(|(i, score)| SearchHit {
                id: self.ids[i],
                score,
                mode,
            })
            .collect())
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
    /// quantization error, never the byte count. Bit-packing is deferred (F04c).
    pub fn is_bit_packed(&self) -> bool {
        false
    }

    /// Heap bytes of the cached QJL projection matrix (`projections * dim * 4`,
    /// TurboQuant only; 0 for PolarQuant). It is generated once per store and shared by
    /// every code, sketch and query — never a copy per sketch or per row (A02-F02).
    pub fn projection_cache_bytes(&self) -> usize {
        self.turbo
            .as_ref()
            .map_or(0, |turbo| turbo.projection_cache_bytes())
    }

    /// Heap bytes of the cached d×d rotation matrix (`dim * dim * 4`), held once per
    /// store (directly for PolarQuant, inside the TurboQuantizer for TurboQuant).
    pub fn rotation_matrix_bytes(&self) -> usize {
        self.turbo.as_ref().map_or_else(
            || self.polar.as_ref().map_or(0, |p| p.rotation_memory_bytes()),
            |t| t.rotation_matrix_bytes(),
        )
    }

    /// Memory held by the quantizer configuration, independent of the number of codes:
    /// [`Self::projection_cache_bytes`] + [`Self::rotation_matrix_bytes`], each counted
    /// exactly once.
    pub fn quantizer_memory_usage(&self) -> usize {
        self.projection_cache_bytes() + self.rotation_matrix_bytes()
    }

    /// Memory held by the store in bytes: the heap payload of the codes
    /// (`heap_bytes()`: radius, angle and sign vectors) plus `size_of::<code>()` per
    /// code (struct incl. the `Vec` headers and, for TurboQuant, the inline `f32`
    /// norm, which is therefore not counted a second time), 8 B per id, and the
    /// quantizer caches ([`Self::quantizer_memory_usage`]: rotation matrix and QJL
    /// projection matrix, once each). Excludes allocator slack (`Vec` capacity beyond
    /// `len`), the transient per-query context (`(dim + projections) * 4` B for the
    /// duration of one search) and the cos/sin table of the polar bit width: that table
    /// is process-wide, one per bit width for every quantizer and store (`2^bits * 8` B,
    /// [`PolarCode::trig_table_bytes`], 512 KiB at most at 16 bits), built on the first
    /// score and owned by no store. Adding it per store would count the same table once
    /// per store: to account for it, add `PolarCode::trig_table_bytes(bits)` once per
    /// bit width in use, not `memory_usage() + trig_table_bytes` per store.
    pub fn memory_usage(&self) -> usize {
        // Heap payload of the codes + the inline part of each code (the struct with
        // its Vec headers and, for TurboQuant, the f32 norm) + 8 B per id. The norm
        // lives inline in `TurboCode`, so it must not be counted again through
        // `encoded_bytes()`.
        let (inline, heap) = if self.turbo.is_some() {
            (
                std::mem::size_of::<TurboCode>(),
                self.codes_turbo
                    .iter()
                    .map(|code| code.heap_bytes())
                    .sum::<usize>(),
            )
        } else {
            (
                std::mem::size_of::<PolarCode>(),
                self.codes_polar
                    .iter()
                    .map(|code| code.heap_bytes())
                    .sum::<usize>(),
            )
        };
        heap + self.len() * (inline + std::mem::size_of::<u64>()) + self.quantizer_memory_usage()
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

    /// Measure recall@k of [`Self::search`] (the default mode) against brute-force
    /// ground truth.
    ///
    /// Given a set of query vectors and the full dataset, computes what fraction
    /// of the true top-k neighbors are recovered by quantized search.
    /// Returns recall in [0.0, 1.0] where 1.0 = perfect recall.
    ///
    /// The ground truth is exact L2 and row `i` of `data` must be the vector stored
    /// with id `i` (what `add_batch(&data, 0)` produces): the search returns ids, the
    /// ground truth is made of row indices.
    pub fn measure_recall(&self, queries: &Array2<f32>, data: &Array2<f32>, k: usize) -> f32 {
        self.measure_recall_with_mode(queries, data, k, Self::default_score_mode())
    }

    /// [`Self::measure_recall`] under an explicit `mode`, against the brute-force
    /// ground truth of that mode's own objective (D18, F04e): the exact L2 neighbours
    /// for [`ScoreMode::L2Heuristic`] (the default), the exact top-k inner products for
    /// [`ScoreMode::InnerProduct`]. They are recalls of two different objectives and
    /// are never mixed: the recall of the `ip` mode is the one against inner-product
    /// ground truth. Against L2 ground truth whatever the mode, see
    /// [`Self::measure_recall_against_l2_with_mode`] (a cross metric for `ip`).
    pub fn measure_recall_with_mode(
        &self,
        queries: &Array2<f32>,
        data: &Array2<f32>,
        k: usize,
        mode: ScoreMode,
    ) -> f32 {
        let ground_truth = match mode {
            ScoreMode::L2Heuristic => GroundTruth::L2,
            ScoreMode::InnerProduct => GroundTruth::InnerProduct,
        };
        self.recall_against(queries, data, k, mode, ground_truth)
    }

    /// Recall@k of `mode` against exact L2 ground truth, whatever the mode is. For
    /// [`ScoreMode::L2Heuristic`] it is [`Self::measure_recall_with_mode`]; for
    /// [`ScoreMode::InnerProduct`] it is a CROSS metric -- how many L2 neighbours the
    /// inner-product ranking happens to return -- and not the recall of the `ip`
    /// objective. Until F04e `measure_recall_with_mode(.., InnerProduct)` returned this
    /// number.
    pub fn measure_recall_against_l2_with_mode(
        &self,
        queries: &Array2<f32>,
        data: &Array2<f32>,
        k: usize,
        mode: ScoreMode,
    ) -> f32 {
        self.recall_against(queries, data, k, mode, GroundTruth::L2)
    }

    /// Recall@k of `mode` against exact brute-force ground truth (f64 accumulations,
    /// ties by row index): a rejected query contributes zero recall.
    fn recall_against(
        &self,
        queries: &Array2<f32>,
        data: &Array2<f32>,
        k: usize,
        mode: ScoreMode,
        ground_truth: GroundTruth,
    ) -> f32 {
        if self.is_empty() || queries.nrows() == 0 {
            return 0.0;
        }
        let n_queries = queries.nrows();
        let mut total_recall = 0.0;

        for qi in 0..n_queries {
            let query = queries.row(qi);

            // Ground truth: brute-force top-k of the objective. A row without a value
            // (NaN) has no rank.
            let mut gt: Vec<(usize, f64)> = data
                .outer_iter()
                .enumerate()
                .map(|(i, row)| {
                    let value: f64 = match ground_truth {
                        GroundTruth::L2 => row
                            .iter()
                            .zip(query.iter())
                            .map(|(a, b)| {
                                let d = f64::from(*a) - f64::from(*b);
                                d * d
                            })
                            .sum(),
                        GroundTruth::InnerProduct => row
                            .iter()
                            .zip(query.iter())
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum(),
                    };
                    (i, value)
                })
                .filter(|(_, value)| value.is_finite())
                .collect();
            let k = k.min(gt.len());
            if k == 0 {
                continue;
            }
            // Nearest first: the smallest squared distances, the largest inner products
            // (a stable sort over the row order: ties by row index).
            gt.sort_by(|a, b| match ground_truth {
                GroundTruth::L2 => a.1.total_cmp(&b.1),
                GroundTruth::InnerProduct => b.1.total_cmp(&a.1),
            });
            let gt_set: std::collections::HashSet<usize> =
                gt[..k].iter().map(|(i, _)| *i).collect();

            // Quantized search top-k (a rejected query contributes zero recall)
            let qresults = self.search_scored(&query, k, mode).unwrap_or_default();
            let quant_set: std::collections::HashSet<usize> =
                qresults.iter().map(|hit| hit.id as usize).collect();

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
            store.encoded_bytes() - n * std::mem::size_of::<f32>()
                + n * (std::mem::size_of::<TurboCode>() + std::mem::size_of::<u64>())
                + store.quantizer_memory_usage()
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
                .sum::<usize>()
            // The two quantizer matrices are heap allocations too.
            + store.quantizer_memory_usage();
        let payload: usize = store
            .codes_turbo
            .iter()
            .map(|code| code.encoded_bytes())
            .sum();
        assert_eq!(payload, store.encoded_bytes());
        assert_eq!(store.encoded_bytes(), n * store.code_layout_bytes());
        // The norm is inline in TurboCode: counted exactly once (through the
        // struct, not again as payload).
        assert_eq!(
            store.memory_usage() + n * std::mem::size_of::<f32>(),
            payload
                + n * (std::mem::size_of::<TurboCode>() + std::mem::size_of::<u64>())
                + store.quantizer_memory_usage()
        );
        assert!(store.memory_usage() <= reserved);
    }

    #[test]
    fn quantizer_memory_is_counted_once_not_per_code() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let empty = QuantizedStore::new(dim, config.clone()).unwrap();
        assert_eq!(empty.memory_usage(), empty.quantizer_memory_usage());
        // Rotation matrix + QJL projection matrix, one independent count.
        assert_eq!(
            empty.quantizer_memory_usage(),
            dim * dim * 4 + (dim / 8) * dim * 4
        );
        assert_eq!(empty.projection_cache_bytes(), (dim / 8) * dim * 4);
        let mut full = QuantizedStore::new(dim, config).unwrap();
        full.add_batch(&random_vectors(100, dim, 1), 0).unwrap();
        let per_code = std::mem::size_of::<TurboCode>() + std::mem::size_of::<u64>()
            - std::mem::size_of::<f32>();
        assert_eq!(
            full.memory_usage(),
            full.encoded_bytes() + 100 * per_code + empty.quantizer_memory_usage()
        );
        // PolarQuant has no QJL stage: only the rotation matrix.
        let polar = QuantizedStore::new(
            dim,
            QuantConfig {
                bits: 4,
                projections: 0,
                seed: 42,
                algorithm: QuantAlgorithm::PolarQuant,
            },
        )
        .unwrap();
        assert_eq!(polar.projection_cache_bytes(), 0);
        assert_eq!(polar.quantizer_memory_usage(), dim * dim * 4);
    }
}
