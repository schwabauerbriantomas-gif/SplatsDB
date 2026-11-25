//! TurboQuant: two-stage vector compression combining PolarQuant and QJL.
//!
//! 1. PolarQuant stage (bits-1): Compress vector via polar encoding.
//! 2. QJL stage (1 bit per projection): Sketch the residual for unbiased correction.
//!
//! Combined estimator: <x, y> ~ IP_polar(x_hat) + IP_qjl(residual), with x_hat the
//! polar reconstruction (deterministic) and residual = x - x_hat. Averaged over the
//! Gaussian projection matrix the QJL term equals <residual, y> for any residual norm,
//! so the sum is an unbiased estimate of <x, y> (property test in tests/quant_contracts:
//! mean over 1000 projection seeds within 2%). This holds since the QJL scale fix;
//! before it, the residual term was biased by sqrt(pi/2)/||residual||.
//!
//! # Ranking modes (D18)
//!
//! [`TurboQuantizer::score`] selects the objective explicitly:
//!
//! * [`ScoreMode::InnerProduct`]: `polar_est + unbiased QJL term` (the estimator above).
//! * [`ScoreMode::L2Heuristic`] (default of a search): `polar_est + legacy QJL term
//!   sqrt(pi/2)/m * sum_i sign_i (g_i.q)`, i.e. the order every version <= 2.11.0 used.
//!   It is NOT `(polar_est + unbiased term) / ||x||`: the legacy order applied the
//!   `1/||residual||` factor to the residual term only, and that is the order that is
//!   reproduced bit for bit. A heuristic for L2 candidate generation, not an estimator.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use super::error::{QuantError, Result};
use super::polar::{check_finite, PolarCode, PolarQuantizer};
use super::qjl::{QjlQuantizer, QjlSketch, ScoreMode};

/// A TurboQuant-compressed vector: polar code + QJL residual sketch.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TurboCode {
    pub polar_code: PolarCode,
    pub residual_sketch: QjlSketch,
}

impl<'de> Deserialize<'de> for TurboCode {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            polar_code: PolarCode,
            residual_sketch: QjlSketch,
        }
        let repr = Repr::deserialize(deserializer)?;
        // Both halves are validated by their own deserializers; what only this one can
        // see is that they describe the same vector (F04e).
        if repr.polar_code.dim != repr.residual_sketch.dim {
            return Err(serde::de::Error::custom(format!(
                "malformed TurboCode: polar code dim {} but residual sketch dim {}",
                repr.polar_code.dim, repr.residual_sketch.dim
            )));
        }
        Ok(TurboCode {
            polar_code: repr.polar_code,
            residual_sketch: repr.residual_sketch,
        })
    }
}

impl TurboCode {
    /// Heap bytes actually held by this code (polar vectors + QJL signs).
    pub fn heap_bytes(&self) -> usize {
        self.polar_code.heap_bytes() + self.residual_sketch.heap_bytes()
    }

    pub fn encoded_bytes(&self) -> usize {
        self.polar_code.encoded_bytes() + self.residual_sketch.encoded_bytes()
    }

    pub fn compression_ratio(&self) -> f32 {
        let original = self.polar_code.dim * std::mem::size_of::<f32>();
        original as f32 / self.encoded_bytes().max(1) as f32
    }
}

/// `polar_est + qjl_term`, rejected if the sum overflowed f32 (F04e).
fn combine(polar_est: f32, qjl_term: f32) -> Result<f32> {
    let score = polar_est + qjl_term;
    if !score.is_finite() {
        return Err(QuantError::NonFiniteScore);
    }
    Ok(score)
}

/// Everything in a score that depends only on the query (A02-F02): the rotated query
/// `R·q` for the polar stage and `g_i·q` for the QJL stage. Built once per search by
/// [`TurboQuantizer::prepare_query`] and shared by every code, so the O(d²) rotation
/// and the O(m·d) projection are paid once per query instead of once per code.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboQueryContext {
    rotated_query: Vec<f32>,
    projected_query: Vec<f32>,
}

impl TurboQueryContext {
    /// Heap bytes of the context: `(dim + projections) * 4`. Transient — it lives for
    /// one search and is not part of the store's memory.
    pub fn memory_bytes(&self) -> usize {
        (self.rotated_query.len() + self.projected_query.len()) * std::mem::size_of::<f32>()
    }

    /// Both halves finite (F04e): what [`TurboQuantizer::prepare_query`] guarantees.
    fn check_finite(&self) -> Result<()> {
        check_finite(&self.rotated_query)?;
        check_finite(&self.projected_query)
    }
}

/// TurboQuant compressor: encodes vectors and estimates inner products.
///
/// Configuration `(dim, bits, projections, seed)` fully determines state.
/// Only these four values need to be persisted; the rotation and projection
/// matrices are rebuilt from them.
#[derive(Debug, Clone, Serialize)]
pub struct TurboQuantizer {
    dim: usize,
    bits: u8,
    projections: usize,
    seed: u64,
    polar: PolarQuantizer,
    qjl: QjlQuantizer,
}

impl<'de> Deserialize<'de> for TurboQuantizer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Same wire fields as the derive this replaces.
        #[derive(Deserialize)]
        struct Repr {
            dim: usize,
            bits: u8,
            projections: usize,
            seed: u64,
            polar: PolarQuantizer,
            qjl: QjlQuantizer,
        }
        let repr = Repr::deserialize(deserializer)?;
        // `new` ties the four values together (polar stage at `bits - 1`); a payload must
        // not bypass it with a configuration that was never built (F04e).
        let consistent = repr.dim == repr.polar.dim()
            && repr.dim == repr.qjl.dim()
            && repr.bits == repr.polar.bits() + 1
            && repr.projections == repr.qjl.projections();
        if !consistent {
            return Err(serde::de::Error::custom(format!(
                "malformed TurboQuantizer: dim {}, bits {}, projections {} do not match the polar stage (dim {}, bits {}) and the QJL stage (dim {}, projections {})",
                repr.dim,
                repr.bits,
                repr.projections,
                repr.polar.dim(),
                repr.polar.bits(),
                repr.qjl.dim(),
                repr.qjl.projections()
            )));
        }
        Ok(Self {
            dim: repr.dim,
            bits: repr.bits,
            projections: repr.projections,
            seed: repr.seed,
            polar: repr.polar,
            qjl: repr.qjl,
        })
    }
}

impl TurboQuantizer {
    pub fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        if !dim.is_multiple_of(2) {
            return Err(QuantError::OddDimension { got: dim });
        }
        if !(2..=16).contains(&bits) {
            return Err(QuantError::InvalidBitWidth { got: bits });
        }
        if projections == 0 {
            return Err(QuantError::ZeroProjectionCount);
        }

        let polar_seed = seed;
        let qjl_seed = seed.wrapping_add(0xCAFE_BABE_0000_0001);

        let polar = PolarQuantizer::new(dim, bits - 1, polar_seed)?;
        let qjl = QjlQuantizer::new(dim, projections, qjl_seed)?;

        Ok(Self {
            dim,
            bits,
            projections,
            seed,
            polar,
            qjl,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn projections(&self) -> usize {
        self.projections
    }

    /// Heap bytes of the cached QJL projection matrix: `projections * dim * 4`, held
    /// once per quantizer (shared by every code and query).
    pub fn projection_cache_bytes(&self) -> usize {
        self.qjl.projection_bytes()
    }

    /// Heap bytes of the cached d×d rotation matrix: `dim * dim * 4`, held once per
    /// quantizer.
    pub fn rotation_matrix_bytes(&self) -> usize {
        self.polar.rotation_memory_bytes()
    }

    /// Heap bytes of the cos/sin table the polar stage scores through (`bits - 1` angle
    /// bits): process-wide, one per bit width, built on the first score.
    pub fn trig_table_bytes(&self) -> usize {
        self.polar.trig_table_bytes()
    }

    /// Encode a vector into a TurboCode.
    pub fn encode(&self, vector: &[f32]) -> Result<TurboCode> {
        if vector.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let polar_code = self.polar.encode(vector)?;
        let reconstruction = self.polar.decode(&polar_code)?;
        let residual: Vec<f32> = vector
            .iter()
            .zip(reconstruction.iter())
            .map(|(o, r)| o - r)
            .collect();
        let residual_sketch = self.qjl.sketch(&residual)?;

        Ok(TurboCode {
            polar_code,
            residual_sketch,
        })
    }

    /// Encode a batch of vectors in parallel.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Result<TurboCode>> {
        vectors.par_iter().map(|v| self.encode(v)).collect()
    }

    /// Unbiased estimate of `<original_vector, query>` ([`ScoreMode::InnerProduct`]).
    /// Unbiased in expectation over the projection matrix (see module docs); not a
    /// per-code guarantee.
    pub fn inner_product_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        self.score(code, query, ScoreMode::InnerProduct)
    }

    /// Score of `code` against `query` under `mode` (see the module docs and
    /// [`ScoreMode`]). A NaN/±Inf query is rejected before it is scored.
    pub fn score(&self, code: &TurboCode, query: &[f32], mode: ScoreMode) -> Result<f32> {
        let context = self.prepare_query(query)?;
        self.score_projected(code, &context, mode)
    }

    /// Build the per-query context once and score every code with
    /// [`Self::score_projected`]: O(d² + m·d) once per query, O(d + m) per code.
    pub fn prepare_query(&self, query: &[f32]) -> Result<TurboQueryContext> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let context = TurboQueryContext {
            rotated_query: self.polar.rotated_query(query)?,
            projected_query: self.qjl.projected_query(query)?,
        };
        // A finite query can overflow the rotation or the projections: such a context is
        // an `Err` here, once per search, before any code is scored (F04e).
        context.check_finite()?;
        Ok(context)
    }

    /// [`Self::score`] against a query already prepared by [`Self::prepare_query`].
    /// Bit-identical to the per-code path.
    pub fn score_projected(
        &self,
        code: &TurboCode,
        context: &TurboQueryContext,
        mode: ScoreMode,
    ) -> Result<f32> {
        let polar_est = self
            .polar
            .inner_product_estimate_rotated(&code.polar_code, &context.rotated_query)?;
        let qjl_term =
            self.qjl
                .score_projected(&code.residual_sketch, &context.projected_query, mode)?;
        combine(polar_est, qjl_term)
    }

    /// [`Self::score_projected`] for a context built by [`Self::prepare_query`] (finite
    /// by construction) and the codes a store holds: each was produced by [`Self::encode`],
    /// which only emits valid codes, so the per-code invariants are checked once at
    /// ingestion and not again per query. A score that overflows is still an `Err`,
    /// never a non-finite number (F04e).
    pub(crate) fn score_projected_validated(
        &self,
        code: &TurboCode,
        context: &TurboQueryContext,
        mode: ScoreMode,
    ) -> Result<f32> {
        debug_assert_eq!(context.rotated_query.len(), self.dim);
        debug_assert_eq!(context.projected_query.len(), self.projections);
        let polar_est = self
            .polar
            .score_validated(&code.polar_code, &context.rotated_query)?;
        let qjl_term = self.qjl.score_projected_validated(
            &code.residual_sketch,
            &context.projected_query,
            mode,
        )?;
        combine(polar_est, qjl_term)
    }

    /// Estimate squared L2 distance.
    pub fn l2_distance_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        let ip = self.inner_product_estimate(code, query)?;
        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let code_norm_sq: f32 = code.polar_code.radii.iter().map(|r| r * r).sum();
        // The estimate can overflow even when `ip` did not (a huge query norm); it is
        // checked before the clamp, which would otherwise turn a NaN into a 0 distance.
        let estimate = query_norm_sq + code_norm_sq - 2.0 * ip;
        if !estimate.is_finite() {
            return Err(QuantError::NonFiniteScore);
        }
        Ok(estimate.max(0.0))
    }

    /// Decode to approximate reconstruction (PolarQuant component only).
    pub fn decode_approximate(&self, code: &TurboCode) -> Result<Vec<f32>> {
        self.polar.decode(&code.polar_code)
    }
}

/// Compression statistics for a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchStats {
    pub count: usize,
    pub total_encoded_bytes: usize,
    pub total_original_bytes: usize,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    #[test]
    fn encode_is_deterministic() {
        let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
        let x = random_vector(16, 1);
        let c1 = q.encode(&x).unwrap();
        let c2 = q.encode(&x).unwrap();
        assert_eq!(c1.polar_code, c2.polar_code);
        assert_eq!(c1.residual_sketch, c2.residual_sketch);
    }

    #[test]
    fn nearest_neighbor_ordering_preserved() {
        let q = TurboQuantizer::new(16, 8, 16, 7).unwrap();
        let query = random_vector(16, 99);
        let close: Vec<f32> = query.iter().map(|x| x + 0.05).collect();
        let far1 = random_vector(16, 200);
        let far2 = random_vector(16, 201);

        let ip_close = q
            .inner_product_estimate(&q.encode(&close).unwrap(), &query)
            .unwrap();
        let ip_far1 = q
            .inner_product_estimate(&q.encode(&far1).unwrap(), &query)
            .unwrap();
        let ip_far2 = q
            .inner_product_estimate(&q.encode(&far2).unwrap(), &query)
            .unwrap();

        assert!(
            ip_close > ip_far1 && ip_close > ip_far2,
            "close={ip_close:.3}, far1={ip_far1:.3}, far2={ip_far2:.3}"
        );
    }

    #[test]
    fn compression_ratio_positive() {
        let q = TurboQuantizer::new(64, 8, 32, 0).unwrap();
        let x = random_vector(64, 1);
        let code = q.encode(&x).unwrap();
        assert!(code.compression_ratio() > 1.0);
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(TurboQuantizer::new(0, 8, 16, 0).is_err());
        assert!(TurboQuantizer::new(7, 8, 16, 0).is_err());
        assert!(TurboQuantizer::new(8, 1, 16, 0).is_err());
        assert!(TurboQuantizer::new(8, 8, 0, 0).is_err());
    }

    #[test]
    fn prepared_query_scores_are_bit_identical_in_both_modes() {
        let dim = 32;
        let q = TurboQuantizer::new(dim, 4, 64, 11).unwrap();
        let query = random_vector(dim, 77);
        let context = q.prepare_query(&query).unwrap();
        assert_eq!(context.memory_bytes(), (dim + 64) * 4);
        for i in 0..1000 {
            let x = random_vector(dim, 5000 + i);
            let code = q.encode(&x).unwrap();
            for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
                assert_eq!(
                    q.score(&code, &query, mode).unwrap().to_bits(),
                    q.score_projected(&code, &context, mode).unwrap().to_bits(),
                    "code {i} {mode:?}"
                );
            }
        }
    }

    #[test]
    fn cache_sizes_and_serde_roundtrip() {
        let dim = 16;
        let q = TurboQuantizer::new(dim, 4, 32, 3).unwrap();
        assert_eq!(q.projection_cache_bytes(), 32 * dim * 4);
        assert_eq!(q.rotation_matrix_bytes(), dim * dim * 4);
        let x = random_vector(dim, 8);
        let query = random_vector(dim, 9);
        let code = q.encode(&x).unwrap();
        let back: TurboQuantizer =
            serde_json::from_str(&serde_json::to_string(&q).unwrap()).unwrap();
        assert_eq!(back.encode(&x).unwrap(), code);
        for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
            assert_eq!(
                back.score(&code, &query, mode).unwrap().to_bits(),
                q.score(&code, &query, mode).unwrap().to_bits()
            );
        }
    }

    #[test]
    fn non_finite_query_is_rejected() {
        let q = TurboQuantizer::new(16, 4, 16, 0).unwrap();
        let code = q.encode(&[1.0f32; 16]).unwrap();
        let mut query = vec![0.25f32; 16];
        query[3] = f32::NEG_INFINITY;
        assert!(matches!(
            q.score(&code, &query, ScoreMode::InnerProduct),
            Err(QuantError::NonFiniteValue { index: 3 })
        ));
        assert!(q.prepare_query(&query).is_err());
    }
}
