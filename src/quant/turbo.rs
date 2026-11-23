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
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::error::{QuantError, Result};
use super::polar::{PolarCode, PolarQuantizer};
use super::qjl::{QjlQuantizer, QjlSketch};

/// A TurboQuant-compressed vector: polar code + QJL residual sketch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurboCode {
    pub polar_code: PolarCode,
    pub residual_sketch: QjlSketch,
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

/// Everything in an estimate that depends only on the query (A02-F02): the rotated
/// query `R·q` for the polar stage and `g_i·q` for the QJL stage. Built once per
/// search by [`TurboQuantizer::prepare_query`] and shared by every code, so the O(d²)
/// rotation and the O(m·d) projection are paid once per query instead of once per code.
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
}

/// TurboQuant compressor: encodes vectors and estimates inner products.
///
/// Configuration `(dim, bits, projections, seed)` fully determines state.
/// Only these four values need to be persisted; the rotation and projection
/// matrices are rebuilt from them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantizer {
    dim: usize,
    bits: u8,
    projections: usize,
    seed: u64,
    polar: PolarQuantizer,
    qjl: QjlQuantizer,
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

    /// Estimate <original_vector, query> from a TurboCode and raw query.
    /// Unbiased in expectation over the projection matrix (see module docs); not a
    /// per-code guarantee.
    pub fn inner_product_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        let context = self.prepare_query(query)?;
        self.inner_product_estimate_projected(code, &context)
    }

    /// Build the per-query context once and score every code with
    /// [`Self::inner_product_estimate_projected`]: O(d² + m·d) once per query,
    /// O(d + m) per code.
    pub fn prepare_query(&self, query: &[f32]) -> Result<TurboQueryContext> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        Ok(TurboQueryContext {
            rotated_query: self.polar.rotated_query(query)?,
            projected_query: self.qjl.projected_query(query)?,
        })
    }

    /// [`Self::inner_product_estimate`] against a query already prepared by
    /// [`Self::prepare_query`]. Bit-identical to the per-code path.
    pub fn inner_product_estimate_projected(
        &self,
        code: &TurboCode,
        context: &TurboQueryContext,
    ) -> Result<f32> {
        let polar_est = self
            .polar
            .inner_product_estimate_rotated(&code.polar_code, &context.rotated_query)?;
        let qjl_correction = self
            .qjl
            .inner_product_estimate_projected(&code.residual_sketch, &context.projected_query)?;
        Ok(polar_est + qjl_correction)
    }

    /// Estimate squared L2 distance.
    pub fn l2_distance_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        let ip = self.inner_product_estimate(code, query)?;
        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let code_norm_sq: f32 = code.polar_code.radii.iter().map(|r| r * r).sum();
        Ok((query_norm_sq + code_norm_sq - 2.0 * ip).max(0.0))
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
    fn prepared_query_estimates_are_bit_identical() {
        let dim = 32;
        let q = TurboQuantizer::new(dim, 4, 64, 11).unwrap();
        let query = random_vector(dim, 77);
        let context = q.prepare_query(&query).unwrap();
        assert_eq!(context.memory_bytes(), (dim + 64) * 4);
        for i in 0..1000 {
            let x = random_vector(dim, 5000 + i);
            let code = q.encode(&x).unwrap();
            assert_eq!(
                q.inner_product_estimate(&code, &query).unwrap().to_bits(),
                q.inner_product_estimate_projected(&code, &context)
                    .unwrap()
                    .to_bits(),
                "code {i}"
            );
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
        assert_eq!(
            back.inner_product_estimate(&code, &query)
                .unwrap()
                .to_bits(),
            q.inner_product_estimate(&code, &query).unwrap().to_bits()
        );
    }
}
