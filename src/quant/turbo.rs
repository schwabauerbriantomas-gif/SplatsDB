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
use serde::{Deserialize, Serialize};

use super::error::{QuantError, Result};
use super::polar::{bit_stream_bytes, read_bits, PackedPolarCode, PolarCode, PolarQuantizer};
use super::qjl::{pack_signs, sign_stream_bytes, unpack_signs, QjlQuantizer, QjlSketch, ScoreMode};

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

/// A [`TurboCode`] in the packed storage layout — what the
/// [`QuantizedStore`](crate::quantization::QuantizedStore) holds.
///
/// | part | depends on | stored as |
/// |---|---|---|
/// | polar radii | `dim` | `dim / 2` f32, exact (unchanged) |
/// | polar angle indices | `dim`, `bits` (`bits - 1` bits each) | one bit stream |
/// | QJL signs | `projections` | one bit stream, 1 bit each (`1` = -1) |
/// | residual norm | — | f32, inline (read by [`ScoreMode::InnerProduct`] only) |
///
/// The two streams share one buffer and each starts at a byte boundary. Produced by
/// [`TurboQuantizer::encode_packed`] / [`TurboQuantizer::pack`], read back with
/// [`TurboQuantizer::unpack_packed`]; [`TurboQuantizer::score_packed`] reproduces the
/// f32 score bits of [`TurboQuantizer::score_projected`] on the unpacked code.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTurboCode {
    /// Angle resolution of the stream (the quantizer's `bits - 1`).
    polar_bits: u8,
    /// `dim / 2` radii — exactly the f32 values of the unpacked code.
    radii: Vec<f32>,
    /// Angle stream ([`Self::angle_stream_bytes`] B) followed by the sign stream
    /// ([`sign_stream_bytes(projections)`](super::qjl::sign_stream_bytes) B).
    packed: Vec<u8>,
    /// `||x - polar reconstruction||` (f32, inline).
    norm: f32,
}

impl PackedTurboCode {
    /// Payload bytes of one packed code for this configuration.
    pub const fn layout_bytes(dim: usize, bits: u8, projections: usize) -> usize {
        let pairs = dim / 2;
        pairs * std::mem::size_of::<f32>()
            + bit_stream_bytes(pairs, bits.saturating_sub(1))
            + sign_stream_bytes(projections)
            + std::mem::size_of::<f32>()
    }

    /// Number of (radius, angle) pairs, i.e. `dim / 2`.
    pub fn pairs(&self) -> usize {
        self.radii.len()
    }

    /// Angle resolution of the stream (`bits - 1`).
    pub fn polar_bits(&self) -> u8 {
        self.polar_bits
    }

    /// Length of the angle stream in bytes (the sign stream starts right after it).
    pub fn angle_stream_bytes(&self) -> usize {
        bit_stream_bytes(self.radii.len(), self.polar_bits)
    }

    /// The f32 radius of pair `i` (identical to `TurboCode::polar_code.radii[i]`).
    pub fn radius(&self, i: usize) -> f32 {
        self.radii[i]
    }

    /// The polar angle index of pair `i` in `[0, 2^(bits-1))`.
    pub fn angle_index(&self, i: usize) -> u32 {
        let width = self.polar_bits as usize;
        read_bits(&self.packed, i * width, self.polar_bits)
    }

    /// The QJL sign of projection `i`: `1` = -1, `0` = +1.
    pub fn sign_bit(&self, i: usize) -> u32 {
        read_bits(&self.packed, self.angle_stream_bytes() * 8 + i, 1)
    }

    /// `||residual||` of the sketch.
    pub fn norm(&self) -> f32 {
        self.norm
    }

    /// Heap bytes held: the radii and the packed streams.
    pub fn heap_bytes(&self) -> usize {
        self.radii.len() * std::mem::size_of::<f32>() + self.packed.len()
    }

    /// Heap bytes reserved (capacity) by the two buffers: what the allocator was asked
    /// for, beyond `heap_bytes()` when a `Vec` keeps spare capacity.
    pub fn reserved_bytes(&self) -> usize {
        self.radii.capacity() * std::mem::size_of::<f32>() + self.packed.capacity()
    }

    /// Payload bytes: [`Self::heap_bytes`] + the 4 B of the inline `f32` norm (same
    /// convention as [`TurboCode::encoded_bytes`]: the norm is not counted again
    /// through `size_of::<Self>()`).
    pub fn encoded_bytes(&self) -> usize {
        self.heap_bytes() + std::mem::size_of::<f32>()
    }

    /// Raw f32 bytes divided by the real encoded bytes.
    pub fn compression_ratio(&self) -> f32 {
        let original = self.pairs() * 2 * std::mem::size_of::<f32>();
        original as f32 / self.encoded_bytes().max(1) as f32
    }
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

    /// Payload bytes of one code in the packed storage layout
    /// ([`PackedTurboCode::layout_bytes`] for this configuration).
    pub fn code_layout_bytes(&self) -> usize {
        PackedTurboCode::layout_bytes(self.dim, self.bits, self.projections)
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

    /// Encode into the packed storage layout. Same code as [`Self::encode`] — packed
    /// before it is returned; the unpacked form lives only inside this call.
    pub fn encode_packed(&self, vector: &[f32]) -> Result<PackedTurboCode> {
        self.pack(&self.encode(vector)?)
    }

    /// Pack an unpacked code of this quantizer — a bijection. The polar code and the
    /// residual sketch must belong to this quantizer and be well formed (finite radii
    /// and norm, angle indices in range, signs +-1); anything else is an error.
    pub fn pack(&self, code: &TurboCode) -> Result<PackedTurboCode> {
        let polar = self.polar.pack(&code.polar_code)?;
        let sketch = &code.residual_sketch;
        if sketch.dim != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: sketch.dim,
            });
        }
        if sketch.signs.len() != self.projections {
            return Err(QuantError::MalformedCode(format!(
                "QJL sketch has {} signs, this quantizer has {} projections",
                sketch.signs.len(),
                self.projections
            )));
        }
        if !sketch.norm.is_finite() || sketch.norm < 0.0 {
            return Err(QuantError::MalformedCode(
                "QJL sketch norm must be finite and non-negative".to_string(),
            ));
        }
        let (radii, mut packed) = polar.into_parts();
        packed.extend_from_slice(&pack_signs(&sketch.signs)?);
        Ok(PackedTurboCode {
            polar_bits: self.bits - 1,
            radii,
            packed,
            norm: sketch.norm,
        })
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
        Ok(TurboQueryContext {
            rotated_query: self.polar.rotated_query(query)?,
            projected_query: self.qjl.projected_query(query)?,
        })
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
        Ok(polar_est + qjl_term)
    }

    /// Exact inverse of [`Self::pack`]: the original [`TurboCode`].
    pub fn unpack_packed(&self, code: &PackedTurboCode) -> Result<TurboCode> {
        self.validate_packed(code)?;
        let (angles, signs) = code.packed.split_at(code.angle_stream_bytes());
        let polar_code = self.polar.unpack_packed(&PackedPolarCode::new(
            code.polar_bits,
            code.radii.clone(),
            angles.to_vec(),
        )?)?;
        Ok(TurboCode {
            polar_code,
            residual_sketch: QjlSketch {
                dim: self.dim,
                projections: self.projections,
                signs: unpack_signs(signs, self.projections),
                norm: code.norm,
            },
        })
    }

    /// [`Self::score_projected`] for a packed code: bit-identical ids, order and f32
    /// score bits (same dequantization, same accumulation order, same scale). Reads the
    /// packed streams in place — no unpacked copy is materialized.
    pub fn score_packed(
        &self,
        code: &PackedTurboCode,
        context: &TurboQueryContext,
        mode: ScoreMode,
    ) -> Result<f32> {
        self.validate_packed(code)?;
        let (angles, signs) = code.packed.split_at(code.angle_stream_bytes());
        let polar_est = self.polar.inner_product_from_stream(
            code.polar_bits,
            &code.radii,
            angles,
            &context.rotated_query,
        )?;
        let qjl_term = self
            .qjl
            .score_packed(signs, code.norm, &context.projected_query, mode)?;
        Ok(polar_est + qjl_term)
    }

    /// A packed code must come from a quantizer with the same `dim`, `bits` and
    /// `projections`: anything else is rejected before a stream is read.
    fn validate_packed(&self, code: &PackedTurboCode) -> Result<()> {
        if code.polar_bits != self.bits - 1 {
            return Err(QuantError::MalformedCode(format!(
                "code polar bits={}, quantizer bits-1={}",
                code.polar_bits,
                self.bits - 1
            )));
        }
        if code.pairs() * 2 != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: code.pairs() * 2,
            });
        }
        let expected = code.angle_stream_bytes() + sign_stream_bytes(self.projections);
        if code.packed.len() != expected {
            return Err(QuantError::MalformedCode(format!(
                "packed code holds {} bytes, expected {} (angles) + {} (signs)",
                code.packed.len(),
                code.angle_stream_bytes(),
                sign_stream_bytes(self.projections)
            )));
        }
        Ok(())
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
