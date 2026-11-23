//! Quantized Johnson-Lindenstrauss (QJL) transform for unbiased inner product estimation.
//!
//! Projects a d-dimensional vector x onto m random hyperplanes g_i ~ N(0, I) and records
//! the sign of each projection. [`QjlSketch`] keeps one i8 per sign plus the f32 norm of
//! x (unpacked: public API and serde wire format). The store holds the packed form — the
//! signs as 1 bit each (`1` = -1, `0` = +1) plus the same f32 norm — a bijection of the
//! sketch ([`pack_signs`] / [`unpack_signs`]); [`QjlQuantizer::score_packed`] reproduces
//! the f32 score bits of [`QjlQuantizer::score_projected`].
//!
//! Unbiased estimator (expectation over g, for ANY ||x||):
//!
//!   E[sign(g.x)(g.y)] = sqrt(2/pi) * <x,y> / ||x||
//!   =>  <x, y> ~ sqrt(pi/2) * ||x|| / m * sum_i sign(g_i.x) * (g_i.y)
//!
//! Until schema v1 the scale was pi/(2m) without the ||x|| factor, i.e. the estimate
//! equaled sqrt(pi/2)/||x|| times the true inner product (unbiased only at
//! ||x|| = sqrt(pi/2)). Verified numerically in tests/quant_contracts (mean over 2000
//! deterministic pairs per norm group, tolerance 2%).
//!
//! # Two separate contracts: estimator vs ranking objective (D18)
//!
//! [`ScoreMode::InnerProduct`] ranks by the unbiased estimator above. It is NOT kept as
//! the default order of a search that generates L2 candidates: an unbiased inner product
//! is not an L2 ranking. [`ScoreMode::L2Heuristic`] (the default) ranks by the legacy
//! order `estimate / ||x||`, i.e. `sqrt(pi/2)/m * sum_i sign(g_i.x)(g_i.q)`, which is
//! what every version <= 2.11.0 computed and is reproduced bit for bit. It is a
//! heuristic for L2 candidate generation, not an unbiased estimator of anything; the
//! two modes are selected explicitly and the mode travels with the result.
//!
//! Reference: "Quantized Johnson-Lindenstrauss is Optimal for Inner Products"
//! Zandieh et al., AAAI 2025.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize};
use std::f32::consts::PI;
use std::sync::Arc;

use super::error::{QuantError, Result};
use super::polar::{bit_stream_bytes, read_bits, write_bits};

/// Wire schema version of [`QjlSketch`]. v1 adds `norm`; v0 (<= 2.11.0: `{dim,
/// projections, signs}` only) is rejected on deserialization and never reinterpreted.
pub const QJL_SKETCH_SCHEMA_VERSION: u32 = 1;

/// Objective a score ranks by. The unbiased inner-product estimator and the ranking
/// used to generate L2 candidates are separate contracts (D18): neither is a bug fix
/// of the other, and a search states which one it used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScoreMode {
    /// Default order of approximate L2 candidate generation: the legacy
    /// `estimate / ||x||`, i.e. `sqrt(pi/2)/m * sum_i sign(g_i.x)(g_i.q)` — exactly what
    /// versions <= 2.11.0 ranked by. A heuristic: it is `sqrt(pi/2) * cos(x, q)`, not an
    /// unbiased estimator of the inner product nor of the L2 distance.
    ///
    /// `||x|| == 0` needs no special case: there is no division. Every sign of the zero
    /// vector is +1 and the score is `sqrt(pi/2)/m * sum_i (g_i.q)`, finite and
    /// deterministic — the same value the legacy estimator returned for it.
    #[default]
    L2Heuristic,
    /// Ranks by the unbiased inner-product estimate `sqrt(pi/2) * ||x|| / m * sum_i ...`
    /// (zero for the zero vector).
    InnerProduct,
}

impl ScoreMode {
    /// Stable wire name (CLI/JSON): `l2_heuristic` | `ip`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2Heuristic => "l2_heuristic",
            Self::InnerProduct => "ip",
        }
    }

    /// Parses a wire name (case-sensitive; surrounding whitespace ignored).
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim() {
            "l2_heuristic" => Some(Self::L2Heuristic),
            "ip" => Some(Self::InnerProduct),
            _ => None,
        }
    }
}

impl std::fmt::Display for ScoreMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A QJL sketch: the sign of each random projection (+1 or -1) plus the norm of the
/// sketched vector `x`, which the unbiased estimator needs as a multiplicative factor.
///
/// Serde compatibility: v1 = v0 + `norm: f32`. A v0 payload fails to deserialize with an
/// explicit error — its estimates were biased and `||x||` cannot be recovered from the
/// signs, so there is no migration: re-encode the source vectors. Byte cost of v1 vs v0:
/// +4 bytes per sketch; signs stay 1 byte each (unpacked layout).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QjlSketch {
    pub dim: usize,
    pub projections: usize,
    pub signs: Vec<i8>,
    /// `||x||` of the sketched vector (schema v1).
    pub norm: f32,
}

impl<'de> Deserialize<'de> for QjlSketch {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            dim: usize,
            projections: usize,
            signs: Vec<i8>,
            // Absent in v0 payloads: sentinel, turned into an explicit error below.
            #[serde(default = "missing_norm")]
            norm: f32,
        }
        fn missing_norm() -> f32 {
            f32::NAN
        }

        let repr = Repr::deserialize(deserializer)?;
        let norm = repr.norm;
        if norm.is_nan() {
            return Err(serde::de::Error::custom(
                "QJL sketch schema v0 (no `norm` field): its inner-product estimates were biased by sqrt(pi/2)/||x|| and the norm cannot be recovered from the signs; re-encode the source vectors",
            ));
        }
        if repr.signs.len() != repr.projections {
            return Err(serde::de::Error::custom(format!(
                "malformed QJL sketch: {} signs for {} projections",
                repr.signs.len(),
                repr.projections
            )));
        }
        if !norm.is_finite() || norm < 0.0 {
            return Err(serde::de::Error::custom(
                "malformed QJL sketch: norm must be finite and non-negative",
            ));
        }
        Ok(QjlSketch {
            dim: repr.dim,
            projections: repr.projections,
            signs: repr.signs,
            norm,
        })
    }
}

impl QjlSketch {
    /// Payload bytes of one sketch in the unpacked representation: one i8 per sign plus
    /// the f32 norm — independent of `bits` (QJL is 1 bit per projection by design).
    /// The packed storage layout costs [`Self::packed_layout_bytes`].
    pub const fn layout_bytes(projections: usize) -> usize {
        projections * std::mem::size_of::<i8>() + std::mem::size_of::<f32>()
    }

    pub fn encoded_bytes(&self) -> usize {
        Self::layout_bytes(self.projections)
    }

    /// Heap bytes actually held by this sketch: the sign vector only.
    ///
    /// `norm` is an inline field of the struct, counted once through
    /// `size_of::<TurboCode>()` and not here; `encoded_bytes()` =
    /// `heap_bytes()` + the 4 B of the norm.
    pub fn heap_bytes(&self) -> usize {
        self.signs.len() * std::mem::size_of::<i8>()
    }

    /// Payload bytes of one sketch in the packed storage layout: 1 bit per sign plus
    /// the f32 norm.
    pub const fn packed_layout_bytes(projections: usize) -> usize {
        bit_stream_bytes(projections, 1) + std::mem::size_of::<f32>()
    }
}

/// Bytes of the packed sign stream of a sketch with `projections` signs (1 bit each).
pub const fn sign_stream_bytes(projections: usize) -> usize {
    bit_stream_bytes(projections, 1)
}

/// Pack signs into their bit stream: `-1` -> `1`, `+1` -> `0`. Any other value is an
/// error (a corrupt sketch is never packed as if it held signs).
pub fn pack_signs(signs: &[i8]) -> Result<Vec<u8>> {
    if let Some(index) = signs.iter().position(|&s| s != 1 && s != -1) {
        return Err(QuantError::MalformedCode(format!(
            "QJL sign at projection {index} is {}, not +1 or -1",
            signs[index]
        )));
    }
    let mut data = vec![0u8; sign_stream_bytes(signs.len())];
    for (i, &sign) in signs.iter().enumerate() {
        if sign < 0 {
            write_bits(&mut data, i, 1, 1);
        }
    }
    Ok(data)
}

/// The first `projections` signs of a packed sign stream (`1` -> -1, `0` -> +1). The
/// padding bits of the last byte are never read.
pub fn unpack_signs(data: &[u8], projections: usize) -> Vec<i8> {
    (0..projections)
        .map(|i| if read_bits(data, i, 1) == 1 { -1 } else { 1 })
        .collect()
}

/// Projects vectors to QJL sketches and scores them against a query.
///
/// The `projections x dim` Gaussian matrix is a pure function of `(dim, projections,
/// seed)`. It is generated exactly once, in [`Self::new`], and shared (never copied per
/// sketch or per row) by every `sketch`, score and search of this quantizer (A02-F02);
/// [`Self::projection_bytes`] is what it costs.
#[derive(Debug, Clone, Serialize)]
pub struct QjlQuantizer {
    dim: usize,
    projections: usize,
    seed: u64,
    /// Row-major `projections x dim` matrix. Not serialized: it is rebuilt from
    /// `(dim, projections, seed)` on deserialization, so the wire format is unchanged.
    #[serde(skip)]
    projection: Arc<[f32]>,
}

impl<'de> Deserialize<'de> for QjlQuantizer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Same wire fields as before the cache existed; `new` regenerates the matrix.
        #[derive(Deserialize)]
        struct Repr {
            dim: usize,
            projections: usize,
            seed: u64,
        }
        let repr = Repr::deserialize(deserializer)?;
        Self::new(repr.dim, repr.projections, repr.seed).map_err(serde::de::Error::custom)
    }
}

impl QjlQuantizer {
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        if projections == 0 {
            return Err(QuantError::ZeroProjectionCount);
        }
        // Generated once per quantizer configuration (A02-F02): before, every sketch()
        // and every estimate sampled m*d normals again.
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF_1234_5678));
        let mut matrix = Vec::with_capacity(projections * dim);
        for _ in 0..projections {
            for _ in 0..dim {
                matrix.push(StandardNormal.sample(&mut rng));
            }
        }
        Ok(Self {
            dim,
            projections,
            seed,
            projection: matrix.into(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn projections(&self) -> usize {
        self.projections
    }

    /// Heap bytes of the cached projection matrix: `projections * dim * 4`, held once
    /// per quantizer regardless of how many sketches or queries use it.
    pub fn projection_bytes(&self) -> usize {
        self.projection.len() * std::mem::size_of::<f32>()
    }

    pub fn sketch(&self, vector: &[f32]) -> Result<QjlSketch> {
        if vector.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        // A non-finite vector has no meaningful projection signs.
        if let Some(index) = vector.iter().position(|v| !v.is_finite()) {
            return Err(QuantError::NonFiniteValue { index });
        }
        let signs: Vec<i8> = self
            .projection
            .chunks_exact(self.dim)
            .map(|row| {
                let dot: f32 = row.iter().zip(vector.iter()).map(|(g, x)| g * x).sum();
                if dot >= 0.0 {
                    1i8
                } else {
                    -1i8
                }
            })
            .collect();
        let norm = vector
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt() as f32;
        if !norm.is_finite() {
            // Squares overflow f32 even though every component is finite.
            return Err(QuantError::MalformedCode(
                "vector norm is not finite (f32 overflow)".to_string(),
            ));
        }
        Ok(QjlSketch {
            dim: self.dim,
            projections: self.projections,
            signs,
            norm,
        })
    }

    /// Unbiased estimate of `<x, query>` ([`ScoreMode::InnerProduct`]).
    pub fn inner_product_estimate(&self, sketch: &QjlSketch, query: &[f32]) -> Result<f32> {
        self.score(sketch, query, ScoreMode::InnerProduct)
    }

    /// Score of `sketch` against `query` under `mode` (see [`ScoreMode`]).
    pub fn score(&self, sketch: &QjlSketch, query: &[f32], mode: ScoreMode) -> Result<f32> {
        let projected = self.projected_query(query)?;
        self.score_projected(sketch, &projected, mode)
    }

    /// `g_i . query` for every projection row `i` (`projections` values): the only part
    /// of a score that depends on the query. Compute it once per query and score any
    /// number of sketches against it with [`Self::score_projected`] — the per-sketch cost
    /// drops from `m*d` to `m` (A02-F02). The values are bit-identical to the dot
    /// product the per-sketch path used to recompute.
    pub fn projected_query(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        // A NaN/Inf query has no meaningful score: reject it before anything is ranked.
        if let Some(index) = query.iter().position(|v| !v.is_finite()) {
            return Err(QuantError::NonFiniteValue { index });
        }
        Ok(self
            .projection
            .chunks_exact(self.dim)
            .map(|row| row.iter().zip(query.iter()).map(|(g, q)| g * q).sum())
            .collect())
    }

    /// [`Self::score`] against a query already projected by [`Self::projected_query`].
    pub fn score_projected(
        &self,
        sketch: &QjlSketch,
        projected_query: &[f32],
        mode: ScoreMode,
    ) -> Result<f32> {
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
        if projected_query.len() != self.projections {
            return Err(QuantError::MalformedCode(format!(
                "projected query has {} entries, this quantizer has {} projections",
                projected_query.len(),
                self.projections
            )));
        }
        let m = self.projections as f32;
        // Same accumulation order as the per-sketch estimator (bit-identical values).
        let estimate: f32 = sketch
            .signs
            .iter()
            .zip(projected_query.iter())
            .map(|(&sign, &g_dot_query)| sign as f32 * g_dot_query)
            .sum();
        let scale = match mode {
            // E[sign(g.x)(g.q)] = sqrt(2/pi) * <x,q>/||x||  =>  scale = sqrt(pi/2) * ||x|| / m
            ScoreMode::InnerProduct => (PI / 2.0).sqrt() * sketch.norm / m,
            // Legacy order (<= 2.11.0, scale pi/(2m)) = sqrt(pi/2) * estimate / ||x||;
            // no division, so ||x|| == 0 is finite.
            ScoreMode::L2Heuristic => PI / (2.0 * m),
        };
        Ok(scale * estimate)
    }

    /// [`Self::score_projected`] for a packed sketch: `signs` is its sign stream of
    /// [`sign_stream_bytes(projections)`](sign_stream_bytes) bytes and `norm` its
    /// `||x||`. Bit-identical to the unpacked path: sign `i` against projection `i`,
    /// sequential f32 accumulation, same scale. No unpacked copy is materialized.
    pub fn score_packed(
        &self,
        signs: &[u8],
        norm: f32,
        projected_query: &[f32],
        mode: ScoreMode,
    ) -> Result<f32> {
        if signs.len() != sign_stream_bytes(self.projections) {
            return Err(QuantError::MalformedCode(format!(
                "packed QJL sketch has {} sign bytes, this quantizer needs {}",
                signs.len(),
                sign_stream_bytes(self.projections)
            )));
        }
        if projected_query.len() != self.projections {
            return Err(QuantError::MalformedCode(format!(
                "projected query has {} entries, this quantizer has {} projections",
                projected_query.len(),
                self.projections
            )));
        }
        if !norm.is_finite() || norm < 0.0 {
            return Err(QuantError::MalformedCode(
                "QJL sketch norm must be finite and non-negative".to_string(),
            ));
        }
        let m = self.projections as f32;
        let estimate: f32 = (0..self.projections)
            .map(|i| {
                let sign = if read_bits(signs, i, 1) == 1 {
                    -1.0f32
                } else {
                    1.0f32
                };
                sign * projected_query[i]
            })
            .sum();
        let scale = match mode {
            ScoreMode::InnerProduct => (PI / 2.0).sqrt() * norm / m,
            ScoreMode::L2Heuristic => PI / (2.0 * m),
        };
        Ok(scale * estimate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    #[test]
    fn sketch_is_deterministic() {
        let q = QjlQuantizer::new(16, 32, 42).unwrap();
        let x = random_vector(16, 1);
        let s1 = q.sketch(&x).unwrap();
        let s2 = q.sketch(&x).unwrap();
        assert_eq!(s1.signs, s2.signs);
    }

    #[test]
    fn inner_product_estimate_scales_with_the_norm_of_x() {
        // Doubling x leaves every sign unchanged, so an unbiased estimator must double.
        // (The full statistical property lives in tests/quant_contracts/properties.rs.)
        let q = QjlQuantizer::new(16, 64, 0).unwrap();
        let x = random_vector(16, 10);
        let y = random_vector(16, 20);
        let x2: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
        let e1 = q
            .inner_product_estimate(&q.sketch(&x).unwrap(), &y)
            .unwrap();
        let e2 = q
            .inner_product_estimate(&q.sketch(&x2).unwrap(), &y)
            .unwrap();
        assert!((e2 - 2.0 * e1).abs() <= 1e-5 * e2.abs(), "e1={e1}, e2={e2}");
    }

    #[test]
    fn sketch_with_wrong_projection_count_is_rejected() {
        let q = QjlQuantizer::new(8, 4, 0).unwrap();
        let other = QjlQuantizer::new(8, 8, 0).unwrap();
        let sketch = other.sketch(&[1.0f32; 8]).unwrap();
        assert!(q.inner_product_estimate(&sketch, &[1.0f32; 8]).is_err());
    }

    #[test]
    fn score_mode_wire_names_round_trip() {
        assert_eq!(ScoreMode::default(), ScoreMode::L2Heuristic);
        for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
            assert_eq!(ScoreMode::parse(mode.as_str()), Some(mode));
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!(ScoreMode::parse("inner_product"), None);
        assert_eq!(ScoreMode::parse(""), None);
    }

    #[test]
    fn projected_query_reuse_is_bit_identical_for_both_modes() {
        let dim = 32;
        let q = QjlQuantizer::new(dim, 64, 9).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        let query: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
        let projected = q.projected_query(&query).unwrap();
        for i in 0..1000 {
            let mut rng = ChaCha8Rng::seed_from_u64(10_000 + i);
            let x: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
            let sketch = q.sketch(&x).unwrap();
            for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
                assert_eq!(
                    q.score(&sketch, &query, mode).unwrap().to_bits(),
                    q.score_projected(&sketch, &projected, mode)
                        .unwrap()
                        .to_bits(),
                    "pair {i} {mode:?}"
                );
            }
        }
        // A projected query of the wrong length is rejected, not misread.
        assert!(q
            .score_projected(
                &q.sketch(&query).unwrap(),
                &projected[..63],
                ScoreMode::InnerProduct
            )
            .is_err());
    }

    #[test]
    fn projection_cache_is_built_once_and_not_serialized() {
        let dim = 16;
        let q = QjlQuantizer::new(dim, 32, 5).unwrap();
        assert_eq!(q.projection_bytes(), 32 * dim * 4);
        let x = random_vector(dim, 3);
        let y = random_vector(dim, 4);
        let sketch = q.sketch(&x).unwrap();
        let wire = serde_json::to_string(&q).unwrap();
        assert!(!wire.contains("\"projection\":"), "{wire}");
        let back: QjlQuantizer = serde_json::from_str(&wire).unwrap();
        // The deserialized quantizer rebuilt the same matrix: same signs, same scores.
        assert_eq!(back.sketch(&x).unwrap(), sketch);
        assert_eq!(
            back.score(&sketch, &y, ScoreMode::L2Heuristic)
                .unwrap()
                .to_bits(),
            q.score(&sketch, &y, ScoreMode::L2Heuristic)
                .unwrap()
                .to_bits()
        );
        assert_eq!(back.projection_bytes(), q.projection_bytes());
    }

    #[test]
    fn non_finite_query_is_rejected() {
        let q = QjlQuantizer::new(8, 16, 0).unwrap();
        let sketch = q.sketch(&[1.0f32; 8]).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut query = vec![0.5f32; 8];
            query[6] = value;
            for mode in [ScoreMode::L2Heuristic, ScoreMode::InnerProduct] {
                assert!(matches!(
                    q.score(&sketch, &query, mode),
                    Err(QuantError::NonFiniteValue { index: 6 })
                ));
            }
            assert!(matches!(
                q.inner_product_estimate(&sketch, &query),
                Err(QuantError::NonFiniteValue { index: 6 })
            ));
        }
    }
}
