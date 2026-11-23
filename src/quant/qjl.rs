//! Quantized Johnson-Lindenstrauss (QJL) transform for unbiased inner product estimation.
//!
//! Projects a d-dimensional vector x onto m random hyperplanes g_i ~ N(0, I) and records
//! the sign of each projection. The current layout stores one i8 per sign plus the f32
//! norm of x (unpacked; bit-packing is out of scope).
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
//! Reference: "Quantized Johnson-Lindenstrauss is Optimal for Inner Products"
//! Zandieh et al., AAAI 2025.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize};
use std::f32::consts::PI;

use super::error::{QuantError, Result};

/// Wire schema version of [`QjlSketch`]. v1 adds `norm`; v0 (<= 2.11.0: `{dim,
/// projections, signs}` only) is rejected on deserialization and never reinterpreted.
pub const QJL_SKETCH_SCHEMA_VERSION: u32 = 1;

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
    pub fn encoded_bytes(&self) -> usize {
        self.projections + std::mem::size_of::<f32>()
    }
}

/// Projects vectors to QJL sketches and estimates inner products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QjlQuantizer {
    dim: usize,
    projections: usize,
    seed: u64,
}

impl QjlQuantizer {
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        if projections == 0 {
            return Err(QuantError::ZeroProjectionCount);
        }
        Ok(Self {
            dim,
            projections,
            seed,
        })
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
        let g = self.projection_matrix();
        let signs: Vec<i8> = g
            .iter()
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

    pub fn inner_product_estimate(&self, sketch: &QjlSketch, query: &[f32]) -> Result<f32> {
        if sketch.dim != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: sketch.dim,
            });
        }
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if sketch.signs.len() != self.projections {
            return Err(QuantError::MalformedCode(format!(
                "QJL sketch has {} signs, this quantizer has {} projections",
                sketch.signs.len(),
                self.projections
            )));
        }
        let g = self.projection_matrix();
        let m = self.projections as f32;
        // E[sign(g.x)(g.q)] = sqrt(2/pi) * <x,q>/||x||  =>  scale = sqrt(pi/2) * ||x|| / m
        let scale = (PI / 2.0).sqrt() * sketch.norm / m;
        let estimate: f32 = g
            .iter()
            .zip(sketch.signs.iter())
            .map(|(row, &sign)| {
                let g_dot_query: f32 = row.iter().zip(query.iter()).map(|(g, q)| g * q).sum();
                sign as f32 * g_dot_query
            })
            .sum();
        Ok(scale * estimate)
    }

    fn projection_matrix(&self) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(0xDEAD_BEEF_1234_5678));
        (0..self.projections)
            .map(|_| {
                (0..self.dim)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect()
            })
            .collect()
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
        let sketch = other.sketch(&vec![1.0f32; 8]).unwrap();
        assert!(q.inner_product_estimate(&sketch, &[1.0f32; 8]).is_err());
    }
}
