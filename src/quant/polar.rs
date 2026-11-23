//! PolarQuant: vector compression via polar coordinate encoding.
//!
//! Converts Cartesian pairs into polar form (radius, angle), then uniformly
//! quantizes angles. After rotation, angles are uniformly distributed on [-pi, pi],
//! making uniform quantization near-optimal with no calibration needed.
//!
//! Algorithm: Given rotated vector y = R·x, group into d/2 pairs,
//! convert to polar (r, theta), quantize theta to `bits` levels.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::OnceLock;

use super::error::{QuantError, Result};
use super::rotation::StoredRotation;

/// A compressed vector in polar form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolarCode {
    pub dim: usize,
    pub bits: u8,
    /// Per-pair radii (f32, lossless).
    pub radii: Vec<f32>,
    /// Quantized angle indices in [0, 2^bits).
    pub angle_indices: Vec<u16>,
}

impl PolarCode {
    pub fn pair_count(&self) -> usize {
        self.dim / 2
    }

    /// Dequantized angle of pair `i` (radians, f32).
    pub fn dequantize_angle(&self, i: usize) -> f32 {
        Self::dequantize_index(self.angle_indices[i], self.bits)
    }

    /// The dequantized angle of `index` at `bits`: `(index / 2^bits) * 2pi - pi`, f32
    /// throughout. The only place the angle is defined: the cos/sin table of a bit width
    /// and every direct evaluation consume exactly this value.
    pub fn dequantize_index(index: u16, bits: u8) -> f32 {
        let levels = 1u32 << bits;
        (index as f32 / levels as f32) * (2.0 * PI) - PI
    }

    /// Heap bytes of the cos/sin table a score at `bits` goes through: `2^bits` entries of
    /// two f32 (16 B at 1 bit, 64 B at 3 bits, 1 KiB at 7 bits, 512 KiB at 16 bits). There
    /// is one table per bit width for the whole process, built on the first score and
    /// shared by every quantizer and store of that bit width: it is not owned by a store,
    /// so it is not part of `QuantizedStore::memory_usage()`.
    pub const fn trig_table_bytes(bits: u8) -> usize {
        (1usize << bits) * std::mem::size_of::<[f32; 2]>()
    }

    /// Payload bytes of one code in the current unpacked layout: an f32 radius and a
    /// u16 angle index per pair (6 B per pair) — independent of `bits`. Bit-packing is
    /// not implemented; until it is, this is what a code really costs.
    pub const fn layout_bytes(dim: usize) -> usize {
        dim / 2 * (std::mem::size_of::<f32>() + std::mem::size_of::<u16>())
    }

    pub fn encoded_bytes(&self) -> usize {
        Self::layout_bytes(self.dim)
    }

    /// Heap bytes actually held by this code (radius and angle vectors).
    pub fn heap_bytes(&self) -> usize {
        self.radii.len() * std::mem::size_of::<f32>()
            + self.angle_indices.len() * std::mem::size_of::<u16>()
    }
}

/// Encodes/decodes vectors using PolarQuant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarQuantizer {
    dim: usize,
    bits: u8,
    rotation: StoredRotation,
}

impl PolarQuantizer {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        if !dim.is_multiple_of(2) {
            return Err(QuantError::OddDimension { got: dim });
        }
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { got: bits });
        }
        let rotation = StoredRotation::new(dim, seed)?;
        Ok(Self {
            dim,
            bits,
            rotation,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn encode(&self, vector: &[f32]) -> Result<PolarCode> {
        if vector.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        // NaN/Inf would otherwise become a NaN radius and a "valid" angle index 0.
        if let Some(index) = vector.iter().position(|v| !v.is_finite()) {
            return Err(QuantError::NonFiniteValue { index });
        }
        let mut rotated = vec![0.0f32; self.dim];
        self.rotation.apply(vector, &mut rotated)?;

        let pairs = self.dim / 2;
        let mut radii = Vec::with_capacity(pairs);
        let mut angle_indices = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let a = rotated[2 * i];
            let b = rotated[2 * i + 1];
            let (r, idx) = encode_pair(a, b, self.bits);
            radii.push(r);
            angle_indices.push(idx);
        }
        Ok(PolarCode {
            dim: self.dim,
            bits: self.bits,
            radii,
            angle_indices,
        })
    }

    pub fn decode(&self, code: &PolarCode) -> Result<Vec<f32>> {
        self.validate_code(code)?;
        let mut rotated = vec![0.0f32; self.dim];
        for i in 0..self.dim / 2 {
            let theta = code.dequantize_angle(i);
            let r = code.radii[i];
            rotated[2 * i] = r * theta.cos();
            rotated[2 * i + 1] = r * theta.sin();
        }
        let mut output = vec![0.0f32; self.dim];
        self.rotation.apply_inverse(&rotated, &mut output)?;
        Ok(output)
    }

    /// `<reconstruction(code), query>`: the exact inner product of the polar
    /// reconstruction (no QJL stage; this is what PolarQuant ranks by in every
    /// [`ScoreMode`](super::qjl::ScoreMode)).
    pub fn inner_product_estimate(&self, code: &PolarCode, query: &[f32]) -> Result<f32> {
        let rotated_query = self.rotated_query(query)?;
        self.inner_product_estimate_rotated(code, &rotated_query)
    }

    /// Rotate a query once (`R·query`) so that many codes can be scored against it
    /// without repeating the O(d²) rotation per code (A02-F02). A NaN/±Inf query is
    /// rejected here, before anything is ranked.
    pub fn rotated_query(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if let Some(index) = query.iter().position(|v| !v.is_finite()) {
            return Err(QuantError::NonFiniteValue { index });
        }
        let mut rotated = vec![0.0f32; self.dim];
        self.rotation.apply(query, &mut rotated)?;
        Ok(rotated)
    }

    /// [`Self::inner_product_estimate`] against a query already rotated by
    /// [`Self::rotated_query`]: O(d) per code. Bit-identical to the one-shot path.
    ///
    /// The cos/sin of every pair come from the table of this bit width, whose entries are
    /// the direct evaluation of the same dequantized angle (F04d): the f32 operations and
    /// their order are those of the direct loop, so the result is bit-identical.
    pub fn inner_product_estimate_rotated(
        &self,
        code: &PolarCode,
        rotated_query: &[f32],
    ) -> Result<f32> {
        if rotated_query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: rotated_query.len(),
            });
        }
        self.validate_code(code)?;
        let pairs = self.dim / 2;
        let trig = trig_table(self.bits);
        let mut estimate = 0.0f32;
        for i in 0..pairs {
            let r = code.radii[i];
            let q_a = rotated_query[2 * i];
            let q_b = rotated_query[2 * i + 1];
            let index = code.angle_indices[i];
            let [cos, sin] = trig
                .and_then(|table| table.get(index as usize).copied())
                .unwrap_or_else(|| {
                    // An index outside [0, 2^bits) (a malformed code is not rejected
                    // before scoring) or a bit width without a table: the direct value,
                    // never a new panic.
                    let theta = PolarCode::dequantize_index(index, self.bits);
                    [theta.cos(), theta.sin()]
                });
            estimate += r * (q_a * cos + q_b * sin);
        }
        Ok(estimate)
    }

    /// Heap bytes of the cached d×d rotation matrix: `dim * dim * 4`, held once per
    /// quantizer and shared by every code and query.
    pub fn rotation_memory_bytes(&self) -> usize {
        self.rotation.memory_bytes()
    }

    /// Heap bytes of the cos/sin table this quantizer scores through
    /// ([`PolarCode::trig_table_bytes`] at `bits`): process-wide, one per bit width, built
    /// on the first score and shared by every quantizer and store of that bit width.
    pub fn trig_table_bytes(&self) -> usize {
        PolarCode::trig_table_bytes(self.bits)
    }

    fn validate_code(&self, code: &PolarCode) -> Result<()> {
        if code.dim != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: code.dim,
            });
        }
        if code.bits != self.bits {
            return Err(QuantError::MalformedCode(format!(
                "code bits={}, quantizer bits={}",
                code.bits, self.bits
            )));
        }
        Ok(())
    }
}

/// `[cos, sin]` of every dequantized angle of one bit width.
type TrigTable = Vec<[f32; 2]>;

/// One cos/sin table per bit width for the whole process (F04d). Entry `i` is
/// `[cos(theta_i), sin(theta_i)]` with `theta_i = PolarCode::dequantize_index(i, bits)`: the
/// dequantized angle depends only on the index and the bit width, so the cos/sin the score
/// used to evaluate once per (code, pair, query) are evaluated once per bit width.
///
/// Bit-identical by construction: an entry is `f32::cos` / `f32::sin` (the same libm calls)
/// of the same f32 angle the direct path consumed. The table is built at runtime — never
/// const-evaluated, approximated or reordered — and never serialized: it is derived state,
/// like the rotation matrix. Thread-safe and deterministic through [`OnceLock`]: a lock-free
/// read once initialized, no lock in the scoring loop.
///
/// Slots 0..=16, one per polar bit width (spelled out because a repeated element in a
/// static requires `Copy`; slot 0 is never used by a valid quantizer).
static TRIG_TABLES: [OnceLock<TrigTable>; 17] = [
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
    OnceLock::new(),
];

/// The cos/sin table of `bits`, built on first use; `None` above the highest bit width a
/// quantizer accepts, whose (foreign) codes keep the direct evaluation.
fn trig_table(bits: u8) -> Option<&'static TrigTable> {
    TRIG_TABLES
        .get(bits as usize)
        .map(|slot| slot.get_or_init(|| build_trig_table(bits)))
}

fn build_trig_table(bits: u8) -> TrigTable {
    (0..1u32 << bits)
        .map(|index| {
            let theta = PolarCode::dequantize_index(index as u16, bits);
            [theta.cos(), theta.sin()]
        })
        .collect()
}

fn encode_pair(a: f32, b: f32, bits: u8) -> (f32, u16) {
    let r = (a * a + b * b).sqrt();
    let theta = b.atan2(a);
    let levels = 1u32 << bits;
    let normalized = (theta + PI) / (2.0 * PI);
    let idx = (normalized * levels as f32).floor() as u32 % levels;
    (r, idx as u16)
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
    fn encode_decode_roundtrip() {
        let q = PolarQuantizer::new(8, 16, 42).unwrap();
        let x = vec![1.0f32, 2.0, -1.5, 0.5, 3.0, -2.0, 0.1, -0.8];
        let code = q.encode(&x).unwrap();
        let decoded = q.decode(&code).unwrap();
        for (orig, dec) in x.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={orig:.4}, decoded={dec:.4}"
            );
        }
    }

    #[test]
    fn inner_product_accurate_at_high_bits() {
        let q = PolarQuantizer::new(16, 16, 7).unwrap();
        let x = random_vector(16, 1);
        let y = random_vector(16, 2);
        let code = q.encode(&x).unwrap();
        let estimated = q.inner_product_estimate(&code, &y).unwrap();
        let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rel_error = (estimated - exact).abs() / (exact.abs() + 1e-6);
        assert!(
            rel_error < 0.02,
            "rel_error={rel_error:.4}, est={estimated:.4}, exact={exact:.4}"
        );
    }

    #[test]
    fn deterministic() {
        let q = PolarQuantizer::new(8, 8, 0).unwrap();
        let x = vec![1.0f32; 8];
        let c1 = q.encode(&x).unwrap();
        let c2 = q.encode(&x).unwrap();
        assert_eq!(c1.angle_indices, c2.angle_indices);
        assert_eq!(c1.radii, c2.radii);
    }

    #[test]
    fn odd_dim_rejected() {
        assert!(PolarQuantizer::new(7, 8, 0).is_err());
    }

    #[test]
    fn trig_table_is_bit_identical_to_direct_cos_sin_for_every_bit_width() {
        for bits in 1..=16u8 {
            let table = trig_table(bits).unwrap();
            assert_eq!(table.len(), 1usize << bits);
            assert_eq!(
                PolarCode::trig_table_bytes(bits),
                table.len() * std::mem::size_of::<[f32; 2]>()
            );
            for (index, &[cos, sin]) in table.iter().enumerate() {
                let theta = PolarCode::dequantize_index(index as u16, bits);
                assert_eq!(
                    cos.to_bits(),
                    theta.cos().to_bits(),
                    "cos bits={bits} index={index}"
                );
                assert_eq!(
                    sin.to_bits(),
                    theta.sin().to_bits(),
                    "sin bits={bits} index={index}"
                );
            }
        }
        // No table above the highest bit width a quantizer accepts (slot 0 exists but no
        // valid quantizer has 0 bits).
        assert!(trig_table(17).is_none() && trig_table(255).is_none());
        // One table per bit width for the whole process, never per quantizer.
        assert!(std::ptr::eq(trig_table(4).unwrap(), trig_table(4).unwrap()));
        assert!(!std::ptr::eq(
            trig_table(3).unwrap(),
            trig_table(4).unwrap()
        ));
        assert_eq!(PolarCode::trig_table_bytes(3), 64);
        assert_eq!(PolarCode::trig_table_bytes(7), 1024);
        assert_eq!(PolarCode::trig_table_bytes(16), 512 * 1024);
    }

    #[test]
    fn dequantize_index_is_the_raw_expression_and_feeds_dequantize_angle() {
        for bits in [1u8, 3, 8, 16] {
            for index in [0u16, 1, 3, 7, 100, u16::MAX] {
                let levels = 1u32 << bits;
                let raw = (index as f32 / levels as f32) * (2.0 * PI) - PI;
                assert_eq!(
                    PolarCode::dequantize_index(index, bits).to_bits(),
                    raw.to_bits()
                );
            }
        }
        let quantizer = PolarQuantizer::new(8, 3, 1).unwrap();
        let code = quantizer
            .encode(&[0.3f32, -1.2, 2.5, 0.7, -0.1, 0.4, 1.0, 1.0])
            .unwrap();
        assert_eq!(quantizer.trig_table_bytes(), 64);
        for i in 0..code.pair_count() {
            assert_eq!(
                code.dequantize_angle(i).to_bits(),
                PolarCode::dequantize_index(code.angle_indices[i], code.bits).to_bits()
            );
        }
    }
}
