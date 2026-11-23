//! PolarQuant: vector compression via polar coordinate encoding.
//!
//! Converts Cartesian pairs into polar form (radius, angle), then uniformly
//! quantizes angles. After rotation, angles are uniformly distributed on [-pi, pi],
//! making uniform quantization near-optimal with no calibration needed.
//!
//! Algorithm: Given rotated vector y = R·x, group into d/2 pairs,
//! convert to polar (r, theta), quantize theta to `bits` levels.
//!
//! # Storage layout (F04c): what in a code depends on `bits`
//!
//! * `radii` — `dim / 2` f32 values. Exact and NOT quantized: every one of their bits
//!   reaches the score (`r * (...)`), so no lossless packing can shrink them. Stored as
//!   f32 in both layouts.
//! * `angle_indices` — `dim / 2` integers in `[0, 2^bits)`. The only field `bits`
//!   governs. [`PolarCode`] keeps one `u16` each (public API and serde wire format,
//!   unchanged); [`PackedPolarCode`] keeps `bits` bits each in one LSB-first stream.
//!
//! The mapping is a bijection: `pack` rejects an index outside `[0, 2^bits)` instead of
//! truncating it, `unpack_packed` returns the original code, and both scoring paths
//! dequantize through [`PolarCode::dequantize_index`] — a packed code reproduces the
//! f32 score bits of the unpacked one.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::error::{QuantError, Result};
use super::rotation::StoredRotation;

/// A compressed vector in polar form (unpacked: one `u16` per angle index — the public
/// API and serde representation; the store holds the [`PackedPolarCode`] layout).
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

    /// Quantized angle (radians) of pair `i`: level `angle_indices[i]` of `2^bits`.
    pub fn dequantize_angle(&self, i: usize) -> f32 {
        Self::dequantize_index(self.angle_indices[i] as u32, self.bits)
    }

    /// Angle of quantization level `index` at `bits` resolution. The only place the
    /// index -> angle arithmetic is written: the unpacked and the packed scoring paths
    /// both call it, so they dequantize to the same f32.
    pub fn dequantize_index(index: u32, bits: u8) -> f32 {
        let levels = 1u32 << bits;
        (index as f32 / levels as f32) * (2.0 * PI) - PI
    }

    /// Payload bytes of one code in the unpacked representation: an f32 radius and a
    /// u16 angle index per pair (6 B per pair) — independent of `bits`. The
    /// [`PackedPolarCode`] storage layout costs [`Self::packed_layout_bytes`].
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

    /// Payload bytes of one code in the packed storage layout ([`PackedPolarCode`]).
    pub const fn packed_layout_bytes(dim: usize, bits: u8) -> usize {
        PackedPolarCode::layout_bytes(dim, bits)
    }
}

/// Bytes of a bit stream that holds `count` values of `width` bits each.
///
/// Stream layout (the packed storage format): value `i` occupies the bits
/// `[i * width, (i + 1) * width)`, counted from the least significant bit of byte
/// `i * width / 8` onwards; the last byte is zero-padded. Defined on bytes, so the
/// format does not depend on the host endianness.
pub const fn bit_stream_bytes(count: usize, width: u8) -> usize {
    (count * width as usize).div_ceil(8)
}

/// The `width`-bit (1..=16) value at bit offset `pos` of a
/// [bit stream](bit_stream_bytes).
///
/// Never reads outside `data`: the window is zero-filled past the end, and callers only
/// ask for values the stream holds (`pos + width <= data.len() * 8`).
#[inline]
pub fn read_bits(data: &[u8], pos: usize, width: u8) -> u32 {
    debug_assert!(
        (1..=16).contains(&width),
        "bit width must be 1..=16, got {width}"
    );
    debug_assert!(
        pos + width as usize <= data.len() * 8,
        "read of {width} bits at {pos} past the end of a {}-byte stream",
        data.len()
    );
    let byte = pos >> 3;
    if byte >= data.len() {
        return 0;
    }
    let word = if byte + 8 <= data.len() {
        u64::from_le_bytes(data[byte..byte + 8].try_into().expect("8 bytes"))
    } else {
        let mut tail = [0u8; 8];
        tail[..data.len() - byte].copy_from_slice(&data[byte..]);
        u64::from_le_bytes(tail)
    };
    ((word >> (pos & 7)) & ((1u64 << width) - 1)) as u32
}

/// Write `value` (< `2^width`) at bit offset `pos`; the other bits are kept.
#[inline]
pub fn write_bits(data: &mut [u8], pos: usize, width: u8, value: u32) {
    debug_assert!(
        (1..=16).contains(&width),
        "bit width must be 1..=16, got {width}"
    );
    debug_assert!(
        (value as u64) < (1u64 << width),
        "{value} does not fit in {width} bits"
    );
    debug_assert!(
        pos + width as usize <= data.len() * 8,
        "write of {width} bits at {pos} past the end of a {}-byte stream",
        data.len()
    );
    let byte = pos >> 3;
    if byte >= data.len() {
        return;
    }
    let end = (byte + 8).min(data.len());
    let mut window = [0u8; 8];
    window[..end - byte].copy_from_slice(&data[byte..end]);
    let shift = pos & 7;
    let mask = ((1u64 << width) - 1) << shift;
    let word = (u64::from_le_bytes(window) & !mask) | (((value as u64) << shift) & mask);
    data[byte..end].copy_from_slice(&word.to_le_bytes()[..end - byte]);
}

/// Pack `values` (< `2^width` each) into a new bit stream. A value that does not fit
/// is an error, never a silent truncation.
pub fn pack_values(values: &[u16], width: u8) -> Result<Vec<u8>> {
    if !(1..=16).contains(&width) {
        return Err(QuantError::InvalidBitWidth { got: width });
    }
    let limit = 1u32 << width;
    if let Some(index) = values.iter().position(|&v| u32::from(v) >= limit) {
        return Err(QuantError::MalformedCode(format!(
            "angle index {} at pair {index} is out of range for {width} bits",
            values[index]
        )));
    }
    let mut data = vec![0u8; bit_stream_bytes(values.len(), width)];
    for (i, &value) in values.iter().enumerate() {
        write_bits(&mut data, i * width as usize, width, u32::from(value));
    }
    Ok(data)
}

/// The first `count` `width`-bit values of a bit stream (padding bits are never read).
pub fn unpack_values(data: &[u8], width: u8, count: usize) -> Vec<u16> {
    (0..count)
        .map(|i| read_bits(data, i * width as usize, width) as u16)
        .collect()
}

/// A [`PolarCode`] in the packed storage layout: the same f32 radii plus the angle
/// indices as `bits` bits each in one bit stream. Produced by
/// [`PolarQuantizer::encode_packed`] / [`PolarQuantizer::pack`] and read back with
/// [`PolarQuantizer::unpack_packed`]; the store holds this layout, never both.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedPolarCode {
    /// Angle resolution of the stream (the quantizer's `bits`).
    bits: u8,
    /// `dim / 2` radii — exactly the f32 values of the unpacked code.
    radii: Vec<f32>,
    /// [`bit_stream_bytes(dim / 2, bits)`](bit_stream_bytes) bytes of angle indices.
    angles: Vec<u8>,
}

impl PackedPolarCode {
    /// Payload bytes of one packed code: radii + angle stream.
    pub const fn layout_bytes(dim: usize, bits: u8) -> usize {
        dim / 2 * std::mem::size_of::<f32>() + bit_stream_bytes(dim / 2, bits)
    }

    pub(crate) fn new(bits: u8, radii: Vec<f32>, angles: Vec<u8>) -> Result<Self> {
        if angles.len() != bit_stream_bytes(radii.len(), bits) {
            return Err(QuantError::MalformedCode(format!(
                "packed polar code has {} angle bytes for {} pairs at {bits} bits",
                angles.len(),
                radii.len()
            )));
        }
        Ok(Self {
            bits,
            radii,
            angles,
        })
    }

    pub(crate) fn into_parts(self) -> (Vec<f32>, Vec<u8>) {
        (self.radii, self.angles)
    }

    /// Number of (radius, angle) pairs, i.e. `dim / 2`.
    pub fn pairs(&self) -> usize {
        self.radii.len()
    }

    /// Angle resolution of the stream.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The f32 radius of pair `i` (identical to `PolarCode::radii[i]`).
    pub fn radius(&self, i: usize) -> f32 {
        self.radii[i]
    }

    /// The quantized angle index of pair `i` in `[0, 2^bits)`.
    pub fn angle_index(&self, i: usize) -> u32 {
        read_bits(&self.angles, i * self.bits as usize, self.bits)
    }

    /// Heap bytes held: the radii and the angle stream.
    pub fn heap_bytes(&self) -> usize {
        self.radii.len() * std::mem::size_of::<f32>() + self.angles.len()
    }

    /// Payload bytes (the whole code is heap; nothing inline).
    pub fn encoded_bytes(&self) -> usize {
        self.heap_bytes()
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
        let mut estimate = 0.0f32;
        for i in 0..pairs {
            let theta = code.dequantize_angle(i);
            let r = code.radii[i];
            let q_a = rotated_query[2 * i];
            let q_b = rotated_query[2 * i + 1];
            estimate += r * (q_a * theta.cos() + q_b * theta.sin());
        }
        Ok(estimate)
    }

    /// Heap bytes of the cached d×d rotation matrix: `dim * dim * 4`, held once per
    /// quantizer and shared by every code and query.
    pub fn rotation_memory_bytes(&self) -> usize {
        self.rotation.memory_bytes()
    }

    /// Payload bytes of one code in the packed storage layout.
    pub fn code_layout_bytes(&self) -> usize {
        PackedPolarCode::layout_bytes(self.dim, self.bits)
    }

    /// Encode into the packed storage layout. Same quantization as [`Self::encode`] (it
    /// is that code, packed); the unpacked form is dropped when this returns.
    pub fn encode_packed(&self, vector: &[f32]) -> Result<PackedPolarCode> {
        self.pack(&self.encode(vector)?)
    }

    /// Pack an unpacked code of this quantizer — a bijection. The code must belong to
    /// this quantizer (`dim`, `bits`), hold finite radii and angle indices inside
    /// `[0, 2^bits)`; anything else is an error, never a truncated index.
    pub fn pack(&self, code: &PolarCode) -> Result<PackedPolarCode> {
        self.validate_code(code)?;
        let pairs = self.dim / 2;
        if code.radii.len() != pairs || code.angle_indices.len() != pairs {
            return Err(QuantError::MalformedCode(format!(
                "polar code has {} radii and {} angle indices, expected {pairs} each",
                code.radii.len(),
                code.angle_indices.len()
            )));
        }
        if let Some(i) = code.radii.iter().position(|r| !r.is_finite()) {
            return Err(QuantError::MalformedCode(format!(
                "polar code radius at pair {i} is not finite"
            )));
        }
        Ok(PackedPolarCode {
            bits: self.bits,
            radii: code.radii.clone(),
            angles: pack_values(&code.angle_indices, self.bits)?,
        })
    }

    /// Exact inverse of [`Self::pack`]: the original [`PolarCode`].
    pub fn unpack_packed(&self, code: &PackedPolarCode) -> Result<PolarCode> {
        self.validate_packed(code)?;
        Ok(PolarCode {
            dim: self.dim,
            bits: self.bits,
            radii: code.radii.clone(),
            angle_indices: unpack_values(&code.angles, self.bits, code.pairs()),
        })
    }

    /// [`Self::inner_product_estimate_rotated`] for a packed code. Bit-identical to
    /// the unpacked path: same dequantization, same accumulation order.
    pub fn inner_product_estimate_packed(
        &self,
        code: &PackedPolarCode,
        rotated_query: &[f32],
    ) -> Result<f32> {
        self.validate_packed(code)?;
        self.inner_product_from_stream(code.bits, &code.radii, &code.angles, rotated_query)
    }

    /// Same, from the bare parts of a packed code (the packed TurboQuant code keeps
    /// its radii and its angle stream inside one buffer). No allocation.
    pub(crate) fn inner_product_from_stream(
        &self,
        bits: u8,
        radii: &[f32],
        angles: &[u8],
        rotated_query: &[f32],
    ) -> Result<f32> {
        if bits != self.bits {
            return Err(QuantError::MalformedCode(format!(
                "code bits={bits}, quantizer bits={}",
                self.bits
            )));
        }
        let pairs = self.dim / 2;
        if radii.len() != pairs || angles.len() != bit_stream_bytes(pairs, bits) {
            return Err(QuantError::MalformedCode(format!(
                "packed polar code has {} radii and {} angle bytes, expected {pairs} and {}",
                radii.len(),
                angles.len(),
                bit_stream_bytes(pairs, bits)
            )));
        }
        if rotated_query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: rotated_query.len(),
            });
        }
        let width = bits as usize;
        let mut estimate = 0.0f32;
        for i in 0..pairs {
            let theta = PolarCode::dequantize_index(read_bits(angles, i * width, bits), bits);
            let r = radii[i];
            let q_a = rotated_query[2 * i];
            let q_b = rotated_query[2 * i + 1];
            estimate += r * (q_a * theta.cos() + q_b * theta.sin());
        }
        Ok(estimate)
    }

    /// [`Self::decode`] for a packed code: bit-identical reconstruction.
    pub fn decode_packed(&self, code: &PackedPolarCode) -> Result<Vec<f32>> {
        self.decode(&self.unpack_packed(code)?)
    }

    /// A packed code must come from a quantizer with the same `dim` and `bits`, and
    /// hold exactly the stream that configuration implies.
    fn validate_packed(&self, code: &PackedPolarCode) -> Result<()> {
        if code.bits != self.bits {
            return Err(QuantError::MalformedCode(format!(
                "code bits={}, quantizer bits={}",
                code.bits, self.bits
            )));
        }
        if code.pairs() * 2 != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: code.pairs() * 2,
            });
        }
        if code.angles.len() != bit_stream_bytes(code.pairs(), code.bits) {
            return Err(QuantError::MalformedCode(format!(
                "packed polar code has {} angle bytes, expected {}",
                code.angles.len(),
                bit_stream_bytes(code.pairs(), code.bits)
            )));
        }
        Ok(())
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
}
