//! F04c (A02-F04): lossless bit-packing of the stored codes.
//!
//! The store holds the packed layout (`PackedTurboCode` / `PackedPolarCode`: angle
//! indices at `bits` bits each, QJL signs at 1 bit, exact f32 radii and norm). Lossless
//! means encode -> packed -> decode/score reproduces ids, order and f32 score bits of
//! the unpacked representation, which stays the public API (`encode` + `score_projected`)
//! and the serde wire format and is used here as the reference. Edge cases: bits
//! {2, 4, 8}, dimensions that are not a multiple of a word, norms 0 / 1e4, atomic
//! `add_batch`, non-finite queries, legacy and corrupt serde payloads.
//!
//! Run: `cargo test --no-default-features --test quant_contracts packing`

use ndarray::s;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use splatsdb::quant::polar::{bit_stream_bytes, pack_values, unpack_values, PackedPolarCode};
use splatsdb::quant::qjl::{pack_signs, sign_stream_bytes, unpack_signs};
use splatsdb::quant::turbo::PackedTurboCode;
use splatsdb::quant::{
    PolarCode, PolarQuantizer, QjlQuantizer, QjlSketch, QuantError, TurboCode, TurboQuantizer,
};
use splatsdb::quantization::{QuantAlgorithm, QuantConfig, QuantizedStore, ScoreMode, SearchHit};

const MODES: [ScoreMode; 2] = [ScoreMode::L2Heuristic, ScoreMode::InnerProduct];

fn gaussian(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len).map(|_| StandardNormal.sample(&mut rng)).collect()
}

/// Same generators as the F04b datasets: uniform [0,1) or centred gaussian.
fn matrix(rows: usize, cols: usize, seed: u64, uniform: bool) -> ndarray::Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    ndarray::Array2::from_shape_fn((rows, cols), |(_, _)| {
        if uniform {
            rng.gen::<f32>()
        } else {
            StandardNormal.sample(&mut rng)
        }
    })
}

/// A vector of norm `norm` (direction from the seeded Gaussian).
fn with_norm(dim: usize, norm: f32, seed: u64) -> Vec<f32> {
    let x = gaussian(dim, seed);
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    x.iter().map(|v| v / n * norm).collect()
}

/// `(id, f32 score bits)` of a full ranking, sorted exactly like the store sorts it.
fn sorted_bits(mut scored: Vec<(usize, f32)>) -> Vec<(u64, u32)> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .map(|(i, score)| (i as u64, score.to_bits()))
        .collect()
}

fn hits_bits(hits: &[SearchHit]) -> Vec<(u64, u32)> {
    hits.iter()
        .map(|hit| (hit.id, hit.score.to_bits()))
        .collect()
}

// ---------------------------------------------------------------------------
// The bit stream itself.
// ---------------------------------------------------------------------------

#[test]
fn bit_stream_roundtrips_every_width_at_word_and_non_word_counts() {
    // 1, 7, 31, 63, 64, 639, 640 values: every edge around the 64-bit read window.
    let counts = [1usize, 7, 31, 63, 64, 639, 640, 1000];
    for width in 1u8..=16 {
        let limit = 1u64 << width;
        for &count in &counts {
            let values: Vec<u16> = (0..count)
                .map(|i| ((i as u64 * 2_654_435_761 + u64::from(width)) % limit) as u16)
                .collect();
            let packed = pack_values(&values, width).unwrap();
            assert_eq!(
                packed.len(),
                bit_stream_bytes(count, width),
                "w={width} n={count}"
            );
            let used = count * usize::from(width);
            if used % 8 != 0 {
                let last = packed[packed.len() - 1];
                assert_eq!(
                    u32::from(last) >> (used % 8),
                    0,
                    "padding bits must stay zero (w={width} n={count})"
                );
            }
            assert_eq!(
                unpack_values(&packed, width, count),
                values,
                "w={width} n={count}"
            );
        }
        for pattern in 0..3 {
            let values: Vec<u16> = (0..65)
                .map(|i| match pattern {
                    0 => 0,
                    1 => (limit - 1) as u16,
                    _ if i % 2 == 0 => (limit - 1) as u16,
                    _ => 0,
                })
                .collect();
            let packed = pack_values(&values, width).unwrap();
            assert_eq!(
                unpack_values(&packed, width, 65),
                values,
                "w={width} pattern={pattern}"
            );
        }
    }
    // A value that does not fit is rejected, never truncated.
    assert!(matches!(
        pack_values(&[8, 0], 3),
        Err(QuantError::MalformedCode(_))
    ));
    assert!(matches!(
        pack_values(&[0], 0),
        Err(QuantError::InvalidBitWidth { got: 0 })
    ));
    assert!(matches!(
        pack_values(&[0], 17),
        Err(QuantError::InvalidBitWidth { got: 17 })
    ));
    assert_eq!(pack_values(&[], 3).unwrap(), Vec::<u8>::new());
    assert_eq!(unpack_values(&[], 3, 0), Vec::<u16>::new());
}

// ---------------------------------------------------------------------------
// PolarQuant: pack -> unpack is the identity, scores and decode are bit-identical.
// ---------------------------------------------------------------------------

#[test]
fn packed_polar_code_is_lossless_for_every_bit_width_and_word_edge_dim() {
    // pairs = 1, 3, 7, 31, 32, 63, 64, 319, 320: value counts around every word edge.
    // (Odd dims are rejected by the algorithm itself, see the test below.)
    let dims = [2usize, 6, 14, 62, 64, 126, 128, 638, 640];
    for &dim in &dims {
        let widths: &[u8] = if dim >= 128 {
            &[1, 3, 7, 16]
        } else {
            &[1, 2, 3, 4, 7, 8, 16]
        };
        for &bits in widths {
            let quantizer = PolarQuantizer::new(dim, bits, 7).unwrap();
            assert_eq!(
                quantizer.code_layout_bytes(),
                PackedPolarCode::layout_bytes(dim, bits)
            );
            for seed in 0..3u64 {
                let x = match seed {
                    0 => vec![0.0; dim],
                    1 => with_norm(dim, 1e4, 2),
                    _ => gaussian(dim, seed),
                };
                let code = quantizer.encode(&x).unwrap();
                let packed = quantizer.encode_packed(&x).unwrap();
                assert_eq!(packed, quantizer.pack(&code).unwrap(), "dim={dim} b={bits}");
                assert_eq!(quantizer.unpack_packed(&packed).unwrap(), code);
                assert_eq!(
                    packed.encoded_bytes(),
                    dim / 2 * 4 + bit_stream_bytes(dim / 2, bits)
                );
                let rotated = quantizer.rotated_query(&gaussian(dim, 100 + seed)).unwrap();
                let unpacked = quantizer
                    .inner_product_estimate_rotated(&code, &rotated)
                    .unwrap();
                let via_packed = quantizer
                    .inner_product_estimate_packed(&packed, &rotated)
                    .unwrap();
                assert_eq!(
                    unpacked.to_bits(),
                    via_packed.to_bits(),
                    "dim={dim} b={bits}"
                );
                assert!(via_packed.is_finite());
                let decoded = quantizer.decode(&code).unwrap();
                let decoded_packed = quantizer.decode_packed(&packed).unwrap();
                for (a, b) in decoded.iter().zip(decoded_packed.iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "decode dim={dim} b={bits}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TurboQuant: pack -> unpack is the identity, both score modes are bit-identical.
// ---------------------------------------------------------------------------

#[test]
fn packed_turbo_code_is_lossless_in_both_modes_including_norm_edges() {
    let shapes = [
        (16usize, 2u8, 7usize),
        (16, 4, 1),
        (16, 16, 31),
        (64, 8, 64),
        (64, 3, 13),
        (128, 4, 256),
    ];
    for (dim, bits, projections) in shapes {
        let quantizer = TurboQuantizer::new(dim, bits, projections, 42).unwrap();
        let queries = [gaussian(dim, 90), gaussian(dim, 91), vec![0.0; dim]];
        for seed in 0..24u64 {
            let x = match seed % 4 {
                0 => vec![0.0; dim],
                1 => with_norm(dim, 1e4, seed),
                2 => with_norm(dim, 1e-4, seed),
                _ => gaussian(dim, seed),
            };
            let code = quantizer.encode(&x).unwrap();
            let packed = quantizer.encode_packed(&x).unwrap();
            assert_eq!(quantizer.pack(&code).unwrap(), packed);
            assert_eq!(quantizer.unpack_packed(&packed).unwrap(), code);
            assert_eq!(packed.encoded_bytes(), quantizer.code_layout_bytes());
            assert_eq!(
                packed.encoded_bytes(),
                PackedTurboCode::layout_bytes(dim, bits, projections)
            );
            for (i, query) in queries.iter().enumerate() {
                let context = quantizer.prepare_query(query).unwrap();
                for mode in MODES {
                    let reference = quantizer.score(&code, query, mode).unwrap();
                    let via_packed = quantizer.score_packed(&packed, &context, mode).unwrap();
                    assert_eq!(reference.to_bits(), via_packed.to_bits());
                    assert!(
                        via_packed.is_finite(),
                        "dim={dim} bits={bits} m={projections} seed={seed} q={i} {mode:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn odd_dims_are_rejected_by_the_algorithm_not_by_the_packing_and_qjl_packs_any_count() {
    // PolarQuant pairs the dimensions: odd dims were rejected before the packing
    // existed and still are — by the quantizer, not by the bit stream.
    for dim in [1usize, 7, 31, 63, 639] {
        assert!(matches!(
            PolarQuantizer::new(dim, 4, 0),
            Err(QuantError::OddDimension { got }) if got == dim
        ));
        assert!(matches!(
            TurboQuantizer::new(dim, 4, 8, 0),
            Err(QuantError::OddDimension { got }) if got == dim
        ));
    }
    // The QJL stage has no pairing: every projection count packs and scores identically.
    let dim = 8;
    for projections in [1usize, 7, 31, 63, 64, 639, 640] {
        let quantizer = QjlQuantizer::new(dim, projections, 3).unwrap();
        let sketch = quantizer.sketch(&gaussian(dim, 4)).unwrap();
        let signs = pack_signs(&sketch.signs).unwrap();
        assert_eq!(signs.len(), sign_stream_bytes(projections));
        assert_eq!(sketch.encoded_bytes(), projections + 4);
        assert_eq!(
            QjlSketch::packed_layout_bytes(projections),
            sign_stream_bytes(projections) + 4
        );
        assert_eq!(unpack_signs(&signs, projections), sketch.signs);
        // The padding bits of the last byte are ignored whatever they hold (only the
        // bits above the `projections` sign bits are padding).
        let mut dirty = signs.clone();
        if projections % 8 != 0 {
            *dirty.last_mut().unwrap() |= 0xFFu8 << (projections % 8);
        }
        assert_eq!(unpack_signs(&dirty, projections), sketch.signs);
        let projected = quantizer.projected_query(&gaussian(dim, 5)).unwrap();
        for mode in MODES {
            assert_eq!(
                quantizer
                    .score_packed(&dirty, sketch.norm, &projected, mode)
                    .unwrap()
                    .to_bits(),
                quantizer
                    .score_projected(&sketch, &projected, mode)
                    .unwrap()
                    .to_bits()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The store (packed) against the unpacked API, on the F04b datasets.
// ---------------------------------------------------------------------------

#[test]
fn store_matches_the_unpacked_reference_on_the_f04b_datasets() {
    // Same families and seeds as F04b (uniform [0,1) and centred gaussian, d=640) at a
    // smaller n, plus two rows at the norm edges: the zero vector and a vector of norm
    // 1e4. TurboQuant 2/4/8-bit and PolarQuant 3-bit, both modes, FULL ranking.
    const DIM: usize = 640;
    const N: usize = 120;
    const N_QUERIES: usize = 8;
    let configs = [
        ("search_4bit", QuantConfig::for_search(DIM)),
        ("default_8bit", QuantConfig::default()),
        (
            "turbo_2bit_m256",
            QuantConfig {
                bits: 2,
                projections: 256,
                seed: 42,
                algorithm: QuantAlgorithm::TurboQuant,
            },
        ),
        ("polar_3bit", QuantConfig::for_max_compression(DIM)),
    ];
    // The reference quantizers: the unpacked API, built once per configuration.
    let turbo_refs: Vec<Option<TurboQuantizer>> = configs
        .iter()
        .map(|(_, config)| {
            (config.algorithm == QuantAlgorithm::TurboQuant).then(|| {
                TurboQuantizer::new(DIM, config.bits, config.projections, config.seed).unwrap()
            })
        })
        .collect();
    let polar_refs: Vec<Option<PolarQuantizer>> = configs
        .iter()
        .map(|(_, config)| {
            (config.algorithm == QuantAlgorithm::PolarQuant)
                .then(|| PolarQuantizer::new(DIM, config.bits, config.seed).unwrap())
        })
        .collect();

    let families = [
        ("uniform01", true, 1001, 2001),
        ("gaussian", false, 1001, 2001),
    ];
    for (family, uniform, data_seed, query_seed) in families {
        let mut data = ndarray::Array2::<f32>::zeros((N + 2, DIM));
        data.slice_mut(s![..N, ..])
            .assign(&matrix(N, DIM, data_seed, uniform));
        // Row N stays all zero (norm 0); row N + 1 has norm 1e4.
        data.slice_mut(s![N + 1, ..])
            .assign(&ndarray::Array1::from(with_norm(DIM, 1e4, 77)));
        let queries = matrix(N_QUERIES, DIM, query_seed, uniform);
        let rows: Vec<Vec<f32>> = data.outer_iter().map(|row| row.to_vec()).collect();

        for (((name, config), turbo_ref), polar_ref) in
            configs.iter().zip(turbo_refs.iter()).zip(polar_refs.iter())
        {
            let mut store = QuantizedStore::new(DIM, config.clone()).unwrap();
            store.add_batch(&data, 0).unwrap();
            assert!(store.is_bit_packed());
            assert_eq!(store.len(), N + 2);

            // Reference codes: the unpacked representation, one encode per row.
            let turbo_codes: Vec<TurboCode> = turbo_ref
                .as_ref()
                .map(|q| rows.iter().map(|row| q.encode(row).unwrap()).collect())
                .unwrap_or_default();
            let polar_codes: Vec<PolarCode> = polar_ref
                .as_ref()
                .map(|q| rows.iter().map(|row| q.encode(row).unwrap()).collect())
                .unwrap_or_default();

            for qi in 0..N_QUERIES {
                let query = queries.row(qi).to_vec();
                let context = turbo_ref.as_ref().map(|q| q.prepare_query(&query).unwrap());
                let rotated = polar_ref.as_ref().map(|q| q.rotated_query(&query).unwrap());
                for mode in MODES {
                    let scored: Vec<(usize, f32)> =
                        if let (Some(q), Some(ctx)) = (turbo_ref, context.as_ref()) {
                            turbo_codes
                                .iter()
                                .enumerate()
                                .map(|(i, code)| (i, q.score_projected(code, ctx, mode).unwrap()))
                                .collect()
                        } else {
                            let q = polar_ref.as_ref().unwrap();
                            let rotated = rotated.as_ref().unwrap();
                            polar_codes
                                .iter()
                                .enumerate()
                                .map(|(i, code)| {
                                    (i, q.inner_product_estimate_rotated(code, rotated).unwrap())
                                })
                                .collect()
                        };
                    let expected = sorted_bits(scored);
                    let got = store
                        .search_scored(&queries.row(qi), usize::MAX, mode)
                        .unwrap();
                    assert_eq!(got.len(), N + 2);
                    assert_eq!(
                        hits_bits(&got),
                        expected,
                        "{family}/{name} query {qi} {mode:?}: ids, order or score bits differ"
                    );
                }
            }
        }
    }
}

#[test]
fn add_batch_is_atomic_and_the_store_metrics_follow_the_packed_layout() {
    const DIM: usize = 64;
    const M: usize = 16;
    let config = QuantConfig {
        bits: 4,
        projections: M,
        seed: 42,
        algorithm: QuantAlgorithm::TurboQuant,
    };
    let mut store = QuantizedStore::new(DIM, config).unwrap();
    let query = ndarray::Array1::from(gaussian(DIM, 9));
    store.add_batch(&matrix(10, DIM, 30, true), 0).unwrap();
    let ranking = |store: &QuantizedStore| {
        hits_bits(
            &store
                .search_scored(&query.view(), usize::MAX, ScoreMode::L2Heuristic)
                .unwrap(),
        )
    };
    let metrics = |store: &QuantizedStore| {
        (
            store.len(),
            store.encoded_bytes(),
            store.memory_usage(),
            store.compression_ratio(),
            store.code_layout_bytes(),
        )
    };
    let before = ranking(&store);
    assert_eq!(before.len(), 10);
    let snapshot = metrics(&store);
    assert_eq!(snapshot.4, PackedTurboCode::layout_bytes(DIM, 4, M));
    assert_eq!(snapshot.1, 10 * snapshot.4);
    assert!(store.is_bit_packed());

    let mut nan = matrix(5, DIM, 31, true);
    nan[[3, 7]] = f32::NAN;
    let mut inf = matrix(5, DIM, 32, true);
    inf[[0, DIM - 1]] = f32::INFINITY;
    let mut huge = matrix(5, DIM, 34, true);
    huge[[2, 4]] = 1.0e30; // its squares overflow f32: no finite norm exists
    for bad in [matrix(5, DIM + 1, 33, true), nan, inf, huge] {
        assert!(store.add_batch(&bad, 10).is_err());
        assert_eq!(
            metrics(&store),
            snapshot,
            "a rejected batch must not touch the store"
        );
    }
    assert_eq!(ranking(&store), before);

    // A non-finite query never reaches the packed codes.
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = query.to_owned();
        bad[DIM / 2] = value;
        for mode in MODES {
            assert!(matches!(
                store.search_scored(&bad.view(), 3, mode),
                Err(QuantError::NonFiniteValue { index }) if index == DIM / 2
            ));
        }
        assert!(store.search(&bad.view(), 3).is_empty());
    }
}

#[test]
fn pack_rejects_corrupt_codes_and_packed_codes_of_other_quantizers() {
    let x = gaussian(16, 3);
    let quantizer = TurboQuantizer::new(16, 4, 8, 42).unwrap();
    let code = quantizer.encode(&x).unwrap();
    let packed = quantizer.encode_packed(&x).unwrap();

    // Angle index 8 needs 4 bits and this code has 3: rejected, never truncated.
    let mut bad = code.clone();
    bad.polar_code.angle_indices[3] = 8;
    assert!(matches!(
        quantizer.pack(&bad),
        Err(QuantError::MalformedCode(_))
    ));
    // A sign that is not +-1, a negative norm, a non-finite radius.
    let mut bad = code.clone();
    bad.residual_sketch.signs[2] = 0;
    assert!(quantizer.pack(&bad).is_err());
    let mut bad = code.clone();
    bad.residual_sketch.norm = -1.0;
    assert!(quantizer.pack(&bad).is_err());
    let mut bad = code.clone();
    bad.polar_code.radii[1] = f32::NAN;
    assert!(quantizer.pack(&bad).is_err());
    // A code of another projection count.
    let foreign = TurboQuantizer::new(16, 4, 16, 42)
        .unwrap()
        .encode(&x)
        .unwrap();
    assert!(quantizer.pack(&foreign).is_err());

    // A packed code is only read by a quantizer with the same dim, bits and projections.
    let others = [
        TurboQuantizer::new(16, 5, 8, 42).unwrap(),
        TurboQuantizer::new(16, 4, 16, 42).unwrap(),
        TurboQuantizer::new(32, 4, 8, 42).unwrap(),
    ];
    for other in &others {
        let context = other.prepare_query(&gaussian(other.dim(), 5)).unwrap();
        assert!(other.unpack_packed(&packed).is_err());
        for mode in MODES {
            assert!(matches!(
                other.score_packed(&packed, &context, mode),
                Err(QuantError::MalformedCode(_)) | Err(QuantError::DimensionMismatch { .. })
            ));
        }
    }

    let polar = PolarQuantizer::new(16, 3, 1).unwrap();
    let polar_code = polar.encode(&x).unwrap();
    let packed_polar = polar.encode_packed(&x).unwrap();
    assert_eq!(polar.unpack_packed(&packed_polar).unwrap(), polar_code);
    let mut bad = polar_code.clone();
    bad.angle_indices[0] = 8;
    assert!(matches!(
        polar.pack(&bad),
        Err(QuantError::MalformedCode(_))
    ));
    let other = PolarQuantizer::new(16, 4, 1).unwrap();
    assert!(other.unpack_packed(&packed_polar).is_err());
    assert!(other
        .inner_product_estimate_packed(&packed_polar, &other.rotated_query(&x).unwrap())
        .is_err());

    // QJL: a sign stream of the wrong length, a NaN or a negative norm.
    let qjl = QjlQuantizer::new(16, 16, 3).unwrap();
    let sketch = qjl.sketch(&x).unwrap();
    let signs = pack_signs(&sketch.signs).unwrap();
    let projected = qjl.projected_query(&x).unwrap();
    assert!(qjl
        .score_packed(&signs[..1], 1.0, &projected, ScoreMode::InnerProduct)
        .is_err());
    assert!(qjl
        .score_packed(&signs, f32::NAN, &projected, ScoreMode::InnerProduct)
        .is_err());
    assert!(qjl
        .score_packed(&signs, -1.0, &projected, ScoreMode::InnerProduct)
        .is_err());
}
