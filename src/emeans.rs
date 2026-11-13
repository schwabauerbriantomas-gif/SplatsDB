//! e-means trainer — port de faiss contrib `e_means` (Meta FAIR,
//! arXiv:2611.09501), validado numéricamente contra la referencia Python
//! (receta batch/α₀/épocas idéntica; ver /mnt/d/emeans-port/results_*.json).
//!
//! Alternativa a Lloyd para el fit jerárquico HRM2: mini-batches por ley de
//! batch (Ec. 7), EMA por masa con step law (Ec. 6) y annealing coseno
//! (Ec. 5), reactivación de clusters muertos cada 5 updates (Apéndice J) y
//! finalize con medias no ponderadas del último pase. Init por filas
//! aleatorias (el paper: k-means++ secuencial es contraproducente en GPU y
//! no aporta en CPU a este presupuesto).
//!
//! CPU-only: HRM2 corre en el arranque del serve y su presupuesto (k≤64,
//! n≤10⁵) no justifica kernels dedicados; los kernels GPU del port quedan
//! en /mnt/d/emeans-port para el escenario IVF >50k.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const ALPHA_MIN_FACTOR: f64 = 0.1;
const CHECK_UNUSED_EVERY: usize = 5;
const UNUSED_THRESHOLD: f32 = 0.1;
const NO_REVIVAL_MAX_D: usize = 2;
const AUTO_QUALITY_MAX_D: usize = 16;
const SPLIT_EPSILON: f32 = 1e-4;
const INIT_USAGE: f32 = 1.0 / 65536.0;
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);
const BATCH_LAW_THROUGHPUT: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);
const BATCH_LAW_QUALITY: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);

/// Resultado del fit: centroides (k·d), labels (n), inercia post-hoc total
/// (f64, misma semántica que KMeans::inertia pero en doble precisión) y
/// pasos de update ejecutados.
pub struct EmeansFit {
    pub centroids: Vec<f32>,
    pub labels: Vec<usize>,
    pub inertia: f64,
    pub n_steps: usize,
}

fn round_power_of_two(v: f64) -> usize {
    if v <= 1.0 {
        return 1;
    }
    let e = v.log2().floor();
    let lower = 2f64.powi(e as i32);
    let upper = lower * 2.0;
    let out = if v - lower >= upper - v { upper } else { lower };
    if out.is_finite() && out >= 1.0 {
        out as usize
    } else {
        usize::MAX
    }
}

/// Batch law Ec. 7 con auto-resolución por dimensión (base.py:89).
fn resolve_batch(n: usize, k: usize, nepochs: f64, d: usize) -> usize {
    let law = if d <= AUTO_QUALITY_MAX_D {
        BATCH_LAW_QUALITY
    } else {
        BATCH_LAW_THROUGHPUT
    };
    let raw = law.0 * (k as f64).powf(law.1) * (n as f64).powf(law.2) * nepochs.powf(law.3);
    let mut batch = round_power_of_two(raw);
    if d > AUTO_QUALITY_MAX_D {
        batch = batch.min(round_power_of_two(n as f64 / 2.0));
    } else {
        batch = batch.min(n / 2);
    }
    batch.max(1).min(n)
}

/// Step law Ec. 6 con clamp [1e-4, 1].
fn step_law(n: usize, k: usize, nepochs: f64, batch: usize) -> f64 {
    let (c, a_k, a_b, a_n, a_t) = ALPHA_LAW;
    (c * (k as f64).powf(a_k) * (batch as f64).powf(a_b) * (n as f64).powf(a_n) * nepochs.powf(a_t))
        .clamp(1e-4, 1.0)
}

/// Annealing coseno hacia 0.1·α₀ (Ec. 5).
fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let frac = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cos = 0.5 * (1.0 + (std::f64::consts::PI * frac).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cos)
}

/// Épocas fraccionarias (base.py:156): T=2.6 → [n, n, 0.6n redondeado].
fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor() as usize;
    let rem = ((nepochs - whole as f64) * n as f64).round() as usize;
    if whole == 0 {
        return vec![rem.min(n)];
    }
    let mut rows = vec![n; whole.saturating_sub(1)];
    if rem > 0 {
        rows.push(rem.min(n));
    }
    rows.push(n);
    rows
}

/// Reactivación de clusters muertos: split donante→muerto ∝ masa (Apéndice J,
/// Ec. 8; signo alternante por coordenada con eps 1e-4).
fn revive_unused(
    centroids: &mut [f32],
    usage: &mut [f32],
    k: usize,
    d: usize,
    rng: &mut ChaCha8Rng,
) -> usize {
    if d <= NO_REVIVAL_MAX_D {
        return 0;
    }
    let total: f32 = usage.iter().sum();
    let thresh = total * UNUSED_THRESHOLD / k as f32;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < thresh).collect();
    let donors: Vec<usize> = (0..k)
        .filter(|&j| usage[j] >= thresh && usage[j] > 0.0)
        .collect();
    let nsplit = unused.len().min(donors.len());
    if nsplit == 0 {
        return 0;
    }
    let mut keys: Vec<(f64, usize)> = donors
        .iter()
        .map(|&j| {
            let u = rng.gen::<f64>().max(f64::MIN_POSITIVE);
            ((usage[j] as f64).ln() - (-u.ln()).ln(), j)
        })
        .collect();
    keys.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let sources: Vec<usize> = keys.into_iter().take(nsplit).map(|(_, j)| j).collect();
    let targets: Vec<usize> = unused.into_iter().take(nsplit).collect();
    for (si, &src) in sources.iter().enumerate() {
        let tgt = targets[si];
        let half = usage[src] / 2.0;
        for t in 0..d {
            let sign = if t % 2 == 0 { 1.0f32 } else { -1.0f32 };
            let v = centroids[src * d + t];
            centroids[src * d + t] = v * (1.0 - SPLIT_EPSILON * sign);
            centroids[tgt * d + t] = v * (1.0 + SPLIT_EPSILON * sign);
        }
        usage[src] = half;
        usage[tgt] = half;
    }
    nsplit
}

/// Fit e-means sobre datos planos row-major (n·d). `nepochs` = pases
/// (5.0 = punto operativo validado; 2.6 = paridad con Lloyd-25 del paper).
/// Determinista dado `seed`.
#[allow(clippy::too_many_arguments)]
pub fn fit_emeans(
    x: &[f32],
    n: usize,
    dim: usize,
    k_req: usize,
    nepochs: f64,
    seed: u64,
) -> EmeansFit {
    let k = k_req.min(n).max(1);
    let d = dim;
    let batch = resolve_batch(n, k, nepochs, d);
    let alpha0 = step_law(n, k, nepochs, batch);
    let epochs = epoch_rows(n, nepochs);
    let total_steps: usize = epochs.iter().map(|r| r.div_ceil(batch)).sum();

    // init: filas aleatorias (sin k-means++)
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..n).collect();
    if n != k {
        order.shuffle(&mut rng);
    }
    let mut centroids = vec![0f32; k * d];
    for (j, &src) in order.iter().take(k).enumerate() {
        centroids[j * d..(j + 1) * d].copy_from_slice(&x[src * d..(src + 1) * d]);
    }
    let mut usage = vec![INIT_USAGE; k];
    let mut split_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_A5A5);
    let mut step = 0usize;
    let mut n_splits = 0usize;

    for (epoch, &rows) in epochs.iter().enumerate() {
        let last = epoch == epochs.len() - 1;
        let ep_seed = seed.wrapping_add((epoch as u64) << 32);
        let mut ep_rng = ChaCha8Rng::seed_from_u64(ep_seed);
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut ep_rng);
        let mut begin = 0usize;
        while begin < rows {
            let end = (begin + batch).min(rows);
            let idx = &perm[begin..end];

            // assignment serial (k pequeño en HRM2; determinista)
            let mut sums = vec![0f32; k * d];
            let mut counts = vec![0u32; k];
            for &i in idx {
                let row = &x[i * d..(i + 1) * d];
                let mut best = 0usize;
                let mut bd = f32::INFINITY;
                for j in 0..k {
                    let mut acc = 0f32;
                    let cent = &centroids[j * d..(j + 1) * d];
                    for t in 0..d {
                        let v = row[t] - cent[t];
                        acc += v * v;
                    }
                    if acc < bd {
                        bd = acc;
                        best = j;
                    }
                }
                counts[best] += 1;
                let s = &mut sums[best * d..(best + 1) * d];
                for t in 0..d {
                    s[t] += row[t];
                }
            }

            if last {
                // último pase: medias no ponderadas (finalize, base.py)
                for (j, &cnt) in counts.iter().enumerate() {
                    if cnt > 0 {
                        let w = 1.0 / cnt as f32;
                        for t in 0..d {
                            centroids[j * d + t] = sums[j * d + t] * w;
                        }
                        usage[j] = usage[j] * (1.0 - alpha0 as f32) + alpha0 as f32;
                    }
                }
            } else {
                let alpha = scheduled_alpha(alpha0, step, total_steps) as f32;
                for (j, &cnt) in counts.iter().enumerate() {
                    if cnt == 0 {
                        continue;
                    }
                    let w = alpha / cnt as f32;
                    for t in 0..d {
                        centroids[j * d + t] += w * (sums[j * d + t] - centroids[j * d + t]);
                    }
                    usage[j] = usage[j] * (1.0 - alpha) + alpha;
                }
            }

            step += 1;
            if step.is_multiple_of(CHECK_UNUSED_EVERY) {
                n_splits += revive_unused(&mut centroids, &mut usage, k, d, &mut split_rng);
            }
            begin = end;
        }
    }
    let _ = n_splits;

    // labels finales: nearest centroid (semántica Lloyd: assignment del
    // estado final) + inercia post-hoc
    let mut labels = vec![0usize; n];
    let mut inertia = 0f64;
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for j in 0..k {
            let cent = &centroids[j * d..(j + 1) * d];
            let mut acc = 0f32;
            for t in 0..d {
                let v = row[t] - cent[t];
                acc += v * v;
            }
            if acc < bd {
                bd = acc;
                best = j;
            }
        }
        labels[i] = best;
        inertia += bd as f64;
    }
    EmeansFit {
        centroids,
        labels,
        inertia,
        n_steps: step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture estructurado: 4 blobs gaussianos en R^d (como embeddings reales).
    fn blobs_fixture(n_per: usize, d: usize, seed: u64) -> Vec<f32> {
        use rand_distr::{Distribution, Normal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 0.15).unwrap();
        let mut x = Vec::with_capacity(4 * n_per * d);
        let mut centers = vec![vec![0f32; d]; 4];
        for (c, cent) in centers.iter_mut().enumerate() {
            for t in 0..d {
                cent[t] = if t == c * 7 % d {
                    2.0
                } else {
                    0.1 * (c as f32 + 1.0)
                };
            }
        }
        for c in 0..4usize {
            for _ in 0..n_per {
                for t in 0..d {
                    x.push(centers[c][t] + normal.sample(&mut rng));
                }
            }
        }
        x
    }

    #[test]
    fn deterministic_same_seed() {
        let x = blobs_fixture(50, 32, 7);
        let a = fit_emeans(&x, 200, 32, 4, 5.0, 99);
        let b = fit_emeans(&x, 200, 32, 4, 5.0, 99);
        assert_eq!(a.centroids, b.centroids);
        assert_eq!(a.labels, b.labels);
        assert!((a.inertia - b.inertia).abs() < 1e-9);
    }

    #[test]
    fn labels_in_range() {
        let x = blobs_fixture(50, 32, 7);
        let fit = fit_emeans(&x, 200, 32, 4, 5.0, 11);
        assert!(fit.labels.iter().all(|&l| l < 4));
        assert_eq!(fit.centroids.len(), 4 * 32);
    }

    #[test]
    fn batch_and_step_laws_match_reference() {
        // Valores verificados 1:1 contra base.py real (13-nov-2026):
        // _resolve_recipe(248, 31, 5.0, 'throughput', None, None)
        // → batch=32, alpha0=0.270245
        assert_eq!(resolve_batch(248, 31, 5.0, 768), 32);
        let a = step_law(248, 31, 5.0, 32);
        assert!((a - 0.270245).abs() < 1e-6, "alpha0={a}");
        // clamp inferior del step law (n enorme fuerza el clamp)
        assert_eq!(step_law(1_000_000, 100_000, 1.0, 1), 1e-4);
    }

    #[test]
    fn epoch_rows_matches_reference() {
        // T=5.0, n=248 → [248×5] (verificado contra resolve_recipe_pub del port)
        assert_eq!(epoch_rows(248, 5.0), vec![248, 248, 248, 248, 248]);
        // T=2.6 → whole=2, rem=round(0.6·248)=149 → [248, 149, 248]
        assert_eq!(epoch_rows(248, 2.6), vec![248, 149, 248]);
    }

    #[test]
    fn kmeans_fit_with_emeans_dispatch() {
        // La vía de integración real de hrm2_engine: KMeans + fit_with.
        use crate::clustering::{KMeans, TrainerKind};
        let x = blobs_fixture(50, 32, 7);
        let arr = ndarray::Array2::from_shape_vec((200, 32), x.clone()).expect("shape");
        let mut km = KMeans::new(4, 100, 42);
        km.fit_with(TrainerKind::Emeans, &arr);
        let centroids = km.centroids.as_ref().expect("centroids tras fit_with");
        let labels = km.labels.as_ref().expect("labels tras fit_with");
        assert_eq!(centroids.nrows(), 4);
        assert_eq!(labels.len(), 200);
        assert!(labels.iter().all(|&l| l < 4));
        // Lloyd (camino histórico) sigue operativo y es la referencia
        let mut km2 = KMeans::new(4, 100, 42);
        km2.fit_with(TrainerKind::Lloyd, &arr);
        let lloyd_inertia = km2.inertia;
        // e-means MEIDO peor en blobs bien separados (contra-caso del paper:
        // GMM k=k_true +51.7%; acá +25.8%) — el gate de calidad real es el
        // A/B sobre embeddings del hub (medido −0.56% a favor de e-means).
        // El unit test solo pide el techo del contra-caso conocido.
        assert!(
            km.inertia <= lloyd_inertia * 1.35,
            "emeans={} lloyd={}",
            km.inertia,
            lloyd_inertia
        );
    }

    #[test]
    fn trainer_kind_parse_roundtrip() {
        use crate::clustering::TrainerKind;
        assert_eq!(TrainerKind::parse("emeans"), Some(TrainerKind::Emeans));
        assert_eq!(TrainerKind::parse("LLOYD"), Some(TrainerKind::Lloyd));
        assert_eq!(TrainerKind::parse("otro"), None);
        assert_eq!(TrainerKind::Emeans.as_str(), "emeans");
    }

    #[test]
    fn trainer_kind_serde_lowercase() {
        // contrato con config TOML/JSON: lowercase
        use crate::clustering::TrainerKind;
        assert_eq!(
            serde_json::to_string(&TrainerKind::Emeans).unwrap(),
            "\"emeans\""
        );
        let t: TrainerKind = serde_json::from_str("\"lloyd\"").unwrap();
        assert_eq!(t, TrainerKind::Lloyd);
    }
}
