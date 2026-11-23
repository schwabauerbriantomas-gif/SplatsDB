//! F02f — the CLI loads the persisted energy state (alpha/kappa) exactly as
//! `serve` does since F02e1.
//!
//! Before the fix every CLI command went through `load_or_create_store`,
//! which only fed the shard to `add_splat`: alpha/kappa stayed at the init
//! values (`SplatsDBConfig::default()`: 1.0 / 10.0) that `ingest` had replaced
//! on disk. This test drives the real binary — a real `ingest`, then
//! `search --format json`, whose rows carry each neighbour's alpha/kappa —
//! against the reference ranking computed from `energy/landscape.bin`, before
//! any restart and after a `serve` session compacted the shard.

#[path = "cli_durability_common/mod.rs"]
mod common;

use std::path::Path;

use common::*;
use serde_json::{json, Value};

/// Splats (clusters) of the ingested corpus — rows of the `default` shard.
const K: usize = 6;
/// `SplatsDBConfig::default()`: what an unloaded energy state looks like.
const INIT_ALPHA: f32 = 1.0;
const INIT_KAPPA: f32 = 10.0;

/// One row of `search --format json` as the CLI rounds it (4 decimals).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Row {
    index: usize,
    distance: f32,
    alpha: f32,
    kappa: f32,
}

fn d2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum()
}

/// `energy/landscape.bin` as `persistence::save_energy_state` writes it:
/// `[u64 rows][u64 cols][f32 mu][u64 n_alpha][f32 alpha][u64 n_kappa][f32 kappa]`.
fn read_landscape(path: &Path) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    let u64at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize;
    let (rows, cols) = (u64at(0), u64at(8));
    let mu: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| f32_at(&b, 16 + (r * cols + c) * 4))
                .collect()
        })
        .collect();
    let o = 16 + rows * cols * 4;
    let n_alpha = u64at(o);
    let alpha: Vec<f32> = (0..n_alpha).map(|i| f32_at(&b, o + 8 + i * 4)).collect();
    let o = o + 8 + n_alpha * 4;
    let n_kappa = u64at(o);
    let kappa: Vec<f32> = (0..n_kappa).map(|i| f32_at(&b, o + 8 + i * 4)).collect();
    assert_eq!(
        o + 8 + n_kappa * 4,
        b.len(),
        "trailing bytes in landscape.bin"
    );
    (mu, alpha, kappa)
}

/// Server-faithful gaussian top-k: the score in f32 exactly like
/// `SplatStore::gaussian_score` (`alpha · exp(−kappa · d²)`), score
/// descending, ties → ascending id.
fn gaussian_topk(
    mu: &[Vec<f32>],
    alpha: &[f32],
    kappa: &[f32],
    q: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = mu
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let d2 = d2(m, q) as f32;
            (i, alpha[i] * (-(kappa[i] * d2)).exp())
        })
        .collect();
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

/// Exact L2 top-k: `(id, d²)`, ascending, ties → ascending id.
fn l2_topk(mu: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = mu.iter().enumerate().map(|(i, m)| (i, d2(m, q))).collect();
    all.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

/// Neighbours whose score did not underflow to 0.0 (f32): the ones a ranking
/// recomputed from the CLI's rounded rows can order robustly.
fn live(scores: &[(usize, f32)]) -> usize {
    scores.iter().take_while(|r| r.1 > 0.0).count()
}

struct Probe {
    q: Vec<f32>,
    live: usize,
}

/// A query whose gaussian top-k (under the persisted alpha/kappa) differs
/// from its L2 ranking and from index order, and whose live neighbours are at
/// least 1% apart — the CLI prints distance/alpha/kappa with 4 decimals, so a
/// score recomputed from them is off by well under 0.1%.
fn discriminating_probe(
    centers: &[Vec<f32>],
    mu: &[Vec<f32>],
    alpha: &[f32],
    kappa: &[f32],
) -> Probe {
    (0..K)
        .flat_map(|a| (0..K).filter(move |b| *b != a).map(move |b| (a, b)))
        .flat_map(|(a, b)| [0.3, 0.5, 0.7, 0.9, 1.2, 1.6].map(move |t| (a, b, t)))
        .find_map(|(a, b, t)| {
            let d = d2(&centers[a], &centers[b]).sqrt();
            let q: Vec<f32> = (0..DIM)
                .map(|i| centers[a][i] + ((t / d) as f32) * (centers[b][i] - centers[a][i]))
                .collect();
            let g = gaussian_topk(mu, alpha, kappa, &q, K);
            let n = live(&g);
            let s: Vec<f64> = g[..n].iter().map(|r| f64::from(r.1)).collect();
            let robust = n >= 4 && s.windows(2).all(|w| (w[0] - w[1]).abs() > 1e-2 * w[0]);
            let ids: Vec<usize> = g[..n].iter().map(|r| r.0).collect();
            let l2: Vec<usize> = l2_topk(mu, &q, n).iter().map(|r| r.0).collect();
            let discriminating = ids != l2 && ids != (0..n).collect::<Vec<_>>();
            (robust && discriminating).then_some(Probe { q, live: n })
        })
        .expect("no query separates the gaussian ranking from L2 for this corpus")
}

fn copy_dir(src: &Path, dst: &Path) {
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let target = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            std::fs::create_dir_all(&target).unwrap();
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// `POST /search?rescore=gaussian` on a running serve: the neighbour ids.
fn serve_search(port: u16, embedding: &[f32], top_k: usize) -> Vec<usize> {
    let body = json!({ "query": "f02f-probe", "embedding": embedding, "top_k": top_k }).to_string();
    let (code, text) = http(port, "POST", "/search?rescore=gaussian", Some(&body))
        .unwrap_or_else(|| panic!("/search on :{port} did not answer"));
    assert_eq!(
        code, 200,
        "POST /search?rescore=gaussian on :{port}: {text}"
    );
    let v: Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("/search json: {e}: {text}"));
    v["results"]
        .as_array()
        .unwrap_or_else(|| panic!("results array missing: {v}"))
        .iter()
        .map(|r| r["index"].as_u64().unwrap() as usize)
        .collect()
}

/// The CLI's own rounding of `search --format json`.
fn round4(x: f32) -> f32 {
    (x * 10000.0).round() / 10000.0
}

/// `search --format json --k k` on `data_dir`: its rows (L2 order) and stderr.
fn cli_rows(data_dir: &Path, q: &[f32], k: usize) -> (Vec<Row>, String) {
    let (query, k) = (csv(q), k.to_string());
    let (status, out, err) = cli(
        data_dir,
        DIM,
        &["search", "--query", &query, "--k", &k, "--format", "json"],
    );
    assert!(
        status.success(),
        "search failed: stdout={out}\nstderr={err}"
    );
    let v: Value = serde_json::from_str(out.trim())
        .unwrap_or_else(|e| panic!("search output is not JSON ({e}): {out}"));
    let rows: Vec<Row> = v
        .as_array()
        .expect("search --format json prints an array of rows")
        .iter()
        .map(|r| Row {
            index: r["index"].as_u64().expect("index") as usize,
            distance: r["distance"].as_f64().expect("distance") as f32,
            alpha: r["alpha"].as_f64().expect("alpha") as f32,
            kappa: r["kappa"].as_f64().expect("kappa") as f32,
        })
        .collect();
    (rows, err)
}

/// The gaussian ranking implied by what the CLI printed:
/// `alpha · exp(−kappa · distance²)` per row, score descending, ties by id.
fn derived_gaussian(rows: &[Row], live: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = rows
        .iter()
        .map(|r| {
            let score =
                f64::from(r.alpha) * (-f64::from(r.kappa) * f64::from(r.distance).powi(2)).exp();
            (r.index, score)
        })
        .collect();
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(live);
    all
}

fn ids_of(scores: &[(usize, f32)]) -> Vec<usize> {
    scores.iter().map(|r| r.0).collect()
}

/// Every row must carry the alpha/kappa of the state on disk (as rounded by
/// the CLI) — never the init values where the persisted state differs.
fn assert_persisted_energy(rows: &[Row], alpha: &[f32], kappa: &[f32], what: &str) {
    for r in rows {
        let (want_alpha, want_kappa) = (round4(alpha[r.index]), round4(kappa[r.index]));
        assert!(
            (r.alpha - want_alpha).abs() < 1e-6 && (r.kappa - want_kappa).abs() < 1e-6,
            "{what}: row {} reports alpha={} kappa={} but the persisted state holds \
             alpha={want_alpha} kappa={want_kappa} (the init is {INIT_ALPHA}/{INIT_KAPPA}); rows: {rows:?}",
            r.index,
            r.alpha,
            r.kappa
        );
    }
}

/// `load_or_create_store` (every CLI reader/writer) loads the alpha/kappa
/// `ingest` persisted: the CLI reports the same energy state and gaussian
/// ranking as `serve`, before any restart and after a `serve` session
/// compacted the shard.
#[test]
fn cli_reports_the_persisted_energy_state_before_and_after_a_serve_restart() {
    let root = scratch("energy");
    let seed = root.join("seed");
    let raw = root.join("raw.bin");
    let raw = {
        write_rows(&raw, &clustered_corpus());
        raw.to_string_lossy().into_owned()
    };
    let (status, out, err) = cli(
        &seed,
        DIM,
        &[
            "ingest",
            "--input",
            &raw,
            "--n-clusters",
            "6",
            "--seed",
            "7",
        ],
    );
    assert!(
        status.success(),
        "ingest failed: stdout={out}\nstderr={err}"
    );
    let mu = read_bin(&shard_path(&seed));
    let (landscape_mu, alpha, kappa) = read_landscape(&landscape_path(&seed));
    assert_eq!(
        landscape_mu, mu,
        "the landscape's mu must be the shard's mu"
    );
    assert_eq!(mu.len(), K, "ingest must leave {K} splats: {mu:?}");
    assert_ne!(
        (alpha.clone(), kappa.clone()),
        (vec![INIT_ALPHA; K], vec![INIT_KAPPA; K]),
        "the persisted energy state must differ from the init values to be observable: \
         alpha={alpha:?} kappa={kappa:?}"
    );
    eprintln!("[f02f] persisted energy state: alpha={alpha:?} kappa={kappa:?}");

    let probe = discriminating_probe(&centres(), &mu, &alpha, &kappa);
    let gauss = gaussian_topk(&mu, &alpha, &kappa, &probe.q, K);
    let gauss_ids = ids_of(&gauss[..probe.live]);
    let l2: Vec<usize> = l2_topk(&mu, &probe.q, K).iter().map(|r| r.0).collect();
    eprintln!(
        "[f02f] probe q={:?} live={} reference: gaussian={gauss:?} l2={l2:?}",
        probe.q, probe.live
    );

    // ── Before any restart: a fresh CLI process on the data-dir `ingest` left.
    let (rows, err) = cli_rows(&seed, &probe.q, K);
    assert_persisted_energy(&rows, &alpha, &kappa, "before the restart");
    assert!(
        err.contains(&format!(
            "energy state cargado (alpha/kappa de {K}/{K} splats"
        )),
        "the CLI must report the loaded energy state: {err}"
    );
    let ids: Vec<usize> = rows.iter().map(|r| r.index).collect();
    assert_eq!(
        ids, l2,
        "A5: `search` (L2) must keep its ranking — only the reported alpha/kappa change"
    );
    let before = derived_gaussian(&rows, probe.live);
    assert_eq!(
        before.iter().map(|r| r.0).collect::<Vec<_>>(),
        gauss_ids,
        "the gaussian ranking implied by the CLI rows must be the persisted one"
    );
    assert_ne!(
        gauss_ids, l2,
        "the probe must discriminate gaussian from L2"
    );

    // ── A `serve` session on a copy: one /store, then SIGTERM compacts the
    //    shard to K+1 rows and leaves the landscape alone.
    let data = root.join("data");
    copy_dir(&seed, &data);
    let port = probe_port();
    let mut server = spawn_serve(&data, &root.join("serve-1.log"), port);
    wait_healthy(port, &mut server);
    let served = serve_search(port, &probe.q, K);
    assert_eq!(
        served[..probe.live],
        gauss_ids,
        "serve before /store must answer the persisted ranking: {served:?}"
    );
    let far: Vec<f32> = (0..DIM).map(|d| 40.0 + 0.25 * d as f32).collect();
    serve_store(port, &far);
    let code = shutdown(&mut server);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&server)
    );
    assert_eq!(
        read_bin(&shard_path(&data)).len(),
        K + 1,
        "the shutdown compacts the shard"
    );
    assert_eq!(
        read_landscape(&landscape_path(&data)),
        (mu.clone(), alpha.clone(), kappa.clone()),
        "serve never rewrites the landscape"
    );

    // ── After the restart: the CLI still sees the ingested splats' state;
    //    the row /store appended keeps the init values, exactly as in serve.
    let (rows, err) = cli_rows(&data, &probe.q, K + 1);
    assert!(
        err.contains(&format!(
            "energy state cargado (alpha/kappa de {K}/{} splats",
            K + 1
        )),
        "the CLI must load the K-row prefix of the landscape: {err}"
    );
    let mut alpha_post = alpha.clone();
    alpha_post.push(INIT_ALPHA);
    let mut kappa_post = kappa.clone();
    kappa_post.push(INIT_KAPPA);
    assert_persisted_energy(&rows, &alpha_post, &kappa_post, "after the restart");
    let mut mu_post = mu.clone();
    mu_post.push(far.clone());
    let l2_post: Vec<usize> = l2_topk(&mu_post, &probe.q, K + 1)
        .iter()
        .map(|r| r.0)
        .collect();
    assert_eq!(
        rows.iter().map(|r| r.index).collect::<Vec<_>>(),
        l2_post,
        "A5: `search` (L2) after the restart"
    );
    let gauss_post = gaussian_topk(&mu_post, &alpha_post, &kappa_post, &probe.q, K + 1);
    let live_post = live(&gauss_post);
    let after = derived_gaussian(&rows, live_post);
    assert_eq!(
        after.iter().map(|r| r.0).collect::<Vec<_>>(),
        ids_of(&gauss_post[..live_post]),
        "the gaussian ranking implied by the CLI rows after the restart"
    );
    for ((_, got), (_, want)) in after.iter().zip(&gauss_post[..live_post]) {
        assert!(
            (got - f64::from(*want)).abs() <= 1e-2 * f64::from(*want),
            "recomputed score {got} vs reference {want}"
        );
    }

    // ── A restarted `serve` reads the same files and answers the same
    //    gaussian ranking the CLI reported.
    let mut server = spawn_serve(&data, &root.join("serve-2.log"), port);
    wait_healthy(port, &mut server);
    let restarted = serve_search(port, &probe.q, K + 1);
    assert_eq!(
        restarted[..live_post],
        after.iter().map(|r| r.0).collect::<Vec<_>>(),
        "serve after the restart vs the CLI rows"
    );
    let code = shutdown(&mut server);
    assert_eq!(
        code,
        Some(0),
        "serve must exit 0 after SIGTERM: {}",
        stderr_of(&server)
    );
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", port)).is_err(),
        "no listener may remain on the probe port"
    );
    std::fs::remove_dir_all(&root).ok();
}
