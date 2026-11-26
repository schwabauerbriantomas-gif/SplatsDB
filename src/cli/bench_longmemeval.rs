//! LongMemEval benchmark — SplatsDB native methodology
//!
//! Two scopes (`--scope`):
//!
//! `wing` (default, legacy):
//!   1. Load session embeddings + query embeddings from binary
//!   2. Load metadata JSON (wing=question group, answer_session_ids)
//!   3. Create SplatStore with HNSW (M=32, ef_construction=400)
//!   4. Insert all sessions, register spatial metadata (wing=question_idx)
//!   5. Build index (HNSW + HRM2)
//!   6. For each query:
//!      a. SpatialIndex.filter(wing=qi) → ~48 candidate indices
//!      b. find_neighbors_filtered(query, candidates, k) → exact re-rank
//!   7. Measure: session recall (answer session in top-k), latency, QPS
//!
//!   The wing filter IS the question's own haystack: a benchmark shortcut
//!   that pre-selects ~48 sessions per question. An agent's single memory
//!   cannot apply it, so this scope never measures retrieval difficulty.
//!
//! `global`:
//!   4'. Insert all sessions (no spatial metadata)
//!   6'. For each query, two unfiltered paths over the whole store:
//!        - ANN:  find_neighbors_fused(query, rows) — HNSW candidates +
//!          exact re-rank (the production search path)
//!        - exact: find_neighbors(query, rows) — linear scan
//!      `rows = max(k, 10) * ROWS_PER_K`. Results are aggregated to the
//!      first k DISTINCT session ids by rank, so a chunk-level store
//!      (several rows per session id) is scored per session.
//!   7. Measure both paths: session recall, latency, QPS. `--dump` writes
//!      the per-query row indices so the scoring can be redone outside.
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array1;

use splatsdb::splats::NeighborResult;

/// Recall cut-offs reported for every run.
const K_VALUES: [usize; 4] = [1, 3, 5, 10];
/// Global scope: rows fetched per result slot before distinct-session
/// aggregation (a chunk-level store needs several rows per session).
const ROWS_PER_K: usize = 4;

/// Session metadata entry
#[derive(serde::Deserialize)]
struct SessionMeta {
    i: usize,    // global vector index
    qi: usize,   // question index (= wing)
    sid: String, // session ID
}

/// Query metadata entry
#[derive(serde::Deserialize)]
struct QueryMeta {
    qi: usize,
    answer_sids: Vec<String>,
    qtype: String,
}

/// Full metadata
#[derive(serde::Deserialize)]
struct BenchMeta {
    dim: usize,
    n_sessions: usize,
    n_queries: usize,
    sessions: Vec<SessionMeta>,
    queries: Vec<QueryMeta>,
}

/// Read SplatsDB binary vectors: u64 n + u64 dim + f32[n][dim]
fn read_vectors(path: &str) -> (usize, usize, Vec<f32>) {
    const MAX_LOAD_ELEMENTS: usize = 1_000_000_000;
    let mut f = fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("Cannot open {}: {}", path, e);
        std::process::exit(1);
    });
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8).unwrap();
    let n_raw = u64::from_le_bytes(buf8);
    let n = usize::try_from(n_raw).unwrap_or_else(|_| {
        eprintln!("Invalid vector count {} in {}", n_raw, path);
        std::process::exit(1);
    });
    f.read_exact(&mut buf8).unwrap();
    let dim_raw = u64::from_le_bytes(buf8);
    let dim = usize::try_from(dim_raw).unwrap_or_else(|_| {
        eprintln!("Invalid dimension {} in {}", dim_raw, path);
        std::process::exit(1);
    });
    let total = n.checked_mul(dim).unwrap_or_else(|| {
        eprintln!("Overflow in n*dim ({}*{}) in {}", n, dim, path);
        std::process::exit(1);
    });
    if total > MAX_LOAD_ELEMENTS {
        eprintln!(
            "Too many elements {} (limit {}) in {}",
            total, MAX_LOAD_ELEMENTS, path
        );
        std::process::exit(1);
    }
    let nbytes = total.checked_mul(4).unwrap_or_else(|| {
        eprintln!("Overflow in n*dim*4 in {}", path);
        std::process::exit(1);
    });
    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    if bytes.len() != nbytes {
        eprintln!("Byte count mismatch in {}", path);
        std::process::exit(1);
    }
    f.read_exact(bytes).unwrap();
    (n, dim, data)
}

/// First `k` distinct session ids in rank order.
///
/// Several rows may carry the same session id (chunk-level store): a
/// session counts once, at its best-ranked row. Rows without a known
/// session id are skipped.
fn first_k_distinct(
    results: &[NeighborResult],
    k: usize,
    idx_to_sid: &HashMap<usize, String>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(k);
    let mut seen: HashSet<String> = HashSet::new();
    for r in results {
        if out.len() >= k {
            break;
        }
        if let Some(sid) = idx_to_sid.get(&r.index) {
            if seen.insert(sid.clone()) {
                out.push(sid.clone());
            }
        }
    }
    out
}

/// A query is a hit when any of its answer sessions is among the results.
fn hit_any(result_sids: &[String], answer_sids: &HashSet<String>) -> bool {
    result_sids.iter().any(|sid| answer_sids.contains(sid))
}

/// Add one (hit, total) observation for a cut-off and a question type.
fn tally(
    counts: &mut HashMap<usize, (usize, usize)>,
    per_type: &mut HashMap<String, HashMap<usize, (usize, usize)>>,
    kv: usize,
    qtype: &str,
    hit: bool,
) {
    let entry = counts.entry(kv).or_insert((0, 0));
    if hit {
        entry.0 += 1;
    }
    entry.1 += 1;

    let pt = per_type.entry(qtype.to_string()).or_default();
    let pte = pt.entry(kv).or_insert((0, 0));
    if hit {
        pte.0 += 1;
    }
    pte.1 += 1;
}

fn rate(hits: usize, total: usize) -> f64 {
    hits as f64 / total.max(1) as f64
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_bench_longmemeval(
    sessions_path: String,
    queries_path: String,
    meta_path: String,
    k: usize,
    ef_search: usize,
    over_fetch: usize,
    scope: String,
    dump: Option<PathBuf>,
) {
    let global = match scope.as_str() {
        "wing" => false,
        "global" => true,
        other => {
            eprintln!("scope inválido: '{}' (usar wing|global)", other);
            std::process::exit(2);
        }
    };
    if k == 0 {
        eprintln!("k debe ser >= 1");
        std::process::exit(2);
    }

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  LongMemEval — SplatsDB Native Pipeline Benchmark           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    if global {
        eprintln!("  Scope: global — whole store, no filter");
    } else {
        eprintln!("  Scope: wing — question haystack pre-filter (benchmark shortcut)");
    }

    // ── Load metadata ──
    let meta: BenchMeta = serde_json::from_str(&fs::read_to_string(&meta_path).expect("read meta"))
        .expect("parse meta");
    eprintln!(
        "  Metadata: {} sessions, {} queries, dim={}",
        meta.n_sessions, meta.n_queries, meta.dim
    );

    // ── Load vectors ──
    eprintln!("[1/5] Loading vectors...");
    let (n_sess, dim_s, sess_data) = read_vectors(&sessions_path);
    let (n_q, dim_q, query_data) = read_vectors(&queries_path);
    assert_eq!(dim_s, meta.dim, "session dim mismatch");
    assert_eq!(dim_q, meta.dim, "query dim mismatch");
    assert_eq!(n_sess, meta.n_sessions, "session count mismatch");
    assert_eq!(n_q, meta.n_queries, "query count mismatch");
    eprintln!(
        "  Sessions: {} x {}d, Queries: {} x {}d",
        n_sess, dim_s, n_q, dim_q
    );

    // ── Build mappings ──
    // qi → set of answer session IDs
    let mut qi_to_answer_sids: HashMap<usize, HashSet<String>> = HashMap::new();
    // global_idx → session_id
    let mut idx_to_sid: HashMap<usize, String> = HashMap::new();

    for s in &meta.sessions {
        idx_to_sid.insert(s.i, s.sid.clone());
    }
    for q in &meta.queries {
        qi_to_answer_sids.insert(q.qi, q.answer_sids.iter().cloned().collect());
    }

    // ── Build SplatStore ──
    eprintln!("[2/5] Building SplatStore...");
    let t0 = Instant::now();

    let mut config = splatsdb::config::SplatsDBConfig::advanced(None);
    config.latent_dim = meta.dim;
    config.max_splats = meta.n_sessions + 1000;
    config.enable_hnsw = true;
    config.hnsw_m = 32;
    config.hnsw_ef_construction = 400;
    config.hnsw_ef_search = ef_search;
    config.hnsw_metric = "l2".to_string();
    config.over_fetch = over_fetch;

    let mut store = splatsdb::splats::SplatStore::new(config.clone());

    // Insert all sessions
    for i in 0..n_sess {
        let vec: &[f32] = &sess_data[i * dim_s..(i + 1) * dim_s];
        store.insert(vec);
    }
    eprintln!(
        "  Inserted {} vectors in {:.1}s",
        n_sess,
        t0.elapsed().as_secs_f64()
    );

    // Build index (HNSW + HRM2)
    let t_build = Instant::now();
    store.build_index();
    let build_ms = t_build.elapsed().as_millis();
    eprintln!(
        "  Index built in {}ms ({:.1}s)",
        build_ms,
        build_ms as f64 / 1000.0
    );

    // Build spatial index: register each session with wing=question_idx
    // (wing scope only — the global scope is the agent-like setting and
    // must not see which sessions belong to the question).
    let mut spatial = splatsdb::spatial::SpatialIndex::new();
    if !global {
        for s in &meta.sessions {
            spatial.register_doc(
                &format!("sess_{}", s.i),
                Some(&format!("wing_{}", s.qi)), // wing = question group
                None,                            // room = auto from clustering
                None,                            // hall = not used here
            );
        }
    }

    let total_setup_ms = t0.elapsed().as_millis();
    eprintln!(
        "  Total setup (insert + build + spatial): {:.1}s",
        total_setup_ms as f64 / 1000.0
    );

    // ── Run benchmark ──
    if global {
        eprintln!("[3/5] Running {} queries over the whole store...", n_q);
        eprintln!("  ANN:   find_neighbors_fused(query, rows) — HNSW candidates + exact re-rank",);
        eprintln!("  exact: find_neighbors(query, rows) — linear scan");
    } else {
        eprintln!(
            "[3/5] Running {} queries (spatial pre-filter → vector search)...",
            n_q
        );
        eprintln!(
            "  Method: SpatialIndex.filter(wing=qi) → find_neighbors_filtered(query, candidates, k)"
        );
    }

    let mut k_values: Vec<usize> = K_VALUES.to_vec();
    if !k_values.contains(&k) {
        k_values.push(k);
        k_values.sort_unstable();
    }
    let rows = *k_values.last().unwrap_or(&10);
    let fetch_rows = rows * ROWS_PER_K;

    let mut recall_counts: HashMap<usize, (usize, usize)> = HashMap::new(); // (hits, total)
    let mut per_type_recall: HashMap<String, HashMap<usize, (usize, usize)>> = HashMap::new();
    let mut recall_counts_exact: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut per_type_recall_exact: HashMap<String, HashMap<usize, (usize, usize)>> = HashMap::new();
    let mut latencies: Vec<f64> = Vec::with_capacity(n_q);
    let mut latencies_exact: Vec<f64> = Vec::with_capacity(n_q);
    let mut dump_file = dump
        .as_ref()
        .map(|p| fs::File::create(p).expect("create dump"));

    let t_search = Instant::now();

    for q in &meta.queries {
        let qi = q.qi;
        let query_vec: &[f32] = &query_data[qi * dim_q..(qi + 1) * dim_q];
        let query_arr = Array1::from_vec(query_vec.to_vec());

        // Main path
        let t_q = Instant::now();
        let results = if global {
            // Production search path: HNSW candidates → exact re-rank
            store.find_neighbors_fused(&query_arr.view(), fetch_rows)
        } else {
            // STEP 1: Spatial pre-filter — get candidates for this wing
            let filter = splatsdb::spatial::SpatialFilter {
                wing: Some(format!("wing_{}", qi)),
                room: None,
                hall: None,
            };
            let filtered_doc_ids = spatial.filter(&filter);
            let candidate_indices: Vec<usize> = filtered_doc_ids
                .iter()
                .filter_map(|id| {
                    id.strip_prefix("sess_")
                        .and_then(|s| s.parse::<usize>().ok())
                })
                .collect();
            // STEP 2: Vector search on filtered candidates (exact re-rank)
            store.find_neighbors_filtered(&query_arr.view(), &candidate_indices, rows)
        };
        latencies.push(t_q.elapsed().as_micros() as f64 / 1000.0);

        // Reference path over the whole store (global scope only)
        let exact = if global {
            let t_e = Instant::now();
            let e = store.find_neighbors(&query_arr.view(), fetch_rows);
            latencies_exact.push(t_e.elapsed().as_micros() as f64 / 1000.0);
            Some(e)
        } else {
            None
        };

        let answer_sids = qi_to_answer_sids.get(&qi).cloned().unwrap_or_default();

        // Check recall at each k
        for &kv in &k_values {
            let result_sids: Vec<String> = if global {
                first_k_distinct(&results, kv, &idx_to_sid)
            } else {
                results
                    .iter()
                    .take(kv)
                    .filter_map(|r| idx_to_sid.get(&r.index).cloned())
                    .collect()
            };
            tally(
                &mut recall_counts,
                &mut per_type_recall,
                kv,
                &q.qtype,
                hit_any(&result_sids, &answer_sids),
            );
            if let Some(ref e) = exact {
                let exact_sids = first_k_distinct(e, kv, &idx_to_sid);
                tally(
                    &mut recall_counts_exact,
                    &mut per_type_recall_exact,
                    kv,
                    &q.qtype,
                    hit_any(&exact_sids, &answer_sids),
                );
            }
        }

        if let Some(f) = dump_file.as_mut() {
            let row_idx =
                |r: &[NeighborResult]| -> Vec<usize> { r.iter().map(|n| n.index).collect() };
            let mut line = serde_json::Map::new();
            line.insert("qi".into(), serde_json::json!(qi));
            line.insert("qtype".into(), serde_json::json!(q.qtype));
            line.insert("ann".into(), serde_json::json!(row_idx(&results)));
            if let Some(ref e) = exact {
                line.insert("exact".into(), serde_json::json!(row_idx(e)));
            }
            writeln!(f, "{}", serde_json::Value::Object(line)).expect("write dump");
        }

        if qi % 100 == 0 {
            eprintln!("  Query {}/{}...", qi, n_q);
        }
    }

    let search_ms = t_search.elapsed().as_millis();
    drop(dump_file);

    // ── Results ──
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  RESULTS — SplatsDB Native Pipeline                         ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!("\n  Methodology:");
    if global {
        eprintln!(
            "    scope=global: no filter, {} candidates per query",
            n_sess
        );
        eprintln!(
            "    ANN   = find_neighbors_fused(query, {} rows) → first k distinct sessions",
            fetch_rows
        );
        eprintln!(
            "    exact = find_neighbors(query, {} rows) → first k distinct sessions",
            fetch_rows
        );
    } else {
        eprintln!("    scope=wing: SpatialIndex.filter(wing=question_group) → ~48 candidates");
        eprintln!("    find_neighbors_filtered(query, candidates, k) → exact re-rank");
    }
    eprintln!("    3. Check if answer session in top-k results");
    eprintln!(
        "\n  SplatsDB config: HNSW M=32, ef_construction=400, ef_search={}, over_fetch={}x",
        ef_search, over_fetch
    );
    eprintln!("  Embeddings: all-MiniLM-L6-v2 ({}d, normalized)", meta.dim);

    eprintln!("\n  Session Recall{}:", if global { " (ANN)" } else { "" });
    for &kv in &k_values {
        let (hits, total) = recall_counts.get(&kv).copied().unwrap_or((0, 1));
        eprintln!(
            "    Recall@{:2}: {}/{} = {:.1}%",
            kv,
            hits,
            total,
            rate(hits, total) * 100.0
        );
    }
    if global {
        eprintln!("\n  Session Recall (exact scan, whole store):");
        for &kv in &k_values {
            let (hits, total) = recall_counts_exact.get(&kv).copied().unwrap_or((0, 1));
            eprintln!(
                "    Recall@{:2}: {}/{} = {:.1}%",
                kv,
                hits,
                total,
                rate(hits, total) * 100.0
            );
        }
    }

    let mut latencies_sorted = latencies.clone();
    latencies_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_sorted[latencies_sorted.len() / 2];
    let p95 = latencies_sorted[(latencies_sorted.len() as f64 * 0.95) as usize];
    let qps = n_q as f64 / (search_ms as f64 / 1000.0);

    eprintln!("\n  Speed:");
    eprintln!("    Index build:  {:.1}s", build_ms as f64 / 1000.0);
    eprintln!(
        "    Total search: {:.1}s ({} queries, both paths)",
        search_ms as f64 / 1000.0,
        n_q
    );
    eprintln!("    P50 latency:  {:.3}ms", p50);
    eprintln!("    P95 latency:  {:.3}ms", p95);
    eprintln!("    QPS:          {:.0}", qps);
    let exact_p50 = if latencies_exact.is_empty() {
        None
    } else {
        let mut l = latencies_exact.clone();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(l[l.len() / 2])
    };
    if let Some(e) = exact_p50 {
        eprintln!("    Exact P50:    {:.3}ms", e);
    }

    let mut types_sorted: Vec<_> = per_type_recall.iter().collect();
    types_sorted.sort_by_key(|(t, _)| t.as_str());
    eprintln!("\n  Per-type Recall@{}:", k);
    for (qtype, kmap) in &types_sorted {
        let (hits, total) = kmap.get(&k).copied().unwrap_or((0, 1));
        let mut line = format!(
            "    {:<35}: {:3}/{:3} = {:5.1}%",
            qtype,
            hits,
            total,
            rate(hits, total) * 100.0
        );
        if global {
            if let Some((eh, et)) = per_type_recall_exact.get(*qtype).and_then(|m| m.get(&k)) {
                line.push_str(&format!(
                    "   exact {:3}/{:3} = {:5.1}%",
                    eh,
                    et,
                    rate(*eh, *et) * 100.0
                ));
            }
        }
        eprintln!("{}", line);
    }

    // ── JSON output ──
    let mut output = serde_json::Map::new();
    output.insert(
        "benchmark".into(),
        serde_json::Value::String("LongMemEval-S-cleaned — SplatsDB Native Pipeline".into()),
    );
    output.insert(
        "scope".into(),
        serde_json::Value::String(if global { "global" } else { "wing" }.into()),
    );
    output.insert(
        "methodology".into(),
        serde_json::Value::String(
            if global {
                "find_neighbors_fused (HNSW + exact re-rank) over the whole store, first k distinct session ids; exact = find_neighbors linear scan".into()
            } else {
                "SpatialIndex.filter(wing) → find_neighbors_filtered (exact re-rank)".into()
            },
        ),
    );
    output.insert(
        "dim".into(),
        serde_json::Value::Number(serde_json::Number::from(meta.dim)),
    );
    output.insert(
        "n_sessions".into(),
        serde_json::Value::Number(serde_json::Number::from(meta.n_sessions)),
    );
    output.insert(
        "n_queries".into(),
        serde_json::Value::Number(serde_json::Number::from(meta.n_queries)),
    );

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "hnsw_m".into(),
        serde_json::Value::Number(serde_json::Number::from(32)),
    );
    cfg.insert(
        "hnsw_ef_construction".into(),
        serde_json::Value::Number(serde_json::Number::from(400)),
    );
    cfg.insert(
        "hnsw_ef_search".into(),
        serde_json::Value::Number(serde_json::Number::from(ef_search)),
    );
    cfg.insert(
        "over_fetch".into(),
        serde_json::Value::Number(serde_json::Number::from(over_fetch)),
    );
    cfg.insert("metric".into(), serde_json::Value::String("l2".into()));
    output.insert("splatsdb_config".into(), serde_json::Value::Object(cfg));

    output.insert(
        "build_time_s".into(),
        serde_json::json!(build_ms as f64 / 1000.0),
    );
    output.insert(
        "search_time_s".into(),
        serde_json::json!(search_ms as f64 / 1000.0),
    );

    // `recall.@k` stays a plain number (legacy readers); counts live in
    // `recall_counts`.
    let counts_of =
        |counts: &HashMap<usize, (usize, usize)>| -> serde_json::Map<String, serde_json::Value> {
            k_values
                .iter()
                .map(|&kv| {
                    let (h, t) = counts.get(&kv).copied().unwrap_or((0, 1));
                    (format!("@{}", kv), serde_json::json!({ "hits": h, "n": t }))
                })
                .collect()
        };
    let recall_map: serde_json::Map<String, serde_json::Value> = k_values
        .iter()
        .map(|&kv| {
            let (h, t) = recall_counts.get(&kv).copied().unwrap_or((0, 1));
            (format!("@{}", kv), serde_json::json!(rate(h, t)))
        })
        .collect();
    output.insert("recall".into(), serde_json::Value::Object(recall_map));
    output.insert(
        "recall_counts".into(),
        serde_json::Value::Object(counts_of(&recall_counts)),
    );

    if global {
        let exact_map: serde_json::Map<String, serde_json::Value> = k_values
            .iter()
            .map(|&kv| {
                let (h, t) = recall_counts_exact.get(&kv).copied().unwrap_or((0, 1));
                (format!("@{}", kv), serde_json::json!(rate(h, t)))
            })
            .collect();
        output.insert("recall_exact".into(), serde_json::Value::Object(exact_map));
        output.insert(
            "recall_exact_counts".into(),
            serde_json::Value::Object(counts_of(&recall_counts_exact)),
        );
        output.insert(
            "rows_fetched".into(),
            serde_json::Value::Number(serde_json::Number::from(fetch_rows)),
        );
    }

    output.insert("qps".into(), serde_json::json!(qps));
    output.insert("p50_ms".into(), serde_json::json!(p50));
    output.insert("p95_ms".into(), serde_json::json!(p95));
    if let Some(e) = exact_p50 {
        output.insert("exact_p50_ms".into(), serde_json::json!(e));
    }

    let per_type: serde_json::Map<String, serde_json::Value> = types_sorted
        .iter()
        .map(|(qt, kmap)| {
            let (h, t) = kmap.get(&k).copied().unwrap_or((0, 1));
            let mut obj = serde_json::Map::new();
            obj.insert("recall".into(), serde_json::json!(rate(h, t)));
            obj.insert("hits".into(), serde_json::json!(h));
            obj.insert("n".into(), serde_json::json!(t));
            if global {
                if let Some((eh, et)) = per_type_recall_exact.get(*qt).and_then(|m| m.get(&k)) {
                    obj.insert("recall_exact".into(), serde_json::json!(rate(*eh, *et)));
                    obj.insert("hits_exact".into(), serde_json::json!(eh));
                }
            }
            ((*qt).clone(), serde_json::Value::Object(obj))
        })
        .collect();
    output.insert("per_type".into(), serde_json::Value::Object(per_type));

    let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(output)).unwrap();
    println!("{}", json_str);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(index: usize) -> NeighborResult {
        NeighborResult {
            mu: vec![],
            alpha: 0.0,
            kappa: 0.0,
            index,
            distance: 0.0,
            score: 0.0,
        }
    }

    fn mapping(pairs: &[(usize, &str)]) -> HashMap<usize, String> {
        pairs.iter().map(|(i, s)| (*i, s.to_string())).collect()
    }

    #[test]
    fn distinct_sessions_count_once_at_best_rank() {
        let m = mapping(&[(7, "A"), (9, "B"), (11, "A"), (12, "C")]);
        let results = [res(7), res(11), res(9), res(12)];
        assert_eq!(
            first_k_distinct(&results, 2, &m),
            vec!["A".to_string(), "B".to_string()]
        );
        assert_eq!(first_k_distinct(&results, 10, &m).len(), 3);
    }

    #[test]
    fn rows_without_session_id_are_skipped() {
        let m = mapping(&[(2, "A")]);
        let results = [res(1), res(3), res(2)];
        assert_eq!(first_k_distinct(&results, 5, &m), vec!["A".to_string()]);
    }

    #[test]
    fn fewer_rows_than_k_yield_fewer_sessions() {
        let m = mapping(&[(1, "A")]);
        assert_eq!(first_k_distinct(&[res(1)], 10, &m).len(), 1);
        assert!(first_k_distinct(&[], 10, &m).is_empty());
    }

    #[test]
    fn hit_requires_an_answer_session_in_results() {
        let answers: HashSet<String> = ["A".to_string()].into_iter().collect();
        assert!(hit_any(&["C".to_string(), "A".to_string()], &answers));
        assert!(!hit_any(&["C".to_string()], &answers));
        assert!(!hit_any(&[], &answers));
    }

    #[test]
    fn tally_counts_hits_and_totals() {
        let mut counts = HashMap::new();
        let mut per_type = HashMap::new();
        for hit in [true, false, true] {
            tally(&mut counts, &mut per_type, 10, "t", hit);
        }
        assert_eq!(counts.get(&10), Some(&(2, 3)));
        assert_eq!(per_type.get("t").unwrap().get(&10), Some(&(2, 3)));
    }
}
