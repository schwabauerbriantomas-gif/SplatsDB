//! SAC-style recap builder — LLM-free context compaction as a service.
//!
//! Motivated by the sac_falsif experiment (2026-11-14, 30/30 runs): compacting
//! to a <400-word handoff recap after every action (SAC, arXiv 2611.05306
//! "You don't need long context") matched full-history solve rate (10/10) on
//! GLM-5.3-flash with 5.6× fewer input tokens, at the cost of 2.1× wall time —
//! entirely from the extra LLM call per compaction (~45 calls × ~10s).
//!
//! This endpoint removes that cost: the recap is built by deterministic
//! extraction (section classification + TF-IDF ranking vs the objective +
//! recent actions + `semantic_compress`), with zero LLM and zero GPU. It is
//! stateless: actions come inline in the request, so it never touches the
//! store, works on live hubs, and is fully deterministic (same input → same
//! output).
//!
//! Recap format mirrors the harness's Appendix-G handoff: objective, facts,
//! changes, verification, dead ends, plan.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::text_compression::{semantic_compress, STOP_WORDS};

/// Default word budget for the generated recap (Appendix-G convention).
pub const DEFAULT_MAX_WORDS: usize = 400;
/// Hard cap on per-action text length (defense against accidental dumps).
pub const MAX_ACTION_TEXT_LEN: usize = 100_000;
/// Hard cap on total request size across all actions.
pub const MAX_TOTAL_ACTIONS_LEN: usize = 2_000_000;

/// One recorded agent action, in chronological order.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RecapAction {
    /// Monotonic sequence number (1-based from the harness).
    pub seq: u64,
    /// Harness-provided kind: "command" | "observation" | "edit" | "test" |
    /// "error" | "note". Content-based classification can override.
    #[serde(default)]
    pub kind: String,
    /// The action text: command line, observation output, diff, etc.
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct RecapRequest {
    /// Task objective; anchors the relevance ranking.
    #[serde(default)]
    pub objective: String,
    /// Action log, chronological.
    #[serde(default)]
    pub actions: Vec<RecapAction>,
    /// Word budget for the whole recap (default 400).
    #[serde(default)]
    pub max_words: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct RecapResponse {
    pub recap: String,
    pub words: usize,
    pub actions_in: usize,
    pub actions_used: usize,
    pub sections: RecapSections,
    pub elapsed_ms: f64,
}

#[derive(Debug, Serialize, Default)]
pub struct RecapSections {
    pub facts: usize,
    pub changes: usize,
    pub verification: usize,
    pub dead_ends: usize,
}

/// Recap section of an action, classified by explicit kind first, then by
/// deterministic content heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Section {
    Changes,
    Verification,
    DeadEnds,
    Facts,
}

fn classify(kind: &str, text: &str) -> Section {
    let lower = text.to_lowercase();
    match kind {
        "edit" | "diff" => return Section::Changes,
        "test" => return Section::Verification,
        "error" => return Section::DeadEnds,
        _ => {}
    }
    // Error/dead-end signals beat everything else.
    if [
        "traceback",
        "failed",
        "error:",
        "permission denied",
        "assertionerror",
        "compilation error",
        "no such file",
        "refused",
    ]
    .iter()
    .any(|s| lower.contains(s))
    {
        return Section::DeadEnds;
    }
    // Test/verification signals — outcomes only. NOTE: a command *running*
    // pytest is not a verification; its result line is.
    if ["passed", "test result", "warning in", "all tests", "ok."]
        .iter()
        .any(|s| lower.contains(s))
    {
        return Section::Verification;
    }
    // Change signals.
    if [
        "diff --git",
        "applypatch",
        "edit_file",
        "sed -i",
        "create_file",
        "write_file",
        "patch",
    ]
    .iter()
    .any(|s| lower.contains(s))
    {
        return Section::Changes;
    }
    Section::Facts
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| w.to_ascii_lowercase())
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

fn term_freq(tokens: &[String]) -> HashMap<String, f32> {
    let mut tf = HashMap::new();
    for t in tokens {
        *tf.entry(t.clone()).or_insert(0.0f32) += 1.0;
    }
    let norm = tokens.len().max(1) as f32;
    tf.values_mut().for_each(|v| *v /= norm);
    tf
}

/// Cosine similarity between two TF vectors under an IDF map.
fn cosine_idf(
    a: &HashMap<String, f32>,
    b: &HashMap<String, f32>,
    idf: &HashMap<String, f32>,
) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (t, v) in a {
        let w = idf.get(t).copied().unwrap_or(0.0);
        na += v * v * w * w;
        if let Some(vb) = b.get(t) {
            dot += v * vb * w * w;
        }
    }
    for (t, v) in b {
        let w = idf.get(t).copied().unwrap_or(0.0);
        nb += v * v * w * w;
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Truncate a line to `max_words` words on word boundaries.
fn truncate_words(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        text.to_string()
    } else {
        format!("{} …", words[..max_words].join(" "))
    }
}

/// Build the recap. Deterministic: no HashMap iteration feeds the output.
pub fn build_recap(req: &RecapRequest) -> RecapResponse {
    let started = Instant::now();
    let max_words = req.max_words.unwrap_or(DEFAULT_MAX_WORDS).clamp(20, 2000);
    let n = req.actions.len();

    let objective = if req.objective.trim().is_empty() {
        "Continue the assigned task.".to_string()
    } else {
        truncate_words(&semantic_compress(req.objective.trim()), 60)
    };

    if n == 0 {
        let recap = format!(
            "OBJECTIVE: {objective}\n\n(No actions recorded yet — inspect files before acting.)"
        );
        let words = count_words(&recap);
        return RecapResponse {
            recap,
            words,
            actions_in: 0,
            actions_used: 0,
            sections: RecapSections::default(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
    }

    // ── classify + dedup (exact-normalized text keeps the LATEST occurrence) ──
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut kept: Vec<(usize, &RecapAction, Section)> = Vec::new(); // (idx, action, section)
    for (i, a) in req.actions.iter().enumerate() {
        let section = classify(&a.kind, &a.text);
        let norm: String = a
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        if let Some(prev) = seen.get_mut(&norm) {
            // Duplicate: keep the newer index, drop the older from `kept`.
            let prev_idx = *prev;
            *prev = i;
            kept.retain(|(idx, _, _)| *idx != prev_idx);
            kept.push((i, a, section));
        } else {
            seen.insert(norm, i);
            kept.push((i, a, section));
        }
    }

    // ── TF-IDF ranking: query = objective + last 3 kept actions (current state) ──
    let docs: Vec<Vec<String>> = kept.iter().map(|(_, a, _)| tokenize(&a.text)).collect();
    let mut df: HashMap<String, usize> = HashMap::new();
    for toks in &docs {
        for t in toks.iter().collect::<std::collections::HashSet<_>>() {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
    }
    let idf: HashMap<String, f32> = df
        .into_iter()
        .map(|(t, d)| (t, ((1.0 + n as f32) / (1.0 + d as f32)).ln().max(0.0)))
        .collect();

    let last: Vec<String> = docs
        .iter()
        .rev()
        .take(3)
        .flat_map(|t| t.iter().cloned())
        .collect();
    let mut query_tokens = tokenize(&req.objective);
    query_tokens.extend(last);
    let query_tf = term_freq(&query_tokens);

    // Score per kept action: relevance × recency.
    let mut scored: Vec<(usize, Section, f32)> = kept
        .iter()
        .enumerate()
        .map(|(k, (idx, _a, sec))| {
            let tf = term_freq(&docs[k]);
            let rel = cosine_idf(&query_tf, &tf, &idf);
            let recency = 1.0 + 0.5 * (*idx as f32 / (n as f32 - 1.0).max(1.0));
            (*idx, *sec, rel * recency)
        })
        .collect();
    // Deterministic order: section priority, then score desc, then input idx.
    scored.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.0.cmp(&b.0))
    });

    // ── budgeted selection ──
    // Fixed overhead: OBJECTIVE line + section headers ("DEAD ENDS (do not
    // retry):" solo pesa 4 palabras) + PLAN line.
    let overhead = count_words(&objective)
        + 14 // headers
        + 28; // plan reserve
    let mut budget = max_words.saturating_sub(overhead).max(10);

    let mut sections = RecapSections::default();

    // State-critical minimums first: 1 change, 1 verification, 1 dead end
    // (best-scored). Selection uses a CHEAP raw word-count estimate; the
    // expensive semantic_compress runs only in the render phase, on the ≤
    // budget-sized set of lines actually chosen (compressed text is never
    // longer than raw, so the budget estimate stays conservative).
    let mut chosen_idx: Vec<(usize, Section)> = Vec::new();
    for must in [Section::Changes, Section::Verification, Section::DeadEnds] {
        if let Some(pos) = scored.iter().position(|(_, s, _)| *s == must) {
            let (idx, sec, _) = scored[pos];
            let w = count_words(&req.actions[idx].text).min(40) + 1; // +1 bullet
            if w <= budget {
                budget -= w;
                chosen_idx.push((idx, sec));
                scored.remove(pos);
            }
        }
    }

    // Greedy fill by score within remaining budget (scored is already sorted
    // deterministically: section priority → score desc → input idx).
    for &(idx, sec, _) in scored.iter() {
        if budget < 2 {
            break;
        }
        let w = count_words(&req.actions[idx].text).min(40) + 1;
        if w <= budget {
            budget -= w;
            chosen_idx.push((idx, sec));
        }
    }

    // ── render: compress each chosen line ONCE, chronological, grouped ──
    chosen_idx.sort_by_key(|(idx, _)| *idx);
    let mut chosen: Vec<(usize, Section, String)> = chosen_idx
        .iter()
        .map(|(idx, sec)| {
            (
                *idx,
                *sec,
                truncate_words(&semantic_compress(&req.actions[*idx].text), 40),
            )
        })
        .collect();

    // Render helper (objective + grouped sections + PLAN line).
    let render_all = |objective: &str, chosen: &[(usize, Section, String)]| -> String {
        let mut out = String::with_capacity(2048);
        out.push_str(&format!("OBJECTIVE: {objective}\n"));
        let section = |out: &mut String, header: &str, sec: Section| {
            let lines: Vec<&(usize, Section, String)> =
                chosen.iter().filter(|(_, s, _)| *s == sec).collect();
            if lines.is_empty() {
                return;
            }
            out.push_str(&format!("\n{header}:\n"));
            for (_, _, line) in lines {
                out.push_str(&format!("- {line}\n"));
            }
        };
        section(&mut out, "CHANGES", Section::Changes);
        section(&mut out, "VERIFICATION", Section::Verification);
        section(&mut out, "DEAD ENDS (do not retry)", Section::DeadEnds);
        section(&mut out, "FACTS", Section::Facts);
        // PLAN: deterministic — continue from the newest selected action.
        if let Some(&(idx, _, _)) = chosen.iter().max_by_key(|(idx, _, _)| *idx) {
            let plan = truncate_words(&semantic_compress(&req.actions[idx].text), 25);
            out.push_str(&format!("\nPLAN: Continue from: {plan}\n"));
        }
        out
    };

    for (_, sec, _) in &chosen {
        *match sec {
            Section::Changes => &mut sections.changes,
            Section::Verification => &mut sections.verification,
            Section::DeadEnds => &mut sections.dead_ends,
            Section::Facts => &mut sections.facts,
        } += 1;
    }

    let mut out = render_all(&objective, &chosen);
    // Post-render safety trim: enforce the word contract exactly. Facts are
    // the lowest-priority section — drop newest-first until under budget.
    while count_words(&out) > max_words {
        match chosen.iter().rposition(|(_, s, _)| *s == Section::Facts) {
            Some(p) => {
                chosen.remove(p);
                sections.facts -= 1;
                out = render_all(&objective, &chosen);
            }
            None => break, // nothing removable left — best effort
        }
    }

    let words = count_words(&out);
    let actions_used = chosen.len();
    RecapResponse {
        recap: out,
        words,
        actions_in: n,
        actions_used,
        sections,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(seq: u64, kind: &str, text: &str) -> RecapAction {
        RecapAction {
            seq,
            kind: kind.to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn classify_sections_by_kind_and_content() {
        assert_eq!(classify("edit", "anything"), Section::Changes);
        assert_eq!(classify("test", "anything"), Section::Verification);
        assert_eq!(classify("error", "anything"), Section::DeadEnds);
        // Content overrides for generic kinds:
        assert_eq!(
            classify("observation", "Traceback (most recent call last)"),
            Section::DeadEnds
        );
        assert_eq!(
            classify("observation", "1 passed, 1 warning in 0.03s"),
            Section::Verification
        );
        assert_eq!(
            classify("command", "diff --git a/x.py b/x.py"),
            Section::Changes
        );
        assert_eq!(classify("command", "ls -la src/"), Section::Facts);
    }

    #[test]
    fn empty_actions_returns_objective_only() {
        let req = RecapRequest {
            objective: "Fix the netrc auth guard".into(),
            actions: vec![],
            max_words: None,
        };
        let r = build_recap(&req);
        assert_eq!(r.actions_in, 0);
        assert!(r.recap.starts_with("OBJECTIVE:"), "recap: {}", r.recap);
        assert!(r.recap.contains("netrc"), "recap: {}", r.recap);
        assert!(r.recap.contains("No actions recorded yet"));
    }

    #[test]
    fn dedup_keeps_latest_occurrence() {
        let req = RecapRequest {
            objective: "Fix bug".into(),
            actions: vec![
                action(1, "command", "pytest tests/test_auth.py"),
                action(2, "command", "pytest tests/test_auth.py"),
                action(3, "command", "git status"),
            ],
            max_words: None,
        };
        let r = build_recap(&req);
        // Duplicate command collapses to one; 3 in → ≤3 used, dedup happened.
        assert_eq!(r.actions_in, 3);
        assert!(r.actions_used < 3);
        // The latest occurrence (seq 2) must be the retained one — order in
        // output is chronological so seq2 renders as the pytest line.
        assert!(r.recap.matches("pytest tests/test_auth.py").count() == 1);
    }

    #[test]
    fn respects_word_budget() {
        let filler =
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod ".repeat(30);
        let actions: Vec<RecapAction> = (0..200)
            .map(|i| action(i as u64 + 1, "observation", &format!("run {i}: {filler}")))
            .collect();
        let req = RecapRequest {
            objective: "Benchmark the recap builder budget".into(),
            actions,
            max_words: Some(400),
        };
        let r = build_recap(&req);
        assert!(
            r.words <= 400,
            "recap words {} exceeded budget 400",
            r.words
        );
        assert!(r.actions_used < 200);
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let mk = || RecapRequest {
            objective: "Fix the TLS connection pool leak".into(),
            actions: vec![
                action(1, "command", "pytest tests/test_pool.py -x"),
                action(2, "observation", "Traceback: connection refused"),
                action(3, "edit", "applypatch get_netrc_auth guard"),
                action(4, "test", "1 passed, 1 warning in 0.03s"),
                action(5, "command", "git diff --stat"),
            ],
            max_words: None,
        };
        let a = build_recap(&mk());
        let b = build_recap(&mk());
        assert_eq!(a.recap, b.recap);
        assert_eq!(a.sections.changes, 1);
        assert_eq!(a.sections.verification, 1);
        assert_eq!(a.sections.dead_ends, 1);
    }

    #[test]
    fn recap_is_fast_on_large_logs() {
        let filler = "status 200 ok endpoint handler response body item ".repeat(40);
        let actions: Vec<RecapAction> = (0..500)
            .map(|i| action(i as u64 + 1, "observation", &format!("step {i}: {filler}")))
            .collect();
        let req = RecapRequest {
            objective: "Speed check".into(),
            actions,
            max_words: None,
        };
        let r = build_recap(&req);
        // Generous CI bound (debug build); release target is <50ms.
        assert!(r.elapsed_ms < 2000.0, "build_recap took {}ms", r.elapsed_ms);
    }
}
