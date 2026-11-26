#!/usr/bin/env python3
"""LongMemEval-S scope ladder: haystack-restricted vs one global memory.

Measures session retrieval — is an answer session in the top-k? — on the same
embeddings in two scopes and three representations. The SplatsDB engine is
NOT involved here; for that run `splatsdb bench-long-mem-eval --scope global`.

Scopes
  within  candidates = the question's own haystack sessions (~48). The
          historical 96.6% headline used this scope: it pre-selects the
          answer's neighbourhood, and an agent cannot apply it.
  global  candidates = every unique session of the dataset, no filter.

Representations
  head       one MiniLM vector per session, model-default truncation
  chunk-max  turn-aligned chunks of <= 200 word-pieces, max cosine per session
  keyword    SQLite FTS5 (unicode61), OR-query, bm25() ranking

Usage
  python benchmarks/longmemeval_scope.py --dataset longmemeval_s_cleaned.json \\
      --cache-dir /tmp/lm_cache [--reuse] [--hybrid-exploratory] [--out r.json]

`--reuse` loads embeddings from --cache-dir when they exist and skips the GPU.
Only aggregate numbers are written: no message or question text is published.
"""
import argparse
import collections
import hashlib
import json
import os
import platform
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_BUDGET = 200
SESSION_BLOCK = 500
KS = (1, 3, 5, 10)
BOOT_N, BOOT_SEED = 10000, 20261205
SEED = 20261206
RRF_K, POOL = 60, 100
FILES = {
    "head": "head_sessions.npy",
    "queries": "queries.npy",
    "chunks": "chunk_vectors.npy",
    "chunk_session": "chunk_session.npy",
    "chunk_tokens": "chunk_token_counts.npy",
    "model": "model_info.json",
    "corpus": "corpus_index.json",
}


# ── pure helpers (unit-tested) ─────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def stream_json_array(path, chunk=1 << 22):
    """Yield the objects of a JSON array one at a time (bounded memory)."""
    decoder = json.JSONDecoder()
    handle = Path(path).open("r", encoding="utf-8")
    buf = ""
    pos = 0

    def more():
        nonlocal buf
        data = handle.read(chunk)
        if not data:
            raise EOFError("unexpected EOF inside JSON array")
        buf += data

    def skip_ws():
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                return buf[pos]
            more()

    try:
        if skip_ws() != "[":
            raise ValueError("expected JSON array")
        pos += 1
        while True:
            c = skip_ws()
            if c == "]":
                pos += 1
                break
            if c == ",":
                pos += 1
                continue
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    more()
            yield obj
            pos = end
            if pos > (1 << 21):
                buf = buf[pos:]
                pos = 0
    finally:
        handle.close()


def session_lines(msgs):
    return [f"{m.get('role') or ''}: {m.get('content') or ''}" for m in msgs]


def assemble_pieces(line_session, encoded, budget=CHUNK_BUDGET):
    """Turn-aligned chunks: (session index, token ids) with <= budget ids.

    Consecutive lines of one session are packed until the budget is reached; a
    single line longer than the budget is split by tokens. No overlap.
    """
    pieces, cur, cur_sess = [], [], None
    for sess, ids in zip(line_session, encoded):
        if cur_sess is not None and sess != cur_sess:
            if cur:
                pieces.append((cur_sess, cur))
            cur = []
        cur_sess = sess
        if not ids:
            continue
        if len(ids) > budget:
            if cur:
                pieces.append((cur_sess, cur))
                cur = []
            pieces.extend((cur_sess, ids[s:s + budget]) for s in range(0, len(ids), budget))
            continue
        if len(cur) + len(ids) > budget:
            pieces.append((cur_sess, cur))
            cur = []
        cur.extend(ids)
    if cur_sess is not None and cur:
        pieces.append((cur_sess, cur))
    return pieces


def topk(scores, restrict, k):
    """Top-k indices by score desc; ties broken by ascending index."""
    idx = np.arange(scores.shape[0]) if restrict is None else np.asarray(restrict, dtype=np.int64)
    order = np.lexsort((idx, -scores[idx]))
    return idx[order[:k]]


def hits_at(rank, answers, ks=KS):
    """Hit@k = any answer session among the first k of `rank`."""
    out = {}
    for k in ks:
        out[k] = bool(set(int(i) for i in rank[:k]) & set(answers))
    return out


def fts_query(text):
    toks = sorted(set(re.findall(r"[A-Za-z0-9]+", (text or "").lower())))
    return " OR ".join(f'"{t}"' for t in toks)


def rrf(rankings, k_const=RRF_K):
    """Reciprocal-rank fusion; ties broken by ascending id."""
    score = {}
    for rank in rankings:
        for pos, d in enumerate(int(i) for i in rank):
            score[d] = score.get(d, 0.0) + 1.0 / (k_const + pos + 1)
    return sorted(score, key=lambda d: (-score[d], d))


def summarize(hit, questions):
    by_type = collections.defaultdict(list)
    for h, q in zip(hit, questions):
        by_type[q["qtype"]].append(bool(h))
    return {
        "n": int(len(hit)),
        "hits": int(hit.sum()),
        "rate": float(hit.mean()) if len(hit) else None,
        "per_type": {t: {"n": len(v), "hits": int(sum(v)), "rate": float(np.mean(v))} for t, v in sorted(by_type.items())},
    }


def ci95(hit, n_boot=BOOT_N, seed=BOOT_SEED):
    if not len(hit):
        return None
    rng = np.random.default_rng(seed)
    means = hit[rng.integers(0, len(hit), size=(n_boot, len(hit)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_delta(a, b, n_boot=BOOT_N, seed=BOOT_SEED):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


# ── pipeline ────────────────────────────────────────────────────────────────

def build_corpus(dataset):
    sid_to_idx, session_texts, questions = {}, [], []
    n_slots = 0
    for q in stream_json_array(dataset):
        hay_ids, sessions = q.get("haystack_session_ids", []), q.get("haystack_sessions", [])
        slots = []
        for sid, msgs in zip(hay_ids, sessions):
            n_slots += 1
            if sid not in sid_to_idx:
                sid_to_idx[sid] = len(session_texts)
                session_texts.append("\n".join(session_lines(msgs)))
            slots.append(sid_to_idx[sid])
        ans = [sid_to_idx[a] for a in q.get("answer_session_ids", []) if a in sid_to_idx]
        questions.append({
            "question_id": q.get("question_id"),
            "qtype": q.get("question_type"),
            "question": q.get("question"),
            "slots": slots,
            "answers": sorted(set(ans)),
            "answers_missing": len(q.get("answer_session_ids", [])) - len(ans),
        })
    return session_texts, questions, n_slots


def build_chunk_corpus(tokenizer, session_texts):
    chunk_texts, chunk_session, token_counts = [], [], []
    n = len(session_texts)
    for b0 in range(0, n, SESSION_BLOCK):
        line_texts, line_session = [], []
        for si in range(b0, min(n, b0 + SESSION_BLOCK)):
            for line in session_texts[si].split("\n"):
                line_texts.append(line)
                line_session.append(si)
        enc = tokenizer(line_texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        pieces = assemble_pieces(line_session, enc)
        chunk_texts.extend(tokenizer.batch_decode([p[1] for p in pieces], skip_special_tokens=True))
        chunk_session.extend(p[0] for p in pieces)
        token_counts.extend(len(p[1]) for p in pieces)
        del enc, pieces, line_texts
    return chunk_texts, np.asarray(chunk_session, dtype=np.int64), np.asarray(token_counts, dtype=np.int64)


def embed(model, texts, batch):
    t = time.perf_counter()
    vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.ascontiguousarray(vecs, dtype=np.float32), time.perf_counter() - t


def build_fts(texts, path):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE VIRTUAL TABLE fts USING fts5(text, tokenize='unicode61')")
    conn.executemany("INSERT INTO fts(rowid, text) VALUES (?, ?)", ((i + 1, t) for i, t in enumerate(texts)))
    conn.commit()
    return conn


def keyword_rank(conn, question, cand):
    """Global ranking and its restriction to `cand`, best first."""
    match = fts_query(question)
    if not match:
        return [], []
    rows = conn.execute("SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ?", (match,)).fetchall()
    if not rows:
        return [], []
    arr = np.asarray(rows, dtype=np.float64)
    idx = arr[:, 0].astype(np.int64) - 1
    glob = idx[np.lexsort((idx, arr[:, 1]))]  # bm25: lower is better
    within = glob[np.isin(glob, cand)]
    return glob.tolist(), within.tolist()


def model_info(model, batch):
    import sentence_transformers
    import torch
    return {
        "name": MODEL_NAME,
        "max_seq_length": int(model.max_seq_length),
        "param_dtype": str(next(model.parameters()).dtype),
        "batch_size": batch,
        "seed": SEED,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def mem_available_gib():
    p = Path("/proc/meminfo")
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return None


def guard(stage, floor):
    avail = mem_available_gib()
    if avail is None:
        return
    print(f"  [mem] {stage}: MemAvailable={avail:.2f} GiB", flush=True)
    if floor > 0 and avail < floor:
        print(f"FATAL_MEMORY_FLOOR at {stage}: MemAvailable={avail:.2f} GiB < {floor}", flush=True)
        sys.exit(3)


def dense_config(name, fn, within, questions, cand, answers):
    out, hit10 = {}, None
    for k in KS:
        hit = np.array([hits_at(topk(fn(i), cand[i] if within else None, k).tolist(), answers[i], (k,))[k] for i in range(len(questions))], dtype=bool)
        out[f"@{k}"] = summarize(hit, questions)
        if k == KS[-1]:
            hit10 = hit
    out["ci95_rate@10"] = ci95(hit10)
    return out, hit10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="longmemeval_s_cleaned.json (read-only)")
    ap.add_argument("--cache-dir", required=True, help="where embeddings and the FTS index live")
    ap.add_argument("--out", default=None, help="results JSON (default: <cache-dir>/longmemeval_scope_results.json)")
    ap.add_argument("--reuse", action="store_true", help="reuse embeddings found in --cache-dir")
    ap.add_argument("--hybrid-exploratory", action="store_true", help="also report untuned RRF fusion (post-protocol, exploratory)")
    ap.add_argument("--mem-floor-gib", type=float, default=0.0, help="abort if MemAvailable drops below this (0 disables)")
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else cache / "longmemeval_scope_results.json"
    t_start = time.perf_counter()
    guard("start", args.mem_floor_gib)

    print("STAGE corpus", flush=True)
    session_texts, questions, n_slots = build_corpus(args.dataset)
    n_sess = len(session_texts)
    cand = [np.unique(np.asarray(q["slots"], dtype=np.int64)) for q in questions]
    answers = [set(q["answers"]) for q in questions]
    if not all(answers) or sum(q["answers_missing"] for q in questions):
        print("FATAL: a question has no resolvable answer session", flush=True)
        sys.exit(2)
    print(f"  unique sessions={n_sess} slots={n_slots} questions={len(questions)}", flush=True)
    guard("corpus", args.mem_floor_gib)

    have = all((cache / name).exists() for name in FILES.values())
    if args.reuse and have:
        info = json.loads((cache / FILES["model"]).read_text())
        head, qvec = np.load(cache / FILES["head"]), np.load(cache / FILES["queries"])
        chunks, chunk_session = np.load(cache / FILES["chunks"]), np.load(cache / FILES["chunk_session"])
        timings = {"head_embeddings": None, "chunking": None, "chunk_embeddings": None, "reused_arrays": True}
        print("STAGE embeddings: reused from cache", flush=True)
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(SEED)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"STAGE model: loading {MODEL_NAME} on {device}", flush=True)
        model = SentenceTransformer(MODEL_NAME, device=device)
        info = model_info(model, 128)
        guard("model-loaded", args.mem_floor_gib)
        head, t_head = embed(model, session_texts, 128)
        qvec, t_q = embed(model, [q["question"] for q in questions], 128)
        np.save(cache / FILES["head"], head)
        np.save(cache / FILES["queries"], qvec)
        (cache / FILES["model"]).write_text(json.dumps(info, indent=2, sort_keys=True))
        print(f"STAGE head embeddings: {head.shape} t={t_head + t_q:.1f}s", flush=True)
        guard("head-embeddings", args.mem_floor_gib)
        t0 = time.perf_counter()
        chunk_texts, chunk_session, token_counts = build_chunk_corpus(model.tokenizer, session_texts)
        t_pieces = time.perf_counter() - t0
        print(f"STAGE chunks: {len(chunk_texts)} t={t_pieces:.1f}s", flush=True)
        guard("chunks-built", args.mem_floor_gib)
        chunks, t_chunk = embed(model, chunk_texts, 128)
        print(f"STAGE chunk embeddings: {chunks.shape} t={t_chunk:.1f}s", flush=True)
        np.save(cache / FILES["chunks"], chunks)
        np.save(cache / FILES["chunk_session"], chunk_session)
        np.save(cache / FILES["chunk_tokens"], token_counts)
        del chunk_texts, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        timings = {"head_embeddings": round(t_head + t_q, 3), "chunking": round(t_pieces, 3), "chunk_embeddings": round(t_chunk, 3), "reused_arrays": False}
    token_counts = np.load(cache / FILES["chunk_tokens"])
    (cache / FILES["corpus"]).write_text(json.dumps({
        "n_sessions": n_sess,
        "session_slots": n_slots,
        "questions": [{k: q[k] for k in ("question_id", "qtype", "slots", "answers")} for q in questions],
    }))
    guard("arrays-ready", args.mem_floor_gib)

    counts = np.bincount(chunk_session, minlength=n_sess)
    if counts.min() == 0:
        print("FATAL: a session produced no chunk", flush=True)
        sys.exit(2)
    starts = np.searchsorted(chunk_session, np.arange(n_sess))

    def head_scores(i):
        return head @ qvec[i]

    def chunk_scores(i):
        sess = np.empty(n_sess, dtype=np.float32)
        np.maximum.reduceat(chunks @ qvec[i], starts, out=sess)
        return sess

    results, hits10 = {}, {}
    for name, fn, within in (("head_within", head_scores, True), ("head_global", head_scores, False), ("chunkmax_within", chunk_scores, True), ("chunkmax_global", chunk_scores, False)):
        results[name], hits10[name] = dense_config(name, fn, within, questions, cand, answers)
        print(f"STAGE eval {name}: hit@10={results[name]['@10']['rate']:.4f}", flush=True)

    t0 = time.perf_counter()
    fts_path = cache / "fts5.sqlite"
    if args.reuse and fts_path.exists():
        conn = sqlite3.connect(str(fts_path))
    else:
        conn = build_fts(session_texts, fts_path)
    t_fts = time.perf_counter() - t0
    kw_global, kw_within = [], []
    for i, q in enumerate(questions):
        g, w = keyword_rank(conn, q["question"], cand[i])
        kw_global.append(g)
        kw_within.append(w)
    conn.close()
    for name, ranks in (("keyword_within", kw_within), ("keyword_global", kw_global)):
        out = {}
        for k in KS:
            hit = np.array([hits_at(r, answers[i], (k,))[k] for i, r in enumerate(ranks)], dtype=bool)
            out[f"@{k}"] = summarize(hit, questions)
            if k == KS[-1]:
                hits10[name] = hit
        out["ci95_rate@10"] = ci95(hits10[name])
        results[name] = out
        print(f"STAGE eval {name}: hit@10={out['@10']['rate']:.4f}", flush=True)

    paired = {
        "chunkmax_global_minus_keyword_global@10": paired_delta(hits10["chunkmax_global"], hits10["keyword_global"]),
        "head_global_minus_keyword_global@10": paired_delta(hits10["head_global"], hits10["keyword_global"]),
        "chunkmax_within_minus_head_within@10": paired_delta(hits10["chunkmax_within"], hits10["head_within"]),
        "chunkmax_global_minus_chunkmax_within@10": paired_delta(hits10["chunkmax_global"], hits10["chunkmax_within"]),
    }

    exploratory = None
    if args.hybrid_exploratory:
        ex_hits = {name: [] for name in ("rrf_keyword_chunkmax", "rrf_keyword_chunkmax_head", "rrf_keyword_head")}
        for i, q in enumerate(questions):
            cm = topk(chunk_scores(i), None, POOL)
            hd = topk(head_scores(i), None, POOL)
            kw = np.asarray(kw_global[i][:POOL], dtype=np.int64)
            fused = {
                "rrf_keyword_chunkmax": rrf([kw, cm]),
                "rrf_keyword_chunkmax_head": rrf([kw, cm, hd]),
                "rrf_keyword_head": rrf([kw, hd]),
            }
            for name, ranked in fused.items():
                ex_hits[name].append(hits_at(ranked, answers[i], (10,))[10])
        exploratory = {
            "exploratory_post_protocol": True,
            "settings": {"rrf_k": RRF_K, "pool": POOL},
            "hit@10": {name: {"hits": int(sum(v)), "n": len(v), "rate": float(np.mean(v))} for name, v in ex_hits.items()},
            "paired_vs_keyword_global@10": {name: paired_delta(np.array(v, dtype=bool), hits10["keyword_global"]) for name, v in ex_hits.items()},
        }

    receipt = {
        "schema_version": 1,
        "benchmark": "LongMemEval-S cleaned — scope ladder",
        "status": "COMPLETE",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_involved": False,
        "content_published": False,
        "settings": {"chunk_budget": CHUNK_BUDGET, "ks": list(KS), "bootstrap_resamples": BOOT_N, "bootstrap_seed": BOOT_SEED, "seed": SEED, "fts_tokenizer": "unicode61", "fts_query": "OR of quoted tokens, bm25()"},
        "dataset": {"bytes": Path(args.dataset).stat().st_size, "sha256": sha256_file(args.dataset)},
        "script_sha256": sha256_file(Path(__file__)),
        "corpus": {
            "questions": len(questions),
            "session_slots": n_slots,
            "unique_sessions": n_sess,
            "haystack_candidates_median": float(np.median([len(c) for c in cand])),
            "chunks": int(chunk_session.shape[0]),
            "chunks_per_session_median": float(np.median(counts)),
        },
        "model": info,
        "timings_seconds": dict(timings, fts_build_or_open=round(t_fts, 3), total=round(time.perf_counter() - t_start, 3)),
        "arrays_sha256": {v: sha256_file(cache / v) for v in FILES.values() if (cache / v).exists() and v.endswith(".npy")},
        "results": results,
        "paired": paired,
        "exploratory_hybrid_rrf": exploratory,
    }
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"\n{'config':17s}{'@1':>8s}{'@3':>8s}{'@5':>8s}{'@10':>8s}")
    for name, r in results.items():
        print(f"{name:17s}" + "".join(f"{r[f'@{k}']['rate']:8.4f}" for k in KS))
    if exploratory:
        for name, r in exploratory["hit@10"].items():
            print(f"{name:28s} hit@10={r['rate']:.4f} (exploratory)")
    print(f"DONE results={out_path} total={receipt['timings_seconds']['total']}s", flush=True)


if __name__ == "__main__":
    main()
