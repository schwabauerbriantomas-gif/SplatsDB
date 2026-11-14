#!/usr/bin/env python3
"""sac_recall_digest.py — complemento de sac_recap: qué memoria sirvió.

Lee el log de recalls (/mnt/d/WSL/orch/hermes-recalls.jsonl, escrito por el
hook de memoria desde 14-nov-2026) y el state.db del perfil (RO), detecta
uso post-inyección con el mismo detector del análisis SkillDuet (tokens
significativos del hit en la ventana de asistentes posteriores), y emite
un digest determinista por sesión: inyecciones, tasa de uso, calidad de
hits, warnings de clutter.

Uso:
  python3 sac_recall_digest.py --last                # última sesión con recalls
  python3 sac_recall_digest.py --session-id <KEY>
  python3 sac_recall_digest.py --last --with-recap   # recap SAC + digest juntos

Solo lectura. Sin LLM. Sin GPU. Fail-open con exit != 0 solo si no hay datos.
"""
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LOG = Path("/mnt/d/WSL/orch/hermes-recalls.jsonl")
STATE_DB = Path.home() / ".hermes/profiles/deepharness/state.db"
SAC_RECAP = Path(__file__).parent / "sac_recap.py"

STOP = set(("para como que del los las una uno unas unos con por sin sobre entre "
            "desde hasta durante segun esto esta este estos esas eso esa ese hay "
            "son fue era ser estar tiene tienen the and for with that this from "
            "have has was were are you your not but all can will what when where "
            "how why who its into than then them they been being").split())


def sig_toks(text, k=14):
    # solo tokens que ARRANCAN con letra: los que empiezan con dígito son
    # timestamps/ids (20261114, proc_...) y aparecen en cualquier output
    # de herramienta — falso positivo sistemático (medido 14-nov: inflaba
    # el uso a 88% con sids como fuente)
    ws = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_./-]{3,}", text.lower())
          if w not in STOP]
    seen, out = set(), []
    for w in sorted(ws, key=len, reverse=True):
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out


HDR = re.compile(
    r"\[sesión \S+ ([0-9a-f_]{12,20})\] \((\w+)\)[^\n]*\n|\[código ([^\s\]]+)/([^\s\]]+)\][^\n]*\n")


def block_snippets(api_content):
    """Snippets de los hits del bloque inyectado (metodología del análisis)."""
    idx = api_content.find("[Memoria vectorial")
    if idx < 0:
        return []
    block = api_content[idx:]
    ms = list(HDR.finditer(block))
    out = []
    for i, m in enumerate(ms):
        end = ms[i+1].start() if i+1 < len(ms) else min(len(block), m.end()+400)
        snip = " ".join(block[m.end():end].split())[:400]
        out.append(snip)
    return out


def get_api_content(sid, ts):
    """Mensaje user con bloque de memoria más cercano al ts del hook (±60s).
    El hook corre ~3-6s antes de la persistencia del mensaje (medido)."""
    db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    row = db.execute(
        "SELECT timestamp, api_content FROM messages WHERE session_id=? AND "
        "role='user' AND timestamp BETWEEN ? AND ? AND api_content LIKE "
        "'%Memoria vectorial%' ORDER BY ABS(timestamp-?) LIMIT 1",
        (sid, ts-60, ts+60, ts)).fetchone()
    db.close()
    return row[1] if row else None


def load_log():
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def session_windows(sid):
    """(ts, content assistant) de una sesión, desde state.db RO."""
    db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT timestamp, content FROM messages WHERE session_id=? "
        "AND role IN ('assistant','tool') ORDER BY timestamp", (sid,)).fetchall()
    db.close()
    return rows


def usage_for(entry, seq, api_content=None):
    """Detector de uso: tokens del snippet del hit (contenido real, no sid)
    presentes en la ventana post-inyección. Fuente de contenido: api_content
    (persistido por el gateway); fallback: nada — sin contenido no hay veredicto."""
    details = []
    t0 = entry["ts"]
    post = [c for ts, c in seq if ts > t0 and c][:4]
    window = " ".join(post).lower()
    if not window or not api_content:
        return 0, details
    snips = block_snippets(api_content)
    if not snips:
        return 0, details
    qtail = api_content[:api_content.find("[Memoria vectorial")].lower()
    used = 0
    for snip in snips:
        toks = sig_toks(snip)
        new = [t for t in toks if t not in qtail]
        if not new:
            details.append(("hit", 0.0))
            continue
        frac = sum(1 for t in new if t in window) / len(new)
        details.append(("hit", round(frac, 2)))
        if frac >= 0.2:
            used = 1
            break
    return used, details


def build_digest(entries, title_lookup):
    by_sid = defaultdict(list)
    for e in entries:
        by_sid[e.get("session_id") or "?"].append(e)
    lines = []
    lines.append("RECALL DIGEST — qué memoria se inyectó y qué mostró uso "
                 "(detector de tokens, post-hoc, desde 14-nov)")
    lines.append("")
    total_inj = sum(len(v) for v in by_sid.values())
    total_hits = sum(len(e.get("hits", [])) for e in entries)
    used_inj = 0
    for sid, evs in sorted(by_sid.items(), key=lambda kv: kv[1][-1]["ts"]):
        seq = session_windows(sid)
        title = title_lookup.get(sid, "")
        u_this = 0
        no_verdict = 0
        kinds = defaultdict(int)
        dists = []
        for e in evs:
            api = get_api_content(sid, e["ts"])
            u, _ = usage_for(e, seq, api)
            if api is None:
                no_verdict += 1
            else:
                u_this += 1 if u else 0
            for h in e.get("hits", []):
                kinds[h.get("kind", "?")] += 1
                if h.get("dist") is not None:
                    dists.append(h["dist"])
        used_inj += u_this
        med = f"{statistics.median(dists):.2f}" if dists else "-"
        lines.append(f"* `{sid[:24]}`{(' — ' + title[:50]) if title else ''}")
        nv = f" · sin veredicto {no_verdict}" if no_verdict else ""
        lines.append(f"  inyecciones {len(evs)} · con uso {u_this}{nv} · "
                     f"hits {sum(len(e.get('hits', [])) for e in evs)} · "
                     f"dist med {med} · tipos {dict(kinds)}")
    lines.append("")
    lines.append(f"TOTAL: {total_inj} inyecciones · {total_hits} hits · "
                 f"{used_inj} con uso detectado ({used_inj/max(1,total_inj):.0%})")
    n_sess = len(by_sid)
    dead = 0
    for sid2, evs in by_sid.items():
        seq2 = session_windows(sid2)
        if not any(usage_for(e, seq2, get_api_content(sid2, e["ts"]))[0]
                   for e in evs):
            dead += 1
    lines.append(f"sesiones con recalls: {n_sess} · sin uso detectado: {dead}")
    lines.append("(detector: tokens del snippet en ventana de 4 asistentes; "
                 "conservador — subestima por paráfrasis)")
    return "\n".join(lines), {"injections": total_inj, "used": used_inj,
                              "sessions": n_sess, "dead_sessions": dead}


def main():
    args = sys.argv[1:]
    with_recap = "--with-recap" in args
    sid = None
    last = "--last" in args
    if "--session-id" in args:
        sid = args[args.index("--session-id") + 1]
    entries = load_log()
    if not entries:
        print("sin recalls loggeados aún en", LOG)
        sys.exit(1)
    if sid:
        entries = [e for e in entries if e.get("session_id") == sid]
        if not entries:
            print(f"sesión {sid} sin recalls loggeados")
            sys.exit(1)
    elif last:
        last_ts = max(e["ts"] for e in entries)
        sid = next(e["session_id"] for e in entries if e["ts"] == last_ts)
        entries = [e for e in entries if e["session_id"] == sid]

    db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    titles = dict(db.execute("SELECT id, title FROM sessions").fetchall())
    db.close()

    digest, stats = build_digest(entries, titles)
    if with_recap and sid:
        import subprocess
        r = subprocess.run([sys.executable, str(SAC_RECAP),
                            "--session-id", sid], capture_output=True, text=True)
        print("=== SAC RECAP (qué pasó) ===")
        print(r.stdout.strip() or f"(sin recap: {r.stderr.strip()[:120]})")
        print()
        print("=== RECALL DIGEST (qué memoria sirvió) ===")
        print(digest)
    else:
        print(digest)


if __name__ == "__main__":
    main()
