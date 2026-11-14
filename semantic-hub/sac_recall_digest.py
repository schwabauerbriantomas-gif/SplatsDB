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
    """Snippets de los hits del bloque inyectado (metodología del análisis).
    Líneas tipo-header que no matchean el regex igual abren snippet nuevo:
    evita que su texto contamine el snippet del hit anterior."""
    idx = api_content.find("[Memoria vectorial")
    if idx < 0:
        return []
    block = api_content[idx:]
    ms = list(HDR.finditer(block))
    # posiciones de líneas header-like sin match (fallback anti-contaminación)
    matched = {m.start() for m in ms}
    for m2 in re.finditer(r"^\[(?:sesión|código) ", block, re.M):
        if m2.start() not in matched:
            ms.append(m2)
    ms.sort(key=lambda m: m.start())
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
            e = json.loads(line)
        except Exception:
            continue
        # schema-safe: JSON válido pero malformado se degrada, no crashea
        if not isinstance(e, dict):
            continue
        ts = e.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        out.append(e)
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
    # dedupe turn-level: el hook dispara por API-call, no por turno de
    # usuario (medido 14-nov: 1 qhash = hasta 29 fires) — 1 inyección real
    # = (session_id, qhash) único. Los cortes (arm=-1) nunca inyectaron:
    # se listan aparte, no cuentan como inyecciones.
    turns = {}
    cuts = []
    fires = 0
    for e in entries:
        fires += 1
        if e.get("arm") == -1:
            cuts.append(e)
            continue
        key = (e.get("session_id") or "?", e.get("qhash"))
        if key not in turns:
            turns[key] = e
    entries = list(turns.values())

    lines = []
    lines.append("RECALL DIGEST — qué memoria se inyectó y qué mostró uso "
                 "(detector de tokens, post-hoc, desde 14-nov)")
    lines.append("")
    total_inj = len(entries)
    total_hits = sum(len(e.get("hits", [])) for e in entries)
    used_inj = 0
    # agrupado por sesión: un header por sesión, filas de turno debajo
    by_sid = defaultdict(list)
    for (sid, _qh), e in turns.items():
        by_sid[sid].append(e)
    dead_sess = 0
    for sid, evs in sorted(by_sid.items(), key=lambda kv: min(e["ts"] for e in kv[1])):
        seq = session_windows(sid)
        title = title_lookup.get(sid, "")
        u_sess = 0
        lines.append(f"* `{sid[:24]}`{(' — ' + title[:50]) if title else ''}")
        for e in sorted(evs, key=lambda e: e["ts"]):
            api = get_api_content(sid, e["ts"])
            u, _ = usage_for(e, seq, api)
            if api is None:
                verdict = "sin veredicto"
            elif u:
                verdict = "CON USO"
                u_sess += 1
            else:
                verdict = "sin uso"
            kinds = defaultdict(int)
            dists = []
            for h in e.get("hits", []):
                kinds[h.get("kind", "?")] += 1
                if h.get("dist") is not None:
                    dists.append(h["dist"])
            med = f"{statistics.median(dists):.2f}" if dists else "-"
            qh = e.get("qhash") or "?"
            lines.append(f"  - turno `{qh}` · {verdict} · hits "
                         f"{len(e.get('hits', []))} · dist med {med} · "
                         f"tipos {dict(kinds)}")
        used_inj += u_sess
        if u_sess == 0:
            dead_sess += 1
    lines.append("")
    if cuts:
        cds = [c.get("cut_dist") for c in cuts if c.get("cut_dist") is not None]
        lines.append(f"CORTES banda medium (arm=-1, nunca inyectaron): "
                     f"{len(cuts)} — cut_dist {cds}")
    lines.append(f"TOTAL: {total_inj} turnos con inyección "
                 f"(de {fires} fires del hook) · {total_hits} hits · "
                 f"{used_inj} con uso detectado ({used_inj/max(1,total_inj):.0%})")
    n_sess = len(by_sid)
    lines.append(f"sesiones con recalls: {n_sess} · sin uso detectado: "
                 f"{dead_sess} (sesión sin NINGÚN turno con uso)")
    lines.append("(detector: tokens del snippet en ventana de 4 asistentes; "
                 "levemente más liberal que el análisis histórico por usar "
                 "top-3 snippets — acuerdo 87% validado 14-nov)")
    return "\n".join(lines), {"injections": total_inj, "used": used_inj,
                              "sessions": n_sess, "dead_sessions": dead_sess,
                              "fires": fires, "cuts": len(cuts)}


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
    if "--json" in args:
        # consumo programático (cron/agentes): sin parsear texto
        print(json.dumps(stats, ensure_ascii=False, indent=1))
        return
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
