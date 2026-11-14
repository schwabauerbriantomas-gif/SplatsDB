#!/usr/bin/env python3
"""sac_recap — recap SAC del estado de una sesión Hermes vía splatDB /recap.

Lee los mensajes de una sesión desde el state.db del perfil (RO: copia a
/tmp, jamás abre la DB viva), arma el action log (tool_calls = comandos,
resultados de tool = observaciones, mensajes del usuario = notas) y lo
envía a POST /recap del hub. Devuelve el recap compacto (<400 palabras)
que permite retomar el trabajo con contexto mínimo — la variante agéntica
del experimento sac_falsif (ver ~/sac_falsif/STATE.md: SACDB 10/10, wall
162s med, 4.45× más rápido que recap por LLM).

Uso:
  python3 sac_recap.py --last                     # última sesión del perfil
  python3 sac_recap.py --session-id 20261114_042047_13591b44
  python3 sac_recap.py --last --tail 100 --out /tmp/recap.md
  python3 sac_recap.py --last --url http://127.0.0.1:8891   # serve scratch

Salida: recap a stdout (y a --out si se pasa). --json agrega stats.
"""
import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

DEFAULT_DB = Path.home() / ".hermes/profiles/deepharness/state.db"
DEFAULT_URL = "http://127.0.0.1:8872"
MAX_ACTIONS = 200          # últimas N acciones (el estado vivo es lo reciente)
MAX_ACTION_CHARS = 1500
MAX_NOTE_CHARS = 600


def load_session(db_path: Path, session_id: str | None, last: bool):
    """Abre el state.db en modo RO (URI, sin copia: 0ms vs 16s de copy)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    if session_id:
        row = cur.execute(
            "SELECT id, title FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        if not row:
            sys.exit(f"sesión {session_id} no encontrada en {db_path}")
    elif last:
        row = cur.execute(
            """SELECT s.id, s.title FROM sessions s
               JOIN messages m ON m.session_id = s.id
               GROUP BY s.id ORDER BY s.started_at DESC LIMIT 1"""
        ).fetchone()
        if not row:
            sys.exit("no hay sesiones con mensajes en la DB")
    else:
        sys.exit("especificar --session-id o --last")

    sid, title = row
    msgs = cur.execute(
        """SELECT role, content, tool_calls, tool_name FROM messages
           WHERE session_id=? AND active=1 AND content IS NOT NULL
           ORDER BY id""", (sid,)
    ).fetchall()
    con.close()
    return sid, title, msgs


def build_actions(msgs):
    """Convierte mensajes en el action log que espera POST /recap."""
    actions = []
    objective = ""
    for role, content, tool_calls, tool_name in msgs:
        text = (content or "").strip()
        if role == "user":
            # saltar ruido de plataforma: gateway JSON, compactions, notificaciones
            if (text.startswith("Gateway message origin")
                    or text.startswith("[CONTEXT COMPACTION")
                    or text.startswith("[IMPORTANT:")
                    or text.startswith("[ASYNC DELEGATION")
                    or text.startswith("[Background process")):
                continue
            if not objective:
                objective = text[:400]
            else:
                actions.append({"seq": len(actions) + 1, "kind": "note",
                                "text": text[:MAX_NOTE_CHARS]})
        elif role == "assistant" and tool_calls:
            try:
                calls = json.loads(tool_calls)
            except (TypeError, json.JSONDecodeError):
                calls = []
            for tc in calls if isinstance(calls, list) else []:
                fn = (tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args[:300]}
                if isinstance(args, dict):
                    cmd = args.get("command") or args.get("code") \
                        or args.get("path") or json.dumps(args)[:300]
                else:
                    cmd = str(args)[:300]
                actions.append({"seq": len(actions) + 1, "kind": "command",
                                "text": f"{fn.get('name', '?')}: {cmd}"[:MAX_ACTION_CHARS]})
        elif role == "tool":
            actions.append({"seq": len(actions) + 1, "kind": "observation",
                            "text": f"[{tool_name}] {text}"[:MAX_ACTION_CHARS]})
        # role=assistant sin tool_calls: razonamiento/respuestas — no es acción
    return objective, actions[-MAX_ACTIONS:]


def post_recap(url, objective, actions):
    payload = json.dumps(
        {"objective": objective or "Continuar la tarea en curso.",
         "actions": actions, "max_words": 400}
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/recap", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--session-id", help="session_key (ej 20261114_042047_13591b44)")
    ap.add_argument("--last", action="store_true", help="última sesión con mensajes")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--url", default=DEFAULT_URL, help="base del hub (default prod :8872)")
    ap.add_argument("--tail", type=int, default=MAX_ACTIONS, help="últimas N acciones")
    ap.add_argument("--out", type=Path, help="guardar recap además de stdout")
    ap.add_argument("--json", action="store_true", help="imprimir JSON completo")
    args = ap.parse_args()

    sid, title, msgs = load_session(args.db, args.session_id, args.last)
    objective, actions = build_actions(msgs)
    if args.tail != MAX_ACTIONS:
        actions = actions[-args.tail:]
    if not actions:
        sys.exit("la sesión no tiene acciones para compactar")

    out = post_recap(args.url, f"{title}. {objective}".strip(". "), actions)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out["recap"])
        print(f"\n[{sid}] palabras={out['words']} acciones={out['actions_in']}"
              f"→{out['actions_used']} server={out['elapsed_ms']:.1f}ms", file=sys.stderr)
    if args.out:
        args.out.write_text(out["recap"], encoding="utf-8")


if __name__ == "__main__":
    main()
