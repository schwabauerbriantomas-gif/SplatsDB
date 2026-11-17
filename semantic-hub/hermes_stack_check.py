#!/usr/bin/env python3
"""hermes_stack_check.py — validación end-to-end de la configuración del
stack de memoria de Hermes (hook + hub + recap + digest).

Un comando, cero LLM: verifica que TODO lo que la memoria necesita esté
vivo y calibrado. Salida: checks PASS/FAIL/WARN + resumen. Exit 1 si hay
FAIL (para cron/agendado), 0 si todo PASS o solo WARN.

Uso: python3 hermes_stack_check.py [--json]
"""
import json
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOOK = Path.home() / ".hermes/agent-hooks/hermes-memory-recall.py"
RECALL_LOG = Path("/mnt/d/WSL/orch/hermes-recalls.jsonl")
STATE_DB = Path.home() / ".hermes/profiles/deepharness/state.db"
DIGEST = Path("/mnt/d/splatdb/semantic-hub/sac_recall_digest.py")
SAC_RECAP = Path("/mnt/d/splatdb/semantic-hub/sac_recap.py")
EMB = "http://127.0.0.1:8860"

results = []


def check(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def get_json(url, timeout=5):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def post_json(url, payload, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:160]


def main():
    t0 = time.time()
    print("hermes_stack_check — configuración del stack de memoria\n")

    # 1. embed server: modelo, device, dims, prompt asimétrico
    try:
        h = get_json(f"{EMB}/health")
        check("embed :8860 health (embeddinggemma-2 cuda 768d)",
              h.get("model") == "google/embeddinggemma-2"
              and h.get("device") == "cuda" and h.get("dims") == 768,
              json.dumps(h)[:100])
        st, r = post_json(f"{EMB}/v1/embeddings",
                          {"input": ["probe de configuración"],
                           "model": "embeddinggemma-2",
                           "prompt_name": "Retrieval-query", "dimensions": 768},
                          timeout=30)
        dim = len(r["data"][0]["embedding"]) if st == 200 else 0
        check("embed query con prompt Retrieval-query (768d)", st == 200 and dim == 768,
              f"status={st}, dims={dim}")
    except Exception as e:
        check("embed server :8860", False, str(e)[:100])

    # 2. hub sesiones :8872 — health + backend gpu + escala de score
    qv = None
    try:
        st, r = post_json(f"{EMB}/v1/embeddings",
                          {"input": ["trainer emeans splatdb"],
                           "model": "embeddinggemma-2",
                           "prompt_name": "Retrieval-query", "dimensions": 768},
                          timeout=30)
        qv = r["data"][0]["embedding"]
    except Exception:
        pass
    try:
        h = get_json("http://127.0.0.1:8872/health")
        ver = h.get("version", "?")
        check("hub sesiones :8872 health", h.get("status") == "ok", f"version={ver}")
    except Exception as e:
        check("hub sesiones :8872", False, str(e)[:100])
    backend = ""
    if qv is None:
        # embed caído: sin query vector los checks /search enrojecen con la
        # causa equivocada (400 sin embedding ≠ embed down) — WARN con causa
        # real (17-nov: el except-pass original ocultaba la falla padre)
        check("hub /search backend=gpu (bandas calibradas sobre d² gpu)",
              False, "salteado: embed :8860 caído (check 1)", warn=True)
        check("escala de score razonable (on-corpus ≤ 0.8)",
              False, "salteado: embed :8860 caído (check 1)", warn=True)
    else:
        try:
            st, r = post_json("http://127.0.0.1:8872/search",
                              {"query": "trainer emeans splatdb", "embedding": qv,
                               "top_k": 3}, timeout=10)
            if not isinstance(r, dict):
                raise ValueError(f"respuesta no-JSON del hub (status {st})")
            backend = str(r.get("backend") or "")
            best = min((x["score"] for x in r.get("results", [])), default=None)
            check("hub /search backend=gpu (bandas calibradas sobre d² gpu)",
                  backend == "gpu", f"backend={backend!r}, best={best}")
            check("escala de score razonable (on-corpus ≤ 0.8)",
                  best is not None and best <= 0.8, f"best={best}")
        except Exception as e:
            check("hub /search", False, str(e)[:100])
    try:
        st, _ = post_json("http://127.0.0.1:8872/search",
                          {"query": "probe sin embedding", "top_k": 3}, timeout=10)
        if st == 400:
            check("hub /search sin embedding → 400 (contrato fa1e528)", True)
        else:
            check("hub /search sin embedding → 400",
                  False, f"status={st}: hub VIEJO sin el commit fa1e528 — "
                  f"restart pendiente (el hook tiene guard backend, riesgo contenido)",
                  warn=True)
    except Exception as e:
        check("hub /search sin embedding", False, str(e)[:100])

    # 3. hub código :8873
    try:
        h = get_json("http://127.0.0.1:8873/health")
        check("hub código :8873 health", h.get("status") == "ok",
              f"version={h.get('version', '?')}")
    except Exception as e:
        check("hub código :8873", False, str(e)[:100])

    # 4. hook: constantes calibradas + guard backend + log schema
    try:
        src = HOOK.read_text()
        check("hook existe", True, f"{HOOK}")
        check("hook STRONG_SESS=0.55 / STRONG_CODE=0.70",
              "STRONG_SESS = 0.55" in src and "STRONG_CODE = 0.70" in src)
        check("hook sin banda medium (corte 14-nov, replay p=0.029)",
              "MEDIUM_SESS = 0.62" not in src)
        check("hook guard backend gpu (fail-closed CPU fallback)",
              'if _backend and _backend != "gpu":' in src)
        check("hook log con backend+drops (telemetría de join roto)",
              '"backend": _backend' in src and '"drops": _drops' in src)
    except Exception as e:
        check("hook", False, str(e)[:100])

    # 5. log de recalls: accesible + schema mínimo de la última línea
    try:
        last = RECALL_LOG.read_text().strip().split("\n")[-1]
        d = json.loads(last)
        keys = {"ts", "session_id", "qhash", "n_injected", "arm", "hits"}
        missing = keys - set(d.keys())
        check("recall log accesible + schema", not missing,
              f"última línea: arm={d.get('arm')}, n={d.get('n_injected')}"
              + (f", faltan {missing}" if missing else ""))
    except Exception as e:
        check("recall log", False, str(e)[:100])

    # 6. state.db RO accesible
    try:
        db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
        n = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        db.close()
        check("state.db RO accesible", n > 0, f"{n} mensajes")
    except Exception as e:
        check("state.db RO", False, str(e)[:100])

    # 7. digest corre (turn-level, cortes aparte)
    try:
        r = subprocess.run([sys.executable, str(DIGEST), "--last"],
                           capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0 and "turnos con inyección" in r.stdout
        tot = re.search(r"TOTAL: (\d+) turnos", r.stdout)
        check("digest corre (--last)", ok,
              (tot.group(0) if tot else r.stdout[:60]))
    except Exception as e:
        check("digest", False, str(e)[:100])

    # 8. sac_recap corre contra el hub vivo
    try:
        r = subprocess.run([sys.executable, str(SAC_RECAP), "--last",
                            "--tail", "3"], capture_output=True, text=True,
                           timeout=60)
        ok = r.returncode == 0 and "OBJECTIVE" in r.stdout
        # sac_recap imprime la línea de stats a stderr (17-nov: grepear stdout
        # nunca encontraba la latencia y el detalle mostraba el fallback)
        ms = re.search(r"server=([\d.]+)ms", r.stderr)
        check("sac_recap corre (POST /recap)", ok,
              f"server={ms.group(1)}ms" if ms else r.stdout[:60])
    except Exception as e:
        check("sac_recap", False, str(e)[:100])

    # 9. crons: ensure */10 + ciclo 9:15
    try:
        cron = subprocess.run(["crontab", "-l"], capture_output=True,
                              text=True).stdout
        check("cron ensure cada 10 min",
              bool(re.search(r"\*/10.*hermes-hub-ensure", cron)))
        check("cron ciclo diario 9:15", bool(re.search(r"15 9.*hermes-cycle", cron)))
    except Exception as e:
        check("crons", False, str(e)[:100])

    fails = sum(1 for s, _, _ in results if s == "FAIL")
    warns = sum(1 for s, _, _ in results if s == "WARN")
    dt = time.time() - t0
    print(f"\n{'PASS' if fails == 0 else 'FAIL'}: {len(results) - fails - warns}"
          f"/{len(results)} checks ({warns} WARN) — {dt:.1f}s")
    if "--json" in sys.argv:
        print(json.dumps([{"status": s, "name": n, "detail": d}
                          for s, n, d in results], ensure_ascii=False, indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
