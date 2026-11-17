#!/usr/bin/env python3
"""Merge del part hermes → hermes-hub/sidecar.db + hermes-hub/data/vectors/default.bin.

Aislado a propósito del merge.py del semantic-hub (8871, memorias m2m legacy +
chunks de repos): el hub de sesiones vive en su propio data-dir y puerto (8872),
así un re-merge + restart del hub de sesiones nunca toca el otro.

Formato del bin: u64 nrows | u64 ncols | f32 row-major (igual que merge.py).
"""
import argparse
import os
import sqlite3
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_lib import DIMS, pack_shard_bin

HERMES_HUB = "/mnt/d/splatdb/hermes-hub"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_merge(hub=HERMES_HUB, parts=("hermes",)):
    parts_dir = os.path.join(hub, "parts")
    master_path = os.path.join(hub, "sidecar.db")
    bin_dir = os.path.join(hub, "data", "vectors")
    os.makedirs(bin_dir, exist_ok=True)

    all_blobs = []
    # rebuild ATÓMICO (14-nov): escribir a .tmp + os.replace — antes el
    # os.remove() dejaba una ventana de crash sin sidecar mientras el bin
    # conservaba las filas viejas → hook sin join vectorial hasta 24h.
    tmp_path = master_path + ".tmp"
    if os.path.exists(tmp_path):
        # borra un .tmp VIEJO de una corrida previa, no un archivo vivo del
        # hub (esos van por os.replace abajo).
        os.remove(tmp_path)  # nosemgrep: non-atomic-file-replace
    con = sqlite3.connect(tmp_path)
    con.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, repo TEXT, "
                "path TEXT, kind TEXT, meta TEXT, text TEXT, vec BLOB)")
    con.execute("CREATE TABLE parts (part TEXT, first_rowid INTEGER, n INTEGER)")
    rowid = 0
    for part in parts:
        p = os.path.join(parts_dir, f"{part}.db")
        if not os.path.exists(p):
            # sqlite3.connect CREARÍA el archivo: un part inexistente dejaría
            # un .db vacío sin tabla chunks que envenena el drift-check del
            # smoke y cualquier merge futuro (17-nov EGM audit).
            raise SystemExit(f"ERROR: part inexistente: {p} — merge abortado")
        src = sqlite3.connect(p)
        rows = src.execute("SELECT rowid, repo, path, kind, meta, text, vec "
                           "FROM chunks ORDER BY rowid").fetchall()
        first = rowid + 1
        for _, repo, path, kind, meta, text, vec in rows:
            rowid += 1
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (rowid, repo, path, kind, meta, text, vec))
            all_blobs.append(bytes(vec))
        con.execute("INSERT INTO parts VALUES (?,?,?)", (part, first, len(rows)))
        src.close()
        log(f"  {part}: {len(rows)} chunks (rowids {first}..{rowid})")
    con.commit()
    con.close()

    # reemplazo del par sidecar+bin: DOS os.replace independientes — NO es
    # atómico como par, pero el ORDEN es el inocuo: sidecar primero (17-nov:
    # sidecar nuevo + bin viejo = chunks nuevos invisibles hasta el próximo
    # merge, los rowids viejos joinean igual; el orden inverso —bin primero—
    # dejaría índices alcanzables SIN texto = joins a _drops de verdad).
    os.replace(tmp_path, master_path)
    bin_path = os.path.join(bin_dir, "default.bin")
    blob = pack_shard_bin(all_blobs)
    bin_tmp = bin_path + ".tmp"
    with open(bin_tmp, "wb") as f:
        f.write(blob)
    os.replace(bin_tmp, bin_path)
    log(f"MASTER: {len(all_blobs)} chunks → {master_path} + "
        f"{bin_path} ({os.path.getsize(bin_path)/1e6:.1f} MB)")
    return len(all_blobs)


def main():
    ap = argparse.ArgumentParser(description="Merge del part hermes → sidecar + shard bin")
    ap.add_argument("--hub", default=HERMES_HUB)
    ap.add_argument("--parts", nargs="+", default=["hermes"],
                    help="parts en orden de merge (los primeros preservan sus "
                         "vector_idx; 'hermes' siempre primero)")
    args = ap.parse_args()
    run_merge(hub=args.hub, parts=tuple(args.parts))


if __name__ == "__main__":
    main()
