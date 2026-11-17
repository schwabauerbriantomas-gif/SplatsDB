#!/usr/bin/env python3
"""check_dead_code.py — gate CI: funciones module-level sin llamador.

AST stdlib-only (diseño del job python-tools: sin pip deps). Heurística
por nombre sobre TODOS los tools del directorio (la misma que validó el
barrido manual 14-nov, que encontró el stats huérfano del digest).

Falsos positivos aceptados y permitidos vía DEAD_CODE_ALLOW: entry
points llamados desde CLI, schedulers o código externo.
"""
import ast
import re
import sys
from pathlib import Path

# entry points legítimos que ningún otro tool referencia por nombre
DEAD_CODE_ALLOW = {"main"}

here = Path(__file__).parent
tools = sorted(here.glob("*.py"))
if not tools:
    print("no hay tools para analizar")
    sys.exit(0)

sources = {p: p.read_text() for p in tools}
all_src = "\n".join(sources.values())

fail = []
n_defs = 0
for p, tree in ((p, ast.parse(src)) for p, src in sources.items()):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_defs += 1
            name = node.name
            if name.startswith("_") or name in DEAD_CODE_ALLOW:
                continue
            # referencias por nombre en TODOS los tools; <=1 = solo la def
            # bordes de palabra: 'recall' no debe 'vivir' dentro de cada
            # mención de recall_any/recall_fts (17-nov: el conteo por
            # subcadena daba falsos negativos estructurales, justo el caso
            # común de este repo: recall*, merge*, check*)
            if len(re.findall(rf"\b{re.escape(name)}\b", all_src)) <= 1:
                fail.append(f"{p.name}:{node.lineno} función sin llamador: {name}()")

if fail:
    print("CÓDIGO MUERTO (funciones module-level sin llamador):")
    for f in fail:
        print(" ", f)
    print(f"\nSi es un entry point externo, agregalo a DEAD_CODE_ALLOW "
          f"({here.name}/check_dead_code.py).")
    sys.exit(1)
print(f"dead-code gate OK: {n_defs} funciones module-level en "
      f"{len(tools)} tools, 0 huérfanos")
