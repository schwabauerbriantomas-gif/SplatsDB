#!/usr/bin/env python3
"""Reconcile the numbers of README.md and BENCHMARKS.md tables with their cited result files.

A number printed in a table is "traceable" when some numeric value of a result
file cited near that table equals it at the printed precision, or equals a
difference or ratio of two such values. Anything else must be listed in
bench-data/claims_untraceable.json, the reviewed list of numbers published
without a preserved artifact. The check fails when a new untraceable number
appears, so nothing unpreserved can be published silently.

    python benchmarks/reconcile_claims.py                   # check (exit 1 on new numbers)
    python benchmarks/reconcile_claims.py --update-baseline # accept the current list

Standard library only.

Limits: a table that cites no result file is not checked, and a number that
coincides with a difference or ratio of two stored values passes. The check
proves the absence of a source, never its presence.
"""
import bisect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "BENCHMARKS.md")
BASELINE = ROOT / "bench-data" / "claims_untraceable.json"
WINDOW = 12  # lines before and after a table where a source file may be cited
CITED = re.compile(r"`((?:bench-data|benchmarks|assets/charts/data)/[A-Za-z0-9_./-]+\.json)`")
CELL = re.compile(r"\*{0,2}([0-9][0-9,]*\.[0-9]+%?|[0-9][0-9,]*%)\*{0,2}")


def numeric_leaves(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from numeric_leaves(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from numeric_leaves(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def matchable(values):
    """Raw values plus every difference and ratio of two of them, sorted."""
    raw = sorted(set(values))
    if len(raw) > 600:
        return raw
    derived = set(raw)
    for i, a in enumerate(raw):
        for b in raw[i + 1:]:
            derived.add(abs(a - b))
            if a and b and abs(a / b) < 1e9:
                derived.add(a / b)
                derived.add(b / a)
    return sorted(derived)


def cell_matches(text, sorted_values):
    """True when the number printed in one table cell is traceable at its precision."""
    value = float(text.rstrip("%").replace(",", ""))
    decimals = text.rstrip("%").split(".")[1] if "." in text else ""
    tolerance = 0.5 * 10 ** (-len(decimals)) + 1e-12
    candidates = [(value, tolerance)]
    if text.endswith("%"):
        candidates.append((value / 100.0, tolerance / 100.0))
    for candidate, tol in candidates:
        index = bisect.bisect_left(sorted_values, candidate - tol)
        if index < len(sorted_values) and sorted_values[index] <= candidate + tol:
            return True
    return False


def table_rows(lines):
    """(first_line, last_line, row_texts) for every Markdown table in a document."""
    start = None
    for number, line in enumerate(lines, 1):
        if line.lstrip().startswith("|"):
            if start is None:
                start = number
        elif start is not None:
            yield start, number - 1, lines[start - 1:number - 1]
            start = None
    if start is not None:
        yield start, len(lines), lines[start - 1:]


def cited_files(lines, blocks):
    """Sources of each table: every citation is assigned to the table nearest to it.

    A citation placed under one table must not make the table after it be checked
    against a file that is not its source.
    """
    sources = [[] for _ in blocks]
    for number, line in enumerate(lines, 1):
        for path in CITED.findall(line):
            if not (ROOT / path).is_file():
                continue
            distances = []
            for index, (first, last, _) in enumerate(blocks):
                gap = 0 if first <= number <= last else min(abs(number - first), abs(number - last))
                distances.append((gap, index))
            gap, index = min(distances)
            if gap <= WINDOW and path not in sources[index]:
                sources[index].append(path)
    return sources


def untraceable_cells(document, root=None):
    root = ROOT if root is None else root
    lines = (root / document).read_text(encoding="utf-8").split("\n")
    missing = []
    blocks = list(table_rows(lines))
    sources = cited_files(lines, blocks)
    for (first, last, rows), files in zip(blocks, sources):
        if not files:
            continue
        values = []
        for path in files:
            values.extend(numeric_leaves(json.loads((root / path).read_text(encoding="utf-8"))))
        if not values:
            continue
        candidates = matchable(values)
        for row in rows[2:]:
            for raw in re.split(r"\s*\|\s*", row.strip().strip("|")):
                match = CELL.fullmatch(raw.strip())
                if match and not cell_matches(match.group(1), candidates):
                    missing.append(f"{document}:{match.group(1)}")
    return sorted(missing)


def main():
    update = "--update-baseline" in sys.argv[1:]
    found = sorted(set(cell for document in DOCS for cell in untraceable_cells(document)))
    listed = json.loads(BASELINE.read_text(encoding="utf-8"))["cells"] if BASELINE.is_file() else []
    new = sorted(set(found) - set(listed))
    stale = sorted(set(listed) - set(found))
    if update:
        BASELINE.write_text(json.dumps({"_note": "Numbers printed in a README.md or BENCHMARKS.md table next to a cited result file "
                                                  "that no cited file contains at the printed precision. Every entry is a published number "
                                                  "without a preserved artifact; review before adding one.",
                                         "cells": found}, indent=1) + "\n", encoding="utf-8")
        print(f"baseline written: {len(found)} untraceable cells")
        return 0
    print(f"tables checked in {', '.join(DOCS)}: {len(found)} untraceable cells, {len(listed)} listed in {BASELINE.relative_to(ROOT)}")
    for cell in new:
        print("NEW (no cited result file contains it):", cell)
    for cell in stale:
        print("listed but now traceable (remove from baseline):", cell)
    if new:
        print("FAILED: cite a result file that contains the number, or review and add it to the baseline with --update-baseline")
        return 1
    print("OK: every untraceable number is on the reviewed list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
