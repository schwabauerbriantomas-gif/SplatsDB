# SplatsDB chart system

Four standalone HTML documents and their PNG renders share one measured-data
style. The visual reference is `../logo-splatsdb.svg`: `#8ca3d4` for primary
series, `#34508e` for secondary series, dark navy surfaces, circular/network
motifs, and flat geometry without decorative shadows. Status colors are reserved
for verdicts; they do not encode vendor or model preference.

## Chart map

| Public PNG | Standalone HTML | Committed measured source | Scope |
|---|---|---|---|
| `../splatsdb-faiss-comparison.png` | `faiss.html` | `../../bench-data/benchmark_results_hardware.json` | Historical v2.5.0 run; exact k=10 and approximate k=64 stay separate |
| `../splatsdb-longmemeval.png` | `longmemeval.html` | `../../bench-data/longmemeval_full_results.json`, `longmemeval_splatdb_native.json`, `longmemeval_baseline_results.json` | NumPy reference and native pipeline are not interchangeable |
| `../splatsdb-m3-context-rerun.png` | `m3-rerun.html` | `data/m3-rerun.json` | GO only under the predeclared protocol; earlier NO_GO remains visible |
| `../splatsdb-m3-context.png` | `m3-historical.html` | `data/m3-historical.json` | Historical NO_GO by coverage; values and verdict unchanged |

The two M3 JSON files are minimal measured subsets of the local `REPORT.json`
receipts. Raw conversations and credentials are not published. The full public
methodology remains in `../../docs/validation/`.

## Rebuild

From the repository root:

```bash
python3 assets/charts/build.py           # regenerate HTML and manifest.json
python3 assets/charts/build.py --render  # also render PNGs; needs Playwright + Chrome
python3 assets/charts/validate.py        # data, links, PNG dimensions, hashes, visible text
```

The HTML contains no JavaScript and no remote fonts. PNG renders are 1800×1400
at device scale 1. A working local Chrome/Chromium is required only for PNG
rendering; the HTML itself opens offline in a browser.

## Editing rules

- Change measured values only in their source JSON; never paint a new number
  into a template.
- Keep the logo palette and the same panel, bar, label, badge and footer system.
- Keep run date, k/search regime, pipeline attribution and limitations visible.
- Never silently replace a historical verdict with a later result.
- After rendering, run `validate.py` and inspect every PNG at full size for
  clipping, low contrast, overlaps and missing footers before publishing.
