# Tests & Benchmarking

Test suites, measurement harnesses, benchmark boards, and a unified
orchestrator that runs everything and produces an HTML dashboard with
per-board visualizations.

## Quick start — the unified harness

The fastest way to run everything is the **unified orchestrator**:

```bash
python tests/run_all.py                       # run all phases
python tests/run_all.py --skip-external       # skip the slow external-boards phase
python tests/run_all.py --only cbb test4      # only specific boards
python tests/run_all.py --rudy-weight 0.3     # enable RUDY in SA
python tests/run_all.py --skip-unit-tests --skip-overlap-regression  # just placement + viz
```

It runs five phases in order, collects all results, and writes a
self-contained HTML dashboard to `tests/output/<run_id>/`:

1. **Unit tests** — `python -m pytest tests/` (falls back to standalone
   `main()` if pytest is missing). Fast correctness gate (~80s).
2. **Placement pipeline** — runs `place_v2` in-process on every board
   in `tests/test_pcbs/`. Captures HPWL, overlaps, density, RUDY,
   cap-IC distances, component-type breakdown, and per-phase timing.
3. **CLI smoke test** — runs `python gridghost.py place` end-to-end on
   each board via subprocess. Catches CLI-only bugs the in-process
   path misses.
4. **Overlap regression** — runs `tests/test_overlap_regression.py`
   (self-report vs independent scan). The standing check from the
   evaluation fix.
5. **External boards** — same as Phase 2 but on
   `tests/external_boards/rl_pcb/*.kicad_pcb` (9 real-world boards).
   Adds ~10 min but catches real-world edge cases.

**Outputs** (under `tests/output/<run_id>/`):
- `dashboard.html` — the main deliverable. Open in any browser.
- `results.json` — raw metrics (for programmatic comparison).
- `<board>.svg` — per-board SVG (also inlined in the dashboard).
- `<board>_placed.kicad_pcb` — placed output (open in KiCad).
- `unit_tests.log`, `cli_smoke_<board>.log` — raw phase logs.

### Dashboard contents

The dashboard is a single self-contained HTML file (no external CSS/JS)
with:

- **Header** — run ID, timestamp, git commit, summary cards (boards,
  passing, total overlaps, total wall-time).
- **Phase summaries** — one card per phase with pass/fail status,
  passed/failed counts, duration.
- **KPI table** — one row per board, columns for components, ICs, caps,
  overlaps, Edge.Cuts present, density, HPWL, RUDY peak, SA time,
  legalize time, total time.
- **Aggregate charts** — overlaps per board (bar), timing breakdown
  (stacked bar), component-type distribution (stacked bar), cap → IC
  distance histogram.
- **Per-board detail** — inline SVG visualization + metric chips +
  overlap-pair list (collapsible) + cap-IC distance stats.

### Visualization features

The SVG visualization (rendered by `tests/visualizer.py`) shows:

- Board outline (Edge.Cuts) as a solid black rectangle.
- Margin ring (light grey fill between outline and interior bbox).
- Interior bbox (dashed green) — the region SA + legalizer clamp to.
- All components colored by type (ICs red, caps blue, resistors green,
  connectors orange, mounting holes grey cross, generic light grey).
- Overlap pairs highlighted with a red fill + red border.
- Decoupling cap → IC leader lines (thin grey dashed).
- Connector pad dots (small black circles) so overhang is visible.
- Mounting hole cross marks.
- Per-component labels for ICs and large components (ref + size).
- Legend in the top-right corner.
- Overlap count badge in the top-left.

### What to look for in the dashboard

When reviewing a run, scan top-to-bottom:

1. **Header summary cards** — if "Total overlaps" is non-zero or
   "Passing" is less than total boards, drill into the KPI table.
2. **Phase cards** — every phase should be PASS. FAIL in unit tests
   means a regression in core functionality. FAIL in placement_test_pcbs
   means at least one board has overlaps. FAIL in CLI smoke means the
   CLI itself is broken (arg parsing, file write, hash-seed re-exec).
   FAIL in overlap_regression means the legalizer's self-report
   disagrees with an independent scan — the original symptom of
   Findings 1 & 2.
3. **KPI table** — sort by "Overlaps" column. Red rows have overlaps;
   green rows are clean. "Edge.Cuts" should be ✓ for every board
   (Finding 1 fix). "Density" should sit between 0.20 and 0.55
   (matching `target_pack_density`).
4. **Aggregate charts**:
   - *Overlaps per board* — any red bar is a board to investigate.
   - *Timing breakdown* — SA dominates (orange). If parse (cyan) is
     >10% of total, the parser may have a regression.
   - *Component-type distribution* — sanity check the parser. A board
     with 0 caps but 5 ICs suggests cap detection is broken.
   - *Cap → IC distance* — most caps should be in the 0-8mm buckets
     (green). Caps in the 12+ bucket (red) violate the decoupling
     proximity rule.
5. **Per-board sections** — open the SVG. Overlapping components are
   outlined in red with a light red fill. Decoupling caps are connected
   to their IC by a thin grey dashed line. Connector pad dots show
   where copper lands. The "⚠ N overlaps" badge in the top-left gives
   the quick verdict.

### Timing expectations

Typical wall-clock per phase on a modern laptop:

| Phase | Test boards (6) | + External boards (9) |
|-------|----------------:|----------------------:|
| Unit tests | ~80-100s | — |
| Placement pipeline | ~30-60s | +5-10 min |
| CLI smoke test | ~30-60s | (skipped by default for external) |
| Overlap regression | ~60-90s | — |
| Dashboard generation | <2s | — |
| **Total** | **~3-5 min** | **~10-15 min** |

For daily use, `--skip-external` brings it to ~5 min. For quick
iteration on a single board, `--only cbb --skip-unit-tests
--skip-overlap-regression --skip-cli-smoke` runs in ~25s.

### Troubleshooting

**"Phase: Placement Test Pcbs — FAIL" with `overlaps=-1`**
The placement pipeline raised an exception. Check the per-board
"error" field in `results.json`. Common causes: malformed `.kicad_pcb`
input, OOM on very large boards, or a regression in `place_v2`.

**"Phase: Cli Smoke — FAIL"**
The CLI itself is broken. Open `cli_smoke_<board>.log` to see stdout.
Common causes: missing dependency (numpy), PYTHONHASHSEED re-exec loop,
or an arg-parsing regression in `gridghost.py`.

**"Phase: Overlap Regression — FAIL"**
The legalizer's self-reported overlap count disagrees with an
independent re-parse-and-scan of the output file. This is the standing
check from the placement-evaluation fix — see `test_overlap_regression.py`'s
docstring. Open `overlap_regression.log` to see which board(s) failed.

**SVG renders but is empty / missing components**
Check that the board's `*_placed.kicad_pcb` was actually written. If
`apply_placement` failed mid-write, the model may have positions but
the file may be missing components. Re-run with `--only <board>` to
isolate.

**Dashboard opens but charts are squished / unreadable**
The dashboard uses CSS grid with `minmax(380px, 1fr)` for charts. If
your browser window is narrower than 760px, charts stack vertically
and may look cramped. Widen the window or zoom out.

### Standalone SVG visualization

The visualizer can be run on any placed `.kicad_pcb` without going
through the full harness:

```bash
# Render a placed board to SVG
python tests/visualizer.py tests/output/run_xxx/cbb_placed.kicad_pcb --out cbb.svg

# Larger SVG, no labels (for thumbnails)
python tests/visualizer.py cbb_placed.kicad_pcb --out cbb.svg --width 1200 --no-labels

# Hide decap leader lines and pad dots (cleaner look for slides)
python tests/visualizer.py cbb_placed.kicad_pcb --out cbb.svg --no-decaps --no-pads
```

Flags: `--width` (pixels, default 800), `--title`, `--no-labels`,
`--no-pads`, `--no-decaps`.

### Programmatic access to results

Each run writes `results.json` alongside `dashboard.html`. Structure:

```json
{
  "run_id": "run_20250726_120000",
  "timestamp": "20250726_120000",
  "git_commit": "7199a7b...",
  "repro_cmd": "python tests/run_all.py ...",
  "phases": {
    "unit_tests":           {"status": "pass", "n_total": 140, "n_passed": 139, ...},
    "placement_test_pcbs":  {"status": "fail", "n_total": 6, "n_passed": 4, ...},
    "cli_smoke":            {"status": "pass", ...},
    "overlap_regression":   {"status": "pass", ...},
    "placement_external":   {"status": "skipped", ...}
  },
  "boards": [
    {
      "name": "cbb",
      "source": "test_pcbs",
      "n_components": 68,
      "n_ics": 4,
      "n_caps": 12,
      "component_types": {"ic": 4, "capacitor": 12, "resistor": 30, ...},
      "overlaps": 0,
      "overlap_pairs": [],
      "edge_cuts_present": true,
      "density": 0.48,
      "hpwl": 6106.5,
      "rudy_peak": 0.458,
      "rudy_penalty": 0.0,
      "cap_ic_distances": [3.2, 4.1, 5.0, ...],
      "timing": {"parse": 0.15, "sa": 23.5, "legalize": 0.0, "total": 23.97, "write_reparse": 0.12},
      "placed_pcb_path": "tests/output/run_xxx/cbb_placed.kicad_pcb"
    },
    ...
  ]
}
```

Compare two runs by diffing their `results.json` files (the `model`
object is stripped from the JSON — only serializable metrics remain).

## Running individual test suites

All `test_*.py` files are collected by **pytest** (their `test_*` functions
live at module level):

```bash
pytest tests/                        # full suite
pytest tests/test_chains.py -v       # one file
pytest tests/test_placement_spread.py -k collapse   # one test
```

The older suites (`test_phase1.py`, `test_cost.py`, `test_macro.py`,
`test_assign_caps.py`) also keep a standalone `main()` harness, so they can be
run directly without pytest:

```bash
python tests/test_cost.py
python tests/test_macro.py
python tests/test_assign_caps.py
python tests/test_phase1.py
```

The newer regression suites (`test_chains.py`, `test_connector_placement.py`,
`test_placement_spread.py`) are pytest-only.

`test_overlap_regression.py` is a **standalone harness** (not pytest-collected).
Run it directly or via the `--skip-...` flags above:

```bash
python tests/test_overlap_regression.py
```

## Test files

| File | What it covers | Style |
|------|----------------|-------|
| `test_phase1.py` | Data model, parser, net clustering, cost function, profiles, SA config, group moves — the broad Phase 1–6 functional suite | standalone + pytest |
| `test_cost.py` | HPWL (clique/star), macro overlap, boundary penalty, `evaluate()` breakdown, power-net exclusion | standalone + pytest |
| `test_macro.py` | `Macro` rigid-body primitives: `alone`/`with_caps`, `translate`/`set_pose`, bounds revert, bbox union, overlap area | standalone + pytest |
| `test_assign_caps.py` | Cap→IC assignment: power/ground detection, round-robin distribution, multi-rail tie-break, determinism | standalone + pytest |
| `test_chains.py` | Signal-flow chain detector (Issue 3): synthetic-graph detection + chain pull-together metric on `cbb` | pytest |
| `test_connector_placement.py` | Connector edge centering (Issue 2): `set_bbox_center`/`rotated_bbox_offset` unit tests + edge group centering & flush placement on `cbb`/`cbbwO` | pytest |
| `test_placement_spread.py` | Interior spread / center-collapse guard (Issue 1): `weighted_attractor_target` per-net weighting + spread floor on `cbb`/`cbbwO` + OOB=0 across boards | pytest |
| `test_overlap_regression.py` | Self-reported overlap count vs independent re-parse-and-scan. The standing check from the placement-evaluation fix. | standalone only |

`test_chains.py`, `test_connector_placement.py`, and `test_placement_spread.py`
are the regression guards for `PLACEMENT_FIX_PLAN.md` (Issues 1–3).

## Harness modules

| File | What it does |
|------|--------------|
| `run_all.py` | Unified orchestrator — runs all 5 phases, generates dashboard. The primary entry point. |
| `dashboard.py` | HTML dashboard generator. Self-contained output (no external CSS/JS). |
| `visualizer.py` | Inline SVG visualizer for placed BoardModel. Also runnable standalone: `python tests/visualizer.py <placed.kicad_pcb> --out board.svg`. |
| `run_benchmark.py` | Comprehensive EE benchmark (legacy). Drives CLI, mines stdout, writes CSV/JSON. Kept for back-compat. |
| `measure_placement_v2.py` | Legacy in-process benchmark using `engine.smart_placement` (not the current macro-v2 default). |
| `dump_placement.py` | Legacy per-component diagnostics. |

## Measurement & benchmark harnesses (legacy)

These pre-date `run_all.py` and are kept for back-compat. `run_all.py`
replaces their common use cases.

| Script | Pipeline | What it reports |
|--------|----------|-----------------|
| `run_benchmark.py` | CLI (`gridghost.py place`) | Comprehensive EE benchmark: stdout metric mining, rotation histogram, decap & connector edge distances, density hotspots, top nets by HPWL, SA replicates with min/median/max, CSV + JSON to `tests/output/`. Use this for cross-board comparisons. |
| `measure_placement_v2.py` | Legacy (in-process) | Phase 4.4 KPI table: HPWL, RUDY congestion, runtime, constraint violations, variance across N seeds; markdown or human-readable. |
| `dump_placement.py` | Legacy (in-process) | Per-component diagnostics: position, distance from center, OOB/keepout flags — for diagnosing centroid shift and OOB regressions at component granularity. |

```bash
python tests/run_benchmark.py --boards cbb cbbwO --skip-greedy
python tests/measure_placement_v2.py --seeds 5 --markdown
python tests/dump_placement.py cbb
```

> **Pipeline note.** `run_benchmark.py` drives the real CLI (macro-first
> pipeline by default). `measure_placement_v2.py` and `dump_placement.py`
> drive the legacy `engine.smart_placement` pipeline in-process and are kept
> for the Phase 4.4 KPI table and per-component detail; they do not reflect
> the current CLI default.

## Subdirectories

### `test_pcbs/`
The in-repo benchmark boards:

- `cbb.kicad_pcb` / `cbbwO.kicad_pcb` — multi-cluster boards with edge
  connectors (the primary regression targets for `PLACEMENT_FIX_PLAN.md`).
- `test4` … `test6`, `th_sensor` — smaller boards used across the suites.

Each board has matching `*_positions.json` and `*_placed_model.json`
artifacts (emitted by `placement_writer.py`). `cbb_placed.kicad_pcb` and
`cbbwO_placed.kicad_pcb` are example placed outputs.

### `debug/`
Ad-hoc diagnostic scripts (not part of the test suite, kept for investigation):

- `diagnose_fix_plan.py` — measures the two `PLACEMENT_FIX_PLAN.md` regression
  metrics (interior spread %, max edge-centering offset) across boards using
  the macro-first `place_v2` pipeline.
- `run_pipeline_v2.py` / `run_all_boards_v2.py` — end-to-end runs of the
  macro-first pipeline on `cbb` / all boards with quality reporting.
- `diag_test4_caps.py` — inspects cap population (values, footprints, nets,
  IC sharing) on a board.

### `external_boards/`
Real-world boards from external datasets, used for broader benchmarking:

- `external_boards/rl_pcb/` — 9 real `.kicad_pcb` files from the
  [LukeVassallo/RL_PCB](https://github.com/LukeVassallo/RL_PCB) dataset
  (MIT-licensed; see its `README.md` for attribution and citation).

### `output/`
Generated at runtime by `run_all.py` (dashboards, per-run JSON, SVGs, placed
PCBs, logs) and `run_benchmark.py` (legacy). Not checked in.

