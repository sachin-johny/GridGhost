# GridGhost

External PCB auto-placement engine for KiCad. Data is extracted from a `.kicad_pcb`
once, optimized completely outside the editor, then written back.

> See [README.md](README.md) for the full feature list, pipeline diagram, cost
> function, and macro model. This file is concise context for an AI assistant:
> current state, where things live, the real CLI, and the load-bearing invariants.

## Project Status (Current)

**Last Updated:** 2026-08-04

- Full pipeline end-to-end: extract → cap classification/assignment → build macros →
  connector perimeter → net-aware initial placement → macro SA → iterative legalization → save.
- **Two pipelines coexist** (mid-refactor):
  - **Macro-first `place_v2`** (`place/pipeline.py`) — **default**. An IC + its decoupling
    caps move as one rigid `Macro` through every stage. Reached via the `place` command
    (`--macro-v2` is on by default).
  - **Legacy grid pipeline** (`engine/smart_placement.py` + `engine/annealer.py`) —
    retained behind `--no-macro-v2` for A/B comparison. Still functional.
- **152 tests across 11 files** (`python -m pytest tests/`). `test_phase1.py` (81 tests)
  covers the legacy engine + data model; the rest cover the macro pipeline. `test_overlap_regression.py`
  is currently disabled (0 collected).
- Bundled boards run clean: `th_sensor.kicad_pcb` (0 overlaps, 0 OOB); larger `cbb.kicad_pcb`
  completes the pipeline.

### Next Improvements
- Tune SA budget/cooling per board size (iterations already scale as `max(1500, 25×N)`).
- Polygon board outline + keepout zone support (currently rectangular outlines only).
- Improve dense-board legalization (residual overlaps on very dense boards).

### Known Issues
- Dense boards still benefit from SA/legalizer tuning (pipeline runs, placement not always optimal).
- Legalization can regress wirelength slightly to reach a legal (overlap-free) placement.
- Abacus legalizer (`--legalizer abacus`) is worse than the default heuristic on every bundled board — kept for comparison only (see `place/abacus_bridge.py` docstring).

---

## Architecture Overview

```
KiCad .kicad_pcb
       ↓
  [parsers] → BoardModel (JSON intermediate)
       ↓
  [assign] → classify + assign decoupling caps to ICs
       ↓
  [place/pipeline] → build rigid Macros → connectors → initial → SA → legalize
       ↓
  Updated .kicad_pcb  (+ placed-model JSON + positions JSON)
```

**Key design:** external optimization vs KiCad API integration — extract once, optimize
fully outside, apply back.

## Directory Structure (current)

```
auto_placer/
├── gridghost.py            # CLI entry point (place / extract / profiles)
├── config.py / config.json # Typed config + defaults
├── models/                 # board_model.py (BoardModel, Component, Net, Pad, BoardOutline),
│                           #   macro.py (Macro: leader + rigid followers)
├── parsers/                # kicad_parser.py, placement_writer.py
├── assign/assign_caps.py   # Cap classification (decoupling/bulk/coupling) + IC assignment
├── cost/                   # cost.py (α·HPWL + β·overlap + γ·boundary), chains.py, incremental.py
├── place/                  # Macro-first pipeline:
│                           #   pipeline.py, initial.py, connectors.py, cluster.py,
│                           #   sa.py, sa_polish.py, legalizer.py, abacus_bridge.py
├── engine/                 # Shared modules (net_clustering, cost_state, congestion,
│                           #   constraint_evaluator, group_moves, subcircuit_patterns,
│                           #   _pure_graph) + legacy-only (cost_function, moves, annealer,
│                           #   grid_placement, quadratic_placement, placement_prepass, smart_placement)
├── legalization/           # legalizer.py (component-level), abacus_legalizer.py,
│                           #   post_legalize.py, spatial_grid.py
├── profiles/board_profiles.py  # Cost weights + constraint rules per board type
├── utils/                  # courtyard.py, density.py, display.py
├── samples/sample_board.py # 18-component MCU peripheral sample board
└── tests/                  # test_*.py + run_all/dashboard/visualizer harness + test_pcbs/
```

> `engine/` is split: **shared** modules are imported by both pipelines (clustering,
> incremental cost, congestion, constraint eval); **legacy-only** modules serve only
> the `--no-macro-v2` path. Don't assume all of `engine/` is dead — check imports.

## Core Concepts

### Cost Function (macro pipeline)
`Total Cost = α·HPWL + β·Overlap + γ·Boundary`
- **HPWL** — Half-perimeter wirelength over all nets. **Power/ground nets are included**
  (key change vs legacy) — caps share rails with their IC, so power-net HPWL is the
  gradient signal keeping caps near their assigned IC during SA.
- **Overlap** — sum of pairwise macro bbox intersection areas (macros = rigid union bboxes).
- **Boundary** — linear distance penalty for bboxes outside the outline; edge connectors (intentional overhang) excluded.

The legacy path additionally applies a **δ·Constraints** term via `engine/cost_function.py`
+ `engine/constraint_evaluator.py` (decoupling proximity, crystal-MCU, thermal, etc.).

### Macro Model
A `Macro` = a leader component + rigidly-attached follower caps. Follower offsets are fixed
at construction (`find_cap_offset`, 8-direction fan bounded by `MAX_CAP_IC_GAP_MM = 4.0` mm
**edge-to-edge** gap). Translating/rotating the leader applies the same rigid transform to
followers, so the cap-IC gap can never grow at runtime. SA moves and the legalizer treat
macros as opaque rigid bodies.

### Component Types
`ic`, `capacitor`, `resistor`, `connector`, `crystal`, `generic` — drive placement heuristics,
spacing, and edge-connector detection (`is_edge_connector`).

### Coordinate System
- Internal: millimeters (float). KiCad file: mm (6 decimals = µm).
- Legalization grid: default 0.5 mm (`config.json` `legalization.grid_mm`), via `--grid-mm`.
- Rotation: KiCad clockwise-positive; pad `(at ...)` rotated with `cos/sin` (negated sin for CW).

---

## CLI Usage

```bash
python gridghost.py place <input.kicad_pcb> [options]   # auto-place (macro-v2 default)
python gridghost.py extract <input.kicad_pcb> [-o out.json]
python gridghost.py profiles
```

Key `place` options (defaults come from `config.json`; see README for the full table):

| Flag | Default | Notes |
|------|---------|-------|
| `-p/--profile` | `auto` | `auto` → `mcu_peripheral` if ICs detected, else `generic` |
| `-m/--margin` | `5.0` | `placement.margin` |
| `--grid-mm` | `0.5` | `legalization.grid_mm` |
| `--sa-iterations` | `max(1500, 25×N)` | scales with macro count |
| `--sa-reheat` | `3` | `annealer.reheat_count` |
| `--alpha/--beta/--gamma` | `1.0 / 25.0 / 8.0` | macro-v2 cost weights |
| `--seed` | `42` | SA/placement RNG seed (`--seed 0` = non-deterministic) |
| `--rudy-weight` | `0.0` | RUDY congestion penalty in SA cost (0 = off) |
| `--legalizer` | `heuristic` | one of `heuristic` / `abacus` / `sa_polish` |
| `--connector-mating-margin` | `5.0` | edge offset for perimeter connectors |
| `--macro-v2/--no-macro-v2` | on | toggle pipeline (`--no-macro-v2` = legacy grid) |
| `--dry-run` / `--debug-bbox` / `-v` | off | skip PCB write / draw bboxes / verbose cost |

---

## Key Data Model

**Component** — `ref, footprint, value, x, y, rotation, layer, width, height, courtyard_margin`
(0.25 mm), `pads, nets, is_fixed, component_type`. Properties: `bbox`, `effective_width/height`
(rotation-aware), `overlaps(other)`, `overlap_area(other)`, `is_edge_connector`.

**Macro** — `leader`, `followers`, `follower_offsets`. Methods: `Macro.alone(leader)`,
`Macro.with_caps(leader, caps)`, `translate(dx,dy,bounds)`, `set_pose(x,y,rot,bounds)`,
`apply_offsets()`, `overlaps(other)`, `overlap_area(other)`, `bbox` (union).

**BoardModel** — components, nets, board outline, `source_file`. Helpers: `get_component`,
`get_net`, `nets_for_component`, `components_on_net`; JSON I/O (`to_json`/`from_json`);
`stats()` (overlap counts, OOB, etc.).

---

## Important Implementation Notes

1. **Legalization is single-pass on the legacy path** — don't call it iteratively inside the
   optimizer loop; it corrupts the gradient/energy signal. The macro-v2 legalizer iterates
   push-apart ↔ clamp internally (up to 5 rounds) — call it once.
2. **Coordinate conversion** — always convert at I/O boundaries; internals are mm.
3. **Grid snapping** — `round(x / grid) * grid` (default grid 0.5 mm).
4. **Use `component_type`** — never stale placement-model fields.
5. **Macros are rigid** — caps follow their leader through every move and the legalizer; the
   cap-IC edge gap is enforced by construction (`MAX_CAP_IC_GAP_MM`), never recomputed at runtime.
6. **Determinism** — SA/initial placement respect `--seed` (default 42). Within one Python
   process runs are reproducible; across processes results can vary because Python's hash
   seed changes set/dict iteration order (see project memory: SA hash-seed determinism).
7. **`patches/` stays untracked** — never `git add patches/`; stage only source files.

## Dependencies

- numpy (vectorized math)
- Python 3.9+

## Design References

- CERC UTexas placement algorithms: https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf
- UCSD PCB placement survey: https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf
- RL for placement (17-21% lower post-routing wirelength): https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf
