# GridGhost

External PCB auto-placement engine for KiCad. Data is extracted from a `.kicad_pcb`
once, optimized completely outside the editor, then written back.

> See [README.md](README.md) for the full feature list, pipeline diagram, cost
> function, and macro model. This file is concise context for an AI assistant:
> current state, where things live, the real CLI, and the load-bearing invariants.

## Project Status (Current)

**Last Updated:** 2026-09-06

- Full pipeline end-to-end: extract → cap classification/assignment → build macros →
  connector perimeter → net-aware initial placement → macro SA → iterative legalization → save.
- **Two pipelines coexist** (mid-refactor):
  - **Macro-first `place_v2`** (`place/pipeline.py`) — **default**. An IC + its decoupling
    caps move as one rigid `Macro` through every stage. Reached via the `place` command
    (`--macro-v2` is on by default).
  - **Legacy grid pipeline** (`engine/smart_placement.py` + `engine/annealer.py`) —
    retained behind `--no-macro-v2` for A/B comparison. Still functional.
- **229 tests across 14 files** (`python -m pytest tests/`). `test_phase1.py` (87 tests)
  covers the legacy engine + data model; the rest cover the macro pipeline, including
  `test_nonrect_outline.py` (26 — polygon outlines), `test_cost.py` (23 — incl. per-pair
  clearance-target exemption tests), `test_keepout.py` (13 — internal
  keepout zones), and `test_pad_rotation.py` (7 — footprint-only rotation rewrite).
  `test_overlap_regression.py` is currently disabled (0 collected).
- Non-rectangular board outlines (notches, mouse-bites, interior holes) and rectangular
  internal keepout zones are now wired into the **default** macro-v2 path end-to-end
  (cost gradient, initial placement, legalizer clamps) — previously data-model/parser-only.
- **Hard-DRC min-gap pass** (2026-09-06) — after the main legalize loop, the default
  legalizer runs `enforce_drc_min_gap` (0.15 mm min gap): pushes sub-DRC passive-pair
  gaps up to the minimum, exempting TestPoint/mechanical pairs and rolling back any
  push that would create a new overlap. Runs unconditionally (sub-DRC gaps aren't
  overlaps, so the overlap pass can't catch them).
- Bundled boards run clean: `th_sensor.kicad_pcb` (0 overlaps, 0 OOB); larger `cbb.kicad_pcb`
  completes the pipeline.

### Next Improvements
- Tune SA budget/cooling per board size (iterations already scale as `max(1500, 25×N)`).
- Expose a `--keepout-weight` CLI flag (currently hardcoded to default to `γ`, not independently tunable).
- Improve dense-board legalization (residual overlaps on very dense boards).
- `--delta` (constraint-rule penalty) is wired but off by default (`0.0`) — evaluate turning it
  on by default now that propose()/total() symmetry is fixed (see IMPROVEMENTS §2.4 in commit history).

### Known Issues
- Dense boards still benefit from SA/legalizer tuning (pipeline runs, placement not always optimal).
- Legalization can regress wirelength slightly to reach a legal (overlap-free) placement.
- Abacus legalizer (`--legalizer abacus`) is worse than the default heuristic on every bundled board — kept for comparison only (see `place/abacus_bridge.py` docstring).
- `RAIL_PAD_SPREAD_THRESHOLD_MM = 3.0` mm (rigid-vs-rail-adjacent decap split) is calibrated
  against one board family (`test4`/U30) — watch for mis-splits on unseen BGA pinouts.

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
├── models/                 # board_model.py (BoardModel, Component, Net, Pad, BoardOutline —
│                           #   rectangle or polygon-with-holes; BoardModel.keepouts),
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
└── tests/                  # test_*.py + run_all/dashboard/visualizer/run_benchmark harness +
                            #   test_pcbs/ + external_boards/rl_pcb/
```

> `engine/` is split: **shared** modules are imported by both pipelines (clustering,
> incremental cost, congestion, constraint eval); **legacy-only** modules serve only
> the `--no-macro-v2` path. Don't assume all of `engine/` is dead — check imports.

## Core Concepts

### Cost Function (macro pipeline)
`Total Cost = α·HPWL + β·Overlap + γ·Boundary + γ·Keepout + rudy·RUDY + pin_density·PinDensity + cap_attraction·CapAttraction + clearance·Clearance`
- **HPWL** — Half-perimeter wirelength over all nets. **Power/ground nets are included**
  (key change vs legacy) — caps share rails with their IC, so power-net HPWL is the
  gradient signal keeping caps near their assigned IC during SA. `--exclude-nets` drops
  named nets (typically global GND/VCC, near-constant HPWL) from this term for SA speed;
  threaded through every `evaluate()` call (initial, per-move, final) and the incremental tracker.
- **Overlap** — sum of pairwise macro bbox intersection areas (macros = rigid union bboxes).
- **Boundary** — distance penalty for bboxes outside the outline, via `board.bbox_overflow()`
  (byte-identical to the old four-sided sum on a rectangle; polygon-aware on notched/hole
  boards). Edge connectors (intentional overhang) excluded.
- **Keepout** — total area of component bboxes intersecting `BoardModel.keepouts` (rectangular
  no-place zones). Weight defaults to `γ`; pass `keepout_weight=0.0` to drop the SA gradient
  (the legalizer's `_keepout_clamp` still evicts violators regardless, via a multi-pass push
  along the shallowest exit axis). Builds on the polygon/keepout data model from `8019d4d`.
- **RUDY** — wire-density congestion (each net's bbox spread uniformly across the grid
  cells it covers). Catches routing choke points where many nets' bboxes overlap — HPWL
  alone is blind to this. Default weight `1.0` (from `config.json` `annealer.rudy_weight`);
  pass `--rudy-weight 0` to disable.
- **Pin density** — signal-pin count per grid cell. Catches pin-escape congestion
  (dense clusters of small passives next to a QFN) that RUDY's wire-density model misses.
  Default weight `0.2` (from `config.json` `annealer.pin_density_weight`); pass
  `--pin-density-weight 0` to disable.
- **Cap attraction** — deadband-linear drift penalty (`max(0, dist − 5mm)`) on
  rail-adjacent freed caps toward their assigned IC. Root-cause fix for shared-rail
  cap drift (rail-bbox HPWL is flat w.r.t. a freed cap once the rail spans several ICs).
  Default weight `1.0`; pass `--cap-attraction-weight 0` to disable.
- **Clearance** — routing halo: pairwise `max(0, target − edge_gap)` charge where the
  pair's target is the **MIN of the two members'** per-type targets
  (`cost/cost.py:_clearance_target_for_pair`, table `_COMPONENT_CLEARANCE_TARGETS_MM`,
  default 1.0 mm). Exempt pairs (0 mm target): mechanical–mechanical
  (mounting_hole/fiducial/test_coupon, via `_is_overlap_exempt`) and any pair with a
  TestPoint member (footprint prefix `TestPoint`/`MeasurementPoint` — probe pads are
  intentionally pinned to IC pins, and the uniform target's false-positive charge
  fought HPWL's pull and lost). Gives SA a "close enough" floor between HPWL's
  monotone pull and the overlap term's cliff at touching (β is 0 the instant bboxes
  stop intersecting, so without this term SA parks components at 0.00mm gaps).
  Intersecting pairs charge exactly the target (β owns depth). Default weight `5.0`,
  auto-tapered by interior density (1.0× ≤ 0.45 → 0.25× ≥ 0.65); pass
  `--clearance-weight 0` to disable. The incremental tracker precomputes per-pair
  targets so the SA hot loop stays a dict lookup.

All routability signals (RUDY, pin density, clearance) are **default-on** so the placer
produces a routable result out of the box. The verbose report (`-v`) always shows
initial + final values, even when a term's weight is set to 0.

The legacy path additionally applies a **δ·Constraints** term via `engine/cost_function.py`
+ `engine/constraint_evaluator.py` (decoupling proximity, crystal-MCU, thermal, etc.).

### Macro Model
A `Macro` = a leader component + rigidly-attached follower caps. Follower offsets are fixed
at construction (`find_cap_offset`, 8-direction fan bounded by `MAX_CAP_IC_GAP_MM = 4.0` mm
**edge-to-edge** gap; offsets use `max(cap_w, cap_h)/2` on both axes — `apply_offsets()`
rotates the offset vector but not the cap, so a cap-height-based offset would land a
non-square cap INSIDE the leader after a 90°/270° leader rotation). Translating/rotating
the leader applies the same rigid transform to followers, so the cap-IC gap can never
grow at runtime. SA moves and the legalizer treat macros as opaque rigid bodies.

### Cap Assignment (`assign/assign_caps.py`)
One classification drives rigid gluing, cap attraction, *and* the legacy `--delta` decoupling-
proximity constraint — no duplicate logic elsewhere.
- **Decoupling (rigid)** — ≤ `DECAP_MAX_VALUE_F = 10µF`, and that (IC, rail) pair's pad spread
  on the IC is ≤ `RAIL_PAD_SPREAD_THRESHOLD_MM = 3.0mm` (a geometric "can this rail host a tight
  ring" test, not a per-IC cap count). All such caps become rigid `Macro` followers.
- **Decoupling (rail-adjacent)** — same value threshold, but pad spread > 3.0mm (e.g. a BGA rail
  landing on opposite corners). Standalone macros held near their IC by the cap-attraction term
  instead of rigid geometry. No per-IC ceiling — one IC can have rigid followers on one rail and
  rail-adjacent caps on another.
- **Bulk** — > 10µF or on a rail no IC shares. **Coupling** — signal-only nets, no power rail.
- Power-rail detection also matches hierarchical labels (`/Buck/VIN`) and KiCad auto-names
  (`Net-(U1-VIN)`) via trailing-token conventions, not just literal net names.

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
| `--alpha/--beta/--gamma` | `1.0 / 25.0 / 8.0` | macro-v2 cost weights (`γ` also defaults the keepout weight) |
| `--delta` | `0.0` | constraint-rule penalty weight (`engine/constraint_evaluator.py`: decoupling proximity, crystal-MCU, thermal, etc.); opt-in, `0` = current default behavior |
| `--seed` | `42` | SA/placement RNG seed (`--seed 0` = non-deterministic) |
| `--rudy-weight` | `1.0` (from config) | RUDY wire-density congestion penalty in SA cost (0 = off) |
| `--pin-density-weight` | `0.2` (from config) | Pin-density congestion penalty in SA cost (0 = off). Complements RUDY. |
| `--cap-attraction-weight` | `1.0` (from config) | cap→IC drift penalty on rail-adjacent freed caps (0 = off) |
| `--clearance-weight` | `5.0` (from config) | routing-halo pairwise clearance in SA cost (0 = off); density-tapered |
| `--exclude-nets` | *(none)* | net names dropped from HPWL during SA, e.g. `--exclude-nets GND +3V3` (speeds up SA on global power nets) |
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
8. **Pad rotation** — footprint rotation rewrites only the footprint's own `(at ...)`; per-pad
   `(at X Y r)` overrides are left alone. Rewriting both double-rotated asymmetric footprints
   (SOIC/QFN/offset-pin-1) — fixed, don't reintroduce a pad-level rewrite.
9. **Polygon outlines are opt-in by data, not by flag** — every polygon/keepout branch is gated
   on `board is not None and board.is_polygon` (or non-empty `board.keepouts`), so a plain
   rectangular board traces the exact pre-polygon code paths. New boundary/keepout logic must
   preserve this fallback rather than assuming rectangular geometry.
10. **DRC min-gap is a post-pass, not a cost weight** — `place/legalizer.py::enforce_drc_min_gap`
    runs after the legalize loop and pushes sub-0.15 mm gaps to the minimum, with rollback
    on overlap regression. Don't "fix" sub-DRC gaps by lowering a pair's clearance target:
    a lower target *weakens* the push-apart gradient (`max(0, target − gap)` shrinks for the
    same gap), so pairs end at the same gap or tighter — the inversion that motivated the
    hard pass (see `_COMPONENT_CLEARANCE_TARGETS_MM` comment in `cost/cost.py`). Tighten via
    a *higher* target or the hard pass, never a lower one.

## Dependencies

- numpy (vectorized math)
- Python 3.9+

## Design References

- CERC UTexas placement algorithms: https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf
- UCSD PCB placement survey: https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf
- RL for placement (17-21% lower post-routing wirelength): https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf
