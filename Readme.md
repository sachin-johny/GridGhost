<div align="center">
  <img src="logo.png" alt="GridGhost Logo" width="150" />
  <h2>GridGhost</h2>
  <p><strong>Macro-first PCB auto-placement engine for KiCad</strong></p>
  <p>Treats each IC + its decoupling caps as a single rigid body (a "macro"), places macros net-aware around their I/O attractors, optimizes via simulated annealing, and legalizes — all outside the KiCad editor.</p>
  <br/>
</div>

## Features

- **Macro abstraction** — an IC + its assigned decoupling caps form one `Macro` that moves rigidly through every stage of the pipeline. Caps rotate around the IC, translate with it, and stay within `MAX_CAP_IC_GAP_MM = 4.0` mm **edge-to-edge gap** of the leader — automatically, because the offset is fixed at construction time.
- **Cap classification + assignment** — caps are classified as *decoupling*, *bulk*, or *coupling* before IC assignment. Decoupling caps are distributed round-robin across ICs (one cap → exactly one IC) based on shared power rails and physical proximity.
- **Net-aware initial placement** — clusters components by net connectivity (reusing `engine/net_clustering`), then shelf-packs each cluster around the centroid of the fixed components + placed connectors it shares nets with. Per-macro gap is proportional to macro size (`max(min_gap, max_dim * 0.5)`), giving SA room to move.
- **Connector-attractor pull** — connectors are placed on the perimeter *first*, so interior placement can use their positions as attractors. Interior macros connected to a given connector get pulled toward it (`attractor_pull` blends with grid-cell fallback to prevent pile-up).
- **Simulated annealing** with four move operators:
  - **Translate** — rigid (dx, dy) on one macro
  - **Rotate** — 90/180/270 around the leader; caps orbit
  - **Swap** — exchange leader positions of two macros
  - **Displace-neighbor** — pick a macro, find a macro it overlaps, push the neighbor along the cheaper axis. Helps SA escape jammed configurations.
- **Calibrated T0** — initial temperature is set from overlap-clean sample moves only, so the HPWL gradient SA follows isn't drowned out by the β=25 overlap penalty.
- **Single cost function** — `α·HPWL + β·Overlap + γ·Boundary`, with **power nets included in HPWL** (the key change vs the legacy pipeline — power-net HPWL is what gives SA gradient signal to keep caps near their assigned IC).
- **End-mating connector orientation** — barrel jacks, USB, RJ45, HDMI, D-Sub and other end-mating connectors are oriented so the mating face points outward. Face-mating connectors (terminal blocks, pin headers, SMA) use the perpendicular-to-pad-column heuristic.
- **Iterative legalizer** — grid snap → push-apart → boundary clamp, iterated up to 5 rounds to settle the push-apart ↔ clamp cycle. Cap-IC distance violations are reported as a separate stat.
- **Pad rotation propagation** — when a footprint is rotated, the rotation delta is applied to each pad's `(at ...)` expression so copper layers match the courtyard orientation.
- **Edge-connector awareness** — horizontal/surface-mount connectors are placed on the board perimeter and excluded from out-of-bounds counts; vertical/THT connectors are treated as interior components.
- **Debug visualization** — `--debug-bbox` draws component bounding boxes and board outline on the `Dwgs.User` layer for visual inspection in KiCad.

## Pipeline (macro-first, `place_v2`)

```text
KiCad .kicad_pcb
       │
  ┌──────────┐   parse S-expressions → BoardModel
  │  parse   │   (components, pads, nets, board outline)
  └────┬─────┘
       │
  ┌──────────┐   1. classify caps (decoupling / bulk / coupling)
  │  build   │   2. assign decoupling caps to ICs (round-robin)
  │ macros   │   3. build Macro per IC leader (+ caps as followers)
  └────┬─────┘   4. connector macros are empty (leader-only)
       │
  ┌──────────┐   place connectors on perimeter FIRST so their
  │connector │   positions can act as attractors for interior
  │ perimeter│   placement
  └────┬─────┘
       │
  ┌──────────┐   cluster interior components by net connectivity,
  │ initial │   shelf-pack each cluster around its attractor point,
  │ interior│   per-macro gap proportional to size
  └────┬─────┘
       │
  ┌──────────┐   simulated annealing on interior macros only
  │   SA     │   (translate / rotate / swap / displace-neighbor)
  │          │   T0 calibrated from overlap-clean sample moves
  └────┬─────┘
       │
  ┌──────────┐   grid snap → push-apart → boundary clamp, iterated
  │ legalize │   reports residual overlaps, boundary failures, and
  │          │   cap-IC distance violations
  └────┬─────┘
       │
  Updated .kicad_pcb  (+  positions JSON)
```

## CLI Usage

```bash
# Auto-place a board (macro-first pipeline is the default)
python gridghost.py place <input.kicad_pcb> [options]

# Extract board data to JSON (no placement)
python gridghost.py extract <input.kicad_pcb> [-o output.json]

# List available board profiles
python gridghost.py profiles
```

### Place options

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--margin` | `5.0` | Board edge margin in mm (`config.json` `placement.margin`) |
| `--grid-mm` | `0.5` | Legalization grid pitch in mm (`config.json` `legalization.grid_mm`) |
| `--sa-iterations` | `max(1500, 25×N)` | SA iterations; scales with macro count when unset (min 1500) |
| `--sa-reheat` | `3` | Number of SA reheat rounds (`config.json` `annealer.reheat_count`) |
| `--alpha` | `1.0` | HPWL weight |
| `--beta` | `25.0` | Overlap penalty weight |
| `--gamma` | `8.0` | Boundary penalty weight |
| `--seed` | `42` | SA RNG seed (deterministic runs) |
| `--connector-mating-margin` | `5.0` | Edge offset for perimeter connectors |
| `--macro-v2` / `--no-macro-v2` | on | Toggle macro-first pipeline. Default on. `--no-macro-v2` falls back to the legacy grid pipeline (`engine/smart_placement.py`), retained for A/B comparison. |
| `--dry-run` | off | Don't write PCB output file |
| `--debug-bbox` | off | Draw bounding boxes on Dwgs.User layer |
| `-v`, `--verbose` | off | Print per-stage cost breakdown |

### Examples

```bash
# Auto-place with defaults
python gridghost.py place board.kicad_pcb

# Bigger SA budget for dense boards
python gridghost.py place board.kicad_pcb --sa-iterations 3000 --sa-reheat 4

# Tighter legalization grid
python gridghost.py place board.kicad_pcb --grid-mm 0.5

# Dry run with verbose cost breakdown + debug bboxes
python gridghost.py place board.kicad_pcb -v --dry-run --debug-bbox

# Extract board data without placing
python gridghost.py extract board.kicad_pcb -o board_model.json
```

## Cost Function

```text
Total Cost = α · HPWL + β · Overlap + γ · Boundary
```

- **HPWL** — Half-perimeter wirelength over all nets. **Power and ground nets are included** — this is the key change vs the legacy pipeline. Caps share power rails with their ICs, so power-net HPWL is the gradient signal that keeps caps near their assigned ICs during SA.
- **Overlap** — Sum of pairwise macro bbox intersection areas. Macros are treated as rigid rectangles (leader + follower union bbox).
- **Boundary** — Linear distance penalty for components whose bbox extends outside the board outline. Edge connectors (intentional overhang) are excluded.

## Macro Model

A `Macro` is the atomic unit of placement. Construction:

1. **Leader**: a component (typically an IC) that's free to move.
2. **Followers**: caps assigned to that IC. Each follower has a *fixed offset* in leader-local coordinates, chosen at construction time by `find_cap_offset` — an 8-direction fan search at increasing edge-to-edge spacings (0.5–4.0 mm) that:
   - Rejects slots where the cap overlaps the leader or any already-placed sibling.
   - Bounds the cap's **edge-to-edge gap** to the leader by `MAX_CAP_IC_GAP_MM = 4.0` mm (the spacing itself). The gap — not the center-to-center distance — is what governs decoupling effectiveness; center-distance is size-dependent and broke every cap on ICs larger than ~9 mm.
   - Falls back to the first leader-clear slot (smallest gap) if no slot is sibling-free.

Because offsets are fixed, the cap-IC gap can **never grow at runtime** — translating or rotating the leader applies the same rigid transform to followers. The gap is enforced by construction.

Move operators and the legalizer all operate on macros as opaque rigid bodies. Caps never get separated from their IC.

## Cap Assignment

`assign/assign_caps.py` classifies each capacitor and assigns decoupling caps to ICs:

- **Decoupling** (≤ `DECAP_MAX_VALUE_F = 1e-6` F, i.e. ≤ 1 µF) — distributed round-robin across ICs. Each cap goes to exactly one IC (the cap-to-IC map has no duplicates). Assignment prefers ICs sharing the cap's power rail; falls back to nets-shared and physical proximity.
- **Bulk** (> 1 µF, on a power rail) — stays standalone, not assigned to any IC.
- **Coupling** (in signal path, e.g. AC-coupling caps) — stays standalone.

A cap assigned to an IC is **only** in that IC's macro — never as a standalone macro. This is critical: if a cap were in two macros, push-apart could move the standalone cap away from its IC.

## Project Structure

```text
GridGhost/
├── gridghost.py                   # CLI entry point
├── config.py                      # Typed config loader (dataclasses)
├── config.json                    # Default config (margin, SA, legalization params)
│
├── models/
│   ├── board_model.py             # BoardModel, Component, Net, Pad, BoardOutline
│   └── macro.py                   # Macro: leader + rigid followers
│                                  # find_cap_offset; MAX_CAP_IC_GAP_MM = 4.0 (edge-to-edge)
│
├── parsers/
│   ├── kicad_parser.py            # S-expression .kicad_pcb parser
│   └── placement_writer.py        # Write placements back to .kicad_pcb / JSON
│
├── assign/
│   └── assign_caps.py             # Classify caps (decoupling / bulk / coupling)
│                                  # and assign decoupling caps to ICs
│
├── cost/
│   ├── cost.py                    # Macro-v2 cost: α·HPWL (incl. power rails)
│   │                              # + β·overlap + γ·boundary
│   ├── chains.py                  # Signal-flow chain grouping for cost/SA
│   └── incremental.py             # Incremental cost delta for macro SA
│
├── place/
│   ├── pipeline.py                # 5-stage orchestrator (build → connector →
│   │                              # initial → SA → legalize)
│   ├── initial.py                 # Net-aware clustered shelf-pack with
│   │                              # connector-attractor pull
│   ├── connectors.py              # Perimeter placement with pad-based rotation
│   ├── cluster.py                 # Net clustering + shelf-pack helpers for initial
│   ├── sa.py                      # Macro-aware SA (translate / rotate / swap /
│   │                              # displace-neighbor); calibrated T0
│   ├── sa_polish.py               # Overlap-weighted SA legalizer strategy (--legalizer sa_polish)
│   ├── legalizer.py               # Grid snap → push-apart → boundary clamp,
│   │                              # iterated up to 5 rounds
│   └── abacus_bridge.py           # Adapter from rigid Macros to the row-based
│                                  # Abacus legalizer (--legalizer abacus)
│
├── engine/                        # Shared + legacy modules.
│   ├── net_clustering.py          # Hypergraph clustering — reused by place/cluster.py
│   ├── cost_state.py              # Incremental cost state — reused by cost/, place/
│   ├── congestion.py              # RUDY wire-density + pin-density congestion maps — reused by cost/, place/sa
│   ├── constraint_evaluator.py    # Constraint penalty evaluation — reused by cost_state
│   ├── group_moves.py             # Macro/group move primitives — reused widely
│   ├── subcircuit_patterns.py     # Subcircuit pattern detection — reused by clustering
│   ├── _pure_graph.py             # Louvain graph core — reused by net_clustering
│   ├── cost_function.py           # Legacy cost function (used by --no-macro-v2 path)
│   ├── moves.py                   # Legacy SA move operators (--no-macro-v2)
│   ├── annealer.py                # Legacy SA engine (--no-macro-v2)
│   ├── placement_prepass.py       # Legacy decap pre-place (--no-macro-v2)
│   ├── grid_placement.py          # Legacy grid / force-directed placement (--no-macro-v2)
│   ├── quadratic_placement.py     # Legacy quadratic placer (--no-macro-v2)
│   └── smart_placement.py         # Legacy grid pipeline (--no-macro-v2)
│
├── legalization/
│   ├── legalizer.py               # Component-level legalizer (grid snap / push-apart /
│   │                              # boundary clamp), used by both pipelines
│   ├── abacus_legalizer.py        # Row-based Abacus DP legalizer
│   ├── post_legalize.py           # Post-legalization HPWL recovery (cell slide / pair swap)
│   └── spatial_grid.py            # Spatial hash grid for overlap acceleration
│
├── profiles/
│   └── board_profiles.py          # Cost weights + constraint rules per board type
│
├── utils/
│   ├── courtyard.py               # Courtyard/bbox helpers for the parser
│   ├── density.py                 # Pack-density target + board-area checks
│   └── display.py                 # CLI formatting (summary, cost, component table)
│
├── samples/
│   └── sample_board.py            # 18-component MCU peripheral sample board
│
├── tests/
│   ├── test_macro.py              # Macro model: rigid moves, bounds, cap gap, bbox
│   ├── test_assign_caps.py        # Cap classification + IC assignment
│   ├── test_cost.py               # HPWL / overlap / boundary cost
│   ├── test_incremental_cost.py   # Incremental cost delta
│   ├── test_chains.py             # Signal-flow chain grouping
│   ├── test_connector_placement.py# Perimeter connector placement + rotation
│   ├── test_placement_spread.py   # Placement spread / coverage
│   ├── test_abacus_bridge.py      # Macro → Abacus legalizer adapter
│   ├── test_sa_polish.py          # SA-polish legalizer strategy
│   ├── test_overlap_regression.py # (currently disabled — 0 collected)
│   ├── test_phase1.py             # 81 tests: legacy pipeline + data model
│   ├── run_all.py / dashboard.py / visualizer.py   # Batch run + result dashboard harness
│   └── test_pcbs/                 # Bundled .kicad_pcb boards + reference JSONs
│
├── patches/                       # (untracked) benchmark scripts + result JSONs
│
├── logo.png
├── CLAUDE.md                      # AI assistant context
└── README.md                      # This file
```

## Data Model

### Component

| Field | Type | Description |
|-------|------|-------------|
| `ref` | str | Reference designator (e.g. "U1", "R3") |
| `footprint` | str | Footprint library ID (e.g. "Package_QFP:LQFP-48") |
| `value` | str | Component value (e.g. "STM32F103C8T6") |
| `x`, `y` | float | Position in mm |
| `rotation` | float | Rotation in degrees (0, 90, 180, 270) |
| `layer` | str | "top" or "bottom" |
| `width`, `height` | float | Bounding box dimensions in mm |
| `courtyard_margin` | float | Courtyard margin in mm (default 0.25) |
| `bbox_offset_x`, `bbox_offset_y` | float | Offset from footprint origin to bbox center |
| `pads` | list[Pad] | Pad list with relative positions and net assignments |
| `nets` | list[str] | Net names connected to this component |
| `is_fixed` | bool | If True, position should not be changed by optimizer |
| `component_type` | str | One of: `ic`, `capacitor`, `resistor`, `connector`, `crystal`, `generic` |

Key properties: `bbox`, `effective_width/height` (rotation-aware), `overlaps(other)`, `overlap_area(other)`, `is_edge_connector`.

### Macro

| Field | Type | Description |
|-------|------|-------------|
| `leader` | Component | The IC (or standalone leader) |
| `followers` | list[Component] | Decoupling caps (rigidly attached) |
| `follower_offsets` | list[(dx, dy)] | Each cap's offset in leader-local coords at rotation=0 |

Key methods: `alone(leader)` (class method), `with_caps(leader, caps)` (class method), `translate(dx, dy, bounds)`, `set_pose(x, y, rot, bounds)`, `apply_offsets()`, `overlaps(other)`, `overlap_area(other)`, `bbox` (union property).

### BoardModel

Central data structure with components, nets, and board outline. Provides lookup helpers (`get_component`, `get_net`, `nets_for_component`, `components_on_net`), JSON serialization (`to_json`, `from_json`), and summary statistics (`stats`).

## Coordinate System

- **Internal**: millimeters (float)
- **KiCad file**: millimeters (float, 6 decimal places = micrometer precision)
- **Legalization grid**: configurable, default 0.5 mm via `--grid-mm` (from `config.json`)
- **Rotation**: KiCad clockwise-positive convention; pad positions use `cos/sin` with negated sin for CW rotation

## Dependencies

- **Python 3.9+**
- **numpy** — vectorized math

## Testing

```bash
# Full suite
python -m pytest tests/

# Macro-first pipeline tests
python -m pytest tests/test_macro.py tests/test_assign_caps.py tests/test_cost.py \
                   tests/test_incremental_cost.py tests/test_chains.py \
                   tests/test_connector_placement.py tests/test_placement_spread.py \
                   tests/test_abacus_bridge.py tests/test_sa_polish.py

# Legacy Phase 1 suite (data model, parser, legacy engine)
python -m pytest tests/test_phase1.py
```

The suite is **152 tests across 11 files** (run `python -m pytest tests/`). Breakdown: `test_phase1.py` (81 — legacy pipeline + data model), `test_cost.py` (12), `test_macro.py` (11), `test_chains.py` (9), `test_connector_placement.py` (10), `test_assign_caps.py` (8), `test_placement_spread.py` (9), `test_sa_polish.py` (5), `test_abacus_bridge.py` (4), `test_incremental_cost.py` (3). `test_overlap_regression.py` is currently disabled (0 collected). The macro-first pipeline is covered by:
- **Macro model**: rigid translation, bounds revert, rotation propagation, `MAX_CAP_IC_GAP_MM` (edge-to-edge cap-IC gap) enforcement, bbox union, overlap detection
- **Cap assignment**: power-net detection, single-IC assignment, round-robin distribution, determinism
- **Cost**: HPWL (2-pin, 3-pin, with/without power), macro overlap area, boundary, edge-connector exclusion, evaluate() returns all components

## Design References

- CERC UTexas placement algorithms: https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf
- UCSD PCB placement survey: https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf
- RL for placement (17-21% lower post-routing wirelength): https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf
