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
- **Routability-aware cost function** — `α·HPWL + β·Overlap + γ·Boundary + γ·Keepout + rudy·RUDY + pin_density·PinDensity + cap_attraction·CapAttraction + clearance·Clearance`, with **power nets included in HPWL** (the key change vs the legacy pipeline — power-net HPWL is what gives SA gradient signal to keep caps near their assigned IC). RUDY, pin-density, cap-attraction and clearance are default-on so the placer produces a routable result out of the box. See [Cost Function](#cost-function) below.
- **Non-rectangular board outlines** — polygon outlines (connector notches, mouse-bites, castellated edges) and interior holes/keepout zones are honored end-to-end: initial placement, SA's boundary gradient, and the legalizer all clamp to the true outline instead of its outer AABB. Plain rectangular boards trace the exact pre-polygon code paths (zero behavior change).
- **Internal keepout zones** — rectangular no-place zones (mounting holes, board-edge fab notes, etc.) are penalized in the cost function and actively evicted by the legalizer's multi-pass keepout clamp, which pushes an offending macro out along its shallowest exit axis.
- **End-mating connector orientation** — barrel jacks, USB, RJ45, HDMI, D-Sub and other end-mating connectors are oriented so the mating face points outward. Face-mating connectors (terminal blocks, pin headers, SMA) use the perpendicular-to-pad-column heuristic.
- **Iterative legalizer** — grid snap → push-apart → boundary clamp, iterated up to 5 rounds to settle the push-apart ↔ clamp cycle. Cap-IC distance violations are reported as a separate stat. A post-pass **hard-DRC min-gap check** then pushes sub-0.15 mm passive-pair gaps up to the DRC minimum — TestPoint and mechanical-feature pairs are exempt (their tight placement is intentional), and any push that would create a new overlap is rolled back. Alternate strategies (`abacus`, `sa_polish`) are selectable via `--legalizer`.
- **Pad rotation propagation** — footprint rotation rewrites only the footprint's own `(at ...)` expression; per-pad `(at X Y r)` overrides are left untouched, avoiding the double-rotation that previously corrupted asymmetric footprints (SOIC/QFN/offset-pin-1 parts).
- **Edge-connector awareness** — horizontal/surface-mount connectors are placed on the board perimeter and excluded from out-of-bounds counts; vertical/THT connectors are treated as interior components.
- **Net exclusion for SA speed** — `--exclude-nets` drops near-constant-HPWL global nets (board-wide GND/VCC) from SA's per-move cost evaluation, cutting evaluation cost without changing what's optimized.
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
  │ legalize │   + hard-DRC min-gap pass (0.15 mm); reports residual
  │          │   overlaps, boundary failures, and cap-IC violations
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
| `--gamma` | `8.0` | Boundary penalty weight (also the default keepout-overlap weight) |
| `--delta` | `0.0` | Constraint-rule penalty weight (decoupling proximity, crystal-MCU, thermal grouping/separation, etc., via `engine/constraint_evaluator.py`). Opt-in — `0` preserves current behavior. |
| `--rudy-weight` | `1.0` | RUDY wire-density congestion penalty (0 = off) |
| `--pin-density-weight` | `0.2` | Pin-escape congestion penalty; complements RUDY (0 = off) |
| `--cap-attraction-weight` | `1.0` | Deadband-linear drift penalty (`max(0, dist − 5mm)`) holding rail-adjacent freed caps near their assigned IC (0 = off) |
| `--clearance-weight` | `5.0` | Pairwise routing-halo clearance charge (`max(0, 1mm − edge_gap)`) between macros; density-tapered (0 = off) |
| `--exclude-nets` | *(none)* | Net names to drop from HPWL during SA (e.g. `--exclude-nets GND +3V3`) — speeds up SA on boards with near-constant, board-wide power nets |
| `--legalizer` | `heuristic` | Overlap-resolution strategy: `heuristic` (grid snap/push-apart/clamp), `abacus` (row-based DP, kept for comparison only), or `sa_polish` (overlap-weighted SA pass) |
| `--seed` | `42` | SA RNG seed (deterministic runs) |
| `--connector-mating-margin` | `5.0` | Edge offset for perimeter connectors |
| `--macro-v2` / `--no-macro-v2` | on | Toggle macro-first pipeline. Default on. `--no-macro-v2` falls back to the legacy grid pipeline (`engine/smart_placement.py`), retained for A/B comparison. |
| `--dry-run` | off | Don't write PCB output file |
| `--debug-bbox` | off | Draw bounding boxes on Dwgs.User layer |
| `-v`, `--verbose` | off | Print per-stage cost breakdown, including routability-term initial/final values even when a term's weight is 0 |

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
Total Cost = α·HPWL + β·Overlap + γ·Boundary + γ·Keepout
             + rudy·RUDY + pin_density·PinDensity
             + cap_attraction·CapAttraction + clearance·Clearance
```

- **HPWL** — Half-perimeter wirelength over all nets. **Power and ground nets are included** — this is the key change vs the legacy pipeline. Caps share power rails with their ICs, so power-net HPWL is the gradient signal that keeps caps near their assigned ICs during SA. `--exclude-nets` can drop specific board-wide nets (e.g. global GND) from this term to speed up SA without changing what it optimizes.
- **Overlap** — Sum of pairwise macro bbox intersection areas. Macros are treated as rigid rectangles (leader + follower union bbox).
- **Boundary** — Distance penalty for components whose bbox extends outside the board outline, computed via the outline's `bbox_overflow()` so it's identical on rectangles and gradient-correct on polygon/notched boards. Edge connectors (intentional overhang) are excluded.
- **Keepout** — Total area of component bboxes intersecting internal keepout zones (mounting holes, fab no-place areas). Weight defaults to `γ` (the boundary weight). Even with this term's SA gradient off, the legalizer's `_keepout_clamp` still actively evicts macros parked inside a keepout, pushing them out along the shallowest exit axis.
- **RUDY** — Wire-density congestion: each net's bounding-box "traffic" spread uniformly across the grid cells it spans, scaled by local component density (0.3× sparse boards, 1.2× dense boards, 1.0× otherwise) so amplification doesn't overshoot already-congested regions. Catches routing choke points HPWL alone is blind to. Default weight `1.0`.
- **Pin density** — Signal-pin count per grid cell (power pins excluded, target adapts to board density). Catches pin-escape congestion — dense clusters of small passives crowding a QFN's pins — that RUDY's wire-density model misses. Default weight `0.2`.
- **Cap attraction** — Deadband-linear drift penalty, `max(0, center_dist − 5mm)`, on *rail-adjacent* freed caps toward their assigned IC. Root-cause fix for shared-rail cap drift: on a rail spanning several ICs the rail's net bbox is already board-wide, so HPWL alone gives a freed cap zero gradient to stay near its IC. Default weight `1.0`.
- **Clearance** — Pairwise routing-halo charge between macro pairs: `max(0, target − edge_gap)`, where the pair's target is the **min of the two members'** per-type targets (default 1.0 mm). Pairs are exempt when both leaders are mechanical features (mounting holes, fiducials, test coupons) or when either member is a TestPoint — probe pads are intentionally pinned to IC pins, and the uniform target charged them for exactly the tight placement the design wants. HPWL's pull and the overlap term's cliff-at-touching otherwise let SA park components at a 0.00 mm gap with no room for a trace; intersecting pairs charge exactly the target so `β` stays the sole "depth" charger. Auto-tapered by interior density (1.0× ≤ 0.45 density, 0.25× ≥ 0.65). Default weight `5.0`.

All four routability terms (Keepout, RUDY, pin density, clearance) plus cap attraction are **default-on**. Pass the matching `--*-weight 0` flag to disable any of them; `-v` always reports each term's initial and final value, even at weight 0.

## Macro Model

A `Macro` is the atomic unit of placement. Construction:

1. **Leader**: a component (typically an IC) that's free to move.
2. **Followers**: caps assigned to that IC. Each follower has a *fixed offset* in leader-local coordinates, chosen at construction time by `find_cap_offset` — an 8-direction fan search at increasing edge-to-edge spacings (0.5–4.0 mm) that:
   - Rejects slots where the cap overlaps the leader or any already-placed sibling.
   - Bounds the cap's **edge-to-edge gap** to the leader by `MAX_CAP_IC_GAP_MM = 4.0` mm (the spacing itself). The gap — not the center-to-center distance — is what governs decoupling effectiveness; center-distance is size-dependent and broke every cap on ICs larger than ~9 mm.
   - Uses `max(cap_w, cap_h)/2` as the cap's half-extent on **both** axes, so the chosen gap survives leader rotation: offsets rotate with the leader but caps don't rotate with it, and a cap-height-based offset would land a non-square cap inside the leader after a 90°/270° rotation.
   - Falls back to the first leader-clear slot (smallest gap) if no slot is sibling-free.

Because offsets are fixed, the cap-IC gap can **never grow at runtime** — translating or rotating the leader applies the same rigid transform to followers. The gap is enforced by construction.

Move operators and the legalizer all operate on macros as opaque rigid bodies. Caps never get separated from their IC.

## Cap Assignment

`assign/assign_caps.py` classifies each capacitor and assigns decoupling caps to ICs. There are four classes:

- **Decoupling (rigid)** (≤ `DECAP_MAX_VALUE_F = 10e-6` F, i.e. ≤ 10 µF) — real decoupling caps whose pins on the IC land in one tight physical cluster (max pairwise pad distance on that net ≤ `RAIL_PAD_SPREAD_THRESHOLD_MM = 3.0` mm). **All** caps the netlist puts on that (IC, rail) pair become rigid `Macro` followers — the count comes from the schematic, not a manual cap — and stay within `MAX_CAP_IC_GAP_MM` edge-to-edge of the leader.
- **Decoupling (rail-adjacent)** — real decoupling caps on a rail whose pins are physically scattered across the IC footprint (e.g. a BGA power net landing on opposite corners). There's no single point to glue a tight rigid ring to, so these become standalone macros; SA holds them near their assigned IC via the **cap→IC attraction** cost term instead of rigid geometry. One IC can have, say, 6 rigid followers on one rail and 2 rail-adjacent on another — the split is per (IC, rail) pair, not a per-IC ceiling.
- **Bulk** (> 10 µF, or on a power rail no IC shares) — stays standalone, not assigned to any IC. Unparseable values default to bulk (conservative).
- **Coupling / signal** — only on signal nets (crystal loads, shields, USB D+/D-, reset filters), no power rail. Standalone.

Decoupling caps are distributed round-robin across the ICs sharing a rail. Power-rail detection also matches hierarchical labels (e.g. `/Buck/VIN`) and KiCad auto-generated net names (e.g. `Net-(U1-VIN)`) via trailing-token power-pin conventions, not just literal `+3V3`/`GND`-style names — deliberately conservative (EN/FB/COMP/SW/BOOT/VG and FET-source pins stay excluded).

A cap assigned to an IC is **only** in that IC's macro — never as a standalone macro. This is critical: if a cap were in two macros, push-apart could move the standalone cap away from its IC. One `assign_caps` classification drives rigid gluing, cap attraction, *and* the legacy `--delta` decoupling-proximity constraint — there's a single source of truth for which caps belong to which IC.

## Project Structure

```text
GridGhost/
├── gridghost.py                   # CLI entry point
├── config.py                      # Typed config loader (dataclasses)
├── config.json                    # Default config (margin, SA, legalization params)
│
├── models/
│   ├── board_model.py             # BoardModel, Component, Net, Pad, BoardOutline
│   │                              # (rectangle or polygon-with-holes; contains/
│   │                              # contains_bbox/fit_bbox_inside/bbox_overflow),
│   │                              # BoardModel.keepouts (rectangular no-place zones)
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
│   ├── cost.py                    # Macro-v2 cost: α·HPWL (incl. power rails) + β·overlap
│   │                              # + γ·boundary + γ·keepout + rudy + pin_density
│   │                              # + cap_attraction + clearance
│   ├── chains.py                  # Signal-flow chain grouping for cost/SA
│   └── incremental.py             # Incremental cost delta for macro SA (all cost terms)
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
│   │                              # iterated up to 5 rounds + hard-DRC min-gap
│   │                              # pass (0.15 mm, overlap-rollback safe)
│   └── abacus_bridge.py           # Adapter from rigid Macros to the row-based
│                                  # Abacus legalizer (--legalizer abacus)
│
├── engine/                        # Shared + legacy modules.
│   ├── net_clustering.py          # Hypergraph clustering — reused by place/cluster.py
│   ├── cost_state.py              # Incremental cost state — reused by cost/, place/
│   ├── congestion.py              # Density-adaptive RUDY wire-density + pin-density
│   │                              # congestion maps — reused by cost/, place/sa
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
│   ├── test_assign_caps.py        # Cap classification + IC assignment (rigid/rail-adjacent split)
│   ├── test_cost.py               # HPWL / overlap / boundary / keepout / cap-attraction / clearance cost
│   ├── test_incremental_cost.py   # Incremental cost delta (all terms)
│   ├── test_chains.py             # Signal-flow chain grouping
│   ├── test_connector_placement.py# Perimeter connector placement + rotation
│   ├── test_placement_spread.py   # Placement spread / coverage
│   ├── test_abacus_bridge.py      # Macro → Abacus legalizer adapter
│   ├── test_sa_polish.py          # SA-polish legalizer strategy
│   ├── test_nonrect_outline.py    # Polygon outline geometry, parser tracing, legalizer clamps
│   ├── test_keepout.py            # Internal keepout cost + legalizer eviction
│   ├── test_pad_rotation.py       # Footprint-only rotation (no double-rotation on pads)
│   ├── test_overlap_regression.py # (currently disabled — 0 collected)
│   ├── test_phase1.py             # 87 tests: legacy pipeline + data model
│   ├── run_all.py / dashboard.py / visualizer.py   # Batch run + result dashboard harness
│   ├── run_benchmark.py           # Benchmark harness for placement quality/timing
│   ├── README.md                  # Test harness usage (run_all, dashboard, visualizer)
│   ├── test_pcbs/                 # Bundled .kicad_pcb boards + reference JSONs
│   └── external_boards/rl_pcb/    # Third-party reference boards (see its own README/LICENSE)
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
                   tests/test_abacus_bridge.py tests/test_sa_polish.py \
                   tests/test_nonrect_outline.py tests/test_keepout.py tests/test_pad_rotation.py

# Legacy Phase 1 suite (data model, parser, legacy engine)
python -m pytest tests/test_phase1.py
```

The suite is **229 tests across 14 files** (run `python -m pytest tests/`). Breakdown: `test_phase1.py` (87 — legacy pipeline + data model), `test_nonrect_outline.py` (26), `test_cost.py` (23), `test_assign_caps.py` (17), `test_keepout.py` (13), `test_macro.py` (11), `test_connector_placement.py` (10), `test_placement_spread.py` (9), `test_chains.py` (9), `test_incremental_cost.py` (8), `test_pad_rotation.py` (7), `test_sa_polish.py` (5), `test_abacus_bridge.py` (4). `test_overlap_regression.py` is currently disabled (0 collected). The macro-first pipeline is covered by:
- **Macro model**: rigid translation, bounds revert, rotation propagation, `MAX_CAP_IC_GAP_MM` (edge-to-edge cap-IC gap) enforcement, bbox union, overlap detection
- **Cap assignment**: power-net detection (incl. hidden/hierarchical rail names), rigid-vs-rail-adjacent split by pad spread, round-robin distribution, determinism
- **Cost**: HPWL (2-pin, 3-pin, with/without power), macro overlap area, boundary (rect + polygon), keepout overlap, cap-attraction drift, pairwise clearance incl. per-pair targets and TestPoint/mechanical exemptions, edge-connector exclusion, evaluate() returns all components
- **Board outline**: polygon containment/clamp/fit, notch + nested-hole geometry, parser outline tracing and rectangle fallback, adversarial legalizer sweeps on notched boards
- **Keepout zones**: cost penalty and multi-pass legalizer eviction from rectangular no-place zones
- **Pad rotation**: footprint-only rotation rewrite on asymmetric footprints (no pad double-rotation)

## Design References

- CERC UTexas placement algorithms: https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf
- UCSD PCB placement survey: https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf
- RL for placement (17-21% lower post-routing wirelength): https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf
