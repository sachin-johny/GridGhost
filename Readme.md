<div align="center">
  <img src="logo.png" alt="GridGhost Logo" width="150" />
  <h2>GridGhost</h2>
  <p><strong>Constraint-aware PCB auto-placement engine for KiCad</strong></p>
  <p>Extracts component data from <code>.kicad_pcb</code> files, optimizes placement using net-aware clustering, simulated annealing, and constraint-driven legalization, then writes results back — all outside the KiCad editor.</p>
  <br/>
</div>

## Features

- **Net-aware hypergraph clustering** — groups components by connectivity with power-rail edges that link decoupling caps to their ICs, ensuring every IC cluster has at least one decoupling capacitor nearby
- **Smart interior-first placement** — places interior components (ICs + passives) via cluster-based grid layout, then positions edge connectors around the interior perimeter with pad-based rotation so they face outward correctly
- **Force-directed placement** — alternative algorithm using attractive (HPWL gradient) and repulsive (inverse-distance) forces with cooling schedule for convergence
- **Simulated annealing (enabled by default)** — adaptive cooling with penalty scaling (hot SA discounts overlaps for exploration, cold SA enforces them), reheating rounds, and greedy refinement with hard overlap rejection. SA auto-disables on tiny (≤6 comps) and large (≥50 comps) boards where greedy is near-optimal or SA is too slow. Use `--no-sa` to force greedy-only.
- **Cap-IC atomic group movement** — decoupling caps move as a unit with their assigned IC through the **entire pipeline** (SA moves, greedy refinement, legalizer snap/clamp/overlap-resolution, abacus row DP, post-legalize slide/swap). Caps stay adjacent to their IC (close but NOT overlapping) without being left behind.
- **End-mating connector orientation** — barrel jacks, USB, RJ45, HDMI, D-Sub and other end-mating connectors are correctly oriented so the mating face points outward. Face-mating connectors (terminal blocks, pin headers, SMA) use the perpendicular-to-pad-column heuristic.
- **Density-aware spreading** — ePlace-style Gini coefficient on a 10×10 cell-occupancy grid adds a soft spreading force so HPWL doesn't collapse everything into a center-of-mass cluster. Weight is adaptive — scaled down for dense/packed boards (no room to spread) and up for sparse boards. A fill-first spread move operator in the interior SA jumps components from overcrowded cells to empty cells within the outline, with SA's Metropolis acceptance filtering HPWL-disastrous jumps
- **Spread floor guard** — rejects SA moves that collapse component spread below 10% of board dimensions, preventing the "SA crams everything into one corner to minimise HPWL" failure mode
- **Dedicated overlap resolver** — after greedy refinement, force-resolves any remaining overlaps by pushing component pairs apart, accepting HPWL increases to guarantee zero overlaps before legalization
- **Incremental cost computation** — O(k) per SA move using sorted sweep-line index and per-net HPWL caching, with snapshot/restore for move rejection. Pre-materialised cap-IC pairs eliminate per-move `get_component` lookups.
- **10 constraint rules** — decoupling proximity, crystal-MCU, connector-edge, thermal grouping, high-current path, bulk cap input, antenna keepout, analog/digital separation, matched length, and ground-plane clearance
- **6 board profiles + auto** — pre-configured cost weights and constraint rules for MCU/peripheral, power supply, RF frontend, mixed-signal, generic, and small/dense boards. `auto` profile selects `mcu_peripheral` if ICs are detected, otherwise `generic`.
- **Rotation-aware bounding boxes** — component dimensions and courtyards update correctly with 90/180/270 rotations; KiCad clockwise rotation convention is respected
- **Pad rotation propagation** — when a footprint is rotated, the rotation delta is applied to each pad's `(at ...)` expression so copper layers match the courtyard orientation
- **Edge connector awareness** — horizontal/surface-mount connectors are placed on the board perimeter and excluded from out-of-bounds counts; vertical/THT connectors are treated as interior components
- **Interior bbox recomputation** — after SA condenses the interior cluster, components are re-centred on the board and the interior bbox is expanded so connectors have room on each edge
- **Debug visualization** — `--debug-bbox` draws component bounding boxes, board outline, and interior bbox rectangles on the `Dwgs.User` layer for visual inspection in KiCad
- **Configurable via JSON** — all tuning parameters (legalization grid, SA iterations, reheat count, spread floor, auto-disable thresholds, force-directed coefficients, clustering bounds, etc.) are loaded from `config.json` as the single source of truth. CLI flags override config values when provided.

## Pipeline

```text
KiCad .kicad_pcb
       │
  ┌────────┐   extract components, pads, nets, board outline
  │ parse  │   (S-expression tokenizer + geometry bbox estimation)
  └───┬────┘
      │
  ┌────────┐   select cost weights & constraint rules
  │profile │   (6 built-in profiles, interactive tuning, YAML custom)
  └───┬────┘
      │
  ┌────────┐   net-based hypergraph clustering + seed positions
  │cluster │   (greedy balanced clustering with power-rail edges)
  └───┬────┘
      │
  ┌────────┐   interior grid/force-directed placement,
  │ place  │   then connectors on interior-bbox perimeter
  └───┬────┘
      │
  ┌────────┐   simulated annealing with penalty scaling,
  │  SA    │   reheating, greedy refinement, overlap resolver
  └───┬────┘
      │
  ┌───────────┐   grid snap + boundary clamp + overlap resolution
  │legalize   │   (interior bbox clamping, edge-connector freeze)
  └─────┬─────┘   post-legalization overlap pass if needed
        │
  Updated .kicad_pcb  (+  JSON model + positions JSON)
```

## CLI Usage

```bash
# Auto-place a board
python gridghost.py place <input.kicad_pcb> [options]

# Extract board data to JSON (no placement)
python gridghost.py extract <input.kicad_pcb> [-o output.json]

# List available board profiles
python gridghost.py profiles
```

### Place options

| Flag | Default | Description |
|------|---------|-------------|
| `-a`, `--algorithm` | `grid` | Placement algorithm: `grid` or `force-directed` |
| `-p`, `--profile` | `auto` | Board profile: `auto`, `mcu_peripheral`, `power_supply`, `rf_frontend`, `mixed_signal`, `generic`, `small_board` |
| `-m`, `--margin` | from config (5.0) | Board edge margin in mm |
| `--no-sa` | off | Disable SA optimization (SA is **on by default**; auto-disables on tiny ≤6 or large ≥50 component boards) |
| `--sa-iterations` | from config (300) | Max SA temperature steps |
| `--sa-reheat` | from config (3) | Number of SA reheat rounds |
| `--dry-run` | off | Don't write PCB output file |
| `--interactive` | off | Interactive profile weight tuning |
| `--debug-bbox` | off | Draw bounding boxes on Dwgs.User layer |
| `--config` | `config.json` | Path to configuration file |

### Configuration

All tunable parameters are loaded from `config.json` (the single source of truth). CLI flags override config values when provided. The `annealer` section controls SA behavior:

```json
{
    "annealer": {
        "max_iterations": 300,
        "reheat_count": 3,
        "reheat_ratio": 0.40,
        "penalty_scale_min": 0.50,
        "greedy_nudge_distances": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "sa_auto_disable_min_components": 6,
        "sa_auto_disable_max_components": 50,
        "spread_floor_fraction": 0.10,
        ...
    }
}
```

Edit `config.json` to tune SA for your board class — no code changes needed.

### Examples

```bash
# Auto-place with defaults (SA enabled, auto profile selection)
python gridghost.py place board.kicad_pcb

# Force-directed algorithm, power supply profile
python gridghost.py place board.kicad_pcb -a force-directed -p power_supply

# Disable SA for fast greedy-only placement
python gridghost.py place board.kicad_pcb --no-sa

# Override SA iterations (config.json value is 300)
python gridghost.py place board.kicad_pcb --sa-iterations 500

# Dry run with debug bounding boxes for visual inspection
python gridghost.py place board.kicad_pcb --dry-run --debug-bbox

# Interactive weight tuning
python gridghost.py place board.kicad_pcb --interactive

# Extract board data without placing
python gridghost.py extract board.kicad_pcb -o board_model.json
```

## Board Profiles

Each profile configures four cost-function weights (alpha, beta, gamma, delta) plus a set of constraint rules with per-rule weights and parameters.

| Profile | Display Name | Focus | Key Constraint Rules |
|---------|-------------|-------|---------------------|
| `mcu_peripheral` | MCU / Peripheral Board | Minimize HPWL | Decoupling caps near power pins (5 mm), crystal proximity (10 mm), connector-edge enforcement (15 mm) |
| `power_supply` | Power Supply Board | Thermal grouping | Thermal grouping (20 mm), high-current path, bulk cap near input |
| `rf_frontend` | RF Frontend Board | Symmetry & isolation | Antenna keepout, analog/digital separation, matched length |
| `mixed_signal` | Mixed-Signal Board | Partitioning | Hard analog/digital boundary, decoupling proximity, ground-plane clearance |
| `generic` | Generic Board | Balanced HPWL | No specific rules — relies on HPWL minimization |
| `small_board` | Small / Dense Board | Overlap prevention | High overlap penalty (beta=15), lower HPWL weight to prevent congestion |

### Interactive Tuning

Pass `--interactive` to adjust cost weights and rule priorities before placement begins. The CLI prompts for each weight and rule parameter, pressing Enter keeps the current value.

## Cost Function

```text
Total Cost = α · HPWL + β · Overlap + γ · Boundary + δ · Σ(wₖ · Cₖ) + w_density · Gini
```

- **HPWL** — Half-perimeter wirelength. Clique model for nets with 4 or fewer pins (pairwise Manhattan distances, normalized), star model for nets with more than 4 pins (distances to center-of-mass auxiliary point). Power/ground nets are excluded since their HPWL is nearly constant regardless of placement.
- **Overlap** — Courtyard intersection area between component pairs. Penalized proportionally so the SA gradient can optimize it continuously.
- **Boundary** — Linear distance penalty for components whose bounding box extends outside the board outline. Edge connectors (horizontal/surface-mount) are excluded since they intentionally overhang the board edge.
- **Constraints** — Rule-based penalties, each producing a continuous non-negative value proportional to violation severity. See the constraint rules section below.
- **Density (Gini)** — Soft spreading force, active when the `decoupling_proximity` rule is enabled. Penalizes inequality of cell-occupancy on a 10×10 grid so components use the full board area instead of collapsing to the center. The weight `w_density` is adaptive — scaled down for dense/packed boards (no room to spread) and up for sparse boards. Decoupling caps assigned to an IC are absorbed into the IC's cell so density doesn't fight the decoupling constraint.

### SA Penalty Scaling

During simulated annealing, the penalty weights are scaled by a temperature-dependent factor:

- **Hot phase** (penalty_scale ≈ 0.65): overlap and boundary penalties are discounted so SA can explore freely through overlaps during global placement
- **Cold phase** (penalty_scale ≈ 1.0): full penalty weights are restored so SA naturally resolves overlaps
- **Greedy refinement**: penalty_scale = 1.0 with hard overlap rejection — greedy never creates new overlaps
- **Overlap resolver**: dedicated pass after greedy that force-resolves remaining overlaps, accepting HPWL increases

## Constraint Rules

GridGhost implements 10 constraint rules that produce continuous penalties, allowing the SA optimizer to follow the gradient toward compliance:

| Rule | Description | Key Parameter |
|------|-------------|---------------|
| `decoupling_proximity` | Decoupling caps must be close to their IC's power pins | `max_distance_mm` (default 5.0) |
| `crystal_mcu` | Crystals must be near their associated MCU/IC | `max_distance_mm` (default 10.0) |
| `connector_edge` | Connectors should be near board edges | `max_edge_distance_mm` (default 15.0) |
| `thermal_grouping` | Thermally-related components (regulators, MOSFETs) that share nets should be grouped | `max_distance_mm` (default 20.0) |
| `high_current_path` | Extra HPWL penalty for high-current nets (VOUT, VBAT, MOTOR, etc.) | `hpwl_weight` (default 2.0) |
| `bulk_cap_input` | Bulk capacitors should be near power-input connectors | `max_distance_mm` (default 15.0) |
| `antenna_keepout` | Keep components out of the antenna area | `keepout_x/y_min/max` |
| `analog_digital_separation` | Analog and digital components should be on opposite sides of the board | `separation_axis`, `margin_mm` |
| `matched_length` | Signal pair HPWL mismatch penalty | `pairs` list of (net1, net2) |
| `ground_plane_clearance` | Keep components clear of ground-plane split boundaries | `zones` list of (x_min, y_min, x_max, y_max) |

## Smart Placement Strategy (Grid Algorithm)

The default `grid` algorithm uses a multi-phase approach that produces better results than simple force-directed:

1. **Classify components** — separate edge connectors (horizontal/surface-mount) from interior components (ICs, passives, vertical/THT connectors)
2. **Net-cluster interior** — build a weighted hypergraph from shared nets, add power-rail edges to link decoupling caps to their ICs, then cluster using greedy balanced partitioning
3. **IC-affinity ordering** — within each cluster, order components so each IC is immediately followed by its closest passives (power-domain round-robin for decoupling caps, then shared-net affinity for remaining passives)
4. **Grid placement** — assign clusters to board sub-regions, place IC groups in center-first grid cells with passives in a ring around each IC
5. **Repulsion pass** — push overlapping components apart with spacing proportional to component size
6. **Interior SA** — simulated annealing on interior components only (connectors excluded to prevent pulling toward original KiCad positions). Includes ePlace-style Gini density penalty and a fill-first spread move operator that jumps components from overcrowded cells to empty cells within the board outline
7. **Re-centre** — shift the interior cluster so its centroid is at the center of the usable board area
8. **Connector perimeter placement** — compute the interior bbox, expand it so connectors fit on each edge, then place connectors on the perimeter with pad-based rotation. End-mating connectors (barrel jacks, USB, RJ45, etc.) use body-long-axis mating direction; face-mating connectors (terminal blocks, pin headers) use perpendicular-to-pad-column heuristic.
9. **Overlap resolution** — resolve all remaining overlaps between connectors and interior components

## Cap-IC Atomic Group Movement

Decoupling capacitors are assigned to ICs based on shared power nets (VCC, VDD, VBAT, etc.). Once assigned, a cap moves as a unit with its IC through **every stage of the pipeline**:

- **SA move operators** (translate, swap, rotate, median) — caps follow the IC's delta; on IC rotation, caps rotate around the IC's center
- **Greedy refinement** — nudges and rotations move the IC + caps as a group
- **Legalizer** — grid snap, boundary clamp, overlap resolution, and abacus row DP all propagate IC deltas to caps via the `propagate_ic_delta` hook
- **Post-legalize** — cell sliding and pair swaps propagate deltas too

Caps are placed **adjacent** to their IC (close but NOT overlapping) — the `_nudge_caps_to_ics` pass finds overlap-free slots in an 8-direction fan around the IC at 0.5–10mm spacing. The `_cleanup_cap_ic_overlaps` pass resolves any remaining cap-IC overlaps as a final step.

## Project Structure

```text
GridGhost/
├── __init__.py                    # Package init
├── gridghost.py                   # CLI entry point (place / extract / profiles commands)
├── config.py                      # Configuration loader with dataclass defaults
├── config.json                    # Runtime configuration (legalization, SA, placement, etc.)
│
├── models/
│   └── board_model.py             # BoardModel, Component, Net, Pad, BoardOutline
│                                  # All coordinates in mm; rotation-aware bboxes;
│                                  # edge-connector detection; courtyard overlap computation
│
├── parsers/
│   ├── kicad_parser.py            # S-expression .kicad_pcb parser
│                                  # Net ID→name resolution; geometry bbox from all layers;
│                                  # component type inference; board outline from Edge.Cuts
│   └── placement_writer.py        # Write placements back to .kicad_pcb / JSON
│                                  # Pad rotation delta propagation; debug bbox visualization
│
├── engine/
│   ├── net_clustering.py          # Hypergraph clustering + seed positions
│                                  # Greedy balanced / Louvain; power-rail edges;
│                                  # IC-has-caps post-processing; orphan cap merging
│   ├── grid_placement.py          # Grid, force-directed, shelf-packing placement
│                                  # Edge-aware connector perimeter placement;
│                                  # corner collision resolution; strong repulsion
│   ├── smart_placement.py         # Smart multi-phase placement engine
│                                  # Interior-first → SA → re-centre → connector perimeter;
│                                  # pad-based facing direction; interior bbox expansion;
│                                  # power-domain-aware decoupling cap grouping;
│                                  # density-aware SA with fill-first spread move operator
│   ├── cost_function.py           # HPWL (clique/star) + overlap + boundary costs
│                                  # Constraint rule integration via constraint_evaluator
│   ├── cost_state.py              # Incremental O(k) cost computation for SA
│                                  # Sorted sweep-line index; per-net HPWL cache;
│                                  # snapshot/restore; penalty scaling;
│                                  # adaptive-weight Gini density penalty
│   ├── constraint_evaluator.py    # 10 constraint rule penalty functions
│                                  # Continuous penalties for SA gradient optimization
│   ├── moves.py                   # SA move operators (translate, swap, rotate, median)
│                                  # Temperature-dependent move selection probabilities
│   ├── annealer.py                # Simulated annealing engine (v8 strategy)
│                                  # Adaptive cooling; reheating; greedy refinement;
│                                  # dedicated overlap resolver; safety-net revert
│   └── simple_optimizer.py        # Greedy pairwise swap optimizer
│
├── legalization/
│   └── legalizer.py               # Grid snap, boundary clamp, overlap resolution
│                                  # Adaptive push strength; interior bbox clamping;
│                                  # edge-connector freeze
│
├── profiles/
│   └── board_profiles.py          # 6 board profiles with cost weights + constraint rules
│                                  # YAML import/export; custom profile support
│
├── utils/
│   └── display.py                 # CLI formatting utilities
│
├── samples/
│   └── sample_board.py            # Sample MCU peripheral board (S-expression)
│
├── tests/
│   └── test_phase1.py             # 42 tests (27 Phase 1 + 15 Phase 6)
│                                  # Data model, parser, clustering, HPWL, placement,
│                                  # legalization, profiles, SA, move operators, CostState
│
├── logo.png                       # Project logo
├── CLAUDE.md                      # AI assistant context / project notes
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

Key properties:
- `bbox` — returns `(x_min, y_min, x_max, y_max)` including courtyard, correctly rotated
- `effective_width` / `effective_height` — physical dimension + 2×courtyard margin, swapped for 90°/270° rotations
- `overlaps(other)` — boolean courtyard intersection check
- `overlap_area(other)` — float area of intersection
- `is_edge_connector` — True for horizontal/surface-mount connectors that overhang the board edge

### BoardModel

Central data structure with components, nets, and board outline. Provides lookup helpers (`get_component`, `get_net`, `nets_for_component`, `components_on_net`), JSON serialization (`to_json`, `from_json`), and summary statistics (`stats` with overlap count, out-of-bounds, etc.).

## Coordinate System

- **Internal**: millimeters (float)
- **KiCad file**: millimeters (float, 6 decimal places = micrometer precision)
- **Legalization grid**: configurable, default 1.8 mm from `config.json`
- **Rotation**: KiCad clockwise-positive convention; pad positions use `cos/sin` with negated sin for CW rotation

## Dependencies

- **Python 3.9+**
- **numpy** — vectorized math in cost function
- **networkx** — net hypergraph construction and clustering (greedy + Louvain)
- **pyyaml** — custom board profile import/export (optional, only if loading YAML profiles)

## Testing

```bash
# Run the full test suite (42 tests)
python tests/test_phase1.py
```

The test suite covers:
- **Data model**: BoardOutline, Component overlap/bbox, Pad absolute positions, JSON round-trip, lookup helpers
- **Parser**: KiCad S-expression parsing, net ID resolution, component type inference
- **Clustering**: Hypergraph construction, clustering, seed positions, power-rail edges
- **Cost function**: HPWL (2-pin, 3-pin, star, auto-model selection), overlap penalty, boundary penalty, full cost evaluation
- **Placement**: Grid placement, edge-aware placement
- **Legalization**: Grid snapping, boundary enforcement, overlap resolution
- **Profiles**: Built-in profiles, rule activation
- **SA engine**: Rotation-aware dimensions, power net detection, CostState incremental/snapshot/restore, move operators (translate, swap, rotate, median), SA no-regression, greedy refinement

## Design References

- CERC UTexas placement algorithms: https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf
- UCSD PCB placement survey: https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf
- RL for placement (17-21% lower post-routing wirelength): https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf
- CadMust-Neo: https://github.com/remiblokker/CadMust-Neo