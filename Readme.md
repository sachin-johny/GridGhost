<div align="center">
  <img src="logo.png" alt="GridGhost Logo" width="200" />
  <h2>GridGhost</h2>
  <p>External PCB auto-placement optimization engine for KiCad.</p>
  <p>Extracts component data from `.kicad_pcb` files, optimizes placement using net-aware clustering, simulated annealing, and constraint-driven legalization, then writes results back.</p>
</div>

## Features

- **Net-aware clustering** — groups components by connectivity for intelligent seed placement
- **Grid & force-directed placement** — two initial placement algorithms with cluster-aware seeding
- **Edge-aware connector placement** — size-aware greedy best-fit distributes connectors across board perimeter
- **Simulated annealing** — adaptive cooling, reheating, translate/swap/rotate move operators
- **Incremental cost computation** — O(k) per SA move for fast convergence
- **Rotation-aware bboxes** — component dimensions and courtyards update with rotation
- **5 board profiles** — pre-configured cost weights for MCU, power, RF, mixed-signal, and generic boards
- **Constraint rules** — decoupling cap proximity, crystal placement, connector-on-edge enforcement
- **Legalization** — grid snapping, boundary clamping, iterative overlap resolution

## Pipeline

```text
KiCad .kicad_pcb
       |
  [parse]  extract components, pads, nets, board outline
       |
  [profile]  select cost weights & constraint rules
       |
  [cluster]  net-based hypergraph clustering + seed positions
       |
  [place]   grid or force-directed initial placement
       |
  [SA]      simulated annealing optimization
       |
  [legalize]  grid snap + overlap resolve + boundary clamp
       |
 Updated .kicad_pcb
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

| Flag               | Default          | Description                                                                          |
| ------------------ | ---------------- | ------------------------------------------------------------------------------------ |
| `-a, --algorithm`  | `force-directed` | Placement algorithm: `force-directed` or `grid`                                     |
| `-p, --profile`    | `generic`        | Board profile: `mcu_peripheral`, `power_supply`, `rf_frontend`, `mixed_signal`, ...  |
| `-m, --margin`     | from config      | Board edge margin in mm                                                              |
| `--no-sa`          | off              | Disable SA optimization                                                              |
| `--sa-iterations`  | 200              | Max SA temperature steps                                                             |
| `--sa-reheat`      | 2                | Number of SA reheat rounds                                                           |
| `--dry-run`        | off              | Don't write PCB output file                                                          |
| `--interactive`    | off              | Interactive profile weight tuning                                                    |
| `--debug-bbox`     | off              | Draw bounding boxes on Dwgs.User layer                                               |
| `--config`         | `config.json`    | Path to configuration file                                                           |

### Examples

```bash
# Quick placement with defaults
python gridghost.py place board.kicad_pcb

# Grid algorithm, MCU profile, no SA
python gridghost.py place board.kicad_pcb -a grid -p mcu_peripheral --no-sa

# Dry run with debug bounding boxes
python gridghost.py place board.kicad_pcb --dry-run --debug-bbox

# Interactive weight tuning
python gridghost.py place board.kicad_pcb --interactive
```

## Board Profiles

| Profile | Focus | Key Constraint Rules |
|---------|-------|---------------------|
| `mcu_peripheral` | Minimize HPWL | Decoupling caps near power pins, crystal proximity |
| `power_supply` | Thermal grouping | High-current trace grouping, bulk cap input |
| `rf_frontend` | Symmetry | Antenna keepout, analog/digital separation |
| `mixed_signal` | Partitioning | Hard analog/digital boundary |
| `generic` | Balanced | No specific rules |

## Cost Function

```text
Total Cost = alpha * HPWL + beta * Overlap + gamma * Boundary + delta * Constraints
```

- **HPWL** — Half-perimeter wirelength (clique model for <=4 pins, star model for >4 pins)
- **Overlap** — Courtyard intersection area between components
- **Boundary** — Distance penalty for components outside board outline
- **Constraints** — Rule-based penalties (decoupling proximity, crystal placement, etc.)

## Project Structure

```text
auto_placer/
├── __init__.py              # Package init
├── gridghost.py              # CLI entry point (place/extract/profiles)
├── models/
│   └── board_model.py       # BoardModel, Component, Net, Pad, BoardOutline
├── parsers/
│   ├── kicad_parser.py      # S-expression .kicad_pcb parser
│   └── placement_writer.py  # Write placements back to .kicad_pcb / JSON
├── engine/
│   ├── net_clustering.py    # Hypergraph clustering + seed positions
│   ├── grid_placement.py    # Grid, force-directed, edge-aware placement
│   ├── cost_function.py     # HPWL (clique/star) + overlap + boundary costs
│   ├── cost_state.py        # Incremental cost computation for SA
│   ├── moves.py             # SA move operators (translate, swap, rotate, median)
│   └── annealer.py          # Simulated annealing engine
├── legalization/
│   └── legalizer.py         # Grid snap, boundary clamp, overlap resolution
├── profiles/
│   └── board_profiles.py    # Board profiles with cost weights + constraint rules
├── utils/
│   └── display.py           # CLI formatting utilities
├── samples/
│   └── sample_board.py      # Sample MCU peripheral board
├── tests/
│   └── test_phase1.py       # 42 tests
├── config.json              # Runtime configuration
├── CLAUDE.md                # AI assistant context
└── README.md                # This file
```

## Dependencies

- Python 3.9+
- numpy

## Coordinate System

- Internal: millimeters (float)
- KiCad: nanometers (int) — conversion via `mm * 1e6`
- Grid: default 0.1mm for legalization
