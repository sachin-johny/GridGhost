# GridGhost

External PCB auto-placement optimization engine for KiCad. Separates data extraction from optimization for better performance and flexibility.

## Project Status (Current: Phase 6 Complete)

**Last Updated:** 2026-05-07

### What Works
- Full pipeline end-to-end: extraction → net clustering → placement → SA optimization → legalization → result saving
- KiCad .kicad_pcb parsing and JSON round-trip support
- HPWL-based placement cost with overlap, boundary, and constraint penalties
- Grid and force-directed placement with cluster-aware seeding and edge-aware connector handling
- Simulated Annealing engine (ported from CadMust-Neo): adaptive cooling, reheating, move operators, greedy refinement
- Rotation-aware component dimensions and bbox computation
- Incremental cost computation O(k) for SA performance
- Power net auto-detection and HPWL exclusion
- 42 tests passing, 0 failures (27 Phase 1 + 15 Phase 6)
- `th_sensor.kicad_pcb` dry-run: 0 overlaps, 0 out-of-bounds
- Larger `cbb.kicad_pcb` now completes placement pipeline

### Next Improvements (High Priority)
- Test SA optimization on real boards and compare HPWL with/without SA
- Tune SA parameters (iterations, reheat, cooling) for different board sizes
- Polygon boundary support and keepout zones

### Known Issues
- Dense boards still benefit from tuning (pipeline works but placement not optimal)
- Legalization can increase wirelength (small regression accepted for legal placement)
- SA on very small test boards can cause boundary penalty spikes

---

## Architecture Overview

```
KiCad .kicad_pcb
       ↓
    [parsers] → BoardModel (JSON intermediate)
       ↓
    [profiles] → Board profile selection
       ↓
    [engine] → Optimization (clustering + grid placement)
       ↓
    [legalization] → Grid snap, overlap resolve, boundary clamp
       ↓
 Updated .kicad_pcb
```

**Key Design:** External optimization vs KiCad API integration - data extracted once, optimized completely outside, then applied back.

---

## Directory Structure

```
auto_placer/
├── __init__.py          # Package init, version 0.1.0
├── gridghost.py        # CLI entry point (place/extract/profiles commands)
├── models/
│   └── board_model.py   # Core data models: BoardModel, Component, Net, Pad, BoardOutline
│                        # All coordinates in mm. bbox/overlaps/effective dimensions included.
├── parsers/
│   ├── kicad_parser.py  # S-expression .kicad_pcb parser with net ID→name resolution
│   └── placement_writer.py  # Write placements back to .kicad_pcb, export positions JSON
├── engine/
│   ├── net_clustering.py    # Hypergraph clustering, seed positions, greedy/Louvain
│   ├── cost_function.py     # HPWL (clique/star), overlap/boundary penalties
│   ├── cost_state.py        # Incremental cost computation O(k) for SA, power net exclusion
│   ├── moves.py             # SA move operators: translate, swap, rotate, median
│   ├── annealer.py          # Simulated Annealing engine with adaptive cooling & reheating
│   └── grid_placement.py    # Grid, force-directed, and shelf-packing placement algorithms
├── legalization/
│   └── legalizer.py         # Grid snap, boundary clamp, overlap resolution
├── profiles/
│   └── board_profiles.py    # 5 profiles: mcu_peripheral, power_supply, rf_frontend,
│                            # mixed_signal, generic (cost weights + constraint rules)
├── utils/
│   └── display.py       # CLI formatting utilities
├── samples/
│   └── sample_board.py  # Sample MCU peripheral board (18 components, 11 nets)
├── tests/
│   └── test_phase1.py   # 42 tests (27 Phase 1 + 15 Phase 6)
├── CLAUDE.md           # This file - project context for Claude
├── README.md           # Detailed architecture and design docs
├── auto_placer_plan.md # Development phases and implementation plan
└── milestone.md        # Current milestone summary
```

---

## Core Concepts

### Cost Function
Total Cost = α·HPWL + β·Overlap + γ·Boundary + δ·Constraints

- **HPWL**: Half-perimeter wirelength - clique model (≤4 pins), star model (>4 pins)
- **Overlap**: Component courtyard intersections
- **Boundary**: Components outside board outline
- **Constraints**: Decoupling caps near power pins, crystals near MCU, connectors on edges

### Board Profiles
Cost weights and rule priorities pre-configured per board type:

| Profile | Focus | Key Rules |
|---------|-------|-----------|
| `mcu_peripheral` | HPWL | Decoupling ≤0.5mm, crystal proximity |
| `power_supply` | Thermal | High-current grouping, bulk cap input |
| `rf_frontend` | Symmetry | Antenna keepout, A/D separation |
| `mixed_signal` | Partition | Hard analog/digital boundary |
| `generic` | Balanced | No specific rules |

### Component Types
`ic`, `capacitor`, `resistor`, `connector`, `crystal`, `generic` - affects placement heuristics and spacing.

### Coordinate System
- Internal: millimeters (float)
- KiCad: nanometers (int) - conversion via `mm * 1e6`
- Grid: default 0.1mm for legalization

---

## CLI Usage

```bash
python gridghost.py place <input.kicad_pcb> [options]
  -a, --algorithm     # force-directed (default) | grid
  -p, --profile       # Board profile (default: generic)
  -m, --margin        # Board edge margin mm (default: 5.0)
  --dry-run           # Don't write PCB file
  --interactive       # Interactive profile weight tuning
  --no-sa             # Disable SA optimization after placement
  --sa-iterations N   # Max SA temperature steps (default: 200)
  --sa-reheat N       # Number of SA reheat rounds (default: 2)

python gridghost.py extract <input.kicad_pcb> [-o output.json]
python gridghost.py profiles
```

---

## Key Data Model Details

**Component:** ref, footprint, value, x, y, rotation, layer, width, height, courtyard_margin (0.25mm), pads, nets, is_fixed, component_type

**bbox property:** Returns (x_min, y_min, x_max, y_max) including courtyard
**overlaps(other):** Boolean check for courtyard intersection
**overlap_area(other):** Float area of intersection

**BoardModel:** board outline, components list, nets list, source_file
- Helpers: get_component(), get_net(), nets_for_component(), components_on_net()
- Serialization: to_json(), from_json(), to_dict(), from_dict()
- stats(): Summary with overlap counts, out-of-bounds, etc.

---

## Testing

**Test File:** `tests/test_phase1.py` - 27 tests covering all Phase 1 functionality
**Samples:** `samples/sample_board.py` - 18 component MCU peripheral board

**Metrics to Track:**
- Total wirelength (HPWL)
- Number of overlaps (pre/post legalization)
- Constraint violations per rule
- Out-of-bounds components
- Visual inspection in KiCad

---

## Development Phases (Reference)

| Phase | Status | Key Output |
|-------|--------|------------|
| 0 | ✅ | Extraction, JSON round-trip |
| 1 | ✅ | Net clustering + grid placement |
| 2 | ✅ | HPWL + overlap + boundary cost |
| 3 | ✅ | Legalization pass |
| 3.5 | ✅ | Grid snap, overlap resolve |
| 4 | ✅ | Constraint rules + board profiles |
| 5 | ✅ | Interactive CLI tuning |
| 6 | ✅ | SA engine + rotation + incremental cost |

**Implementation Order:** 0 → 1 → 2 → 3.5 → 4 → 5 → 6

---

## Important Implementation Notes

1. **Legalization is single-pass** - don't call iteratively in optimizer loop, corrupts gradient/energy signals
2. **Coordinate conversion** - KiCad uses nanometers, internal uses mm. Always convert at boundaries.
3. **Grid snapping** - Round to nearest grid increment (default 0.1mm): `round(x / grid) * grid`
4. **Component type consistency** - Always use the `component_type` field, not stale placement model fields
5. **Edge-aware placement** - Connectors automatically placed near edges if board has connectors
6. **Cluster seeding** - Net-based clustering provides deterministic seed positions, dramatically improves convergence

---

## Dependencies

- numpy (vectorized math)
- Python 3.9+
- (Future: networkx, scipy, numba/jax for acceleration)

---

## Design References

- CERC UTexas placement algorithms: https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf
- UCSD PCB placement survey: https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf
- RL for placement (17-21% lower post-routing wirelength): https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf
