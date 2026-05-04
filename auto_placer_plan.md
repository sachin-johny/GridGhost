# KiCad Smart Auto-Placer (External Optimization Engine)

## Overview

This project implements a constraint-aware PCB auto-placement system for KiCad by separating:
- **Data extraction & application** → via KiCad Python API (`pcbnew`)
- **Optimization & simulation** → external Python environment

The goal is faster, more flexible, and scalable placement than what is feasible inside KiCad's scripting runtime.

***

## Architecture

```
┌──────────────────────┐
│   KiCad (.kicad_pcb) │
└─────────┬────────────┘
          │
 [cerc.utexas](https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf) Extract via pcbnew API
          │
          ▼
┌──────────────────────┐
│   Intermediate Model │
│ (JSON / Python objs) │
└─────────┬────────────┘
          │
 [cseweb.ucsd](https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf) Board Profile Selection (User Feedback Loop)
          │
          ▼
┌──────────────────────┐
│  Optimization Engine │
│  + Legalization Pass │
└─────────┬────────────┘
          │
 [arxiv](https://arxiv.org/pdf/2502.14012.pdf) Apply via pcbnew API
          │
          ▼
┌──────────────────────┐
│  Updated .kicad_pcb  │
└──────────────────────┘
```

***

## Project Goals

- Automate initial component placement
- Reduce manual layout time
- Provide extensible optimization framework
- Support rule-based + cost-based placement

## Non-Goals *(Important)*

- Full replacement for professional EDA placers
- Perfect routing-aware placement
- High-speed / RF-grade optimization (initially)

***

## Core Components

### 1. Data Extraction Layer (KiCad API)

**Input:** `.kicad_pcb`, Netlist (optional, usually embedded)

**Responsibilities:**
- Load board via `pcbnew`
- Extract footprints (reference, position, rotation), pads and nets, board outline, courtyard/bounding boxes, net classes (optional)
- **Layer awareness:** tag each component as top/bottom; affects decoupling cap placement and connector orientation rules

**Output:** Serialized model (JSON or Python dict)

**Key Data Model:**
```json
{
  "board": {
    "outline": [...],
    "width": ...,
    "height": ...
  },
  "components": [
    {
      "ref": "U1",
      "width": 5.0,
      "height": 5.0,
      "x": 10.0,
      "y": 20.0,
      "rotation": 0,
      "layer": "top",
      "nets": ["VCC", "GND", "IO1"]
    }
  ],
  "nets": {
    "VCC": ["U1", "C1"],
    "GND": ["U1", "C1", "J1"]
  }
}
```

***

### 2. User Feedback Loop (Board Profile System)

Runs **before** the optimization engine. Prompts the user to select or tune a board profile, which pre-configures cost weights and active constraint rules.

**Board Profiles:**

| Board Type | Dominant Cost Term | Key Constraint Rules |
|---|---|---|
| `mcu_peripheral` | HPWL (connectivity) | Decoupling caps ≤ 0.5mm from power pins, crystal proximity |
| `power_supply` | Thermal + current path | High-current grouping, thermal relief spacing, bulk cap input |
| `rf_frontend` | Symmetry + isolation | Antenna keepout, analog/digital separation, matched trace lengths |
| `mixed_signal` | Partition penalty | Hard boundary between analog/digital ground domains |

**Profile Config Example:**
```python
BOARD_PROFILES = {
    "mcu_peripheral": {
        "alpha": 1.0,   # HPWL weight
        "beta":  5.0,   # overlap penalty
        "gamma": 3.0,   # boundary penalty
        "delta": 4.0,   # constraint penalty (decoupling, crystal)
        "rules": ["decoupling_proximity", "crystal_mcu", "connector_edge"]
    },
    "power_supply": {
        "alpha": 0.5,
        "beta":  5.0,
        "gamma": 2.0,
        "delta": 6.0,   # thermal dominates
        "rules": ["thermal_grouping", "high_current_path", "bulk_cap_input"]
    },
    "rf_frontend": {
        "alpha": 0.8,
        "beta":  5.0,
        "gamma": 4.0,
        "delta": 7.0,
        "rules": ["antenna_keepout", "analog_digital_separation", "matched_length"]
    }
}
```

**Interactive Tuning (CLI):**

Rather than a single `delta` scalar, each rule gets its own priority weight — exposing this interactively gives you a real constraint priority system:

```
Board profile: mcu_peripheral
Active rules:
   [cerc.utexas](https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf) decoupling_proximity  weight=4.0  → adjust? (enter new value or press Enter to keep)
   [cseweb.ucsd](https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf) crystal_mcu           weight=3.0  → adjust?
   [arxiv](https://arxiv.org/pdf/2502.14012.pdf) connector_edge        weight=2.0  → adjust?
```

This can also be driven by a YAML config file per board session for reproducibility.

***

### 3. Optimization Engine (External Python)

Runs completely outside KiCad.

**Libraries:**
- `numpy` → vectorized math
- `networkx` → graph modeling
- `scipy` → optimization (optional)
- `numba` / `jax` → acceleration (optional)

#### 3.1 Placement Model

- Components → rectangles
- Nets → weighted graph edges
- Board → polygon boundary
- Constraints → penalty functions

#### 3.2 Net Model (Multi-Pin Handling)

HPWL works cleanly for 2-pin nets. For multi-pin nets, choose based on net size: [cerc.utexas](https://www.cerc.utexas.edu/utda/publications/book_tdp.pdf)

| Net Size | Model | Rationale |
|---|---|---|
| ≤ 4 pins | **Clique model** | Exact pairwise distances, sparse enough for matrix solvers |
| > 4 pins | **Star model** | Center-of-mass auxiliary point avoids dense matrix formation |

```python
def net_wirelength(pins, model="auto"):
    if model == "auto":
        model = "clique" if len(pins) <= 4 else "star"
    if model == "clique":
        return sum(dist(p1, p2) for p1, p2 in combinations(pins, 2)) / (len(pins) - 1)
    if model == "star":
        center = np.mean(pins, axis=0)
        return sum(dist(p, center) for p in pins)
```

#### 3.3 Cost Function

\[
\text{Total Cost} = \alpha \cdot W_{\text{HPWL}} + \beta \cdot P_{\text{overlap}} + \gamma \cdot P_{\text{boundary}} + \delta_k \cdot \sum_k w_k \cdot C_k
\]

Where \(\delta_k\) and \(w_k\) are per-rule weights set by the board profile and user tuning. [cseweb.ucsd](https://cseweb.ucsd.edu/classes/fa23/cse248-a/papers/placement/PCBPlacement.pdf)

#### 3.4 Seeded Initial Placement

Random initialization hurts convergence badly for SA/force-directed methods. Use net-clustering as a seed: [arxiv](https://arxiv.org/pdf/2502.14012.pdf)

```
Phase 0 seed:
  1. Build net hypergraph
  2. Cluster by shared nets (spectral or greedy)
  3. Assign cluster centroids to board regions
  4. Use as initial positions for optimizer
```

This dramatically reduces iterations before convergence.

#### 3.5 Constraints (Heuristics Layer)

- Decoupling capacitors near IC power pins
- Connectors near board edges
- Crystals close to MCU
- Group components by shared nets
- Keep analog/digital separated (especially for `mixed_signal` profile)

#### 3.6 Optimization Algorithms

| Phase | Algorithm | Notes |
|---|---|---|
| Phase 1 (MVP) | Net-based clustering + grid placement | Deterministic seed |
| Phase 2 | Force-directed (spring + repulsion) | Springs = net edges, repulsion = overlap |
| Phase 3 | Simulated annealing | Random perturbation + acceptance function |
| Phase 4 | Genetic algorithms / gradient-based | Relaxed continuous model |

#### 3.7 Rotation Handling

- Discrete set: {0°, 90°, 180°, 270°}
- Optional: flip (top/bottom layer)

***

### 4. Legalization Pass *(New — Required Before Application)*

Force-directed and SA outputs frequently produce non-integer, overlapping placements. A legalization pass is mandatory before writing back to KiCad: [forum.kicad](https://forum.kicad.info/t/snapping-to-grid/33669)

**Steps:**
1. **Grid snapping** — round coordinates to KiCad's placement grid (e.g. 0.1mm or 0.05mm)
2. **Overlap resolution** — iteratively shift components to nearest non-overlapping position
3. **Boundary enforcement** — clamp any out-of-bounds components to legal board region
4. **Courtyard DRC check** — verify no courtyard intersections remain

```python
def legalize(components, grid_mm=0.1, board_outline=None):
    for comp in components:
        # Snap to grid
        comp.x = round(comp.x / grid_mm) * grid_mm
        comp.y = round(comp.y / grid_mm) * grid_mm
        # Clamp to board
        if board_outline:
            comp.x = np.clip(comp.x, board_outline.xmin, board_outline.xmax)
            comp.y = np.clip(comp.y, board_outline.ymin, board_outline.ymax)
    resolve_overlaps(components)  # iterative push-apart
    return components
```

> **Important:** Run legalization as a single post-optimization pass, not iteratively inside the optimizer loop — it would corrupt gradient/energy signals.

***

### 5. Placement Application Layer

**Responsibilities:**
- Map optimized coordinates → KiCad nanometer units
- Apply position, rotation, layer assignment
- Save updated board

**Important:** Apply all changes in one pass only.

**Coordinate System:**
- KiCad uses nanometers (int)
- Convert: `kicad_units = int(mm_value * 1e6)`

***

## Development Phases

| Phase | Goal | Key Output |
|---|---|---|
| 0 | Setup: load board, extract, serialize JSON | `board_model.json` |
| 1 | Net clustering seed + grid placement | Basic placed board |
| 2 | HPWL + overlap + boundary cost engine | Cost-scored placements |
| 3 | Force-directed + SA optimizer | Optimized placement |
| 3.5 *(new)* | Legalization pass | Legal, grid-snapped placement |
| 4 | Constraint rule engine + board profiles | Profile-aware placement |
| 5 | User feedback loop (CLI/YAML tuning) | Per-session tunable weights |
| 6 | Rotation optimization + multi-pass | Final refinement |

### Recommended Implementation Order

Use the phases in this order so each step has a stable upstream contract:

1. **Phase 0 first**: make extraction and serialization trustworthy before any placement logic.
2. **Phase 1 next**: cluster components and produce deterministic seed positions on a small board.
3. **Phase 2 next**: validate the cost model against those seed placements.
4. **Phase 3.5 before advanced optimization**: legalization must be reliable before force-directed or SA output is accepted.
5. **Phase 4 and 5 after the core pipeline works**: add constraint rules and interactive tuning once the model is stable.
6. **Phase 6 last**: rotation and layer refinement should sit on top of a working placement/legalization loop.

### Minimum Done Criteria Per Phase

- **Phase 0**: parse a real `.kicad_pcb` into a `BoardModel`, then round-trip to JSON.
- **Phase 1**: generate repeatable non-overlapping seed placements for a sample board.
- **Phase 2**: report HPWL, overlap, and boundary cost separately and consistently.
- **Phase 3.5**: snap to grid, resolve overlaps, and keep all components inside the board outline.
- **Phase 4**: evaluate named rules from the selected board profile.
- **Phase 5**: allow per-session profile tuning from CLI or YAML.
- **Phase 6**: search discrete rotations and layer choices without breaking legalization.

### Immediate Next Step

The first high-value implementation slice is to harden the Phase 0/1 boundary:

- make sure every parsed footprint has enough geometry for placement and legalization,
- ensure seed placement never overlaps fixed components,
- and verify legalization can recover a legal board from a rough seeded placement.

***

## Testing Strategy

Use small boards first (MCU + few peripherals).

**Metrics:**
- Total wirelength (HPWL)
- Number of overlaps (pre/post legalization)
- Constraint violations per rule
- Visual inspection in KiCad

***

## Known Challenges

- Missing design intent
- Approximate DRC rules
- Performance tuning
- Local minima in optimization
- Legalization can increase wirelength (accept small regression for legal placement)

***

## Future Extensions

- Routing-aware placement (estimate congestion)
- Machine learning-based placement (RL agents show 17–21% lower post-routing wirelength vs SA ) [lukevassallo](https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf)
- Interactive GUI for constraint tuning
- Integration as KiCad plugin