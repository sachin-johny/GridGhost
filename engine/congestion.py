"""Routability signals for PCB placement.

Two complementary congestion estimators live here:

1. **RUDY** (Rectangular Uniform wire DensitY) — estimates *wire density*
   by distributing each net's bounding box uniformly across the grid cells
   it covers. RUDY catches routing choke points where many nets' bounding
   boxes overlap (HPWL alone is blind to this — two placements with the
   same HPWL can have very different RUDY peaks).

2. **Pin density** — counts pins per grid cell. Pins are the endpoints a
   router must reach; areas with many pins (e.g. a tight cluster of small
   passives next to a QFN IC) need more routing channels per mm² than
   the wire-density model alone suggests. Pin density catches pin-escape
   congestion that RUDY misses: two pins 0.5 mm apart on the same net
   contribute almost nothing to RUDY (tiny bbox), but if 50 such pins
   share a 2 mm × 2 mm cell the router still has to escape all of them.

Both signals are used as cost terms during SA to discourage placing
components in congested areas, improving routability. They are
default-on (see ``config.json`` ``annealer.rudy_weight`` and
``annealer.pin_density_weight``) so the placer produces a routable
result out of the box; pass ``--rudy-weight 0`` /
``--pin-density-weight 0`` to disable either signal explicitly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component


from engine.cost_state import _is_power_net


def compute_rudy_map(
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[list[list[float]], dict]:
    """Compute the RUDY congestion map.

    Divides the board into a uniform grid (cell size = grid_resolution mm).
    For each non-power net with >= 2 pins, computes the bounding box of all
    pin positions and distributes a unit of wire density uniformly across
    the covered grid cells. Short nets contribute more per cell than long
    nets (1/num_cells_covered), modelling the higher per-area routing
    difficulty of dense local interconnect.

    Args:
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (grid, metadata) where:
        - grid is a 2D list of floats [row][col], congestion values
        - metadata is a dict with x_offset, y_offset, cell_size, nx, ny
    """
    board = model.board
    cell_size = max(0.5, grid_resolution)  # safety: at least 0.5mm

    nx = max(1, int(math.ceil(board.width / cell_size)))
    ny = max(1, int(math.ceil(board.height / cell_size)))

    x_offset = board.x_min
    y_offset = board.y_min

    # Initialize grid with zeros
    grid = [[0.0] * nx for _ in range(ny)]

    # Build a quick ref -> component lookup
    ref_to_comp: dict[str, "Component"] = {}
    for comp in model.components:
        ref_to_comp[comp.ref] = comp

    # For each non-power net, compute bounding box of pin positions
    # and distribute wire density across covered cells
    for net in model.nets:
        if _is_power_net(net.name):
            continue

        # Collect absolute pin positions for this net
        pin_xs: list[float] = []
        pin_ys: list[float] = []
        for ref, pad_name in net.pins:
            comp = ref_to_comp.get(ref)
            if not comp:
                continue
            # Find the pad on this component
            found_pad = False
            for pad in comp.pads:
                if pad.pad_name == pad_name:
                    abs_x, abs_y = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    pin_xs.append(abs_x)
                    pin_ys.append(abs_y)
                    found_pad = True
                    break
            if not found_pad:
                # Fallback: use component center
                pin_xs.append(comp.x)
                pin_ys.append(comp.y)

        if len(pin_xs) < 2:
            continue

        # Bounding box of pins
        bbox_x_min = min(pin_xs)
        bbox_y_min = min(pin_ys)
        bbox_x_max = max(pin_xs)
        bbox_y_max = max(pin_ys)

        # Convert to grid cell indices
        col_min = max(0, int((bbox_x_min - x_offset) / cell_size))
        col_max = min(nx - 1, int((bbox_x_max - x_offset) / cell_size))
        row_min = max(0, int((bbox_y_min - y_offset) / cell_size))
        row_max = min(ny - 1, int((bbox_y_max - y_offset) / cell_size))

        num_cells = (col_max - col_min + 1) * (row_max - row_min + 1)
        if num_cells <= 0:
            continue

        # Uniform distribution: 1.0 / num_cells per cell
        contribution = 1.0 / num_cells

        for row in range(row_min, row_max + 1):
            grid_row = grid[row]
            for col in range(col_min, col_max + 1):
                grid_row[col] += contribution

    metadata = {
        'x_offset': x_offset,
        'y_offset': y_offset,
        'cell_size': cell_size,
        'nx': nx,
        'ny': ny,
    }

    return grid, metadata


def rudy_congestion_penalty(
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[float, float, float, float]:
    """Compute a scalar congestion penalty from the RUDY map.

    The penalty is based on how much the peak congestion exceeds an
    adaptive target, plus total overflow (sum of excess above target
    across all cells). The target is set as a fraction above the
    average congestion, so it adapts to the board's overall density.

    Args:
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (penalty, peak, average, overflow) where:
        - penalty: scalar congestion penalty for the cost function
        - peak: maximum congestion value in any cell
        - average: average congestion across all cells
        - overflow: sum of (cell_value - target) for cells above target
    """
    grid, _meta = compute_rudy_map(model, grid_resolution)
    ny = len(grid)
    if ny == 0:
        return 0.0, 0.0, 0.0, 0.0
    nx = len(grid[0])
    if nx == 0:
        return 0.0, 0.0, 0.0, 0.0

    total = 0.0
    peak = 0.0
    for row in grid:
        for val in row:
            total += val
            if val > peak:
                peak = val

    num_cells = nx * ny
    average = total / num_cells

    # Adaptive target: 1.5x the average congestion.
    # Boards with uniformly distributed congestion will have peak ~ average,
    # so the target provides headroom. Boards with hotspots will have
    # peak >> average, and the penalty kicks in to spread things out.
    target = average * 1.5

    # If target is very small (nearly empty board), skip
    if target < 1e-9:
        return 0.0, peak, average, 0.0

    overflow = 0.0
    for row in grid:
        for val in row:
            if val > target:
                overflow += val - target

    # Penalty: peak * (excess above target) + overflow.
    #
    # The first term is the worst-case signal — it grows when the single
    # hottest cell rises above target. The second term is the total
    # excess across all cells above target. Together they penalise both
    # high peak congestion and widespread overflow.
    #
    # Note: an earlier v2 experiment added a quadratic peak term
    # `0.5 * peak * peak` to discourage SA from trading higher peak for
    # lower overflow. It fixed test6's peak-rising failure mode but
    # caused regressions on th_sensor / test5 / tc_logger_silabs — the
    # changed cost landscape led SA to worse local minima on those
    # boards (both HPWL and peak got worse). The peak-rising failure
    # mode is instead addressed at the threshold level in place/pipeline.py
    # by lowering the dense/congested scale on the affected board.
    excess = max(0.0, peak - target)
    penalty = peak * excess + overflow

    return penalty, peak, average, overflow


def rudy_gradient_for_comp(
    comp: "Component",
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[float, float]:
    """Compute a congestion gradient vector for a component.

    Finds the grid cells occupied by the component's bounding box
    and computes the direction toward lower congestion. Returns
    a unit vector pointing away from congested areas.

    This is used as a gentle force during SA translate moves to
    nudge components away from congested regions.

    Args:
        comp: The component to compute the gradient for.
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (dx, dy) unit vector pointing toward lower congestion.
        Returns (0.0, 0.0) if no gradient can be computed.
    """
    grid, meta = compute_rudy_map(model, grid_resolution)
    ny = len(grid)
    if ny == 0:
        return 0.0, 0.0
    nx = len(grid[0])
    if nx == 0:
        return 0.0, 0.0

    x_offset = meta['x_offset']
    y_offset = meta['y_offset']
    cell_size = meta['cell_size']

    # Get component bounding box
    bbox = comp.bbox
    cx_min, cy_min, cx_max, cy_max = bbox

    # Convert to grid cell indices
    col_min = max(0, int((cx_min - x_offset) / cell_size))
    col_max = min(nx - 1, int((cx_max - x_offset) / cell_size))
    row_min = max(0, int((cy_min - y_offset) / cell_size))
    row_max = min(ny - 1, int((cy_max - y_offset) / cell_size))

    # Compute gradient using central differences on the congestion
    # values surrounding the component's occupied cells.
    # We look one cell beyond the component's bbox in each direction.
    sum_dx = 0.0
    sum_dy = 0.0
    count = 0

    # For each occupied cell, compute the difference between the
    # cell to the left vs right (dx) and above vs below (dy).
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            # Horizontal gradient: right - left
            if col > 0 and col < nx - 1:
                dx = grid[row][col + 1] - grid[row][col - 1]
            elif col > 0:
                dx = grid[row][col] - grid[row][col - 1]
            elif col < nx - 1:
                dx = grid[row][col + 1] - grid[row][col]
            else:
                dx = 0.0

            # Vertical gradient: down - up
            if row > 0 and row < ny - 1:
                dy = grid[row + 1][col] - grid[row - 1][col]
            elif row > 0:
                dy = grid[row][col] - grid[row - 1][col]
            elif row < ny - 1:
                dy = grid[row + 1][col] - grid[row][col]
            else:
                dy = 0.0

            sum_dx += dx
            sum_dy += dy
            count += 1

    if count == 0:
        return 0.0, 0.0

    avg_dx = sum_dx / count
    avg_dy = sum_dy / count

    # The gradient points toward higher congestion.
    # We want to move AWAY from congestion, so negate it.
    move_dx = -avg_dx
    move_dy = -avg_dy

    # Normalize to a unit vector
    magnitude = math.sqrt(move_dx * move_dx + move_dy * move_dy)
    if magnitude < 1e-12:
        return 0.0, 0.0

    return move_dx / magnitude, move_dy / magnitude


# ─────────────────────────────────────────────────────────────────────
# Pin density congestion
# ─────────────────────────────────────────────────────────────────────


def compute_pin_density_map(
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[list[list[float]], dict]:
    """Compute the pin-density congestion map.

    Splits the board into a uniform grid (cell size = grid_resolution mm)
    and counts how many *signal* pins (i.e. pins on non-power nets) fall
    into each cell. The result is a 2D pin-density heatmap that
    complements RUDY: RUDY sees wire-density from net bounding boxes;
    pin density sees local pin-escape demand regardless of net span.

    Power/ground pins are excluded because they're nearly universally
    present (every IC has VCC/GND) and would uniformly inflate every
    cell, washing out the signal.

    Args:
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0; should match
            the RUDY grid for combined reporting).

    Returns:
        (grid, metadata) — same shape contract as
        :func:`compute_rudy_map` so the two maps can be rendered
        side-by-side by the visualizer.
    """
    board = model.board
    cell_size = max(0.5, grid_resolution)

    nx = max(1, int(math.ceil(board.width / cell_size)))
    ny = max(1, int(math.ceil(board.height / cell_size)))

    x_offset = board.x_min
    y_offset = board.y_min

    grid = [[0.0] * nx for _ in range(ny)]

    # Collect the set of power-net names so we can skip their pins.
    # _is_power_net already handles GND/VCC/+3V3/+5V style names.
    power_net_names: set[str] = set()
    for net in model.nets:
        if _is_power_net(net.name):
            power_net_names.add(net.name)

    # Build ref -> component lookup
    ref_to_comp: dict[str, "Component"] = {c.ref: c for c in model.components}

    # For each non-power net, count every pin into its grid cell.
    # We iterate nets (not components) so we only count pins that
    # actually belong to a signal net — a pad tied to a power net
    # doesn't add routing demand, and a pad not on any net doesn't
    # either.
    counted_pads: set[tuple[str, str]] = set()
    for net in model.nets:
        if net.name in power_net_names:
            continue
        for ref, pad_name in net.pins:
            key = (ref, pad_name)
            if key in counted_pads:
                continue
            counted_pads.add(key)
            comp = ref_to_comp.get(ref)
            if comp is None:
                continue
            # Resolve the pad's absolute position
            pin_x = comp.x
            pin_y = comp.y
            for pad in comp.pads:
                if pad.pad_name == pad_name:
                    pin_x, pin_y = pad.absolute_pos(comp.x, comp.y, comp.rotation)
                    break
            col = max(0, min(nx - 1, int((pin_x - x_offset) / cell_size)))
            row = max(0, min(ny - 1, int((pin_y - y_offset) / cell_size)))
            grid[row][col] += 1.0

    metadata = {
        'x_offset': x_offset,
        'y_offset': y_offset,
        'cell_size': cell_size,
        'nx': nx,
        'ny': ny,
    }
    return grid, metadata


def pin_density_penalty(
    model: "BoardModel",
    grid_resolution: float = 2.0,
) -> tuple[float, float, float, float]:
    """Scalar pin-density congestion penalty.

    Mirrors :func:`rudy_congestion_penalty`'s return contract so the two
    signals compose cleanly in the cost function, but uses a pin-density-
    appropriate threshold scheme. RUDY's "1.5× the average over all
    cells" works because wire density is roughly continuous — most cells
    get some contribution from each net whose bbox crosses them. Pin
    density is sparse: most cells are empty, and a single pin in a cell
    is normal, not a hotspot. So the threshold is::

        target = max(PIN_DENSITY_MIN_TARGET,
                     1.5 × average over non-empty cells)

    where ``PIN_DENSITY_MIN_TARGET = 2.0``. A cell with 1 pin never
    fires the penalty (no pin-escape congestion from a lone pin); a
    cell with 3+ pins starts to fire when the rest of the board is
    sparser. On a densely-packed board where every cell has many pins,
    the 1.5× non-empty-average term takes over and prevents the
    penalty from firing on uniformly-dense pin grids.

    Args:
        model: BoardModel with components and nets.
        grid_resolution: Cell size in mm (default 2.0).

    Returns:
        (penalty, peak, average, overflow) — same semantics as
        :func:`rudy_congestion_penalty`. ``penalty`` is the scalar to
        add to the SA cost; the other three are diagnostic only.
        ``average`` here is the average over ALL cells (including
        empty), matching RUDY's contract for side-by-side reporting.
    """
    # A cell needs at least this many pins before the penalty can fire.
    # Below this, a single pin in a cell is normal — no router ever
    # struggles to escape one pin. 2.0 means a cell needs at least 3
    # pins (above the 2.0 target) to start contributing to overflow.
    PIN_DENSITY_MIN_TARGET = 2.0

    grid, _meta = compute_pin_density_map(model, grid_resolution)
    ny = len(grid)
    if ny == 0:
        return 0.0, 0.0, 0.0, 0.0
    nx = len(grid[0])
    if nx == 0:
        return 0.0, 0.0, 0.0, 0.0

    total = 0.0
    peak = 0.0
    non_empty_total = 0.0
    non_empty_count = 0
    for row in grid:
        for val in row:
            total += val
            if val > peak:
                peak = val
            if val > 0.0:
                non_empty_total += val
                non_empty_count += 1

    num_cells = nx * ny
    average = total / num_cells  # over ALL cells — matches RUDY's contract

    # Adaptive target: 1.5× the average over NON-EMPTY cells, with a
    # floor of PIN_DENSITY_MIN_TARGET. On a sparse board (most cells
    # empty), the floor dominates and a single pin per cell doesn't
    # fire. On a dense board, the 1.5× term dominates and a uniformly
    # dense pin distribution doesn't fire either.
    if non_empty_count == 0:
        return 0.0, peak, average, 0.0
    non_empty_avg = non_empty_total / non_empty_count
    target = max(PIN_DENSITY_MIN_TARGET, non_empty_avg * 1.5)

    if peak <= target:
        return 0.0, peak, average, 0.0

    overflow = 0.0
    for row in grid:
        for val in row:
            if val > target:
                overflow += val - target

    # Note: pin-density peak is a raw count (e.g., 6 pins in the worst
    # cell), not a normalized density like RUDY peak. The linear term
    # `peak * excess` is therefore already strongly peak-sensitive
    # (raising peak from 4 to 6 with target=2 increases the term from
    # 8 to 24). Adding the quadratic term used in rudy_congestion_penalty
    # would dominate the linear term and over-penalize legitimately
    # dense pin clusters (e.g., a QFN-32 with 8 pins per cell is normal,
    # not a congestion bug). So pin-density keeps the linear formula.
    excess = max(0.0, peak - target)
    penalty = peak * excess + overflow
    return penalty, peak, average, overflow
