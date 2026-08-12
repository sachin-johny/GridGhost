"""Quadratic analytical initial placement.

Solves a quadratic approximation of wirelength to find near-optimal
component positions. Based on the force-directed placement approach
used in VLSI tools (Kraftwerk, FastPlace).

The quadratic wirelength model:
- Each net contributes a force pulling connected components together
- Force magnitude proportional to distance (quadratic wirelength)
- Fixed components (including pseudo-anchors) provide anchor points
- Solution is found by solving a linear system (sparse Cholesky)

Since we avoid numpy/scipy, the linear system is solved iteratively
using Gauss-Seidel with SOR (successive over-relaxation).

Typical usage:
    from engine.quadratic_placement import quadratic_place
    quadratic_place(model, margin=5.0, n_iterations=3, verbose=True)
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

from engine.cost_state import _is_power_net

if TYPE_CHECKING:
    from models.board_model import BoardModel, Component


# Grid snap resolution (mm) — matches legalizer default
GRID_SNAP = 0.1


def _add_legalization_anchors(
    model: "BoardModel",
    movable: List["Component"],
    movable_refs: Set[str],
    anchor_forces: Dict[str, Tuple[float, float, float]],
    grid_mm: float = 0.1,
    x_min: float = 0.0,
    x_max: float = 100.0,
    y_min: float = 0.0,
    y_max: float = 100.0,
) -> None:
    """Add anchor forces based on estimated legalization displacement.

    Simulates a quick Abacus-style row assignment to estimate where each
    component would end up after legalization, then adds anchor forces
    pulling components toward those estimated positions. This reduces
    the gap between quadratic-optimal and legalized positions.

    The anchor weight is proportional to the expected displacement —
    components that would be moved far get stronger anchors.
    """
    if not movable:
        return

    # Compute row pitch (same as Abacus)
    max_height = max(c.effective_height for c in movable)
    row_pitch = max(math.ceil(max_height / grid_mm) * grid_mm, grid_mm)

    # Quick row assignment by y-coordinate
    row_map: Dict[int, List["Component"]] = {}
    for comp in movable:
        row_idx = round((comp.y - y_min - row_pitch / 2) / row_pitch)
        if row_idx not in row_map:
            row_map[row_idx] = []
        row_map[row_idx].append(comp)

    # For each row, estimate the x-displacement from Abacus-style placement
    for row_idx, row_comps in row_map.items():
        if not row_comps:
            continue

        row_y = y_min + row_idx * row_pitch + row_pitch / 2

        # Sort by x for left-to-right processing
        sorted_comps = sorted(row_comps, key=lambda c: c.x)

        # Simulate Abacus: components are packed left-to-right with gaps
        # Each component's "legalized x" is max(its optimal x, previous right edge + gap)
        gap = grid_mm
        placed_right_edge = x_min

        for comp in sorted_comps:
            ew = comp.effective_width
            optimal_left = comp.x - ew / 2
            # The legalized position is either the optimal or pushed right
            legalized_left = max(optimal_left, placed_right_edge + gap)
            legalized_x = legalized_left + ew / 2

            displacement = abs(legalized_x - comp.x)

            # Add anchor force proportional to expected displacement
            # Stronger for components with large expected displacement
            if displacement > grid_mm * 5:
                anchor_weight = min(displacement * 0.1, 2.0)

                existing = anchor_forces.get(comp.ref, (0.0, 0.0, 0.0))
                w, tx, ty = existing
                anchor_forces[comp.ref] = (
                    w + anchor_weight,
                    tx + anchor_weight * legalized_x,
                    ty + anchor_weight * row_y,
                )

            placed_right_edge = legalized_left + ew


def quadratic_place(
    model: "BoardModel",
    margin: float = 5.0,
    n_iterations: int = 3,
    verbose: bool = False,
) -> None:
    """Quadratic analytical placement for interior components.

    Main entry point.  Identifies movable interior components, builds
    the connectivity-based force model, then alternates between solving
    the quadratic system and adding spreading forces for ``n_iterations``
    rounds.

    Only moves components that are:
    - Not fixed (component.is_fixed == False)
    - Not edge connectors (component.is_edge_connector == False)
    - Not already placed on the board perimeter

    Args:
        model: BoardModel with components to place
        margin: Margin from board edges in mm
        n_iterations: Number of outer solve-spread iterations (3 typical)
        verbose: Print progress information
    """
    board = model.board
    x_min = board.x_min + margin
    x_max = board.x_max - margin
    y_min = board.y_min + margin
    y_max = board.y_max - margin

    # Identify movable interior components
    movable: List["Component"] = []
    fixed: List["Component"] = []
    for comp in model.components:
        if comp.is_fixed or comp.is_edge_connector:
            fixed.append(comp)
        else:
            movable.append(comp)

    if not movable:
        return

    movable_refs = {c.ref for c in movable}
    ref_to_idx: Dict[str, int] = {c.ref: i for i, c in enumerate(movable)}
    n = len(movable)

    if verbose:
        hpwl_before = _compute_hpwl_simple(model, movable_refs)
        print(f"  Quadratic placement: {n} movable, "
              f"{len(fixed)} fixed, HPWL={hpwl_before:.1f}")

    # Build the net-based force model (connectivity)
    force_matrix = _build_force_model(model, movable_refs)

    if not force_matrix:
        if verbose:
            print("  Quadratic placement: no signal nets, skipping")
        return

    # Alternating solve + spread iterations
    for iteration in range(n_iterations):
        # Compute spreading forces based on current overlap
        anchor_forces = _compute_spreading_forces(
            model, movable, movable_refs, force_matrix, iteration, n_iterations,
            x_min, x_max, y_min, y_max,
        )

        # Legalization-aware feedback: simulate row assignment and add
        # displacement-penalty anchor forces for components that would be
        # moved far from their quadratic-optimal positions during legalization.
        # This produces positions that are both wirelength-optimal AND closer
        # to their eventual legalized positions, reducing the HPWL increase
        # caused by the Abacus row-based legalizer.
        if iteration >= 1:  # Only after first iteration (positions need to be reasonable)
            _add_legalization_anchors(
                model, movable, movable_refs, anchor_forces,
                grid_mm=GRID_SNAP, x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
            )

        # Add anchor forces for fixed components' contributions
        _add_fixed_anchors(model, force_matrix, movable_refs, ref_to_idx,
                           anchor_forces)

        # Solve the quadratic system for x and y
        new_x, new_y = _solve_quadratic(
            model, movable, force_matrix, ref_to_idx, anchor_forces,
            x_min, x_max, y_min, y_max,
        )

        # Update component positions with grid snapping
        for i, comp in enumerate(movable):
            comp.x = _grid_snap(new_x[i])
            comp.y = _grid_snap(new_y[i])

        if verbose:
            hpwl_now = _compute_hpwl_simple(model, movable_refs)
            n_overlaps = _count_overlaps(movable)
            print(f"    Iteration {iteration + 1}/{n_iterations}: "
                  f"HPWL={hpwl_now:.1f}, overlaps={n_overlaps}")

    if verbose:
        hpwl_after = _compute_hpwl_simple(model, movable_refs)
        pct = ((hpwl_before - hpwl_after) / hpwl_before * 100
               if hpwl_before > 0 else 0.0)
        print(f"  Quadratic placement done: HPWL "
              f"{hpwl_before:.1f} -> {hpwl_after:.1f} ({pct:+.1f}%)")


# =========================================================================
# Force model construction
# =========================================================================

def _build_force_model(
    model: "BoardModel",
    movable_refs: Set[str],
) -> Dict[Tuple[str, str], float]:
    """Build the quadratic force model from net connectivity.

    For each non-power net:
    - 2-pin net: weight 1.0 between the two components
    - Multi-pin net (P pins, clique model): weight 2/P per pair

    Returns:
        Dictionary mapping (ref_i, ref_j) -> weight for all
        pairs where at least one component is movable.
    """
    force_matrix: Dict[Tuple[str, str], float] = {}

    for net in model.nets:
        # Skip power nets — consistent with HPWL computation
        if _is_power_net(net.name):
            continue

        # Collect component refs on this net (deduplicate — one
        # component may have multiple pads on the same net)
        refs_on_net = list(set(ref for ref, _ in net.pins))
        P = len(refs_on_net)

        if P < 2:
            continue

        # Clique model: for P pins, each of the P*(P-1)/2 pairs
        # gets weight 2/P.  This ensures the total force equals
        # the star-model force and gives the correct quadratic
        # wirelength approximation.
        if P == 2:
            weight = 1.0
        else:
            weight = 2.0 / P

        # Add pairwise forces
        for i in range(P):
            ri = refs_on_net[i]
            for j in range(i + 1, P):
                rj = refs_on_net[j]

                # Only add if at least one is movable
                if ri not in movable_refs and rj not in movable_refs:
                    continue

                key = (ri, rj) if ri < rj else (rj, ri)
                force_matrix[key] = force_matrix.get(key, 0.0) + weight

    return force_matrix


def _add_fixed_anchors(
    model: "BoardModel",
    force_matrix: Dict[Tuple[str, str], float],
    movable_refs: Set[str],
    ref_to_idx: Dict[str, int],
    anchor_forces: Dict[str, Tuple[float, float, float]],
) -> None:
    """Add anchor forces from connections between movable and fixed components.

    When a movable component is connected to a fixed component, the
    fixed component acts as an anchor that pulls the movable one
    toward its position.
    """
    for (ri, rj), weight in force_matrix.items():
        ri_movable = ri in movable_refs
        rj_movable = rj in movable_refs

        if ri_movable and not rj_movable:
            # rj is fixed — it pulls ri toward rj's position
            comp_j = model.get_component(rj)
            if comp_j:
                existing = anchor_forces.get(ri, (0.0, 0.0, 0.0))
                w, tx, ty = existing
                anchor_forces[ri] = (w + weight, tx + weight * comp_j.x,
                                     ty + weight * comp_j.y)
        elif rj_movable and not ri_movable:
            # ri is fixed — it pulls rj toward ri's position
            comp_i = model.get_component(ri)
            if comp_i:
                existing = anchor_forces.get(rj, (0.0, 0.0, 0.0))
                w, tx, ty = existing
                anchor_forces[rj] = (w + weight, tx + weight * comp_i.x,
                                     ty + weight * comp_i.y)


# =========================================================================
# Quadratic solver
# =========================================================================

def _solve_quadratic(
    model: "BoardModel",
    movable: List["Component"],
    force_matrix: Dict[Tuple[str, str], float],
    ref_to_idx: Dict[str, int],
    anchor_forces: Dict[str, Tuple[float, float, float]],
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> Tuple[List[float], List[float]]:
    """Solve the quadratic placement system.

    Builds matrix A and vectors bx, by such that the optimal positions
    minimize the quadratic wirelength:
        minimize  0.5 * x^T A x - bx^T x  (and similarly for y)

    Matrix A is symmetric positive semi-definite.  With anchor forces
    it becomes positive definite (strictly diagonal dominant for the
    anchored rows), so Gauss-Seidel converges.

    For each force (i, j) with weight w:
        A[i,i] += w, A[j,j] += w, A[i,j] -= w, A[j,i] -= w

    For each anchor on component i with weight w, target (tx, ty):
        A[i,i] += w, bx[i] += w*tx, by[i] += w*ty

    Args:
        model: Board model
        movable: List of movable components
        force_matrix: Pairwise force weights
        ref_to_idx: Mapping from component ref to movable index
        anchor_forces: Dict ref -> (weight, target_x_sum, target_y_sum)
        x_min, x_max, y_min, y_max: Board bounds for clamping

    Returns:
        (new_x, new_y) lists of optimal positions for each movable component
    """
    n = len(movable)

    # Build diagonal and off-diagonal of A
    A_diag = [0.0] * n
    A_off_diag: Dict[Tuple[int, int], float] = {}
    bx = [0.0] * n
    by = [0.0] * n

    # Add connectivity forces
    for (ri, rj), weight in force_matrix.items():
        ii = ref_to_idx.get(ri)
        jj = ref_to_idx.get(rj)

        # Both must be movable for us to add matrix entries
        if ii is None or jj is None:
            continue

        A_diag[ii] += weight
        A_diag[jj] += weight

        key = (min(ii, jj), max(ii, jj))
        A_off_diag[key] = A_off_diag.get(key, 0.0) - weight

    # Add anchor forces
    for ref, (w, tx_sum, ty_sum) in anchor_forces.items():
        idx = ref_to_idx.get(ref)
        if idx is None:
            continue
        A_diag[idx] += w
        bx[idx] += tx_sum
        by[idx] += ty_sum

    # Ensure diagonal dominance (small regularization for disconnected
    # components that have no anchor forces)
    for i in range(n):
        if A_diag[i] < 1e-6:
            # Disconnected component — anchor it at its current position
            A_diag[i] += 1.0
            bx[i] += movable[i].x
            by[i] += movable[i].y

    # Solve Ax = bx and Ay = by using Gauss-Seidel
    # Start from current positions
    sol_x = [movable[i].x for i in range(n)]
    sol_y = [movable[i].y for i in range(n)]

    sol_x = _solve_linear_gauss_seidel(n, A_diag, A_off_diag, bx, sol_x)
    sol_y = _solve_linear_gauss_seidel(n, A_diag, A_off_diag, by, sol_y)

    # Clamp to board bounds (component center must allow full bbox inside)
    for i in range(n):
        hw = movable[i].effective_width / 2.0
        hh = movable[i].effective_height / 2.0
        sol_x[i] = max(x_min + hw, min(sol_x[i], x_max - hw))
        sol_y[i] = max(y_min + hh, min(sol_y[i], y_max - hh))

    return sol_x, sol_y


def _solve_linear_gauss_seidel(
    n: int,
    A_diag: List[float],
    A_off_diag: Dict[Tuple[int, int], float],
    b: List[float],
    x0: List[float],
    max_iter: int = 200,
    tol: float = 1e-6,
    omega: float = 1.3,
) -> List[float]:
    """Solve Ax = b using Gauss-Seidel iteration with SOR.

    A is stored as:
    - A_diag[i]: diagonal entry A[i,i]
    - A_off_diag[(i,j)] with i < j: off-diagonal entry A[i,j] = A[j,i]

    SOR (Successive Over-Relaxation) with omega > 1 accelerates
    convergence.  omega=1.3 is a good default for the diagonally-
    dominant matrices arising from force-directed placement.

    Args:
        n: Matrix dimension
        A_diag: Diagonal entries
        A_off_diag: Off-diagonal entries (i < j)
        b: Right-hand side vector
        x0: Initial guess (current positions)
        max_iter: Maximum iterations
        tol: Convergence tolerance (relative residual)
        omega: SOR relaxation factor (1.0 = pure Gauss-Seidel)

    Returns:
        Solution vector x
    """
    # Build row-based adjacency for efficient iteration
    # For each row i, store list of (j, A[i,j]) for off-diagonal entries
    row_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), val in A_off_diag.items():
        row_neighbors[i].append((j, val))
        row_neighbors[j].append((i, val))

    x = list(x0)

    for iteration in range(max_iter):
        max_delta = 0.0

        for i in range(n):
            if A_diag[i] < 1e-12:
                continue

            # Compute residual: r_i = b[i] - sum_j A[i,j] * x[j]
            sigma = 0.0
            for j, aij in row_neighbors[i]:
                sigma += aij * x[j]

            # Gauss-Seidel update with SOR
            x_new = (b[i] - sigma) / A_diag[i]
            x_old = x[i]
            x[i] = x_old + omega * (x_new - x_old)

            delta = abs(x[i] - x_old)
            if delta > max_delta:
                max_delta = delta

        # Check convergence
        if max_delta < tol:
            break

    return x


# =========================================================================
# Spreading forces
# =========================================================================

def _compute_spreading_forces(
    model: "BoardModel",
    movable: List["Component"],
    movable_refs: Set[str],
    force_matrix: Dict[Tuple[str, str], float],
    iteration: int,
    n_iterations: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute forces to spread overlapping components.

    For each overlapping pair (i, j), adds a repulsive anchor force
    pushing them apart.  The force acts along the axis of minimum
    overlap (same strategy as the existing push-apart code) and
    is implemented as an anchor force pulling each component away
    from the other.

    Force magnitude is proportional to overlap area, scaled by a
    factor that increases with iteration:
      - Early iterations: gentle spreading (don't disrupt wirelength opt)
      - Later iterations: stronger spreading (resolve remaining overlaps)

    Also adds position-hold forces that anchor each component to its
    current position, preventing collapse to the connectivity centroid
    and providing damping between iterations.

    Args:
        model: Board model
        movable: List of movable components
        movable_refs: Set of movable component refs
        force_matrix: Pairwise force weights (for hold-weight computation)
        iteration: Current outer iteration (0-based)
        n_iterations: Total outer iterations
        x_min, x_max, y_min, y_max: Board bounds

    Returns:
        Dict ref -> (total_weight, target_x_sum, target_y_sum)
        representing accumulated anchor forces
    """
    anchor_forces: Dict[str, Tuple[float, float, float]] = {}

    # Scale factor: gentle at first, stronger later
    # iteration 0: 0.5, iteration 1: 1.0, iteration 2: 2.0
    progress = (iteration + 1) / n_iterations
    base_scale = 0.5 * progress + 0.5 * progress * progress
    # base_scale ranges from ~0.5 to ~1.5 over the iterations

    n = len(movable)

    for i in range(n):
        ca = movable[i]
        for j in range(i + 1, n):
            cb = movable[j]

            # Check overlap
            ax1, ay1, ax2, ay2 = ca.bbox
            bx1, by1, bx2, by2 = cb.bbox

            # Overlap region
            ox1 = max(ax1, bx1)
            oy1 = max(ay1, by1)
            ox2 = min(ax2, bx2)
            oy2 = min(ay2, by2)

            if ox2 <= ox1 or oy2 <= oy1:
                continue  # No overlap

            overlap_x = ox2 - ox1
            overlap_y = oy2 - oy1
            overlap_area = overlap_x * overlap_y

            # Direction: push along axis of minimum overlap
            # (less displacement needed to resolve)
            dx = cb.x - ca.x
            dy = cb.y - ca.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 0.1:
                # Nearly coincident — pick a deterministic direction.
                # sum-of-ords is PYTHONHASHSEED-independent, unlike hash().
                pair_key = ca.ref + cb.ref
                angle = (sum(ord(c) for c in pair_key) % 360) * math.pi / 180.0
                dx, dy = math.cos(angle), math.sin(angle)
                dist = 1.0
            else:
                dx, dy = dx / dist, dy / dist

            # Force magnitude: proportional to overlap area
            # Scale increases with iteration to avoid early oscillation
            force_mag = overlap_area * base_scale * 2.0

            # Minimum force to ensure some separation
            min_gap = max(ca.effective_width, ca.effective_height,
                          cb.effective_width, cb.effective_height) * 0.3
            if dist < min_gap:
                force_mag += (min_gap - dist) * base_scale

            # Push ca in -direction, cb in +direction
            # Each gets an anchor force pulling it away from the other
            spread = force_mag * 0.5

            # Target for ca: push in -d direction
            target_ax = ca.x - dx * spread
            target_ay = ca.y - dy * spread

            # Target for cb: push in +d direction
            target_bx = cb.x + dx * spread
            target_by = cb.y + dy * spread

            # Anchor weight — higher for larger overlaps
            anchor_weight = force_mag * 0.5

            # Accumulate anchor forces for ca
            if ca.ref in anchor_forces:
                w, tx, ty = anchor_forces[ca.ref]
                anchor_forces[ca.ref] = (w + anchor_weight,
                                          tx + anchor_weight * target_ax,
                                          ty + anchor_weight * target_ay)
            else:
                anchor_forces[ca.ref] = (anchor_weight,
                                          anchor_weight * target_ax,
                                          anchor_weight * target_ay)

            # Accumulate anchor forces for cb
            if cb.ref in anchor_forces:
                w, tx, ty = anchor_forces[cb.ref]
                anchor_forces[cb.ref] = (w + anchor_weight,
                                          tx + anchor_weight * target_bx,
                                          ty + anchor_weight * target_by)
            else:
                anchor_forces[cb.ref] = (anchor_weight,
                                          anchor_weight * target_bx,
                                          anchor_weight * target_by)

    # Position-hold forces: anchor each component to its current
    # position.  This is essential for two reasons:
    #   1. Breaks translation invariance — the quadratic wirelength
    #      objective is invariant to global translation when all
    #      components are movable.  Without hold forces, the solver
    #      collapses everything to a point (the connectivity centroid).
    #   2. Damping — prevents oscillation between iterations by
    #      limiting how far components can jump in one solve step.
    #
    # The hold weight scales with iteration: strong initially (keeps
    # components near their grid-layout positions), weaker later
    # (allows more freedom to optimize wirelength).  This is the
    # standard "previous-position pseudo-net" technique used in
    # Kraftwerk2 and similar tools.
    #
    # The hold weight for each component is proportional to its total
    # connectivity weight (sum of force_matrix entries involving it),
    # scaled by a factor that decreases with iteration progress.
    # This ensures the hold is strong enough to prevent collapse but
    # weak enough to allow the solver to improve positions.

    # Compute per-component connectivity weight
    comp_conn_weight: Dict[str, float] = {c.ref: 0.0 for c in movable}
    for (ri, rj), w in force_matrix.items():
        if ri in comp_conn_weight:
            comp_conn_weight[ri] += w
        if rj in comp_conn_weight:
            comp_conn_weight[rj] += w

    # Hold factor: starts at 0.5 (50% of connectivity weight), drops to
    # 0.15 by the last iteration.  This provides strong initial damping
    # while still allowing significant wirelength optimization.
    hold_factor = 0.5 * (1.0 - progress * 0.7)  # 0.5 -> 0.15

    for comp in movable:
        conn_w = comp_conn_weight.get(comp.ref, 0.0)
        # Minimum hold weight of 0.1 for components with no connectivity
        hold_weight = max(hold_factor * conn_w, 0.1)

        if comp.ref in anchor_forces:
            w, tx, ty = anchor_forces[comp.ref]
            anchor_forces[comp.ref] = (w + hold_weight,
                                        tx + hold_weight * comp.x,
                                        ty + hold_weight * comp.y)
        else:
            anchor_forces[comp.ref] = (hold_weight,
                                        hold_weight * comp.x,
                                        hold_weight * comp.y)

    return anchor_forces


# =========================================================================
# Utility functions
# =========================================================================

def _grid_snap(value: float, grid: float = GRID_SNAP) -> float:
    """Snap a coordinate to the nearest grid point."""
    return round(value / grid) * grid


def _compute_hpwl_simple(
    model: "BoardModel",
    interior_refs: Set[str],
) -> float:
    """Compute HPWL using component centers (fast approximation).

    Unlike the full HPWL in cost_function.py which uses pad positions,
    this uses component centers for speed during iterative placement.
    This is consistent with the quadratic wirelength model which also
    uses component centers.
    """
    total = 0.0
    for net in model.nets:
        if _is_power_net(net.name):
            continue
        xs, ys = [], []
        for ref, _ in net.pins:
            if ref not in interior_refs:
                # Include fixed/edge components as anchor points
                comp = model.get_component(ref)
                if comp:
                    xs.append(comp.x)
                    ys.append(comp.y)
                continue
            comp = model.get_component(ref)
            if comp:
                xs.append(comp.x)
                ys.append(comp.y)
        if len(xs) >= 2:
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _count_overlaps(components: List["Component"]) -> int:
    """Count the number of overlapping component pairs."""
    count = 0
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if components[i].overlaps(components[j]):
                count += 1
    return count
