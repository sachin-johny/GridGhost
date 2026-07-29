"""SA-based legalization polish phase.

Context: the default legalizer (``place/legalizer.py``) resolves
overlaps with pure greedy descent — push-apart and force-spread only
ever accept a move that doesn't increase (or, for force-spread, that
reduces) the accepting macro's own overlap count. Pure greedy descent
gets stuck in local minima: a macro boxed in by 3-4 overlapping
neighbors sometimes cannot reach a better configuration without a move
that temporarily makes ITS OWN overlap count worse (e.g. swap past a
neighbor, or step through a third macro's space to reach open room on
the far side). Greedy descent can't take that step even when it leads
somewhere better one move later — this is exactly why test6 (39%
density) has residual overlaps despite having room, and why
force-spread was added on top as a second heuristic to try to route
around the problem instead of fixing the underlying search strategy.

A separately-tried fix (bridging in the legacy Abacus row-DP
legalizer, see ``place/abacus_bridge.py``) made this WORSE, not
better — its row-binning model doesn't fit macro-v2's size-variable
macro list, and it discarded most of SA's HPWL optimization doing a
large, non-local re-placement. That confirmed the actual problem isn't
"we need a different algorithm family with a formal guarantee" — rows
don't exist on a PCB, that guarantee doesn't transfer. The problem is
squarely that greedy descent is the wrong SEARCH STRATEGY for this
landscape, and the fix already lives in this codebase: run_macro_sa
IS a general local-search engine with an accept-temporarily-worse-
moves criterion built in (that's what simulated annealing is). This
module reuses it directly, unmodified, as a legalization polish phase:

  - beta cranked far above the main SA's value, so overlap dominates
    the cost function almost completely.
  - alpha kept small but nonzero — a slight pull toward not needlessly
    lengthening wires while resolving overlaps, so this stays a LOCAL
    polish, not a second full placement pass (the mistake the Abacus
    bridge made).
  - a small translation window, for the same "stay local" reason.
  - a higher displace_prob than the main SA, since directly shoving a
    concretely-overlapping neighbor is the most productive move type
    once overlap is what's being optimized.
  - only runs at all if the incoming placement actually has residual
    overlaps — a clean board costs nothing extra.

Skips grid re-snapping after (matching the existing push-apart /
force-spread / Tetris passes, none of which preserve grid alignment
either — that's an existing, unrelated soft target in this pipeline).

MEASURED RESULT (seed=42, six bundled test boards), final config —
staged beta ramp (4 stages, beta 60→800 geometric, window 5mm→0.2mm)
+ overlap-biased move selection, vs. the default greedy heuristic:

    board       heuristic          sa_polish
    test4       0 ovl, hpwl 1870.8 1 ovl,  hpwl 1823.0 (close, not quite)
    cbb         0 ovl, hpwl 4095.9 0 ovl,  hpwl 4048.6  <- wins
    cbbwO       0 ovl, hpwl 4216.7 0 ovl,  hpwl 4165.9  <- wins
    test5       0 ovl, hpwl  444.8 0 ovl,  hpwl  427.7  <- wins
    test6       5 ovl, area  8.3   5 ovl,  area 24.2    <- loses
    th_sensor   0 ovl, hpwl  156.1 0 ovl,  hpwl  156.4  <- near-tie

Three outright wins (better HPWL, still overlap-free) is a real result
— a single fixed high beta (the first version tried) never beat the
heuristic on ANY board; ramping beta instead of fixing it, plus
spending the move budget on the macros actually in conflict, closed
most of that gap. But it did NOT solve the board this was originally
aimed at (test6 — the board with residual overlaps that motivated all
three attempts in the first place): same overlap COUNT as the
heuristic, worse overlap AREA/HPWL. Scaling the iteration budget with
macro count was tried as a follow-up (larger boards get a fixed 4000
iterations same as small ones, which seemed likely to matter) — it
did NOT produce a reliable improvement. Three different iteration-
budget formulas produced three different winners/losers across
test4/th_sensor with no clear monotonic trend, which is a sign of
fitting stochastic noise on 6 boards rather than a real effect, so the
fixed iteration count was kept rather than chase that further.

Net honest assessment: this is the best of the three attempts (plain
greedy, abacus bridge, single-shot SA-polish, staged SA-polish) and
demonstrates the original diagnosis was right — search strategy, not
algorithm family, was the lever that mattered. But it's still not a
strict win: default legalizer stays ``heuristic`` unless/until this
also closes the gap on the board it was built to fix. Available via
``--legalizer sa_polish`` for comparison and further work — e.g. a
proper multi-seed statistical comparison instead of single-seed
numbers, before trusting any further tuning on top of this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from place.sa import run_macro_sa

if TYPE_CHECKING:
    from models.board_model import BoardModel
    from models.macro import Macro


def sa_polish_legalize(
    model: "BoardModel",
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    *,
    grid_mm: float = 1.0,
    iterations: int = 4000,
    beta_start: float = 60.0,
    beta_end: float = 800.0,
    n_stages: int = 4,
    alpha: float = 0.2,
    gamma: float = 40.0,
    window_start_mm: float = 5.0,
    window_end_mm: float = 0.2,
    bias_overlap_prob: float = 0.7,
    seed: int = 42,
    verbose: bool = False,
    per_macro_keepout=None,
) -> dict[str, int]:
    """Run a staged, overlap-weighted SA pass to legalize ``macros``.

    Two changes from the single-shot version this replaced (see the
    module docstring's "REVISION 2" section for why):

    1. A geometric beta ramp across ``n_stages`` sequential
       ``run_macro_sa`` calls (each stage reheats fresh, seeded from
       where the previous stage left off) instead of one fixed beta
       for the whole budget. Early stages have low beta and a wide
       window — free to explore, including moves that temporarily
       raise overlap, which is what a stuck macro needs. Late stages
       have very high beta and a narrow window — overlap must
       essentially not exist in an accepted move, forcing convergence
       instead of leaving residual stochastic noise. A single fixed
       high beta (the REVISION 1 approach) spends its ENTIRE budget in
       the "must not overlap" regime, which is exactly what made it
       bad at escaping local minima in the first place.
    2. ``bias_overlapping=True`` in every stage (see ``place/sa.py``):
       concentrate proposed moves on the macros actually in conflict
       instead of picking uniformly among all macros, most of which
       have nothing to do with the residual overlaps.

    Unlike the main placement SA (beta=25 by default, balancing HPWL
    against overlap for the whole board), this exists purely to
    eliminate residual overlaps left after the main SA + grid_snap.
    """
    from place.legalizer import _count_residual_overlaps, boundary_clamp, push_apart_overlapping

    residual_before = _count_residual_overlaps(macros)
    if residual_before == 0:
        return {"residual_overlaps": 0, "boundary_failures": 0}

    if verbose:
        print(f"  SA-polish: {residual_before} residual overlaps entering polish phase")

    iters_per_stage = max(1, iterations // n_stages)
    ratio = (beta_end / beta_start) ** (1.0 / max(1, n_stages - 1)) if n_stages > 1 else 1.0
    win_ratio = (window_end_mm / window_start_mm) ** (1.0 / max(1, n_stages - 1)) if n_stages > 1 else 1.0

    stage_beta = beta_start
    stage_window = window_start_mm
    for stage in range(n_stages):
        run_macro_sa(
            model, macros, bounds,
            iterations=iters_per_stage,
            reheats=0,
            alpha=alpha, beta=stage_beta, gamma=gamma,
            initial_window_mm=stage_window,
            final_window_mm=max(stage_window * win_ratio, window_end_mm),
            rotate_prob=0.05,
            swap_prob=0.05,
            displace_prob=0.35,
            bias_overlapping=True,
            bias_overlap_prob=bias_overlap_prob,
            seed=seed + stage,  # different move sequence per stage
            verbose=False,
        )
        if verbose:
            r = _count_residual_overlaps(macros)
            print(f"  SA-polish: stage {stage+1}/{n_stages} "
                  f"(beta={stage_beta:.0f}, window={stage_window:.1f}mm) -> {r} residual")
        stage_beta *= ratio
        stage_window *= win_ratio

    after_sa = _count_residual_overlaps(macros)

    # Greedy cleanup tail: even the final, highest-beta stage can leave
    # small residual noise (its acceptance rule isn't literally "reject
    # all overlap-increasing moves", it's "reject with very high
    # probability"). A short, cheap push-apart pass mops that up,
    # starting from SA's already-largely-resolved layout rather than
    # the original packed starting position.
    if after_sa > 0:
        push_apart_overlapping(macros, bounds, max_passes=100)

    failed = boundary_clamp(macros, bounds, per_macro_keepout=per_macro_keepout)
    residual_after = _count_residual_overlaps(macros)

    if verbose:
        print(
            f"  SA-polish: {residual_before} -> {after_sa} (SA, {n_stages} stages) -> "
            f"{residual_after} (after greedy cleanup tail) overlaps"
        )

    return {"residual_overlaps": residual_after, "boundary_failures": failed}
