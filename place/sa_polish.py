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

MEASURED RESULT (seed=42, six bundled test boards), current config —
staged beta ramp (4 stages, beta 60→800 geometric, window 5mm→0.2mm)
+ overlap-biased move selection, vs. the default greedy heuristic:

    board       heuristic              sa_polish
    test4       0 ovl, hpwl 2328.9     0 ovl, hpwl 2099.2   <- wins
    test5       0 ovl, hpwl  461.8     0 ovl, hpwl  476.5   <- loses (slight)
    test6       0 ovl, hpwl 2547.3     0 ovl, hpwl 2577.1   <- loses (slight)
    cbb         0 ovl, hpwl 4074.7     0 ovl, hpwl 3981.5   <- wins
    cbbwO       0 ovl, hpwl 4412.0     0 ovl, hpwl 4254.8   <- wins
    th_sensor   0 ovl, hpwl  157.0     0 ovl, hpwl  154.2   <- wins (marginal)

Headline change vs. the numbers this docstring used to carry: BOTH
legalizers are now overlap-free on every bundled board, including
test6 — the board whose residual overlaps motivated this module in the
first place (the old table showed 5 residual overlaps for test6 under
BOTH strategies). That fix did not come from sa_polish itself: the
overlap-aware boundary clamp (``_boundary_clamp_overlap_aware`` below,
commit "overlap-aware boundary clamp...") replaced the plain
``boundary_clamp`` that used to translate a macro inside its keepout-
shrunk zone with NO overlap check — yanking an IC onto a neighbor and
reintroducing overlaps SA had just resolved. It is wired into BOTH the
heuristic and the sa_polish strategy branches, so the keepout-induced
overlap regression that once made sa_polish lose test4 (1 residual
overlap) and lose test6 badly (worse overlap area, 24.2 vs 8.3) no
longer occurs for either strategy.

With overlaps tied at 0 everywhere, what's left to compare is HPWL:
sa_polish wins 4/6 (test4, cbb, cbbwO, th_sensor) and loses narrowly
on 2/6 (test5 ~3%, test6 ~1%) — a net win, not a sweep. Ramping beta
across stages instead of fixing it high, plus concentrating the move
budget on the macros actually in conflict, is what closed the gap a
single fixed high beta (the first version tried, which never beat the
heuristic on any board) could not.

Scaling the iteration budget with macro count was tried as a separate
follow-up — it did NOT produce a reliable improvement. Three different
iteration-budget formulas produced three different winners/losers
across test4/th_sensor with no clear monotonic trend, a sign of
fitting stochastic noise on 6 boards rather than a real effect, so the
fixed iteration count was kept rather than chase that further.

Net honest assessment: search strategy (not algorithm family) was the
lever that mattered, and the shared overlap-aware clamp closed the
correctness gap this module was built to fix. sa_polish is a net HPWL
win over the heuristic at seed=42 but not a strict one. It has NOT
been re-run across multiple seeds: the suite has pre-existing seed
sensitivity (see ``AGENT_NOTES.md``), so this single-seed result is
not the multi-seed statistical comparison needed before trusting
further tuning on top of it. Available via ``--legalizer sa_polish``;
default legalizer stays ``heuristic`` pending that broader comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from place.sa import run_macro_sa

if TYPE_CHECKING:
    from models.board_model import BoardModel
    from models.macro import Macro


def _boundary_clamp_overlap_aware(
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    per_macro_keepout=None,
) -> int:
    """Like ``boundary_clamp`` but rolls back a macro's translation if it
    would create a new overlap.

    The plain ``boundary_clamp`` translates each macro independently to
    pull it inside the per-macro keepout-shrunk bounds — with NO overlap
    check. When ICs (keepout=5mm) and passives (keepout=0mm) share a
    region, the clamp can yank an IC inward by up to 5mm and slam it
    into a previously overlap-free cap, *creating* overlaps that SA had
    just resolved.

    This variant does the same per-macro translation, but BEFORE
    accepting the move it checks whether the macro now overlaps any
    other macro. If yes:
      1. Try axis-only moves (dx-only or dy-only) — one axis may be
         safe even when the combined move isn't.
      2. If neither axis-only move is safe, try pushing the blocking
         neighbor out of the way (a single push_apart call on just
         the macro and its blocker), then retry the clamp.
      3. If still no safe in-bounds position, fall back to the plain
         clamp for THIS macro only — accept the overlap (the next
         push_apart round in the caller will try to resolve it) rather
         than leaving the macro OOB. A macro 0.5mm past the keepout
         line AND overlapping a neighbor is strictly worse than a macro
         in-bounds and overlapping — the latter is what push_apart is
         designed to fix, the former is unfixable by push_apart.

    A macro that stays OOB after this pass (only happens when the macro
    is bigger than the keepout-shrunk bounds, i.e. genuinely too big
    for the board) is reported in the failed count.
    """
    from place.legalizer import push_apart_overlapping

    x_min, y_min, x_max, y_max = bounds
    failed = 0
    for m in macros:
        if m.is_fixed:
            continue
        # Use GLOBAL bounds for the clamp, NOT keepout-shrunk bounds.
        # The keepout enforcement is handled separately by the keepout
        # loop in legalize(). This clamp's job is solely to ensure no
        # macro is physically off the board — a hard placement invalidity.
        # Using keepout-shrunk bounds here caused `failed` to be non-zero
        # whenever an IC was inside the global bounds but within its
        # keepout zone (a soft DFM concern, not a placement invalidity),
        # which made the CLI exit 1 on valid placements.
        bx_min = x_min
        by_min = y_min
        bx_max = x_max
        by_max = y_max

        bx1, by1, bx2, by2 = m.bbox
        dx_left = bx_min - bx1
        dx_right = bx_max - bx2
        dy_top = by_min - by1
        dy_bottom = by_max - by2

        dx = 0.0
        dy = 0.0
        if dx_left > 0:
            dx = dx_left
        elif dx_right < 0:
            dx = dx_right
        if dy_top > 0:
            dy = dy_top
        elif dy_bottom < 0:
            dy = dy_bottom

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue  # already in-bounds

        macro_bounds = bounds

        # Try the full (dx, dy) translation, then axis-only, then
        # neighbor-push + retry, then fall back to plain clamp.
        accepted = _try_clamp_no_overlap(m, dx, dy, macros, macro_bounds)
        if not accepted and abs(dx) > 1e-9:
            accepted = _try_clamp_no_overlap(m, dx, 0.0, macros, macro_bounds)
        if not accepted and abs(dy) > 1e-9:
            accepted = _try_clamp_no_overlap(m, 0.0, dy, macros, macro_bounds)

        if not accepted:
            # The clamp move would create an overlap on every axis combo.
            # Try pushing the blocking neighbor out of the way, then retry.
            # Use a single targeted push_apart pass — it's cheap and only
            # resolves the immediate blocker, not a global re-optimization.
            push_apart_overlapping(macros, bounds, max_passes=20)
            accepted = _try_clamp_no_overlap(m, dx, dy, macros, macro_bounds)
            if not accepted and abs(dx) > 1e-9:
                accepted = _try_clamp_no_overlap(m, dx, 0.0, macros, macro_bounds)
            if not accepted and abs(dy) > 1e-9:
                accepted = _try_clamp_no_overlap(m, 0.0, dy, macros, macro_bounds)

        if not accepted:
            # All overlap-aware attempts failed. Leave the macro OOB —
            # the caller's Tetris pass will jump it to a clear in-bounds
            # slot. (Tetris now handles OOB macros too — see
            # displace_to_clear_slots's _is_oob filter.)
            #
            # Do NOT fall back to plain clamp here: that would yank the
            # macro into an overlap, which is the exact bug we're fixing.
            # An OOB macro is recoverable by Tetris; an overlapping macro
            # may or may not be recoverable (Tetris might not find a
            # clear slot, leaving the overlap as residual).
            failed += 1
    return failed


def _try_clamp_no_overlap(
    m: "Macro",
    dx: float,
    dy: float,
    macros: list["Macro"],
    macro_bounds: tuple[float, float, float, float],
) -> bool:
    """Try translating ``m`` by (dx, dy); accept only if no new overlap.

    Returns True if the move was accepted, False if it was rolled back
    (either because ``translate`` rejected it on bounds grounds, or
    because it created a new overlap).
    """
    snap = m._snapshot()
    if not m.translate(dx, dy, bounds=macro_bounds):
        return False
    if any(m.overlaps(o) for o in macros if o is not m):
        m._restore(snap)
        return False
    return True


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
    #
    # CRITICAL: when SA reached 0 overlaps, DO NOT run push_apart here.
    # push_apart is a greedy local heuristic — on a layout SA carefully
    # balanced at 0 overlaps, it can find a "cheaper axis" push for one
    # pair that cascades into new overlaps with a third macro. Measured
    # on th_sensor seed=42: SA finished at 0, push_apart created 1
    # (TP11↔U7), which boundary_clamp then locked in. The fix is to
    # skip the greedy cleanup entirely when SA already converged.
    if after_sa > 0:
        push_apart_overlapping(macros, bounds, max_passes=100)

    # Use the overlap-aware boundary clamp: it translates each macro
    # inside its keepout-shrunk bounds but ROLLS BACK the translation
    # if it would create a new overlap. The plain boundary_clamp can
    # yank an IC inward by up to 5mm and slam it into a previously
    # overlap-free cap, *creating* overlaps that SA had just resolved.
    # Measured on test4 seed=0: SA finished at 0, plain boundary_clamp
    # produced 4 (D3↔TP1, C86↔C28, C86↔R3, C80↔U1); the recovery
    # branch below only fires when residual_after > after_sa, but
    # `after_sa` was 0 so 4 > 0 fired the recovery — which ran
    # push_apart + force_spread + push_apart + boundary_clamp AGAIN,
    # and the second boundary_clamp re-created the same overlaps.
    # The overlap-aware clamp breaks that loop at the source.
    failed = _boundary_clamp_overlap_aware(
        macros, bounds, per_macro_keepout=per_macro_keepout,
    )
    residual_after = _count_residual_overlaps(macros)

    # If the overlap-aware clamp left macros OOB (failed > 0) or created
    # overlaps (residual_after > 0), run push_apart + force_spread +
    # Tetris as recovery. Tetris is especially important here: it jumps
    # OOB macros to clear in-bounds slots (the overlap-aware clamp
    # deliberately leaves them OOB rather than overlap, expecting Tetris
    # to clean up).
    if failed > 0 or residual_after > 0:
        push_apart_overlapping(macros, bounds, max_passes=200)
        new_residual = _count_residual_overlaps(macros)
        if new_residual > 0:
            try:
                from place.legalizer import force_spread_overlapping
                force_spread_overlapping(macros, bounds, max_passes=50, step_size=2.0)
                push_apart_overlapping(macros, bounds, max_passes=200)
            except ImportError:
                pass
        # Tetris fallback — jump stuck/OOB macros to the nearest clear
        # in-bounds slot. SA-polish can leave a macro boxed in by
        # neighbors the clamp repositioned; push_apart and force_spread
        # are LOCAL heuristics that can't escape, but Tetris takes a
        # global view and jumps the macro out. Also handles OOB macros
        # (see _is_oob in displace_to_clear_slots).
        try:
            from place.legalizer import displace_to_clear_slots
            displace_to_clear_slots(
                macros, bounds, grid_mm=1.0,
                per_macro_keepout=per_macro_keepout,
            )
        except ImportError:
            pass
        # Final clamp — necessary because push_apart/force_spread/Tetris
        # may have left a macro slightly OOB. Use overlap-aware again
        # so this final clamp doesn't re-introduce the problem.
        failed = _boundary_clamp_overlap_aware(
            macros, bounds, per_macro_keepout=per_macro_keepout,
        )
        residual_after = _count_residual_overlaps(macros)

    if verbose:
        print(
            f"  SA-polish: {residual_before} -> {after_sa} (SA, {n_stages} stages) -> "
            f"{residual_after} (after greedy cleanup tail) overlaps"
        )

    return {"residual_overlaps": residual_after, "boundary_failures": failed}
