"""Bridge: run the legacy row-based Abacus DP legalizer against
macro-v2's rigid ``Macro`` objects.

Context (see the code review that led to this file): macro-v2 shipped
with its own from-scratch legalizer (``place/legalizer.py``) — a stack
of greedy heuristics (push-apart, force-spread, Tetris slot search)
accreted one patch at a time, with no correctness guarantee, that
still leaves residual overlaps on real boards. Meanwhile
``legalization/abacus_legalizer.py`` already implements a real,
well-understood algorithm (Brenner 2005 cluster-merge DP: minimum
squared-displacement placement within a row, subject to a hard
non-overlap constraint) — but it operates on raw ``Component`` objects
via the legacy engine's OWN macro-tracking (``engine.group_moves``'s
``decap_map``), not on ``models.macro.Macro``. The two engines don't
speak the same macro representation, so the new pipeline couldn't use
it directly.

This module is the bridge, not a rewrite of either side:

1. Represent each non-fixed ``Macro`` as ONE synthetic rectangular
   proxy ``Component`` sized to the macro's bbox (leader ∪ followers,
   the same union used everywhere else in macro-v2 for overlap
   checks). Fixed macros (connectors, mounting holes) become fixed
   proxies, so Abacus still avoids them as obstacles.
2. Followers are NOT included in the proxy model at all — only one
   proxy per macro. This sidesteps the legacy decap_map/
   propagate_ic_move machinery entirely (verified safe: with no cap
   components present, ``get_decap_map`` returns ``{}`` and every
   propagation call becomes a no-op — see
   ``tests/test_abacus_bridge.py``).
3. Run ``abacus_legalize`` on the proxy model.
4. Abacus only ever changes x/y (never rotation — the row/cluster DP
   has no rotation step). Read back each proxy's (dx, dy) and apply it
   as a RIGID translation to the real macro: move the leader by
   (dx, dy) and call ``macro.apply_offsets()`` so followers are
   re-derived from their (unchanged) offsets. This is exactly what
   ``Macro.translate()`` does, just applied directly since we already
   know the delta is safe (Abacus already resolved overlaps against
   the proxy's own footprint, which IS the macro's true footprint).

This does not claim Abacus is a perfect legalizer either — a row-based
DP legalizer's row model assumes reasonably uniform row heights, and a
handful of wildly different macro sizes (a large MCU next to 0402
caps) can still leave a residual overlap or two. Callers should treat
this as a strictly-better first pass, not a guarantee, and keep
``place.legalizer``'s keepout/Tetris cleanup as a fallback safety net
for whatever this doesn't resolve — see ``place/legalizer.py``'s
``legalize(..., use_abacus=True)``.

MEASURED RESULT (not just a theoretical caveat): on the six bundled
test boards, this bridge is currently WORSE than the greedy heuristic
stack it was meant to replace — more residual overlaps on 2/6 boards,
10-30% worse final HPWL on all 6, with average per-macro displacement
of 3.6-17mm (vs. a board that's often only 50-100mm across). The
"uniform row height" assumption above isn't a minor edge case here —
it's the normal case for macro-v2, where the proxy list mixes lone
0402 passives with 15mm IC+cap clusters in the SAME row-assignment
pass, and ``_assign_rows``'s adaptive row-count/median-height logic
(tuned for the legacy pipeline's much more uniform per-component
lists) reacts by forcing a small number of coarse rows that drag
macros far from the Y position SA already optimized for HPWL. Fixing
this properly means either (a) a row model that accounts for macro
bbox variance instead of a single global median height, or (b) a
"minimal perturbation" DP mode that penalizes distance from the
current position instead of re-bucketing by raw Y — both real
algorithm-design work, not a config tweak. Until one of those lands,
``use_abacus`` defaults to False (see ``place/legalizer.py``); this
module is kept for comparison and as a starting point for that work,
not because it currently wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.board_model import BoardModel, BoardOutline, Component
from legalization.abacus_legalizer import abacus_legalize

if TYPE_CHECKING:
    from models.macro import Macro


def _macro_proxy(macro: "Macro") -> Component:
    """One rectangular Component standing in for the whole rigid macro."""
    x1, y1, x2, y2 = macro.bbox
    leader = macro.leader
    proxy = Component(
        ref=leader.ref,
        footprint=leader.footprint,
        value=leader.value,
        x=(x1 + x2) / 2.0,
        y=(y1 + y2) / 2.0,
        rotation=0.0,
        width=max(x2 - x1, 1e-6),
        height=max(y2 - y1, 1e-6),
        courtyard_margin=0.0,  # bbox already includes each member's margin
        component_type=leader.component_type,
    )
    proxy.is_fixed = macro.is_fixed
    return proxy


def abacus_legalize_macros(
    model: "BoardModel",
    macros: list["Macro"],
    bounds: tuple[float, float, float, float],
    grid_mm: float = 1.0,
    verbose: bool = False,
) -> dict[str, int]:
    """Legalize ``macros`` (rigid rectangles) with the Abacus row DP.

    Applies the resulting positions back to the real macros as rigid
    translations (leader + ``apply_offsets()``), preserving the exact
    cap-IC offset geometry macro-v2 depends on. Only x/y change; macro
    rotation is left untouched (Abacus never rotates).

    Returns ``{"residual_overlaps": int, "boundary_failures": int}``
    measured against the PROXY footprints (i.e. the same rigid-macro
    bboxes used everywhere else — an apples-to-apples comparison with
    ``place.legalizer.legalize()``'s own counts).
    """
    if not macros:
        return {"residual_overlaps": 0, "boundary_failures": 0}

    proxies = [_macro_proxy(m) for m in macros]
    orig_centers = [(p.x, p.y) for p in proxies]

    proxy_board = BoardOutline(bounds[0], bounds[1], bounds[2], bounds[3])
    proxy_model = BoardModel(board=proxy_board, components=proxies, nets=model.nets)

    abacus_legalize(
        proxy_model,
        grid_mm=grid_mm,
        verbose=verbose,
        interior_bbox=bounds,
        cached_decap_map=None,
    )

    x_min, y_min, x_max, y_max = bounds
    for macro, proxy, (old_cx, old_cy) in zip(macros, proxies, orig_centers):
        if macro.is_fixed:
            continue
        dx = proxy.x - old_cx
        dy = proxy.y - old_cy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        macro.leader.x += dx
        macro.leader.y += dy
        macro.apply_offsets()

    residual_overlaps = 0
    for i in range(len(proxies)):
        for j in range(i + 1, len(proxies)):
            if proxies[i].overlaps(proxies[j]):
                residual_overlaps += 1

    boundary_failures = 0
    for macro in macros:
        if macro.is_fixed:
            continue
        bx1, by1, bx2, by2 = macro.bbox
        if bx1 < x_min - 1e-6 or by1 < y_min - 1e-6 or bx2 > x_max + 1e-6 or by2 > y_max + 1e-6:
            boundary_failures += 1

    if verbose:
        print(
            f"  Abacus bridge: {residual_overlaps} residual overlaps, "
            f"{boundary_failures} boundary failures (pre-keepout/Tetris cleanup)"
        )

    return {"residual_overlaps": residual_overlaps, "boundary_failures": boundary_failures}
