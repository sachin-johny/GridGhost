"""Pre-SA placement passes.

Currently provides pre-placement of decoupling caps adjacent to their
assigned ICs. Gives SA a good starting configuration so the group-aware
density grid has correct grouping from t=0, instead of relying on SA
to drag caps into position.
"""

from __future__ import annotations


from models.board_model import BoardModel
from profiles.board_profiles import ConstraintRule
from engine.constraint_evaluator import _build_decoupling_map
from legalization.legalizer import _nudge_caps_to_ics


def preplace_caps_near_ics(
    model: BoardModel,
    rules: list[ConstraintRule],
    interior_bbox: tuple[float, float, float, float] | None = None,
    verbose: bool = False,
) -> int:
    """Snap each decoupling cap to a free slot adjacent to its assigned IC.

    Thin wrapper over the legalizer's ``_nudge_caps_to_ics`` — that function
    already implements the 8-slot fan search (4 cardinal + 4 diagonal at
    increasing spacings) with overlap checking.

    Caps are NOT marked ``is_fixed`` — SA still moves them. Group-aware
    translate (moves.py) keeps them attached to their IC during optimization.

    Returns the number of caps in the decoupling map (i.e. the upper bound
    on caps that may have been repositioned).
    """
    has_decap_rule = any(
        r.name == 'decoupling_proximity' and r.enabled for r in rules
    )
    if not has_decap_rule:
        return 0

    decap_map = _build_decoupling_map(model)
    if not decap_map:
        return 0

    # Snapshot positions so we can count actual moves.
    before = {c.ref: (c.x, c.y, c.rotation) for c in model.components}

    _nudge_caps_to_ics(
        model, rules,
        interior_bbox=interior_bbox,
        verbose=verbose,
        cached_decap_map=decap_map,
    )

    cap_refs = {cap for caps in decap_map.values() for cap in caps}
    moved = 0
    for ref in cap_refs:
        comp = next((c for c in model.components if c.ref == ref), None)
        if comp is None:
            continue
        prev = before.get(ref)
        if prev is None:
            continue
        if (comp.x, comp.y, comp.rotation) != prev:
            moved += 1
    return moved
