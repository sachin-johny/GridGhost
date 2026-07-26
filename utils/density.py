"""Single source of truth for placement density targets (Finding 6 fix).

Previously, three independent "target density" constants lived across
the codebase and disagreed by 2x:

  - ``kicad_parser._infer_board_from_components``:  TARGET_DENSITY = 0.35
    (controls how much the inferred board outline grows to give the
    placer room to push components apart)

  - ``place/legalizer.expand_bounds_to_fit``:       target_density = 0.55
    (controls when the legalizer's bounds expansion kicks in to fit
    macros that can't pack at the current bounds)

  - ``config.json`` legalization.bbox_expansion_density_threshold: 0.75
    (legacy legalizer's expansion threshold; dead code path in the
    active macro-v2 pipeline but still a source of confusion)

These all encode the same underlying design decision ("how packed is
too packed") but disagree. This module gives them ONE config-driven
value so they all agree.

The default 0.55 reflects the active pipeline's legalizer behavior —
55% density is a safe 2D packing target for irregular rectangles
(literature: 50-70% achievable for arbitrary rectangle bin-packing).
The infer-outline routine used to use 0.35 (more conservative — more
empty space), but matching the legalizer's threshold means the inferred
outline already has the right amount of room and the legalizer doesn't
need to expand again.

Override via config.json:
  {
    "placement": {
      "target_pack_density": 0.55
    }
  }
"""
from __future__ import annotations

# Module-level cache of the loaded value. None = not loaded yet.
_TARGET_PACK_DENSITY_CACHE: float | None = None


def target_pack_density() -> float:
    """Return the single shared target pack density (default 0.55).

    Loaded from config.json's ``placement.target_pack_density`` field.
    Falls back to 0.55 if config loading fails or the field is absent.
    """
    global _TARGET_PACK_DENSITY_CACHE
    if _TARGET_PACK_DENSITY_CACHE is not None:
        return _TARGET_PACK_DENSITY_CACHE
    try:
        from config import load_config
        cfg = load_config()
        val = getattr(cfg.placement, "target_pack_density", None)
        if val is None or val <= 0 or val >= 1:
            _TARGET_PACK_DENSITY_CACHE = 0.55
        else:
            _TARGET_PACK_DENSITY_CACHE = float(val)
    except Exception:
        _TARGET_PACK_DENSITY_CACHE = 0.55
    return _TARGET_PACK_DENSITY_CACHE


def reset_cache() -> None:
    """Reset the cached value. Useful for tests that swap config.json."""
    global _TARGET_PACK_DENSITY_CACHE
    _TARGET_PACK_DENSITY_CACHE = None
