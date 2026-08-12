"""Tests for the footprint/pad rotation fix — IMPROVEMENTS §2.2.

KiCad's pad transform is ``world_rot = footprint_R + pad_R``: the
footprint's ``(at x y R)`` rotates the whole body, and each pad's local
``(at X Y r)`` is composed on top of it. A previous version of
``_update_position`` ALSO added the placer's rotation delta to every
pad's local ``at``, giving ``world_rot = footprint_R + pad_R + delta`` —
double-rotating pads in the world frame. Latent on symmetric 2-pin
parts (180° is a world-frame no-op) but visibly wrong on asymmetric
footprints (SOIC, QFN, connectors with offset pin 1).

The fix: update ONLY the footprint's ``(at ...)`` (the first one in the
block) and leave every pad ``(at ...)`` untouched. These tests pin that.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.board_model import BoardModel, BoardOutline, Component
from parsers.placement_writer import _apply_footprint_positions, _update_position


# ---------------------------------------------------------------------------
# A realistic asymmetric footprint: footprint (at) FIRST, then properties
# with their own (at), then four SMD pads each carrying a LOCAL rotation
# (the 3-arg (at X Y r) form — exactly what the old code corrupted).
# ---------------------------------------------------------------------------
SOIC_BLOCK = """(footprint "Test:SOIC-8"
	(layer "F.Cu")
	(at 0 0)
	(property "Reference" "U1" (at 0 -5 0) (layer "F.SilkS"))
	(property "Value" "MCU" (at 0 5 0) (layer "F.Fab"))
	(pad "1" smd rect (at -2 -1.27 90) (size 0.6 0.8) (layers "F.Cu" "F.Mask" "F.Paste"))
	(pad "2" smd rect (at -2 1.27 90) (size 0.6 0.8) (layers "F.Cu" "F.Mask" "F.Paste"))
	(pad "3" smd rect (at 2 -1.27 90) (size 0.6 0.8) (layers "F.Cu" "F.Mask" "F.Paste"))
	(pad "4" smd rect (at 2 1.27 90) (size 0.6 0.8) (layers "F.Cu" "F.Mask" "F.Paste"))
)"""


def _pad_at_strings(block):
    """Return every pad-local (at X Y r) substring, in order."""
    # Only (at ...) that live INSIDE a (pad ...) form — match the pad lines.
    return re.findall(r'\(pad[^)]*\(at\s+[^\)]*\)', block)


# ---------------------------------------------------------------------------
# _update_position — unit level
# ---------------------------------------------------------------------------

def test_footprint_at_updated_with_rotation():
    comp = Component(ref="U1", footprint="SOIC-8", value="MCU",
                     x=10.0, y=20.0, rotation=90.0, component_type="ic")
    out = _update_position(SOIC_BLOCK, comp)
    # The FIRST (at ...) is now the new footprint placement.
    assert "(at 10 20 90)" in out
    # It really is the first one.
    assert re.search(r'\(at 10 20 90\)', out).start() < out.index("pad")


def test_zero_rotation_drops_rotation_field():
    comp = Component(ref="U1", footprint="SOIC-8", value="MCU",
                     x=12.5, y=-3.0, rotation=0.0, component_type="ic")
    out = _update_position(SOIC_BLOCK, comp)
    assert "(at 12.5 -3)" in out
    assert "(at 12.5 -3 0)" not in out


def test_pad_local_at_values_unchanged_by_rotation_delta():
    """The §2.2 regression: writing a 90° footprint rotation must NOT
    mutate any pad's local (at X Y r). Each pad keeps its own 90° — so
    its WORLD rotation is footprint(90) + pad(90) = 180°, not the
    footprint(90) + pad(90) + delta(90) = 270° the old code produced."""
    comp = Component(ref="U1", footprint="SOIC-8", value="MCU",
                     x=10.0, y=20.0, rotation=90.0, component_type="ic")
    out = _update_position(SOIC_BLOCK, comp)

    original_pads = _pad_at_strings(SOIC_BLOCK)
    written_pads = _pad_at_strings(out)
    assert len(written_pads) == len(original_pads) == 4
    for before, after in zip(original_pads, written_pads):
        assert after == before, f"pad (at ...) mutated: {before!r} -> {after!r}"


def test_property_at_values_also_unchanged():
    """Property (at ...) fields come after the footprint (at) but before
    the pads; count=1 must skip them too (they're footprint-relative)."""
    comp = Component(ref="U1", footprint="SOIC-8", value="MCU",
                     x=1.0, y=2.0, rotation=45.0, component_type="ic")
    out = _update_position(SOIC_BLOCK, comp)
    assert "(at 0 -5 0)" in out   # Reference property untouched
    assert "(at 0 5 0)" in out    # Value property untouched


def test_only_first_at_is_replaced():
    """Exactly one (at ...) changes — the footprint's. Count the
    surviving original-form (at 0 0) tokens: should be zero (it was the
    footprint's and got replaced)."""
    comp = Component(ref="U1", footprint="SOIC-8", value="MCU",
                     x=5.0, y=5.0, rotation=180.0, component_type="ic")
    out = _update_position(SOIC_BLOCK, comp)
    # The original footprint (at 0 0) is gone; new placement present.
    assert "(at 0 0)\n" not in out      # footprint's old (at) replaced
    assert "(at 5 5 180)" in out        # new footprint placement


# ---------------------------------------------------------------------------
# _apply_footprint_positions — integration (the writer's real entry path)
# ---------------------------------------------------------------------------

def test_writer_leaves_pads_untouched_end_to_end():
    """Run the full writer regex path on a one-footprint .kicad_pcb text.
    The footprint moves + rotates; every pad (at) survives verbatim."""
    pcb_text = f"""(kicad_pcb
	(version 20260206)
	(layers (0 "F.Cu" signal) (31 "B.Cu" signal) (25 "Edge.Cuts" user))
{SOIC_BLOCK}
)"""
    comp = Component(ref="U1", footprint="SOIC-8", value="MCU",
                     x=33.0, y=44.0, rotation=270.0, component_type="ic")
    board = BoardOutline(0.0, 0.0, 80.0, 60.0)
    model = BoardModel(board=board, components=[comp])

    written = _apply_footprint_positions(pcb_text, {comp.ref: comp})

    # Footprint moved + rotated.
    assert "(at 33 44 270)" in written
    # Every pad (at) byte-identical to the source.
    for before, after in zip(_pad_at_strings(pcb_text), _pad_at_strings(written)):
        assert after == before, f"writer mutated pad (at): {before!r} -> {after!r}"


def test_round_trip_writer_then_parse_preserves_pad_positions(tmp_path):
    """Write a placed file, re-parse it with KiCadParser, and confirm
    the component landed where we asked. (Pad-local rotation isn't
    re-exposed by the parser, but a clean round-trip after the fix is a
    good smoke test that we didn't corrupt the block structurally.)"""
    from parsers.kicad_parser import KiCadParser

    src = tmp_path / "rot.kicad_pcb"
    src.write_text(f"""(kicad_pcb
	(version 20260206)
	(layers (0 "F.Cu" signal) (31 "B.Cu" signal) (25 "Edge.Cuts" user))
	(setup (pad_to_mask_clearance 0))
{SOIC_BLOCK}
	(gr_rect (start 0 0) (end 80 60) (stroke (width 0.1) (type solid)) (fill none) (layer "Edge.Cuts"))
)
""", encoding="utf-8")

    base = KiCadParser(str(src), bbox_margin=0.8).parse()
    u1 = base.get_component("U1")
    u1.x, u1.y, u1.rotation = 25.0, 35.0, 90.0

    from parsers.placement_writer import apply_placement
    out = apply_placement(base, str(src), output_pcb_path=str(tmp_path / "rot_placed.kicad_pcb"),
                          backup=False)
    reparsed = KiCadParser(out, bbox_margin=0.8).parse().get_component("U1")
    assert (reparsed.x, reparsed.y, reparsed.rotation) == (25.0, 35.0, 90.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
