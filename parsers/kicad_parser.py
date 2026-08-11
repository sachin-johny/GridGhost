"""KiCad .kicad_pcb s-expression parser.

Parses the KiCad PCB file format (s-expressions) to extract:
- Board outline (gr_rect or gr_poly on Edge.Cuts)
- Footprints with positions, rotations, bounding boxes
- Pads and their net assignments
- Net definitions

Reference: https://dev-docs.kicad.org/en/file-formats/sexpr-pcb/
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from models.board_model import BoardModel, BoardOutline, Component, Net, Pad
from models.board_model import _polygon_area, _segments_intersect


# ---------------------------------------------------------------------------
# S-expression tokenizer / parser
# ---------------------------------------------------------------------------

def tokenize_sexp(text: str) -> list:
    """Tokenize an s-expression string into a nested list structure.

    Example:
        "(module (at 10 20) (fp_text reference U1))"
        → ["module", ["at", "10", "20"], ["fp_text", "reference", "U1"]]
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in ' \t\n\r':
            i += 1
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        elif c == '"':
            # Quoted string
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == '\\':
                    j += 1  # skip escaped char
                j += 1
            tokens.append(text[i + 1:j])
            i = j + 1
        else:
            # Unquoted token
            j = i
            while j < n and text[j] not in ' \t\n\r()':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse_sexp(tokens: list) -> list:
    """Convert flat token list into nested list structure."""
    result = []
    stack = [result]
    for token in tokens:
        if token == '(':
            new_list = []
            stack[-1].append(new_list)
            stack.append(new_list)
        elif token == ')':
            stack.pop()
        else:
            stack[-1].append(token)
    return result


# ---------------------------------------------------------------------------
# Helper extractors
# ---------------------------------------------------------------------------

def find_all(sexp: list, key: str) -> list[list]:
    """Find all sub-expressions starting with `key`."""
    results = []
    for item in sexp:
        if isinstance(item, list) and len(item) > 0 and item[0] == key:
            results.append(item)
    return results


def find_first(sexp: list, key: str) -> Optional[list]:
    """Find first sub-expression starting with `key`."""
    for item in sexp:
        if isinstance(item, list) and len(item) > 0 and item[0] == key:
            return item
    return None


def find_deep(sexp: list, key: str) -> list[list]:
    """Find all sub-expressions at any depth starting with `key`."""
    results = []
    for item in sexp:
        if isinstance(item, list):
            if len(item) > 0 and item[0] == key:
                results.append(item)
            results.extend(find_deep(item, key))
    return results


def try_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Bounding box estimation from fp_line / fp_rect / fp_circle / pad geometry
# ---------------------------------------------------------------------------

def _extract_fp_geometry(sexp: list, margin_mm: float = 0.5) -> tuple[float, float, float, float]:
    """Extract component bounding box from footprint geometry.

    Collects the outermost extent across ALL relevant KiCad layers
    (CrtYd, Fab, SilkS, Paste, Cu, Mask) plus actual pad sizes.

    Returns: (width, height, bbox_offset_x, bbox_offset_y)
        - width/height: total extent including margin
        - bbox_offset_x/y: offset from footprint origin to geometry center
    """
    RELEVANT = ("CrtYd", "Fab", "Silk", "Paste", ".Cu", "Mask")
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Collect geometry from fp_line, fp_rect, fp_circle, fp_arc, fp_poly
    geom_items = (find_deep(sexp, "fp_line") + find_deep(sexp, "fp_rect") +
                  find_deep(sexp, "fp_circle") + find_deep(sexp, "fp_arc") +
                  find_deep(sexp, "fp_poly"))
    for item in geom_items:
        layer = find_first(item, "layer")
        if not layer or len(layer) < 2:
            continue
        layer_str = str(layer[1])
        if not any(kw in layer_str for kw in RELEVANT):
            continue
        for pt_key in ("start", "end", "center"):
            pt = find_first(item, pt_key)
            if pt and len(pt) >= 3:
                v1, v2 = try_float(pt[1]), try_float(pt[2])
                min_x, min_y = min(min_x, v1), min(min_y, v2)
                max_x, max_y = max(max_x, v1), max(max_y, v2)

    # Collect actual pad sizes (not hardcoded 0.5mm)
    for pad_expr in find_all(sexp, "pad"):
        at_expr = find_first(pad_expr, "at")
        if not (at_expr and len(at_expr) >= 3):
            continue
        px, py = try_float(at_expr[1]), try_float(at_expr[2])
        size_expr = find_first(pad_expr, "size")
        sx = sy = 0.5
        if size_expr and len(size_expr) >= 3:
            sx, sy = try_float(size_expr[1]) / 2.0, try_float(size_expr[2]) / 2.0
        min_x, min_y = min(min_x, px - sx), min(min_y, py - sy)
        max_x, max_y = max(max_x, px + sx), max(max_y, py + sy)

    if min_x == float('inf'):
        return 2.0, 2.0, 0.0, 0.0

    w = max(max_x - min_x, 0.5)
    h = max(max_y - min_y, 0.5)
    # Offset from footprint origin to geometry center
    off_x = (min_x + max_x) / 2.0
    off_y = (min_y + max_y) / 2.0
    return w, h, off_x, off_y


# ---------------------------------------------------------------------------
# Component type inference
# ---------------------------------------------------------------------------

def _infer_component_type(ref: str, footprint: str, value: str) -> str:
    """Infer component type from reference prefix and footprint name.

    Mounting holes (Finding 4 fix): footprints whose library prefix is
    ``MountingHole`` (e.g. ``MountingHole:MountingHole_3.2mm_M3``) or
    whose reference prefix is ``H`` / ``MH`` / ``HS`` are classified as
    ``mounting_hole``. The parser then marks these ``is_fixed=True`` and
    gives them a larger courtyard (1.5mm) so other components keep a
    real assembly clearance from them — not just the 0.25mm default
    courtyard that lets resistors sit touching the hole's copper pad.
    """
    ref_prefix = re.match(r'^([A-Z]+)', ref)
    prefix = ref_prefix.group(1) if ref_prefix else ""

    type_map = {
        "U": "ic",
        "IC": "ic",
        "J": "connector",
        "P": "connector",
        "CN": "connector",
        "Y": "crystal",
        "X": "crystal",
    }
    if prefix in type_map:
        return type_map[prefix]

    # Capacitors and resistors by reference prefix
    if prefix in ("C",):
        return "capacitor"
    if prefix in ("R",):
        return "resistor"

    # Check footprint hints
    fp_lower = footprint.lower()
    # Mounting holes — recognized by library prefix or by H/MH/HS ref prefix.
    # Footprint lib id format is "MountingHole:MountingHole_3.2mm_M3" —
    # the prefix before ':' is the library name. We check both the prefix
    # and the bare name so "MountingHole_3.2mm_M3" without a library also
    # matches.
    if "mountinghole" in fp_lower.replace(":", " ").replace("_", " "):
        return "mounting_hole"
    if prefix in ("H", "MH", "HS"):
        # H / MH / HS are standard mounting-hole / hole-slot ref prefixes.
        # Confirm with footprint hint — a stray "H1" inductor should not
        # be misclassified. If no footprint hint either, trust the ref
        # prefix (most EDA tools use H exclusively for mounting holes).
        if "mountinghole" in fp_lower or "hole" in fp_lower or not fp_lower:
            return "mounting_hole"

    if "qfp" in fp_lower or "qfn" in fp_lower or "bga" in fp_lower or "sop" in fp_lower or "soic" in fp_lower:
        return "ic"
    if "crystal" in fp_lower or "xtal" in fp_lower:
        return "crystal"
    if "connector" in fp_lower or "hdr" in fp_lower or "usb" in fp_lower:
        return "connector"

    return "generic"


# Mounting holes use the same courtyard margin as every other component
# (parser.bbox_margin, default 0.8mm). An earlier version of this fix
# expanded the courtyard to 1.5mm for an "extra keepout", but that
# caused adjacent mounting holes (e.g. test6's H1-H4 stacked 8mm apart
# vertically) to overlap each other. The is_fixed=True flag alone is
# sufficient — other components are pushed away from the hole's natural
# bbox by the legalizer's push-apart pass. A proper "extra clearance
# around fixed hardware" mechanism (per-macro keepout that OTHER macros
# must respect) is a Phase 2 improvement.


# ---------------------------------------------------------------------------
# Board outline extraction
# ---------------------------------------------------------------------------

def _has_edge_cuts(sexp: list) -> bool:
    """Return True if any geometry exists on the Edge.Cuts layer."""
    for tag in ("gr_rect", "gr_line", "gr_poly", "gr_circle"):
        for expr in find_all(sexp, tag):
            layer = find_first(expr, "layer")
            if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
                return True
    return False


def _collect_edge_cuts_shapes(sexp: list) -> list[dict]:
    """Collect every closed shape on Edge.Cuts as a bbox + area record.

    Used by _extract_board_outline_and_keepouts to distinguish the outer
    outline (largest-area closed shape) from internal cutouts (mounting
    holes, connector slots) that must become keepouts.

    Only CLOSED shapes can be cutouts — gr_line segments that form part
    of a polyline outline are handled separately by _extract_board_outline
    which already takes the global AABB.  Here we collect gr_rect,
    gr_poly, and gr_circle because each is a single closed shape with a
    well-defined bbox.

    Returns a list of dicts:
        [{"x_min", "y_min", "x_max", "y_max", "area", "kind": "rect"|"poly"|"circle"}, ...]
    """
    shapes: list[dict] = []

    # gr_rect on Edge.Cuts — closed by definition
    for gr_rect in find_all(sexp, "gr_rect"):
        layer = find_first(gr_rect, "layer")
        if not layer or len(layer) < 2 or "Edge.Cuts" not in str(layer[1]):
            continue
        start_pt = find_first(gr_rect, "start")
        end_pt = find_first(gr_rect, "end")
        if not (start_pt and len(start_pt) >= 3 and end_pt and len(end_pt) >= 3):
            continue
        x1, y1 = try_float(start_pt[1]), try_float(start_pt[2])
        x2, y2 = try_float(end_pt[1]), try_float(end_pt[2])
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        shapes.append({
            "x_min": x_min, "y_min": y_min,
            "x_max": x_max, "y_max": y_max,
            "area": (x_max - x_min) * (y_max - y_min),
            "kind": "rect",
        })

    # gr_poly on Edge.Cuts — closed by definition (polygon)
    for gr_poly in find_all(sexp, "gr_poly"):
        layer = find_first(gr_poly, "layer")
        if not layer or len(layer) < 2 or "Edge.Cuts" not in str(layer[1]):
            continue
        pts_expr = find_first(gr_poly, "pts")
        if not pts_expr:
            continue
        xs, ys = [], []
        for xy in find_all(pts_expr, "xy"):
            if len(xy) >= 3:
                xs.append(try_float(xy[1]))
                ys.append(try_float(xy[2]))
        if len(xs) < 3:
            continue
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # Polygon area via shoelace — signed, take abs.
        area = 0.0
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            area += xs[i] * ys[j] - xs[j] * ys[i]
        shapes.append({
            "x_min": x_min, "y_min": y_min,
            "x_max": x_max, "y_max": y_max,
            "area": abs(area) / 2.0,
            "kind": "poly",
        })

    # gr_circle on Edge.Cuts — closed by definition
    for gr_circle in find_all(sexp, "gr_circle"):
        layer = find_first(gr_circle, "layer")
        if not layer or len(layer) < 2 or "Edge.Cuts" not in str(layer[1]):
            continue
        center = find_first(gr_circle, "center")
        end = find_first(gr_circle, "end")
        if not (center and len(center) >= 3 and end and len(end) >= 3):
            continue
        cx, cy = try_float(center[1]), try_float(center[2])
        ex, ey = try_float(end[1]), try_float(end[2])
        r = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        shapes.append({
            "x_min": cx - r, "y_min": cy - r,
            "x_max": cx + r, "y_max": cy + r,
            "area": math.pi * r * r,
            "kind": "circle",
        })

    return shapes


Point = tuple[float, float]


def _circle_center(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Optional[Point]:
    """Center of the circle through 3 points, or None if (near-)collinear."""
    ax, ay = x2 - x1, y2 - y1
    bx, by = x3 - x1, y3 - y1
    d = 2.0 * (ax * by - ay * bx)
    if abs(d) < 1e-9:
        return None
    ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / d
    uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / d
    return (x1 + ux, y1 + uy)


def _arc_to_polyline(gr_arc: list, steps: int = 8) -> list[Point]:
    """Approximate a gr_arc (KiCad 7+ start/mid/end 3-point format) as a
    short polyline, for outline tracing and bounding-box purposes.

    Falls back to a straight line between start/end if the geometry is
    degenerate (collinear points, or an older/unsupported arc encoding),
    which is still strictly better than ignoring the arc entirely.
    """
    start_pt = find_first(gr_arc, "start")
    mid_pt = find_first(gr_arc, "mid")
    end_pt = find_first(gr_arc, "end")
    pts: list[Point] = []
    if start_pt and len(start_pt) >= 3:
        pts.append((try_float(start_pt[1]), try_float(start_pt[2])))
    if end_pt and len(end_pt) >= 3:
        end = (try_float(end_pt[1]), try_float(end_pt[2]))
    else:
        return pts
    if not (mid_pt and len(mid_pt) >= 3 and pts):
        pts.append(end)
        return pts

    sx, sy = pts[0]
    mx, my = try_float(mid_pt[1]), try_float(mid_pt[2])
    ex, ey = end
    center = _circle_center(sx, sy, mx, my, ex, ey)
    if center is None:
        return [pts[0], end]
    cx, cy = center
    r = math.hypot(sx - cx, sy - cy)

    def norm(a: float) -> float:
        while a < 0:
            a += 2 * math.pi
        while a >= 2 * math.pi:
            a -= 2 * math.pi
        return a

    a0 = math.atan2(sy - cy, sx - cx)
    am = norm(math.atan2(my - cy, mx - cx) - a0)
    a1 = norm(math.atan2(ey - cy, ex - cx) - a0)
    # Sweep from the start angle through the mid-point angle to the end
    # angle, in whichever rotational direction actually passes through it.
    sweep = a1 if am <= a1 else -(2 * math.pi - a1)
    return [
        (cx + r * math.cos(a0 + sweep * i / steps), cy + r * math.sin(a0 + sweep * i / steps))
        for i in range(steps + 1)
    ]


def _collect_edge_cuts_segments(sexp: list) -> list[tuple[Point, Point]]:
    """Every straight edge implied by Edge.Cuts line/rect/arc geometry.

    gr_poly is excluded here — it's already a complete standalone closed
    loop and is handled directly by ``_trace_outer_board_polygon``.
    """
    segments: list[tuple[Point, Point]] = []

    for gr_line in find_all(sexp, "gr_line"):
        layer = find_first(gr_line, "layer")
        if not (layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1])):
            continue
        start_pt = find_first(gr_line, "start")
        end_pt = find_first(gr_line, "end")
        if start_pt and len(start_pt) >= 3 and end_pt and len(end_pt) >= 3:
            a = (try_float(start_pt[1]), try_float(start_pt[2]))
            b = (try_float(end_pt[1]), try_float(end_pt[2]))
            segments.append((a, b))

    for gr_rect in find_all(sexp, "gr_rect"):
        layer = find_first(gr_rect, "layer")
        if not (layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1])):
            continue
        start_pt = find_first(gr_rect, "start")
        end_pt = find_first(gr_rect, "end")
        if start_pt and len(start_pt) >= 3 and end_pt and len(end_pt) >= 3:
            x1, y1 = try_float(start_pt[1]), try_float(start_pt[2])
            x2, y2 = try_float(end_pt[1]), try_float(end_pt[2])
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for i in range(4):
                segments.append((corners[i], corners[(i + 1) % 4]))

    for gr_arc in find_all(sexp, "gr_arc"):
        layer = find_first(gr_arc, "layer")
        if not (layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1])):
            continue
        pts = _arc_to_polyline(gr_arc)
        for i in range(len(pts) - 1):
            segments.append((pts[i], pts[i + 1]))

    return segments


def _chain_segments_to_loops(segments: list[tuple[Point, Point]], tol: float = 1e-3) -> list[list[Point]]:
    """Greedily chain line segments that share endpoints (within ``tol``
    mm) into closed polygon loops.

    Any chain that never closes (an open outline, a stray dangling edge)
    is dropped rather than guessed at — the caller falls back to the
    AABB-based rectangle outline in that case, so an incomplete Edge.Cuts
    drawing degrades to the pre-existing behavior instead of producing a
    wrong shape.
    """
    def key(pt: Point) -> tuple[int, int]:
        return (round(pt[0] / tol), round(pt[1] / tol))

    remaining = list(segments)
    loops: list[list[Point]] = []
    while remaining:
        a, b = remaining.pop(0)
        loop = [a, b]
        while True:
            tail = loop[-1]
            found = False
            for i, (sa, sb) in enumerate(remaining):
                if key(sa) == key(tail):
                    loop.append(sb)
                    remaining.pop(i)
                    found = True
                    break
                if key(sb) == key(tail):
                    loop.append(sa)
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                break
            if key(loop[-1]) == key(loop[0]) and len(loop) > 2:
                break
        if key(loop[-1]) == key(loop[0]) and len(loop) > 2:
            loop.pop()  # drop the duplicate closing vertex
            loops.append(loop)
        # else: never closed — discarded, see docstring.
    return loops


def _is_axis_aligned_rect(poly: list[Point], tol: float = 1e-6) -> bool:
    """True if ``poly`` is (up to vertex order) exactly a 4-corner
    axis-aligned rectangle — the common case, kept on the original
    rectangle code path (with its outward margin) for full backward
    compatibility."""
    if len(poly) != 4:
        return False
    xs = sorted({round(p[0], 6) for p in poly})
    ys = sorted({round(p[1], 6) for p in poly})
    if len(xs) != 2 or len(ys) != 2:
        return False
    expected = {(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])}
    actual = {(round(p[0], 6), round(p[1], 6)) for p in poly}
    return expected == actual


def _polygon_self_intersects(poly: list[Point]) -> bool:
    """True if any two non-adjacent edges of ``poly`` cross. Our
    containment/clamp math assumes a simple (non-self-intersecting)
    polygon, so a self-intersecting trace is rejected by the caller
    rather than silently mishandled."""
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue  # adjacent edges legitimately share a vertex
            b1, b2 = poly[j], poly[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _trace_outer_board_polygon(sexp: list) -> Optional[list[Point]]:
    """Reconstruct the true (possibly non-rectangular) outer board outline
    as a polygon from Edge.Cuts geometry.

    Real boards routinely draw their outline with a connector notch,
    mouse-bite tabs, or a castellated/cutout edge baked directly into the
    Edge.Cuts perimeter — not just as a separate internal keepout shape.
    Taking only the AABB of that geometry (the old behavior) silently
    discards the concavity. This traces the actual loop instead.

    Returns the largest-area closed loop found (interior cutouts are
    smaller and handled separately as keepouts), or None if the Edge.Cuts
    geometry doesn't reduce to a clean simple polygon — callers fall back
    to the AABB rectangle exactly as before this feature existed, so an
    unusual or incomplete drawing degrades gracefully instead of failing.
    """
    loops: list[list[Point]] = []

    for gr_poly in find_all(sexp, "gr_poly"):
        layer = find_first(gr_poly, "layer")
        if not (layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1])):
            continue
        pts_expr = find_first(gr_poly, "pts")
        if not pts_expr:
            continue
        pts = [(try_float(xy[1]), try_float(xy[2])) for xy in find_all(pts_expr, "xy") if len(xy) >= 3]
        if len(pts) >= 3:
            loops.append(pts)

    segments = _collect_edge_cuts_segments(sexp)
    if segments:
        loops.extend(_chain_segments_to_loops(segments))

    loops = [loop for loop in loops if len(loop) >= 3]
    if not loops:
        return None

    loops.sort(key=_polygon_area, reverse=True)
    outer = loops[0]
    if _polygon_self_intersects(outer):
        return None
    return outer


def _extract_board_outline(sexp: list) -> BoardOutline:
    """Extract board outline from Edge.Cuts geometry.

    First tries to trace the actual outer polygon (handles notches,
    mouse-bites, and other non-rectangular outlines drawn directly into
    the board edge). If that traces out to a plain axis-aligned
    rectangle, or if the geometry can't be reduced to a clean simple
    polygon, falls back to the original behavior: the global AABB of all
    Edge.Cuts geometry (rects, lines, polys, arcs, circles), expanded by
    a small margin. Internal cutouts that are entirely inside the outer
    outline are extracted separately by _collect_edge_cuts_shapes and
    turned into keepouts by the parser.
    """
    outer_polygon = _trace_outer_board_polygon(sexp)
    if outer_polygon is not None and not _is_axis_aligned_rect(outer_polygon):
        return BoardOutline(polygon=outer_polygon)

    points_x = []
    points_y = []

    # gr_rect on Edge.Cuts
    for gr_rect in find_all(sexp, "gr_rect"):
        layer = find_first(gr_rect, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            start_pt = find_first(gr_rect, "start")
            end_pt = find_first(gr_rect, "end")
            if start_pt and len(start_pt) >= 3 and end_pt and len(end_pt) >= 3:
                points_x.extend([try_float(start_pt[1]), try_float(end_pt[1])])
                points_y.extend([try_float(start_pt[2]), try_float(end_pt[2])])

    # gr_line on Edge.Cuts
    for gr_line in find_all(sexp, "gr_line"):
        layer = find_first(gr_line, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            start_pt = find_first(gr_line, "start")
            end_pt = find_first(gr_line, "end")
            if start_pt and len(start_pt) >= 3:
                points_x.append(try_float(start_pt[1]))
                points_y.append(try_float(start_pt[2]))
            if end_pt and len(end_pt) >= 3:
                points_x.append(try_float(end_pt[1]))
                points_y.append(try_float(end_pt[2]))

    # gr_poly on Edge.Cuts
    for gr_poly in find_all(sexp, "gr_poly"):
        layer = find_first(gr_poly, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            pts_expr = find_first(gr_poly, "pts")
            if pts_expr:
                for xy in find_all(pts_expr, "xy"):
                    if len(xy) >= 3:
                        points_x.append(try_float(xy[1]))
                        points_y.append(try_float(xy[2]))

    # gr_circle on Edge.Cuts (approximate as bounding box)
    for gr_circle in find_all(sexp, "gr_circle"):
        layer = find_first(gr_circle, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            center = find_first(gr_circle, "center")
            end = find_first(gr_circle, "end")
            if center and len(center) >= 3 and end and len(end) >= 3:
                cx, cy = try_float(center[1]), try_float(center[2])
                ex, ey = try_float(end[1]), try_float(end[2])
                r = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                points_x.extend([cx - r, cx + r])
                points_y.extend([cy - r, cy + r])

    # gr_arc on Edge.Cuts — previously ignored entirely, which under-sized
    # the AABB (and could clip the real board) for any outline using arcs
    # (rounded corners, curved notches). Sampled the same way the polygon
    # tracer above samples arcs.
    for gr_arc in find_all(sexp, "gr_arc"):
        layer = find_first(gr_arc, "layer")
        if layer and len(layer) > 1 and "Edge.Cuts" in str(layer[1]):
            for px, py in _arc_to_polyline(gr_arc):
                points_x.append(px)
                points_y.append(py)

    if not points_x:
        # Default board if no outline found
        return BoardOutline(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

    margin = 2.0  # Board outline margin — must leave room for component courtyards
    return BoardOutline(
        x_min=min(points_x) - margin,
        y_min=min(points_y) - margin,
        x_max=max(points_x) + margin,
        y_max=max(points_y) + margin,
    )


def _extract_keepouts_from_edge_cuts(
    sexp: list,
    outer_outline: BoardOutline,
) -> list[BoardOutline]:
    """Identify internal cutouts on Edge.Cuts and return them as keepouts.

    Strategy: every closed shape on Edge.Cuts whose bbox is STRICTLY
    INSIDE the outer outline (with a small tolerance) is treated as an
    internal cutout — a mounting hole, connector slot, etc.  The largest
    closed shape is assumed to BE the outer outline (or part of it) and
    is NOT turned into a keepout.

    Edge case: if there's only one closed shape and it's the same size
    as the outer outline, no keepouts are produced (the board has no
    internal cutouts).  If there are no closed shapes (only gr_line
    polyline outlines), no keepouts are produced either.
    """
    shapes = _collect_edge_cuts_shapes(sexp)
    if not shapes:
        return []

    # The outer outline is the largest closed shape (by area).  In a
    # typical board, the outer outline is a gr_rect or gr_poly covering
    # the whole board, and any smaller closed shape is a cutout.
    largest_area = max(s["area"] for s in shapes)
    # 5% tolerance — a cutout is at most 95% of the outer outline's area
    # in any realistic board.  (Mounting holes are tiny by comparison.)
    outer_threshold = largest_area * 0.95

    keepouts: list[BoardOutline] = []
    for s in shapes:
        if s["area"] >= outer_threshold:
            continue  # This IS the outer outline, not a cutout
        # Sanity check: the cutout should be strictly inside the outer
        # outline.  If it extends outside (overlapping the board edge),
        # it's probably a slot/notch rather than a hole — still treat
        # it as a keepout because components shouldn't be placed there.
        keepouts.append(BoardOutline(
            x_min=s["x_min"], y_min=s["y_min"],
            x_max=s["x_max"], y_max=s["y_max"],
        ))

    return keepouts


def _infer_board_from_components(components: list[Component]) -> BoardOutline:
    """Infer a board outline from placed components when Edge.Cuts is absent.

    Density-aware: when components are tightly packed (density exceeds
    the shared target pack density — see ``utils.density.target_pack_density``),
    the board is expanded so that the legalizer has room to push components
    apart without cascading overlaps. For sparse boards, the original 25%
    padding is retained.
    """
    if not components:
        return BoardOutline(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    total_comp_area = 0.0

    for comp in components:
        half_w = comp.effective_width / 2.0
        half_h = comp.effective_height / 2.0
        min_x = min(min_x, comp.x - half_w)
        min_y = min(min_y, comp.y - half_h)
        max_x = max(max_x, comp.x + half_w)
        max_y = max(max_y, comp.y + half_h)
        total_comp_area += comp.effective_width * comp.effective_height

    cluster_w = max(max_x - min_x, 1.0)
    cluster_h = max(max_y - min_y, 1.0)
    cluster_area = cluster_w * cluster_h

    default_padding = max(8.0, max(cluster_w, cluster_h) * 0.25)

    # Finding 6 fix: use the shared target_pack_density from utils.density
    # so the inferred outline matches what the legalizer expects. Previously
    # this was a hardcoded 0.35, disagreeing with the legalizer's 0.55 —
    # the inferred outline had more empty space than the legalizer needed,
    # then the legalizer would not expand (because the bounds were already
    # sparse enough), but the placement still came out loose.
    from utils.density import target_pack_density
    TARGET_DENSITY = target_pack_density()
    needed_area = total_comp_area / TARGET_DENSITY

    if needed_area > cluster_area:
        a = 4.0
        b = 2.0 * (cluster_w + cluster_h)
        c = cluster_area - needed_area
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            density_padding = (-b + math.sqrt(disc)) / (2.0 * a)
        else:
            density_padding = 0.0
        padding = max(default_padding, density_padding)
    else:
        padding = default_padding
    return BoardOutline(
        x_min=min_x - padding,
        y_min=min_y - padding,
        x_max=max_x + padding,
        y_max=max_y + padding,
    )


def _needs_inferred_board(board_outline: BoardOutline, components: list[Component]) -> bool:
    """Return True when the parsed outline is just the default fallback board."""
    if not components:
        return False

    is_default_board = (
        math.isclose(board_outline.x_min, 0.0)
        and math.isclose(board_outline.y_min, 0.0)
        and math.isclose(board_outline.x_max, 100.0)
        and math.isclose(board_outline.y_max, 100.0)
    )
    if not is_default_board:
        return False

    for comp in components:
        bx1, by1, bx2, by2 = comp.bbox
        if bx1 < board_outline.x_min or by1 < board_outline.y_min or bx2 > board_outline.x_max or by2 > board_outline.y_max:
            return True

    return False


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class KiCadParser:
    """Parser for .kicad_pcb files."""

    def __init__(self, filepath: str, bbox_margin: float = 0.5):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"PCB file not found: {filepath}")
        self._sexp = None
        self._net_id_to_name: dict[str, str] = {}  # Maps net ID ("1") → name ("VCC")
        self._bbox_margin = bbox_margin  # Margin around bbox in mm

    def parse(self) -> BoardModel:
        """Parse the .kicad_pcb file and return a BoardModel."""
        text = self.filepath.read_text(encoding="utf-8")
        tokens = tokenize_sexp(text)
        parsed = parse_sexp(tokens)
        # The root is the `kicad_pcb` expression
        self._sexp = parsed[0] if parsed else parsed
        sexp = self._sexp or []

        # Build net ID → name mapping from top-level net definitions
        self._build_net_id_map()

        board_outline = _extract_board_outline(sexp)
        has_edge_cuts = _has_edge_cuts(sexp)
        components = self._extract_components()
        nets = self._extract_nets()

        # Link nets to components
        net_map: dict[str, Net] = {n.name: n for n in nets}
        for comp in components:
            comp_nets = set()
            for pad in comp.pads:
                if pad.net and pad.net in net_map:
                    comp_nets.add(pad.net)
            comp.nets = sorted(comp_nets)

        if _needs_inferred_board(board_outline, components):
            board_outline = _infer_board_from_components(components)

        # Extract internal cutouts (mounting holes, slots) as keepouts.
        # Only attempt this when there's a real Edge.Cuts outline —
        # inferred-from-components boards have no cutouts by definition.
        keepouts: list[BoardOutline] = []
        if has_edge_cuts:
            keepouts = _extract_keepouts_from_edge_cuts(sexp, board_outline)

        model = BoardModel(
            board=board_outline,
            components=components,
            nets=nets,
            source_file=str(self.filepath),
            user_defined_outline=has_edge_cuts,
            keepouts=keepouts,
        )
        return model

    def _build_net_id_map(self) -> None:
        """Build mapping from net IDs to net names from top-level (net <id> <name>) definitions."""
        sexp = self._sexp or []
        for net_expr in find_all(sexp, "net"):
            if len(net_expr) >= 3:
                net_id = net_expr[1]
                net_name = net_expr[2]
                self._net_id_to_name[net_id] = net_name

    def _extract_components(self) -> list[Component]:
        """Extract all footprints as Components."""
        components = []
        sexp = self._sexp or []

        for fp_expr in find_all(sexp, "footprint"):
            # Also handle "module" for older KiCad formats
            self._parse_footprint(fp_expr, components)

        for fp_expr in find_all(sexp, "module"):
            self._parse_footprint(fp_expr, components)

        return components

    def _parse_footprint(self, fp_expr: list, components: list[Component]) -> None:
        """Parse a single footprint/module expression into a Component."""
        # Library ID
        lib_id = fp_expr[1] if len(fp_expr) > 1 else ""

        # Position and rotation
        at_expr = find_first(fp_expr, "at")
        if at_expr and len(at_expr) >= 3:
            x = try_float(at_expr[1])
            y = try_float(at_expr[2])
            rotation = try_float(at_expr[3]) if len(at_expr) > 3 else 0.0
        else:
            x, y, rotation = 0.0, 0.0, 0.0

        # Layer
        layer_expr = find_first(fp_expr, "layer")
        layer = "bottom" if layer_expr and len(layer_expr) > 1 and "B." in str(layer_expr[1]) else "top"

        # Reference and value
        ref = ""
        value = ""
        for prop in find_all(fp_expr, "property"):
            if len(prop) > 2:
                if prop[1] == "Reference":
                    ref = prop[2]
                elif prop[1] == "Value":
                    value = prop[2]

        # Older format: fp_text reference / fp_text value
        if not ref:
            for ft in find_all(fp_expr, "fp_text"):
                if len(ft) > 3 and ft[1] == "reference":
                    ref = ft[2]
                elif len(ft) > 3 and ft[1] == "value":
                    value = ft[2]

        if not ref:
            return  # Skip unnamed components

        # Pads
        pads = []
        for pad_expr in find_all(fp_expr, "pad"):
            pad = self._parse_pad(pad_expr, x, y, rotation)
            if pad:
                pads.append(pad)

        # Bounding box from geometry (with margin for spacing)
        width, height, bbox_ox, bbox_oy = _extract_fp_geometry(fp_expr, margin_mm=self._bbox_margin)

        # Hierarchical schematic sheet (KiCad 7+) — the designer's own
        # functional grouping, sitting in the file for free.  Components
        # on the same sheet (e.g. "/MCU/", "/POWER/") get a strong prior
        # edge in the clustering hypergraph so the auto-placer respects
        # the schematic organisation.  Empty string for flat schematics
        # or root-sheet components (sheetname="/").  See AUDIT_PHASE0.md §3.1.
        sheet = ""
        sheet_expr = find_first(fp_expr, "sheetname")
        if sheet_expr and len(sheet_expr) >= 2:
            sheet = str(sheet_expr[1]) if sheet_expr[1] else ""

        # Determine if fixed. Keep connectors movable so the auto-placer can
        # move them to the board perimeter during edge-aware placement.
        # Mounting holes (Finding 4 fix): always fixed — their position is
        # dictated by the enclosure, not the placer. Courtyard margin is
        # the parser default (same as every other component); see the
        # comment near MOUNTING_HOLE_COURTYARD_MARGIN_MM for why we don't
        # expand it.
        comp_type = _infer_component_type(ref, lib_id, value)
        is_fixed = False
        if comp_type == "mounting_hole":
            is_fixed = True

        # Finding 8 fix: size- and type-aware courtyard margin.
        # Previously every component got the flat parser default (0.8mm).
        # Now small passives get 0.5mm, mid-size ICs get 1.0mm, large ICs
        # get 1.5mm — IPC-7351 nominal courtyards. This makes "0 overlaps"
        # actually mean "assembly-safe" rather than "barely touching".
        # See utils/courtyard.py for the full table.
        from utils.courtyard import courtyard_margin_for_component
        courtyard_margin = courtyard_margin_for_component(
            width=width, height=height, component_type=comp_type,
        )

        comp = Component(
            ref=ref,
            footprint=lib_id,
            value=value,
            x=x,
            y=y,
            rotation=rotation,
            layer=layer,
            width=width,
            height=height,
            courtyard_margin=courtyard_margin,
            bbox_offset_x=bbox_ox,
            bbox_offset_y=bbox_oy,
            pads=pads,
            is_fixed=is_fixed,
            component_type=comp_type,
            sheet=sheet,
        )
        components.append(comp)

    def _resolve_net_name(self, net_expr: Optional[list]) -> Optional[str]:
        """Resolve a pad's net expression to a net name.

        Handles two formats:
          - (net <id> <name>)  → use name directly
          - (net <id>)         → look up name from _net_id_to_name map
        """
        if not net_expr or len(net_expr) < 2:
            return None

        net_id = net_expr[1]

        # Format: (net <id> <name>) — name is directly available
        if len(net_expr) >= 3:
            # Check if the third element looks like a net name (not a number)
            potential_name = net_expr[2]
            if potential_name and not potential_name.lstrip('-').replace('.', '').isdigit():
                return potential_name

        # Format: (net <id>) — look up by ID
        if net_id in self._net_id_to_name:
            return self._net_id_to_name[net_id]

        # Last resort: if the ID itself is a name (e.g., net names without quotes)
        if net_id and not net_id.lstrip('-').replace('.', '').isdigit():
            return net_id

        return None

    def _parse_pad(self, pad_expr: list, comp_x: float, comp_y: float, rotation: float) -> Optional[Pad]:
        """Parse a pad expression into a Pad object."""
        if len(pad_expr) < 3:
            return None

        pad_name = pad_expr[1]

        # Position (relative to component origin)
        at_expr = find_first(pad_expr, "at")
        if at_expr and len(at_expr) >= 3:
            px = try_float(at_expr[1])
            py = try_float(at_expr[2])
        else:
            px, py = 0.0, 0.0

        # Net — resolve using the ID→name map
        net_expr = find_first(pad_expr, "net")
        net_name = self._resolve_net_name(net_expr)

        return Pad(pad_name=pad_name, x=px, y=py, net=net_name)

    def _extract_nets(self) -> list[Net]:
        """Extract net definitions."""
        nets = []
        sexp = self._sexp or []
        # From (net <id> <name>) in the nets section
        for net_expr in find_all(sexp, "net"):
            if len(net_expr) >= 3 and isinstance(net_expr[1], str) and isinstance(net_expr[2], str):
                net_name = net_expr[2]
                # Collect pins from all footprints
                pins = []
                for comp_ref, pad_name, pad_net in self._collect_net_pins():
                    if pad_net == net_name:
                        pins.append((comp_ref, pad_name))
                nets.append(Net(net_name, pins))

        # If no net section found, build nets from pad net assignments
        if not nets:
            net_dict: dict[str, list[tuple[str, str]]] = {}
            for comp_ref, pad_name, pad_net in self._collect_net_pins():
                if pad_net:
                    net_dict.setdefault(pad_net, []).append((comp_ref, pad_name))
            for name, pins in net_dict.items():
                nets.append(Net(name, pins))

        return nets

    def _collect_net_pins(self) -> list[tuple[str, str, Optional[str]]]:
        """Collect all (comp_ref, pad_name, net_name) tuples from all footprints."""
        result = []
        seen_refs = set()
        sexp = self._sexp or []

        for fp_expr in find_all(sexp, "footprint") + find_all(sexp, "module"):
            # Get reference
            ref = ""
            for prop in find_all(fp_expr, "property"):
                if len(prop) > 2 and prop[1] == "Reference":
                    ref = prop[2]
            if not ref:
                for ft in find_all(fp_expr, "fp_text"):
                    if len(ft) > 3 and ft[1] == "reference":
                        ref = ft[2]

            if not ref or ref in seen_refs:
                continue
            seen_refs.add(ref)

            for pad_expr in find_all(fp_expr, "pad"):
                if len(pad_expr) < 3:
                    continue
                pad_name = pad_expr[1]
                net_expr = find_first(pad_expr, "net")
                net_name = self._resolve_net_name(net_expr)
                result.append((ref, pad_name, net_name))

        return result
