#!/usr/bin/env python3
"""GridGhost benchmark harness.

Runs `gridghost.py place` on the four test PCBs in greedy and SA modes,
parses stdout for metrics, mines the *_placed_model.json for EE-perspective
metrics (rotation histogram, decoupling cap distances, density hotspots,
top nets by HPWL), and writes structured results to tests/output/.

Outputs (per run):
  benchmark_<board>_<mode>_run<N>.json     structured metrics
  benchmark_<board>_<mode>_run<N>.log      raw stdout
  benchmark_<board>_<mode>_run<N>_model.json  copy of placed_model.json
  benchmark_summary.csv                     flat table of all runs

No changes to the auto-placer itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tests" / "output"
TEST_PCB_DIR = ROOT / "tests" / "test_pcbs"
GRIDGHOST = ROOT / "gridghost.py"

BOARDS = ["cbb", "cbbwO", "test4", "th_sensor"]
SA_REPLICATES = 3


# ---------------------------------------------------------------------------
# stdout parsing
# ---------------------------------------------------------------------------

def parse_output(out: str) -> dict:
    """Mine gridghost.py stdout for all available metrics."""
    m = {
        # board / extraction
        "board_width_mm": None,
        "board_height_mm": None,
        "n_components": None,
        "n_movable": None,
        "n_nets": None,
        "overlaps_post_extract": None,
        "overlap_area_post_extract_mm2": None,
        "oob_post_extract": None,
        # placement
        "overlaps_post_place": None,
        "overlap_area_post_place_mm2": None,
        "oob_post_place": None,
        # optimization
        "opt_mode": None,  # "SA+greedy+swap" or "greedy+swap"
        "density": None,
        "penalty_scale_min": None,
        "sa_initial_cost": None,
        "sa_final_cost": None,
        "sa_initial_hpwl": None,
        "sa_final_hpwl": None,
        "sa_overlaps": None,
        "reheat_count_observed": 0,
        # post-optimization (pre-legalize)
        "hpwl_post_opt": None,
        "overlap_penalty_post_opt": None,
        "boundary_penalty_post_opt": None,
        "constraint_penalty_post_opt": None,
        "total_cost_post_opt": None,
        "overlaps_post_opt": None,
        "oob_post_opt": None,
        # legalization
        "legal_overlaps_in": None,
        "legal_oob_in": None,
        "abacus_displacement_mm": None,
        "legal_overlaps_out": None,
        "legal_oob_out": None,
        # post-legalization
        "hpwl_post_legal": None,
        "overlap_penalty_post_legal": None,
        "boundary_penalty_post_legal": None,
        "constraint_penalty_post_legal": None,
        "total_cost_post_legal": None,
        "overlaps_post_legal": None,
        "oob_post_legal": None,
        "cost_change_from_legalization": None,
    }

    def find(pattern, text, cast=float, group=1):
        m_ = re.search(pattern, text)
        return cast(m_.group(group)) if m_ else None

    # --- Board info (from "After Extraction" section, which is the first
    # block that prints board dimensions). We anchor to the extraction block
    # specifically by limiting to text before "After Placement".
    extract_block = out.split("After Extraction", 1)[0] if "After Extraction" in out else out
    m["board_width_mm"] = find(r"Board size:\s+([\d.]+)\s*x\s*([\d.]+)\s*mm", extract_block, float, 1)
    m["board_height_mm"] = find(r"Board size:\s+([\d.]+)\s*x\s*([\d.]+)\s*mm", extract_block, float, 2)
    m["n_components"] = find(r"Total components:\s+(\d+)", extract_block, int)
    m["n_movable"] = find(r"Movable:\s+(\d+)", extract_block, int)
    m["n_nets"] = find(r"Total nets:\s+(\d+)", extract_block, int)
    m["overlaps_post_extract"] = find(r"Overlaps:\s+(\d+)", extract_block, int)
    m["overlap_area_post_extract_mm2"] = find(r"Overlap area:\s+([\d.]+)", extract_block, float)
    m["oob_post_extract"] = find(r"Out-of-bounds:\s+(\d+)", extract_block, int)

    # --- Placement block
    if "After Placement" in out:
        place_block = out.split("After Placement", 1)[1].split("Step", 1)[0]
        m["overlaps_post_place"] = find(r"Overlaps:\s+(\d+)", place_block, int)
        m["overlap_area_post_place_mm2"] = find(r"Overlap area:\s+([\d.]+)", place_block, float)
        m["oob_post_place"] = find(r"Out-of-bounds:\s+(\d+)", place_block, int)

    # --- SA / greedy block
    # Density and penalty scale printed once at start of step 5
    m["density"] = find(r"density=([\d.]+)", out, float)
    m["penalty_scale_min"] = find(r"penalty_scale_min=([\d.]+)", out, float)

    # Mode label is "SA:" or "Greedy:"
    mode_match = re.search(r"(SA|Greedy):\s*cost\s+([\d.]+)\s*->\s*([\d.]+)", out)
    if mode_match:
        m["opt_mode"] = "SA+greedy+swap" if mode_match.group(1) == "SA" else "greedy+swap"
        m["sa_initial_cost"] = float(mode_match.group(2))
        m["sa_final_cost"] = float(mode_match.group(3))

    hpwl_match = re.search(r"(?:SA|Greedy):\s*HPWL\s+([\d.]+)\s*->\s*([\d.]+)", out)
    if hpwl_match:
        m["sa_initial_hpwl"] = float(hpwl_match.group(1))
        m["sa_final_hpwl"] = float(hpwl_match.group(2))

    ov_match = re.search(r"(?:SA|Greedy):\s*overlaps\s*=\s*(\d+)", out)
    if ov_match:
        m["sa_overlaps"] = int(ov_match.group(1))

    # Count reheat rounds
    m["reheat_count_observed"] = len(re.findall(r"Reheat \d+:", out))

    # Abacus displacement
    m["abacus_displacement_mm"] = find(r"Abacus:\s*total displacement\s*=\s*([\d.]+)\s*mm", out, float)

    # Legalization input/output
    legal_in = re.search(r"Legalization input:\s+(\d+)\s+overlaps,\s+(\d+)\s+out-of-bounds", out)
    if legal_in:
        m["legal_overlaps_in"] = int(legal_in.group(1))
        m["legal_oob_in"] = int(legal_in.group(2))
    legal_out = re.search(r"Legalization output:\s+(\d+)\s+overlaps,\s+(\d+)\s+out-of-bounds", out)
    if legal_out:
        m["legal_overlaps_out"] = int(legal_out.group(1))
        m["legal_oob_out"] = int(legal_out.group(2))

    # --- Cost breakdowns
    def parse_cost_block(label):
        # Find a section like "Placement Cost (Pre-Legalization)" or "(Post-Legalization)".
        # Layout is:
        #   ----- (opening separator) -----
        #   Placement Cost (Pre-Legalization)
        #   ----- (opening separator) -----
        #   HPWL: ...
        #   ...
        #   ----- (small separator) ----
        #   TOTAL COST: ...
        #   Overlaps: ...
        #   Out-of-bounds: ...
        #   ----- (closing separator) -----
        #
        # We grab 600 chars AFTER the label, then search for the closing
        # separator (40+ dashes) that ends the block. The opening separator
        # right under the label is also 40+ dashes, so we skip the FIRST
        # match (which is the opening) and cut at the second.
        idx = out.find(label)
        if idx < 0:
            return {}
        block = out[idx: idx + 700]
        # Find all separator matches
        sep_matches = list(re.finditer(r"\n\s*-{40,}", block))
        # First separator is the opening one immediately after the label;
        # second is the closing one we want to cut at.
        if len(sep_matches) >= 2:
            block = block[: sep_matches[1].start()]
        elif sep_matches:
            block = block[: sep_matches[0].start()]
        return {
            "hpwl": find(r"HPWL:\s+([\d.]+)", block, float),
            "overlap_penalty": find(r"Overlap penalty:\s+([\d.]+)", block, float),
            "boundary_penalty": find(r"Boundary penalty:\s+([\d.]+)", block, float),
            "constraint_penalty": find(r"Constraint penalty:\s*([\d.]+)", block, float),
            "total_cost": find(r"TOTAL COST:\s+([\d.]+)", block, float),
            "overlaps": find(r"Overlaps:\s+(\d+)", block, int),
            "oob": find(r"Out-of-bounds:\s+(\d+)", block, int),
        }

    pre = parse_cost_block("Placement Cost (Pre-Legalization)")
    if pre:
        m["hpwl_post_opt"] = pre["hpwl"]
        m["overlap_penalty_post_opt"] = pre["overlap_penalty"]
        m["boundary_penalty_post_opt"] = pre["boundary_penalty"]
        m["constraint_penalty_post_opt"] = pre["constraint_penalty"]
        m["total_cost_post_opt"] = pre["total_cost"]
        m["overlaps_post_opt"] = pre["overlaps"]
        m["oob_post_opt"] = pre["oob"]

    post = parse_cost_block("Placement Cost (Post-Legalization)")
    if post:
        m["hpwl_post_legal"] = post["hpwl"]
        m["overlap_penalty_post_legal"] = post["overlap_penalty"]
        m["boundary_penalty_post_legal"] = post["boundary_penalty"]
        m["constraint_penalty_post_legal"] = post["constraint_penalty"]
        m["total_cost_post_legal"] = post["total_cost"]
        m["overlaps_post_legal"] = post["overlaps"]
        m["oob_post_legal"] = post["oob"]

    m["cost_change_from_legalization"] = find(
        r"Cost change from legalization:\s+(-?[\d.]+)", out, float
    )

    return m


# ---------------------------------------------------------------------------
# EE-perspective metrics from *_placed_model.json
# ---------------------------------------------------------------------------

def _net_pin_positions(board, net):
    """Return list of (x, y) absolute pin positions for a net."""
    by_ref = {c.ref: c for c in board.components}
    by_ref_lower = {c.ref.lower(): c for c in board.components}
    positions = []
    for ref, pad_name in net.pins:
        c = by_ref.get(ref) or by_ref_lower.get(ref.lower())
        if not c:
            continue
        for pad in c.pads:
            if pad.pad_name == pad_name:
                positions.append(pad.absolute_pos(c.x, c.y, c.rotation))
                break
        else:
            # Pad name not found — fall back to component origin
            positions.append((c.x, c.y))
    return positions


def median_pairwise_hpwl(board) -> float:
    """Compute median per-net HPWL across non-power nets."""
    hpwls = []
    for net in board.nets:
        pins = _net_pin_positions(board, net)
        if len(pins) < 2:
            continue
        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        hpwls.append(hpwl)
    return statistics.median(hpwls) if hpwls else 0.0


def top_nets_by_hpwl(board, n=10):
    """Return top-N nets by HPWL: list of (name, pin_count, hpwl)."""
    out = []
    for net in board.nets:
        pins = _net_pin_positions(board, net)
        if len(pins) < 2:
            continue
        xs = [p[0] for p in pins]
        ys = [p[1] for p in pins]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        out.append((net.name, len(pins), hpwl))
    out.sort(key=lambda x: -x[2])
    return out[:n]


def rotation_histogram(board) -> dict:
    """Count components by rotation bucket."""
    buckets = {0: 0, 90: 0, 180: 0, 270: 0, "other": 0}
    for c in board.components:
        r = round(c.rotation) % 360
        if r in buckets:
            buckets[r] += 1
        else:
            buckets["other"] += 1
    return buckets


def decap_distances(board) -> dict:
    """For each capacitor, distance to nearest IC pin (any pad)."""
    caps = [c for c in board.components if c.component_type == "capacitor"]
    ics = [c for c in board.components if c.component_type == "ic"]
    if not caps or not ics:
        return {"count": 0, "min": None, "median": None, "max": None, "violators": []}

    import math
    # Pre-compute IC pad absolute positions
    ic_pad_positions = []
    for ic in ics:
        for pad in ic.pads:
            ic_pad_positions.append(pad.absolute_pos(ic.x, ic.y, ic.rotation))

    distances = []
    violators = []  # caps >5mm from any IC
    for cap in caps:
        if not ic_pad_positions:
            break
        best = min(
            math.hypot(cap.x - px, cap.y - py)
            for px, py in ic_pad_positions
        )
        distances.append(best)
        if best > 5.0:
            violators.append({"ref": cap.ref, "distance_mm": round(best, 2)})
    return {
        "count": len(distances),
        "min": round(min(distances), 2) if distances else None,
        "median": round(statistics.median(distances), 2) if distances else None,
        "max": round(max(distances), 2) if distances else None,
        "violators_gt_5mm": violators,
    }


def connector_edge_distances(board_model) -> dict:
    """Distance from each connector centroid to nearest board edge."""
    outline = getattr(board_model, "board", None)
    if outline is None or outline.width <= 0 or outline.height <= 0:
        return {"count": 0, "note": "no board outline"}
    connectors = [c for c in board_model.components if c.component_type == "connector"]
    if not connectors:
        return {"count": 0}

    x_min, x_max = outline.x_min, outline.x_max
    y_min, y_max = outline.y_min, outline.y_max

    distances = []
    violators = []
    for conn in connectors:
        d = min(
            conn.x - x_min, x_max - conn.x,
            conn.y - y_min, y_max - conn.y,
        )
        distances.append(d)
        if d > 5.0:
            violators.append({"ref": conn.ref, "distance_mm": round(d, 2)})
    return {
        "count": len(distances),
        "min": round(min(distances), 2) if distances else None,
        "median": round(statistics.median(distances), 2) if distances else None,
        "max": round(max(distances), 2) if distances else None,
        "violators_gt_5mm": violators,
        "outline": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }


def density_hotspot(board_model, window_mm=10.0) -> dict:
    """Max local density over a sliding window_mm x window_mm grid.

    Component footprint area is accumulated into 1mm cells (using each
    component's bbox), then we slide a window_mm window and find the
    maximum percentage of cell area filled.
    """
    if not board_model.components:
        return {"max_local_density_pct": None}
    outline = getattr(board_model, "board", None)
    if outline and outline.width > 0 and outline.height > 0:
        x_min, x_max = outline.x_min, outline.x_max
        y_min, y_max = outline.y_min, outline.y_max
    else:
        xs = [c.x for c in board_model.components]
        ys = [c.y for c in board_model.components]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

    cell = 1.0
    nx = max(1, int((x_max - x_min) / cell) + 1)
    ny = max(1, int((y_max - y_min) / cell) + 1)

    occupied = [[0.0 for _ in range(ny)] for _ in range(nx)]
    for c in board_model.components:
        # Use the rotation-aware courtyard bbox directly
        try:
            bx0, by0, bx1, by1 = c.bbox
        except Exception:
            try:
                w = c.effective_width
                h = c.effective_height
            except AttributeError:
                w, h = c.width or 0.0, c.height or 0.0
            bx0, by0, bx1, by1 = c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2
        area = (bx1 - bx0) * (by1 - by0)
        if area <= 0:
            continue
        # Distribute component area across all cells its bbox covers
        cx_min = max(0, int((bx0 - x_min) / cell))
        cx_max = min(nx - 1, int((bx1 - x_min) / cell))
        cy_min = max(0, int((by0 - y_min) / cell))
        cy_max = min(ny - 1, int((by1 - y_min) / cell))
        n_cells = max(1, (cx_max - cx_min + 1) * (cy_max - cy_min + 1))
        per_cell = area / n_cells
        for ix in range(cx_min, cx_max + 1):
            for iy in range(cy_min, cy_max + 1):
                occupied[ix][iy] += per_cell

    w_cells = max(1, int(window_mm / cell))
    max_density_pct = 0.0
    for ix in range(0, nx - w_cells + 1):
        for iy in range(0, ny - w_cells + 1):
            total_area = 0.0
            for dx in range(w_cells):
                row = occupied[ix + dx]
                for dy in range(w_cells):
                    total_area += row[iy + dy]
            window_area = (w_cells * cell) ** 2
            pct = 100.0 * total_area / window_area if window_area > 0 else 0.0
            if pct > max_density_pct:
                max_density_pct = pct
    return {
        "window_mm": window_mm,
        "max_local_density_pct": round(max_density_pct, 1),
    }


def mine_ee_metrics(model_path: Path) -> dict:
    """Read placed_model.json and compute EE-perspective metrics."""
    try:
        sys.path.insert(0, str(ROOT))
        from models.board_model import BoardModel
        board = BoardModel.from_json(str(model_path))
    except Exception as e:
        return {"error": f"failed to load {model_path}: {e}"}

    return {
        "n_components": len(board.components),
        "n_nets": len(board.nets),
        "median_pairwise_hpwl": round(median_pairwise_hpwl(board), 2),
        "top_nets_by_hpwl": [
            {"net": n, "pins": p, "hpwl": round(h, 2)} for n, p, h in top_nets_by_hpwl(board, 10)
        ],
        "rotation_histogram": rotation_histogram(board),
        "decap_distances": decap_distances(board),
        "connector_edge_distances": connector_edge_distances(board),
        "density_hotspot_10mm": density_hotspot(board, 10.0),
    }


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------

def run_one(board: str, mode: str, profile: str, run_id: int) -> dict:
    """Run gridghost.py once and capture all metrics."""
    pcb_path = TEST_PCB_DIR / f"{board}.kicad_pcb"
    if not pcb_path.exists():
        return {"board": board, "mode": mode, "profile": profile, "run_id": run_id,
                "error": f"missing PCB: {pcb_path}"}

    cmd = [
        sys.executable, str(GRIDGHOST), "place", str(pcb_path),
        "--profile", profile,
        "--sa-iterations", "200",
        "--sa-reheat", "2",
        "--dry-run",
    ]
    if mode == "sa":
        cmd.append("--sa")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        return {"board": board, "mode": mode, "profile": profile, "run_id": run_id,
                "error": f"timeout: {e}"}
    wall_s = time.perf_counter() - t0

    out = proc.stdout + "\n" + proc.stderr

    # Save log
    log_path = OUTPUT_DIR / f"benchmark_{board}_{mode}_run{run_id}.log"
    log_path.write_text(out, encoding="utf-8")

    # Parse stdout metrics
    parsed = parse_output(out)

    # Copy placed_model.json (writer emits it even in --dry-run)
    src_model = TEST_PCB_DIR / f"{board}_placed_model.json"
    dst_model = OUTPUT_DIR / f"benchmark_{board}_{mode}_run{run_id}_model.json"
    if src_model.exists():
        shutil.copy2(src_model, dst_model)
        ee_metrics = mine_ee_metrics(dst_model)
    else:
        ee_metrics = {"error": "no placed_model.json produced"}

    # Derived metrics
    derived = {}
    if parsed["hpwl_post_opt"] and parsed["hpwl_post_legal"]:
        derived["legalizer_hpwl_delta_pct"] = round(
            100.0 * (parsed["hpwl_post_legal"] - parsed["hpwl_post_opt"]) / parsed["hpwl_post_opt"], 2
        )
    if parsed["sa_initial_hpwl"] and parsed["sa_final_hpwl"] and parsed["sa_initial_hpwl"] > 0:
        derived["sa_hpwl_reduction_pct"] = round(
            100.0 * (parsed["sa_initial_hpwl"] - parsed["sa_final_hpwl"]) / parsed["sa_initial_hpwl"], 2
        )
    if parsed["hpwl_post_opt"] and parsed["n_nets"]:
        derived["mean_hpwl_per_net_post_opt"] = round(parsed["hpwl_post_opt"] / parsed["n_nets"], 3)
    if parsed["hpwl_post_legal"] and parsed["n_nets"]:
        derived["mean_hpwl_per_net_post_legal"] = round(parsed["hpwl_post_legal"] / parsed["n_nets"], 3)

    result = {
        "board": board,
        "mode": mode,
        "profile": profile,
        "run_id": run_id,
        "wall_time_s": round(wall_s, 2),
        "exit_code": proc.returncode,
        "stdout_metrics": parsed,
        "derived": derived,
        "ee_metrics": ee_metrics,
        "log_file": str(log_path.relative_to(ROOT)),
        "model_file": str(dst_model.relative_to(ROOT)) if dst_model.exists() else None,
    }

    # Save per-run JSON
    out_json = OUTPUT_DIR / f"benchmark_{board}_{mode}_run{run_id}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate_sa_runs(results: list) -> dict:
    """For SA mode, compute min/median/max across replicates."""
    sa_runs = [r for r in results if r.get("mode") == "sa" and "error" not in r]
    if not sa_runs:
        return {}
    by_board = {}
    for r in sa_runs:
        by_board.setdefault(r["board"], []).append(r)
    summary = {}
    for board, runs in by_board.items():
        hpwls = [r["stdout_metrics"]["hpwl_post_legal"] for r in runs
                 if r["stdout_metrics"].get("hpwl_post_legal") is not None]
        walls = [r["wall_time_s"] for r in runs]
        if hpwls:
            summary[board] = {
                "n_runs": len(hpwls),
                "hpwl_post_legal_min": round(min(hpwls), 2),
                "hpwl_post_legal_median": round(statistics.median(hpwls), 2),
                "hpwl_post_legal_max": round(max(hpwls), 2),
                "hpwl_spread_pct": round(100.0 * (max(hpwls) - min(hpwls)) / statistics.median(hpwls), 2)
                if statistics.median(hpwls) > 0 else None,
                "wall_time_min_s": round(min(walls), 2),
                "wall_time_median_s": round(statistics.median(walls), 2),
                "wall_time_max_s": round(max(walls), 2),
            }
    return summary


def write_summary_csv(results: list, path: Path):
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"board": r["board"], "mode": r["mode"], "profile": r["profile"],
                         "run_id": r["run_id"], "error": r["error"]})
            continue
        s = r["stdout_metrics"]
        d = r.get("derived", {})
        rows.append({
            "board": r["board"], "mode": r["mode"], "profile": r["profile"], "run_id": r["run_id"],
            "wall_s": r["wall_time_s"],
            "n_comp": s.get("n_components"), "n_nets": s.get("n_nets"),
            "density_pct": round(100.0 * s["density"], 1) if s.get("density") is not None else None,
            "hpwl_post_place": s.get("sa_initial_hpwl"),
            "hpwl_post_opt": s.get("hpwl_post_opt"),
            "hpwl_post_legal": s.get("hpwl_post_legal"),
            "legalizer_delta_pct": d.get("legalizer_hpwl_delta_pct"),
            "sa_reduction_pct": d.get("sa_hpwl_reduction_pct"),
            "overlaps_post_place": s.get("overlaps_post_place"),
            "overlaps_post_opt": s.get("overlaps_post_opt"),
            "overlaps_post_legal": s.get("overlaps_post_legal"),
            "oob_post_legal": s.get("oob_post_legal"),
            "abacus_displacement_mm": s.get("abacus_displacement_mm"),
        })
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", nargs="*", default=BOARDS)
    parser.add_argument("--sa-replicates", type=int, default=SA_REPLICATES)
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--skip-sa", action="store_true")
    parser.add_argument("--include-th-sensor-mcu", action="store_true",
                        help="Supplementary mcu_peripheral run on th_sensor.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = []
    for board in args.boards:
        if not args.skip_greedy:
            plan.append((board, "greedy", "generic", 1))
        if not args.skip_sa:
            for i in range(1, args.sa_replicates + 1):
                plan.append((board, "sa", "generic", i))
    if args.include_th_sensor_mcu and "th_sensor" in args.boards:
        plan.append(("th_sensor", "sa", "mcu_peripheral", 1))

    print(f"Running {len(plan)} configurations...")
    results = []
    for i, (board, mode, profile, run_id) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {board} / {mode} / {profile} / run{run_id}", flush=True)
        r = run_one(board, mode, profile, run_id)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            s = r["stdout_metrics"]
            print(f"  wall={r['wall_time_s']:.1f}s hpwl_place={s.get('sa_initial_hpwl')} "
                  f"hpwl_opt={s.get('hpwl_post_opt')} hpwl_legal={s.get('hpwl_post_legal')} "
                  f"overlaps_post_legal={s.get('overlaps_post_legal')}")

    # Aggregate
    sa_summary = aggregate_sa_runs(results)

    # Write summary files
    summary_json = OUTPUT_DIR / "benchmark_summary.json"
    summary_json.write_text(json.dumps({
        "runs": results,
        "sa_aggregate": sa_summary,
    }, indent=2), encoding="utf-8")
    write_summary_csv(results, OUTPUT_DIR / "benchmark_summary.csv")

    print(f"\nWrote {summary_json}")
    print(f"Wrote {OUTPUT_DIR / 'benchmark_summary.csv'}")
    print(f"SA aggregate: {json.dumps(sa_summary, indent=2)}")


if __name__ == "__main__":
    main()
