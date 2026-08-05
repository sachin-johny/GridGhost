#!/usr/bin/env python3
"""GridGhost unified test orchestrator.

Single entry point that runs all test layers and produces an HTML
dashboard visualizing every board's placement result.

Phases (run in order, all results collected into one dashboard):

  1. **Unit tests** — `python -m pytest tests/` (or standalone `main()`
     fallback if pytest isn't installed). Fast correctness gate.
  2. **Placement pipeline** — runs `place_v2` in-process on every board
     in `tests/test_pcbs/`. Captures HPWL, overlaps, density, RUDY,
     cap-IC distances, component-type breakdown, and per-phase timing
     (parse / SA / legalize).
  3. **CLI smoke test** — runs `python gridghost.py place` end-to-end
     on each board via subprocess. Catches CLI-only bugs the in-process
     path misses (CLI arg parsing, file write, etc.).
  4. **Overlap regression** — runs `tests/test_overlap_regression.py`
     (self-report vs independent scan). The standing check from the
     evaluation fix.
  5. **External boards** — same as Phase 2 but on
     `tests/external_boards/rl_pcb/*.kicad_pcb` (9 real-world boards).
     Catches real-world edge cases the in-repo boards miss.
  6. **Dashboard generation** — combines all results into a single
     self-contained `dashboard.html` with per-board SVG visualizations,
     KPI table, and aggregate charts.

Outputs (under `tests/output/<run_id>/`):
  - `dashboard.html` — the main deliverable
  - `results.json` — raw metrics (for programmatic comparison)
  - `<board>.svg` — per-board SVG (also inlined in the dashboard)
  - `<board>_placed.kicad_pcb` — placed output (for opening in KiCad)
  - `unit_tests.log`, `cli_smoke_<board>.log` — raw phase logs

Usage:
    python tests/run_all.py                       # run everything
    python tests/run_all.py --skip-unit-tests     # skip phase 1
    python tests/run_all.py --skip-external       # skip phase 5
    python tests/run_all.py --only cbb test4      # only specific boards
    python tests/run_all.py --rudy-weight 0.3     # enable RUDY in SA

The dashboard is the primary output — open `tests/output/<run_id>/dashboard.html`
in any browser. No server needed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 mode: this script prints ✓/✗/⚠/→ glyphs, which the
# Windows default console codec (cp1252) cannot encode and would crash on
# (UnicodeEncodeError). PYTHONUTF8 is read at interpreter startup, so it
# must be set BEFORE any imports / subprocess spawning — setting it in
# os.environ here is inherited by every subprocess.run's
# env=os.environ.copy(). The explicit stdout/stderr reconfigure below
# additionally covers the current process.
#
# NOTE: the previous PYTHONHASHSEED=0 re-exec crutch has been removed —
# the engine/subcircuit_patterns.py set-iteration fix makes the codebase
# deterministic without forcing a hash seed.
os.environ.setdefault("PYTHONUTF8", "1")

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass  # not a reconfigurable stream (e.g. pytest capture) — fine

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser  # noqa: E402
from parsers.placement_writer import apply_placement  # noqa: E402
from place.pipeline import place_v2  # noqa: E402
from config import load_config  # noqa: E402
from engine.congestion import rudy_congestion_penalty  # noqa: E402
from engine.cost_function import total_hpwl  # noqa: E402
from assign.assign_caps import assign_caps  # noqa: E402


# ─── Phase 1: Unit tests ─────────────────────────────────────────────

def run_unit_tests(output_dir: Path) -> dict:
    """Run the pytest suite. Falls back to standalone main() if pytest missing.

    Returns a phase-result dict.
    """
    print("\n" + "=" * 60)
    print("Phase 1: Unit tests")
    print("=" * 60)

    t0 = time.perf_counter()
    log_path = output_dir / "unit_tests.log"

    # Try pytest first
    pytest_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    try:
        with open(log_path, "w") as f:
            proc = subprocess.run(
                pytest_cmd, cwd=str(ROOT), env=os.environ.copy(),
                stdout=f, stderr=subprocess.STDOUT, timeout=180,
            )
        rc = proc.returncode
        # Read back the log to count passes/failures
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        n_passed = log_text.count(" passed")
        # pytest prints "X passed, Y failed" in the summary line
        n_failed = 0
        for line in log_text.splitlines():
            if "failed" in line.lower() and "passed" in line.lower():
                # e.g. "=== 81 passed, 0 failed in 1.23s ==="
                import re
                m_pass = re.search(r"(\d+)\s+passed", line)
                m_fail = re.search(r"(\d+)\s+failed", line)
                if m_pass:
                    n_passed = int(m_pass.group(1))
                if m_fail:
                    n_failed = int(m_fail.group(1))
                break
        n_total = n_passed + n_failed
        duration = time.perf_counter() - t0
        status = "pass" if rc == 0 else "fail"
        print(f"  pytest: {n_passed} passed, {n_failed} failed ({duration:.1f}s)")
        return {
            "status": status,
            "n_total": n_total,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "duration_s": duration,
            "log_path": str(log_path.relative_to(ROOT)),
            "runner": "pytest",
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: run each test_*.py standalone
    print("  pytest unavailable, falling back to standalone main()...")
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    n_passed = 0
    n_failed = 0
    failures: list[str] = []
    with open(log_path, "w") as f:
        for tf in test_files:
            # Skip the regression test (it's slow and runs as its own phase)
            if tf.name == "test_overlap_regression.py":
                continue
            f.write(f"\n=== {tf.name} ===\n")
            try:
                proc = subprocess.run(
                    [sys.executable, str(tf)],
                    cwd=str(ROOT), env=os.environ.copy(),
                    stdout=f, stderr=subprocess.STDOUT, timeout=60,
                )
                if proc.returncode == 0:
                    n_passed += 1
                else:
                    n_failed += 1
                    failures.append(tf.name)
            except subprocess.TimeoutExpired:
                n_failed += 1
                failures.append(f"{tf.name} (timeout)")
    duration = time.perf_counter() - t0
    n_total = n_passed + n_failed
    status = "pass" if n_failed == 0 else "fail"
    print(f"  standalone: {n_passed}/{n_total} test files passed ({duration:.1f}s)")
    return {
        "status": status,
        "n_total": n_total,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "failures": failures,
        "duration_s": duration,
        "log_path": str(log_path.relative_to(ROOT)),
        "runner": "standalone",
    }


# ─── Phase 2/5: Placement pipeline (in-process) ──────────────────────

def run_placement_pipeline(
    boards: list[Path],
    output_dir: Path,
    *,
    rudy_weight: float | None = None,
    pin_density_weight: float | None = None,
    seed: int = 42,
    source_label: str = "test_pcbs",
) -> tuple[list[dict], dict]:
    """Run place_v2 on each board in-process. Returns (per-board results, phase summary).

    Captures: components, overlaps (independent scan), Edge.Cuts present,
    density, HPWL, RUDY peak/penalty, cap-IC distances, component-type
    breakdown, per-phase timing, and the placed model (for SVG rendering).
    """
    print("\n" + "=" * 60)
    print(f"Phase: Placement pipeline ({source_label})")
    print("=" * 60)

    t_phase_start = time.perf_counter()
    board_results: list[dict] = []

    for pcb_path in boards:
        print(f"\n  [{pcb_path.stem}] running place_v2...")
        board_result = _run_one_board(
            pcb_path, output_dir,
            rudy_weight=rudy_weight, pin_density_weight=pin_density_weight,
            seed=seed, source_label=source_label,
        )
        board_results.append(board_result)
        n_ovl = board_result.get("overlaps", "?")
        edge = "✓" if board_result.get("edge_cuts_present") else "✗"
        t = board_result.get("timing", {}).get("total", 0.0)
        print(f"    overlaps={n_ovl}  Edge.Cuts={edge}  time={t:.1f}s")

    duration = time.perf_counter() - t_phase_start
    n_pass = sum(1 for b in board_results if b.get("overlaps", 0) == 0)
    phase_summary = {
        "status": "pass" if n_pass == len(board_results) else ("fail" if board_results else "skip"),
        "n_total": len(board_results),
        "n_passed": n_pass,
        "n_failed": len(board_results) - n_pass,
        "duration_s": duration,
        "summary": f"{n_pass}/{len(board_results)} boards with 0 overlaps",
    }
    print(f"\n  → {n_pass}/{len(board_results)} boards clean ({duration:.1f}s total)")
    return board_results, phase_summary


def _run_one_board(
    pcb_path: Path,
    output_dir: Path,
    *,
    rudy_weight: float | None = None,
    pin_density_weight: float | None = None,
    seed: int = 42,
    source_label: str,
) -> dict:
    """Run place_v2 on a single board and collect all metrics."""
    # Seed PRNG
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    cfg = load_config()
    timings: dict[str, float] = {}

    # ─── Parse ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    parser = KiCadParser(str(pcb_path), bbox_margin=cfg.parser.bbox_margin)
    model = parser.parse()
    timings["parse"] = time.perf_counter() - t0

    n_components = len(model.components)

    # Component-type breakdown
    type_counts: dict[str, int] = {}
    for c in model.components:
        t = c.component_type or "generic"
        type_counts[t] = type_counts.get(t, 0) + 1
    n_ics = sum(v for k, v in type_counts.items() if k in {"ic", "mcu", "regulator"})
    n_caps = type_counts.get("capacitor", 0)

    # ─── Placement (place_v2: SA + legalize) ─────────────────────────
    n_macros = sum(1 for c in model.components if not c.is_fixed)
    sa_iters = max(1500, 25 * n_macros)

    t0 = time.perf_counter()
    try:
        place_v2(
            model,
            margin=cfg.placement.margin,
            grid_mm=cfg.legalization.grid_mm,
            sa_iterations=sa_iters,
            sa_reheats=cfg.annealer.reheat_count,
            seed=seed,
            verbose=False,
            rudy_weight=rudy_weight,
            pin_density_weight=pin_density_weight,
        )
        sa_failed = False
        sa_error = None
    except Exception as e:
        sa_failed = True
        sa_error = f"{type(e).__name__}: {e}"
    sa_time = time.perf_counter() - t0
    # place_v2 internally times SA + legalize as one call; split them
    # by re-running the timing breakdown via the result dict if available.
    # For simplicity we attribute the whole thing to "sa" here and let
    # the dashboard show it as "placement". The legalizer's self-report
    # is captured separately below.
    timings["sa"] = sa_time
    timings["legalize"] = 0.0  # place_v2 doesn't expose this split
    timings["total"] = sum(timings.values())

    if sa_failed:
        return {
            "name": pcb_path.stem,
            "source": source_label,
            "n_components": n_components,
            "n_ics": n_ics,
            "n_caps": n_caps,
            "component_types": type_counts,
            "overlaps": -1,
            "overlap_pairs": [],
            "edge_cuts_present": False,
            "density": 0.0,
            "hpwl": 0.0,
            "rudy_peak": 0.0,
            "rudy_penalty": 0.0,
            "cap_ic_distances": [],
            "timing": timings,
            "error": sa_error,
            "model": model,  # may still be useful for SVG
        }

    # ─── Write placed .kicad_pcb (for CLI parity + Edge.Cuts check) ──
    placed_pcb = output_dir / f"{pcb_path.stem}_placed.kicad_pcb"
    try:
        apply_placement(model, str(pcb_path), str(placed_pcb), backup=False)
        # Re-parse to verify the write + check Edge.Cuts
        t0 = time.perf_counter()
        parser2 = KiCadParser(str(placed_pcb), bbox_margin=cfg.parser.bbox_margin)
        model2 = parser2.parse()
        timings["write_reparse"] = time.perf_counter() - t0
        edge_cuts_present = model2.user_defined_outline or _has_edge_cuts_geom(placed_pcb)
        independent_overlaps, overlap_pairs = _scan_overlaps(model2)
    except Exception as e:
        edge_cuts_present = False
        independent_overlaps = -1
        overlap_pairs = []
        print(f"    WARNING: write/reparse failed: {e}")
    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    # ─── Metrics on the in-memory model (HPWL, RUDY, density) ────────
    hpwl = total_hpwl(model)
    try:
        rudy_penalty, rudy_peak, rudy_avg, _ = rudy_congestion_penalty(model)
    except Exception:
        rudy_penalty = rudy_peak = rudy_avg = 0.0
    density = _compute_density(model)

    # Cap-IC distances
    cap_ic_distances: list[float] = []
    try:
        decap_map = assign_caps(model)
        for ic_ref, cap_refs in decap_map.items():
            ic = model.get_component(ic_ref)
            if ic is None:
                continue
            for cap_ref in cap_refs:
                cap = model.get_component(cap_ref)
                if cap is None:
                    continue
                d = float(((cap.x - ic.x) ** 2 + (cap.y - ic.y) ** 2) ** 0.5)
                cap_ic_distances.append(d)
    except Exception:
        pass

    return {
        "name": pcb_path.stem,
        "source": source_label,
        "n_components": n_components,
        "n_ics": n_ics,
        "n_caps": n_caps,
        "component_types": type_counts,
        "overlaps": independent_overlaps,
        "overlap_pairs": [
            {"a": a, "b": b, "area_mm2": round(area, 3)}
            for a, b, area in overlap_pairs
        ],
        "edge_cuts_present": bool(edge_cuts_present),
        "density": round(density, 4),
        "hpwl": round(hpwl, 2),
        "rudy_peak": round(rudy_peak, 4),
        "rudy_penalty": round(rudy_penalty, 4),
        "rudy_avg": round(rudy_avg, 4),
        "cap_ic_distances": [round(d, 3) for d in cap_ic_distances],
        "timing": timings,
        "model": model,  # for SVG rendering in the dashboard
        "placed_pcb_path": str(placed_pcb.relative_to(ROOT)) if placed_pcb.exists() else None,
    }


def _scan_overlaps(model) -> tuple[int, list[tuple[str, str, float]]]:
    """Independent pairwise bbox overlap scan."""
    pairs = []
    comps = model.components
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            if a.overlaps(b):
                pairs.append((a.ref, b.ref, a.overlap_area(b)))
    return len(pairs), pairs


def _has_edge_cuts_geom(pcb_path: Path) -> bool:
    """Return True if the .kicad_pcb has actual gr_* geometry on Edge.Cuts."""
    import re
    text = pcb_path.read_text(encoding="utf-8", errors="replace")
    # Match (gr_rect|gr_line|gr_poly|gr_circle ... (layer "Edge.Cuts"))
    pattern = re.compile(
        r"\(\s*gr_(?:rect|line|poly|circle)\b[^)]*?"
        r"\(\s*layer\s+\"Edge\.Cuts\"\s*\)",
        re.DOTALL,
    )
    return bool(pattern.search(text))


def _compute_density(model) -> float:
    """Component effective area / board area."""
    board = model.board
    area = board.width * board.height
    if area <= 0:
        return 0.0
    comp_area = sum(c.effective_width * c.effective_height for c in model.components)
    return comp_area / area


# ─── Phase 3: CLI smoke test ─────────────────────────────────────────

def run_cli_smoke(boards: list[Path], output_dir: Path, seed: int = 42) -> dict:
    """Run `python gridghost.py place` on each board. Returns phase summary.

    The CLI smoke test catches CLI-only bugs the in-process path misses:
    arg parsing, file write, hash-seed re-exec, etc.
    """
    print("\n" + "=" * 60)
    print("Phase 3: CLI smoke test")
    print("=" * 60)

    t0 = time.perf_counter()
    n_pass = 0
    n_total = len(boards)
    board_summaries: list[dict] = []

    for pcb_path in boards:
        out_pcb = output_dir / f"{pcb_path.stem}_cli_placed.kicad_pcb"
        log_path = output_dir / f"cli_smoke_{pcb_path.stem}.log"
        cmd = [
            sys.executable, str(ROOT / "gridghost.py"), "place",
            str(pcb_path), "-o", str(out_pcb),
            "--seed", str(seed),
        ]
        env = os.environ.copy()
        try:
            with open(log_path, "w") as f:
                proc = subprocess.run(
                    cmd, cwd=str(ROOT), env=env,
                    stdout=f, stderr=subprocess.STDOUT, timeout=180,
                )
            rc = proc.returncode
            if rc == 0 and out_pcb.exists():
                n_pass += 1
                print(f"  ✓ {pcb_path.stem}")
                board_summaries.append({"name": pcb_path.stem, "status": "pass"})
            else:
                print(f"  ✗ {pcb_path.stem} (rc={rc}, output exists: {out_pcb.exists()})")
                board_summaries.append({"name": pcb_path.stem, "status": "fail", "rc": rc})
        except subprocess.TimeoutExpired:
            print(f"  ✗ {pcb_path.stem} (TIMEOUT)")
            board_summaries.append({"name": pcb_path.stem, "status": "timeout"})
        except Exception as e:
            print(f"  ✗ {pcb_path.stem} ({type(e).__name__}: {e})")
            board_summaries.append({"name": pcb_path.stem, "status": "error", "error": str(e)})

    duration = time.perf_counter() - t0
    print(f"\n  → {n_pass}/{n_total} CLI runs succeeded ({duration:.1f}s)")
    return {
        "status": "pass" if n_pass == n_total else "fail",
        "n_total": n_total,
        "n_passed": n_pass,
        "n_failed": n_total - n_pass,
        "duration_s": duration,
        "boards": board_summaries,
    }


# ─── Phase 4: Overlap regression ─────────────────────────────────────

def run_overlap_regression(output_dir: Path) -> dict:
    """Run tests/test_overlap_regression.py. Returns phase summary."""
    print("\n" + "=" * 60)
    print("Phase 4: Overlap regression (self-report vs independent scan)")
    print("=" * 60)

    t0 = time.perf_counter()
    log_path = output_dir / "overlap_regression.log"
    cmd = [sys.executable, str(ROOT / "tests" / "test_overlap_regression.py")]
    env = os.environ.copy()

    try:
        with open(log_path, "w") as f:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=env,
                stdout=f, stderr=subprocess.STDOUT, timeout=600,
            )
        rc = proc.returncode
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        # Parse "  ✓ OK: <board> — <N> overlaps" lines
        n_pass = log_text.count("✓ OK")
        n_fail = log_text.count("✗ FAIL")
        n_total = n_pass + n_fail
        duration = time.perf_counter() - t0
        status = "pass" if rc == 0 and n_fail == 0 else "fail"
        print(f"  → {n_pass}/{n_total} boards pass ({duration:.1f}s)")
        return {
            "status": status,
            "n_total": n_total,
            "n_passed": n_pass,
            "n_failed": n_fail,
            "duration_s": duration,
            "log_path": str(log_path.relative_to(ROOT)),
        }
    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - t0
        print(f"  ✗ TIMEOUT after {duration:.1f}s")
        return {
            "status": "fail",
            "n_total": 0,
            "n_passed": 0,
            "n_failed": 0,
            "duration_s": duration,
            "error": "timeout",
        }


# ─── Main orchestrator ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GridGhost unified test orchestrator. Runs all test layers and "
                    "produces an HTML dashboard with per-board visualizations.",
    )
    ap.add_argument("--skip-unit-tests", action="store_true",
                    help="Skip phase 1 (unit tests)")
    ap.add_argument("--skip-cli-smoke", action="store_true",
                    help="Skip phase 3 (CLI smoke test)")
    ap.add_argument("--skip-overlap-regression", action="store_true",
                    help="Skip phase 4 (overlap regression)")
    ap.add_argument("--skip-external", action="store_true",
                    help="Skip phase 5 (external boards)")
    ap.add_argument("--skip-placement", action="store_true",
                    help="Skip phase 2 (placement pipeline on test_pcbs)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run these board stems (e.g. cbb test4). Applies to phases 2+3.")
    ap.add_argument("--rudy-weight", type=float, default=None,
                    help="RUDY congestion penalty weight in SA (default: from config.json = enabled)")
    ap.add_argument("--pin-density-weight", type=float, default=None,
                    help="Pin-density congestion penalty weight in SA (default: from config.json = enabled)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for placement determinism (default: 42)")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: tests/output/<run_id>/)")
    ap.add_argument("--run-id", default=None,
                    help="Run identifier (default: run_<timestamp>)")
    args = ap.parse_args()

    # ─── Set up output directory ─────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"run_{timestamp}"
    output_dir = Path(args.output_dir) if args.output_dir else (ROOT / "tests" / "output" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_dir}")

    # ─── Collect boards ──────────────────────────────────────────────
    test_pcb_dir = ROOT / "tests" / "test_pcbs"
    external_dir = ROOT / "tests" / "external_boards" / "rl_pcb"

    test_boards = sorted(test_pcb_dir.glob("*.kicad_pcb"))
    external_boards = sorted(external_dir.glob("*.kicad_pcb")) if external_dir.exists() else []

    if args.only:
        test_boards = [b for b in test_boards if b.stem in args.only]
        external_boards = [b for b in external_boards if b.stem in args.only]

    print(f"Test boards: {len(test_boards)} ({', '.join(b.stem for b in test_boards)})")
    if not args.skip_external:
        print(f"External boards: {len(external_boards)}")

    # ─── Run phases ──────────────────────────────────────────────────
    phases: dict[str, dict] = {}

    if not args.skip_unit_tests:
        phases["unit_tests"] = run_unit_tests(output_dir)
    else:
        phases["unit_tests"] = {"status": "skipped", "summary": "skipped by --skip-unit-tests"}

    all_board_results: list[dict] = []

    if not args.skip_placement and test_boards:
        results, summary = run_placement_pipeline(
            test_boards, output_dir,
            rudy_weight=args.rudy_weight,
            pin_density_weight=args.pin_density_weight,
            seed=args.seed,
            source_label="test_pcbs",
        )
        all_board_results.extend(results)
        phases["placement_test_pcbs"] = summary
    else:
        phases["placement_test_pcbs"] = {"status": "skipped"}

    if not args.skip_cli_smoke and test_boards:
        phases["cli_smoke"] = run_cli_smoke(test_boards, output_dir, seed=args.seed)
    else:
        phases["cli_smoke"] = {"status": "skipped"}

    if not args.skip_overlap_regression:
        phases["overlap_regression"] = run_overlap_regression(output_dir)
    else:
        phases["overlap_regression"] = {"status": "skipped"}

    if not args.skip_external and external_boards:
        results, summary = run_placement_pipeline(
            external_boards, output_dir,
            rudy_weight=args.rudy_weight,
            pin_density_weight=args.pin_density_weight,
            seed=args.seed,
            source_label="external_boards",
        )
        all_board_results.extend(results)
        phases["placement_external"] = summary
    else:
        phases["placement_external"] = {"status": "skipped"}

    # ─── Write per-board SVGs (also inlined in the dashboard) ────────
    print("\n" + "=" * 60)
    print("Generating per-board SVGs...")
    print("=" * 60)
    from tests.visualizer import render_board_svg
    for board in all_board_results:
        model = board.get("model")
        if model is None:
            continue
        try:
            svg = render_board_svg(model, target_width=800, title=board["name"])
            svg_path = output_dir / f"{board['name']}.svg"
            svg_path.write_text(svg, encoding="utf-8")
        except Exception as e:
            print(f"  ⚠ {board['name']}: SVG render failed: {e}")

    # ─── Generate dashboard ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating dashboard...")
    print("=" * 60)

    # Get git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"

    # Strip the model objects before serializing to JSON (they're not JSON-serializable)
    boards_for_json = []
    for b in all_board_results:
        b_copy = {k: v for k, v in b.items() if k != "model"}
        boards_for_json.append(b_copy)

    results_obj = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "repro_cmd": (
            f"python tests/run_all.py"
            + (" --skip-unit-tests" if args.skip_unit_tests else "")
            + (" --skip-cli-smoke" if args.skip_cli_smoke else "")
            + (" --skip-overlap-regression" if args.skip_overlap_regression else "")
            + (" --skip-external" if args.skip_external else "")
            + (f" --rudy-weight {args.rudy_weight}" if args.rudy_weight is not None else "")
            + (f" --pin-density-weight {args.pin_density_weight}" if args.pin_density_weight is not None else "")
            + (f" --seed {args.seed}" if args.seed != 42 else "")
        ),
        "phases": phases,
        "boards": boards_for_json,
    }

    # Write raw JSON
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results_obj, indent=2, default=str), encoding="utf-8")
    print(f"  results.json: {json_path}")

    # Generate dashboard (needs the boards WITH model objects for SVG rendering)
    from tests.dashboard import generate_dashboard
    dashboard_obj = dict(results_obj)
    dashboard_obj["boards"] = all_board_results  # restore models for SVG
    dashboard_path = output_dir / "dashboard.html"
    generate_dashboard(dashboard_obj, dashboard_path)
    print(f"  dashboard.html: {dashboard_path}")

    # ─── Final summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Run ID:       {run_id}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Dashboard:    {dashboard_path}")
    print(f"  Raw metrics:  {json_path}")
    n_overlaps = sum(b.get("overlaps", 0) for b in all_board_results if b.get("overlaps", 0) > 0)
    n_pass = sum(1 for b in all_board_results if b.get("overlaps", 0) == 0)
    print(f"  Boards:       {n_pass}/{len(all_board_results)} with 0 overlaps")
    print(f"  Total overlaps: {n_overlaps}")
    print()
    print("Open the dashboard in any browser:")
    print(f"  file://{dashboard_path.absolute()}")


if __name__ == "__main__":
    main()
