#!/usr/bin/env python3
"""Run the placement pipeline (SA + legalizer) on selected .kicad_pcb files.

Produces logs under `tests/output/` and a JSON summary per-run.
Exits with non-zero if post-legalization overlaps or OOB are detected.
"""
from __future__ import annotations

import sys
import subprocess
import json
import re
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PCBS = [
    ROOT / "tests" / "test_pcbs" / "cbb.kicad_pcb",
    ROOT / "tests" / "test_pcbs" / "test4.kicad_pcb",
]


def parse_summary(output: str) -> dict:
    summary = {
        "sa_initial_cost": None,
        "sa_final_cost": None,
        "sa_initial_hpwl": None,
        "sa_final_hpwl": None,
        "sa_overlaps": None,
        "legal_overlaps": None,
        "legal_oob": None,
    }

    # SA cost line: "SA: cost 1234.0 -> 1100.0"
    m = re.search(r"SA: cost\s*([0-9.]+)\s*->\s*([0-9.]+)", output)
    if m:
        summary["sa_initial_cost"] = float(m.group(1))
        summary["sa_final_cost"] = float(m.group(2))

    m = re.search(r"SA: HPWL\s*([0-9.]+)\s*->\s*([0-9.]+)", output)
    if m:
        summary["sa_initial_hpwl"] = float(m.group(1))
        summary["sa_final_hpwl"] = float(m.group(2))

    m = re.search(r"SA: overlaps=\s*([0-9]+)", output)
    if m:
        summary["sa_overlaps"] = int(m.group(1))

    # Legalizer prints: "Legalization output: X overlaps, Y out-of-bounds"
    m = re.search(r"Legalization output:\s*([0-9]+) overlaps,\s*([0-9]+) out-of-bounds", output)
    if m:
        summary["legal_overlaps"] = int(m.group(1))
        summary["legal_oob"] = int(m.group(2))

    # Fallback: look for "After Legalization" board summary lines
    if summary["legal_overlaps"] is None:
        m = re.search(r"After Legalization.*?(\d+) overlaps.*?(\d+) out-of-bounds", output, re.S)
        if m:
            summary["legal_overlaps"] = int(m.group(1))
            summary["legal_oob"] = int(m.group(2))

    return summary


def run_one(pcb_path: Path, sa_iterations: int, sa_reheat: int) -> dict:
    gridghost = ROOT / "gridghost.py"
    if not gridghost.exists():
        raise FileNotFoundError(f"gridghost.py not found at {gridghost}")

    cmd = [sys.executable, str(gridghost), "place", str(pcb_path), "--profile", "generic", "--sa-iterations", str(sa_iterations), "--sa-reheat", str(sa_reheat), "--dry-run"]

    print(f"Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = proc.stdout + "\n" + proc.stderr

    base = pcb_path.stem
    log_path = OUTPUT_DIR / f"{base}_run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)

    summary = parse_summary(out)
    summary_path = OUTPUT_DIR / f"{base}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote log: {log_path}")
    print(f"Wrote summary: {summary_path}")
    return {"pcb": str(pcb_path), "log": str(log_path), "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcbs", nargs="*", help="Paths to .kicad_pcb files", default=[str(p) for p in DEFAULT_PCBS])
    parser.add_argument("--sa-iterations", type=int, default=200)
    parser.add_argument("--sa-reheat", type=int, default=2)
    args = parser.parse_args()

    results = []
    exit_code = 0
    for p in args.pcbs:
        pcb = Path(p)
        if not pcb.exists():
            print(f"Skipping missing PCB: {pcb}")
            continue
        res = run_one(pcb, args.sa_iterations, args.sa_reheat)
        results.append(res)
        legal_overlaps = res["summary"].get("legal_overlaps")
        legal_oob = res["summary"].get("legal_oob")
        if legal_overlaps is None:
            print(f"Warning: could not parse legalizer overlaps for {pcb}")
        elif legal_overlaps > 0 or (legal_oob is not None and legal_oob > 0):
            print(f"FAIL: Post-legalization issues for {pcb}: overlaps={legal_overlaps}, oob={legal_oob}")
            exit_code = 2

    summary_all = OUTPUT_DIR / "runs_summary.json"
    with open(summary_all, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
