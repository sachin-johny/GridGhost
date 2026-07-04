#!/usr/bin/env python3
"""Phase 3.1 before/after demo: sheet-aware clustering on th_sensor.kicad_pcb.

Parses th_sensor.kicad_pcb, then runs cluster_components twice:
  1. WITHOUT sheet-aware edges (monkey-patch _add_sheet_edges to no-op)
  2. WITH sheet-aware edges (default behaviour after Phase 3.1)

Prints a compact table showing, for each cluster, which sheets its
components came from.  A "good" sheet-aware clustering has clusters
that are dominated by ONE sheet — most clusters should have ≥80% of
their components from a single sheet.  A pure-net clustering spreads
each sheet across multiple clusters because components on the same
sheet often share only power rails (excluded from signal edges).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.kicad_parser import KiCadParser
from engine.net_clustering import cluster_components


def cluster_sheet_purity(clusters, model):
    """For each cluster, return (dominant_sheet, dominant_count, total,
    sheet_histogram).  Purity = dominant_count/total.
    """
    ref_to_sheet = {c.ref: getattr(c, 'sheet', '') or '' for c in model.components}
    results = []
    for cluster in clusters:
        if not cluster:
            continue
        sheets = [ref_to_sheet.get(r, '') for r in cluster]
        hist = Counter(sheets)
        dominant_sheet, dominant_count = hist.most_common(1)[0]
        total = len(cluster)
        results.append({
            'dominant_sheet': dominant_sheet,
            'purity': dominant_count / total,
            'dominant_count': dominant_count,
            'total': total,
            'hist': dict(hist),
        })
    return results


def print_clusters(label, clusters, model):
    print(f"\n=== {label} ===")
    print(f"  {len(clusters)} clusters total")
    purities = cluster_sheet_purity(clusters, model)
    avg_purity = sum(p['purity'] for p in purities) / len(purities) if purities else 0
    print(f"  Average cluster sheet-purity: {avg_purity:.1%}")
    print(f"  Per-cluster breakdown:")
    for i, p in enumerate(purities):
        sheet_str = ', '.join(f"{s}={n}" for s, n in sorted(p['hist'].items(), key=lambda x: -x[1]))
        print(f"    Cluster {i}: {p['total']} comps, {p['purity']:.0%} {p['dominant_sheet']}  [{sheet_str}]")


def main():
    pcb_path = ROOT / 'tests' / 'test_pcbs' / 'th_sensor.kicad_pcb'
    if not pcb_path.exists():
        print(f"ERROR: {pcb_path} not found")
        return 1

    parser = KiCadParser(str(pcb_path))
    model = parser.parse()

    # Print sheet distribution
    sheet_counts = Counter(getattr(c, 'sheet', '') or '' for c in model.components)
    print(f"Parsed {len(model.components)} components from {pcb_path.name}")
    print(f"Sheet distribution:")
    for sheet, count in sorted(sheet_counts.items(), key=lambda x: -x[1]):
        print(f"  {sheet or '(empty)':<20} {count} comps")

    # === BEFORE: pure-net clustering (sheet-aware edges disabled) ===
    # Monkey-patch _add_sheet_edges to no-op, run clustering, restore.
    import engine.net_clustering as nc
    original_add_sheet_edges = nc._add_sheet_edges
    nc._add_sheet_edges = lambda G, model: None  # disable
    try:
        clusters_before = cluster_components(model)
    finally:
        nc._add_sheet_edges = original_add_sheet_edges
    print_clusters("BEFORE (pure-net clustering, no sheet edges)", clusters_before, model)

    # === AFTER: sheet-aware clustering (default Phase 3.1 behavior) ===
    clusters_after = cluster_components(model)
    print_clusters("AFTER (sheet-aware clustering, Phase 3.1)", clusters_after, model)

    # Summary
    print("\n=== Summary ===")
    purities_before = cluster_sheet_purity(clusters_before, model)
    purities_after = cluster_sheet_purity(clusters_after, model)
    avg_before = sum(p['purity'] for p in purities_before) / len(purities_before) if purities_before else 0
    avg_after = sum(p['purity'] for p in purities_after) / len(purities_after) if purities_after else 0
    print(f"  Avg cluster sheet-purity BEFORE: {avg_before:.1%}")
    print(f"  Avg cluster sheet-purity AFTER:  {avg_after:.1%}")
    print(f"  Improvement: {avg_after - avg_before:+.1%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
