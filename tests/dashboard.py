"""Self-contained HTML dashboard generator for GridGhost test runs.

Takes the metrics dict produced by ``tests/run_all.py`` and emits a
single ``dashboard.html`` file with:

  1. Header — run timestamp, total boards, total overlaps, total time.
  2. KPI table — sortable, one row per board, columns for the core
     metrics (components, overlaps, Edge.Cuts, density, HPWL, RUDY peak,
     SA time, legalize time, total time).
  3. Per-board section — inline SVG visualization + metric chips +
     component-type breakdown bar + cap-IC distance histogram.
  4. Aggregate charts — overlap count bar chart, timing breakdown
     stacked bar, component-type distribution across all boards.
  5. Footer — reproduction command + git commit.

The HTML is fully self-contained: inline ``<style>``, inline SVGs, no
external CSS/JS resources. Opens in any browser, no server needed.

Used by ``tests/run_all.py`` to produce the final dashboard.
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.visualizer import render_board_svg


# ─── Color palette (consistent with visualizer.py) ──────────────────

TYPE_COLORS = {
    "ic":            "#d62728",
    "mcu":           "#d62728",
    "regulator":     "#9467bd",
    "capacitor":     "#1f77b4",
    "resistor":      "#2ca02c",
    "connector":     "#ff7f0e",
    "crystal":       "#17becf",
    "mounting_hole": "#7f7f7f",
    "generic":       "#bcbcbc",
}


def generate_dashboard(
    results: dict[str, Any],
    output_path: Path,
    *,
    title: str = "GridGhost Test Dashboard",
) -> Path:
    """Write a self-contained HTML dashboard to ``output_path``.

    Args:
        results: Dict with structure:
            {
                "run_id": str,
                "timestamp": str (ISO),
                "git_commit": str,
                "phases": {
                    "unit_tests": {...},
                    "placement": {"boards": [...]},
                    "cli_smoke": {"boards": [...]},
                    "overlap_regression": {...},
                    "external_boards": {"boards": [...]},
                },
                "boards": [  # consolidated per-board view
                    {
                        "name": str,
                        "source": "test_pcbs" | "external_boards",
                        "n_components": int,
                        "component_types": {type: count},
                        "overlaps": int,
                        "overlap_pairs": [...],
                        "edge_cuts_present": bool,
                        "density": float,
                        "hpwl": float,
                        "rudy_peak": float,
                        "rudy_penalty": float,
                        "cap_ic_distances": [float, ...],
                        "timing": {"parse": float, "sa": float, "legalize": float, "total": float},
                        "model": BoardModel,  # for SVG rendering
                    },
                    ...
                ],
            }
        output_path: Where to write the HTML.
        title: Dashboard title.

    Returns:
        The output_path (for convenience).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boards = results.get("boards", [])
    total_overlaps = sum(b.get("overlaps", 0) for b in boards)
    total_time = sum(b.get("timing", {}).get("total", 0.0) for b in boards)
    n_passing = sum(1 for b in boards if b.get("overlaps", 0) == 0)

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        _CSS,
        "</style>",
        "</head>",
        "<body>",
    ]

    # ─── Header ──────────────────────────────────────────────────────
    parts.append('<header class="hero">')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(
        f'<div class="meta">Run <code>{html.escape(results.get("run_id", "?"))}</code> '
        f'&middot; {html.escape(results.get("timestamp", "?"))} '
        f'&middot; commit <code>{html.escape(results.get("git_commit", "?")[:12])}</code></div>'
    )
    parts.append('<div class="summary-cards">')
    parts.append(_summary_card("Boards", str(len(boards)), "#1f77b4"))
    parts.append(_summary_card("Passing (0 overlaps)", f"{n_passing}/{len(boards)}",
                               "#28a745" if n_passing == len(boards) else "#ffc107"))
    parts.append(_summary_card("Total overlaps", str(total_overlaps),
                               "#dc3545" if total_overlaps else "#28a745"))
    parts.append(_summary_card("Total wall-time", f"{total_time:.1f}s", "#6c757d"))
    parts.append("</div>")
    parts.append("</header>")

    # ─── Phase summaries ─────────────────────────────────────────────
    parts.append('<section class="phases">')
    parts.append("<h2>Test phases</h2>")
    parts.append('<div class="phase-grid">')
    phases = results.get("phases", {})
    for phase_name, phase_data in phases.items():
        parts.append(_phase_card(phase_name, phase_data))
    parts.append("</div>")
    parts.append("</section>")

    # ─── KPI table ───────────────────────────────────────────────────
    parts.append('<section class="kpi-section">')
    parts.append("<h2>Per-board KPIs</h2>")
    parts.append(_kpi_table(boards))
    parts.append("</section>")

    # ─── Aggregate charts ────────────────────────────────────────────
    parts.append('<section class="agg-charts">')
    parts.append("<h2>Aggregate view</h2>")
    parts.append('<div class="chart-grid">')
    parts.append(_overlap_bar_chart(boards))
    parts.append(_timing_chart(boards))
    parts.append(_component_type_chart(boards))
    parts.append(_cap_ic_distance_chart(boards))
    parts.append("</div>")
    parts.append("</section>")

    # ─── Per-board sections ──────────────────────────────────────────
    parts.append('<section class="boards">')
    parts.append("<h2>Per-board detail</h2>")
    for board in boards:
        parts.append(_board_section(board))
    parts.append("</section>")

    # ─── Footer ──────────────────────────────────────────────────────
    parts.append("<footer>")
    cmd = results.get("repro_cmd", "python tests/run_all.py")
    parts.append(
        f'<p>Reproduce: <code>{html.escape(cmd)}</code></p>'
    )
    parts.append(
        f'<p>Raw metrics: <code>tests/output/{results.get("run_id", "run")}/results.json</code></p>'
    )
    parts.append("</footer>")

    parts.append("</body></html>")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


# ─── Section builders ────────────────────────────────────────────────

def _summary_card(label: str, value: str, color: str) -> str:
    return (
        f'<div class="summary-card" style="border-left-color: {color}">'
        f'<div class="value" style="color: {color}">{html.escape(value)}</div>'
        f'<div class="label">{html.escape(label)}</div>'
        f"</div>"
    )


def _phase_card(name: str, data: dict) -> str:
    status = data.get("status", "unknown")
    status_class = {
        "pass": "ok",
        "fail": "bad",
        "skipped": "skip",
        "unknown": "skip",
    }.get(status, "skip")
    parts = [
        f'<div class="phase-card {status_class}">',
        f'<h3>{html.escape(name.replace("_", " ").title())}</h3>',
        f'<div class="phase-status">{html.escape(status.upper())}</div>',
    ]
    if "n_total" in data:
        parts.append(
            f'<div class="phase-detail">{data.get("n_passed", 0)}/{data["n_total"]} '
            f'passed</div>'
        )
    if "duration_s" in data:
        parts.append(f'<div class="phase-detail">{data["duration_s"]:.1f}s</div>')
    if "summary" in data:
        parts.append(f'<div class="phase-detail">{html.escape(str(data["summary"]))}</div>')
    parts.append("</div>")
    return "".join(parts)


def _kpi_table(boards: list[dict]) -> str:
    """Sortable HTML table of per-board KPIs."""
    headers = [
        ("Board", "name"),
        ("Source", "source"),
        ("Comps", "n_components"),
        ("ICs", "n_ics"),
        ("Caps", "n_caps"),
        ("Overlaps", "overlaps"),
        ("Edge.Cuts", "edge_cuts_present"),
        ("Density", "density"),
        ("HPWL", "hpwl"),
        ("RUDY peak", "rudy_peak"),
        ("SA (s)", "sa_time"),
        ("Legalize (s)", "legalize_time"),
        ("Total (s)", "total_time"),
    ]
    parts = [
        '<table class="kpi-table">',
        "<thead><tr>",
    ]
    for label, _ in headers:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for b in boards:
        row_class = "pass" if b.get("overlaps", 0) == 0 else "fail"
        parts.append(f'<tr class="{row_class}">')
        for _, key in headers:
            val = b.get(key, "")
            if key == "edge_cuts_present":
                cell = "✓" if val else "✗"
            elif key == "density":
                cell = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
            elif key in ("hpwl", "rudy_peak"):
                cell = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
            elif key in ("sa_time", "legalize_time", "total_time"):
                v = b.get("timing", {}).get(key.replace("_time", ""), val)
                cell = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
            else:
                cell = str(val)
            parts.append(f"<td>{html.escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _overlap_bar_chart(boards: list[dict]) -> str:
    """Pure HTML/CSS bar chart of overlap count per board."""
    max_overlaps = max((b.get("overlaps", 0) for b in boards), default=1) or 1
    parts = [
        '<div class="chart-card">',
        "<h3>Overlaps per board</h3>",
        '<div class="bar-chart">',
    ]
    for b in boards:
        n = b.get("overlaps", 0)
        height_pct = (n / max_overlaps) * 100 if max_overlaps > 0 else 0
        color = "#dc3545" if n > 0 else "#28a745"
        parts.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{html.escape(b["name"])}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width: {height_pct:.1f}%; background: {color};" '
            f'title="{n} overlaps"></div>'
            f'<span class="bar-value">{n}</span>'
            f"</div></div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def _timing_chart(boards: list[dict]) -> str:
    """Stacked bar chart of timing breakdown per board."""
    parts = [
        '<div class="chart-card">',
        "<h3>Timing breakdown (parse + SA + legalize)</h3>",
        '<div class="bar-chart">',
    ]
    max_total = max(
        (b.get("timing", {}).get("total", 0.0) for b in boards),
        default=1.0,
    ) or 1.0
    for b in boards:
        t = b.get("timing", {})
        parse_t = t.get("parse", 0.0)
        sa_t = t.get("sa", 0.0)
        legalize_t = t.get("legalize", 0.0)
        total = max(parse_t + sa_t + legalize_t, 0.001)
        # Width as percentage of max_total
        parse_pct = (parse_t / max_total) * 100
        sa_pct = (sa_t / max_total) * 100
        legal_pct = (legalize_t / max_total) * 100
        parts.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{html.escape(b["name"])}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width: {parse_pct:.1f}%; background: #17becf;" '
            f'title="parse: {parse_t:.2f}s"></div>'
            f'<div class="bar-fill" style="width: {sa_pct:.1f}%; background: #ff7f0e;" '
            f'title="SA: {sa_t:.2f}s"></div>'
            f'<div class="bar-fill" style="width: {legal_pct:.1f}%; background: #9467bd;" '
            f'title="legalize: {legalize_t:.2f}s"></div>'
            f'<span class="bar-value">{total:.1f}s</span>'
            f"</div></div>"
        )
    parts.append(
        '<div class="chart-legend">'
        '<span><span class="dot" style="background: #17becf;"></span>Parse</span>'
        '<span><span class="dot" style="background: #ff7f0e;"></span>SA</span>'
        '<span><span class="dot" style="background: #9467bd;"></span>Legalize</span>'
        "</div>"
    )
    parts.append("</div></div>")
    return "".join(parts)


def _component_type_chart(boards: list[dict]) -> str:
    """Stacked bar chart of component-type distribution per board."""
    # Aggregate all types across boards
    all_types = set()
    for b in boards:
        all_types.update(b.get("component_types", {}).keys())
    # Stable sort: IC-like first, then common types, then alphabetically
    type_order = ["ic", "mcu", "regulator", "capacitor", "resistor",
                  "connector", "crystal", "mounting_hole", "generic"]
    types = [t for t in type_order if t in all_types]
    types += sorted(all_types - set(type_order))

    parts = [
        '<div class="chart-card">',
        "<h3>Component-type distribution</h3>",
        '<div class="bar-chart">',
    ]
    max_total = max(
        (sum(b.get("component_types", {}).values()) for b in boards),
        default=1,
    ) or 1
    for b in boards:
        ct = b.get("component_types", {})
        total = sum(ct.values())
        parts.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{html.escape(b["name"])} <span class="muted">({total})</span></div>'
            f'<div class="bar-track">'
        )
        for t in types:
            n = ct.get(t, 0)
            if n == 0:
                continue
            pct = (n / max_total) * 100
            color = TYPE_COLORS.get(t, "#999999")
            parts.append(
                f'<div class="bar-fill" style="width: {pct:.1f}%; background: {color};" '
                f'title="{t}: {n}"></div>'
            )
        parts.append(f'<span class="bar-value">{total}</span></div></div>')
    parts.append('<div class="chart-legend">')
    for t in types:
        color = TYPE_COLORS.get(t, "#999999")
        parts.append(
            f'<span><span class="dot" style="background: {color};"></span>{html.escape(t)}</span>'
        )
    parts.append("</div></div></div>")
    return "".join(parts)


def _cap_ic_distance_chart(boards: list[dict]) -> str:
    """Histogram of decoupling cap → IC distances across all boards."""
    # Collect all distances
    all_dists: list[float] = []
    for b in boards:
        all_dists.extend(b.get("cap_ic_distances", []))
    if not all_dists:
        return (
            '<div class="chart-card"><h3>Cap → IC distances</h3>'
            "<p class='muted'>No decoupling caps assigned.</p></div>"
        )
    # Histogram buckets: 0-2, 2-4, 4-6, 6-8, 8-12, 12+
    buckets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, float("inf"))]
    counts = [0] * len(buckets)
    for d in all_dists:
        for i, (lo, hi) in enumerate(buckets):
            if lo <= d < hi:
                counts[i] += 1
                break
    max_count = max(counts) or 1
    parts = [
        '<div class="chart-card">',
        "<h3>Decoupling cap → IC distance</h3>",
        f'<p class="muted">{len(all_dists)} cap-IC pairs across {len(boards)} boards. '
        f"Target: &lt; 8mm (decoupling effectiveness).</p>",
        '<div class="histogram">',
    ]
    for (lo, hi), n in zip(buckets, counts):
        height_pct = (n / max_count) * 100
        if hi == float("inf"):
            label = f"{lo}+"
        else:
            label = f"{lo}-{hi}"
        # Color: green if <8mm, yellow if 8-12, red if >12
        if lo < 8:
            color = "#28a745"
        elif lo < 12:
            color = "#ffc107"
        else:
            color = "#dc3545"
        parts.append(
            f'<div class="hist-bar">'
            f'<div class="hist-fill" style="height: {height_pct:.1f}%; background: {color};" '
            f'title="{label}mm: {n} caps"></div>'
            f'<div class="hist-label">{label}</div>'
            f'<div class="hist-count">{n}</div>'
            f"</div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def _board_section(board: dict) -> str:
    """Per-board detail section: SVG + metric chips + type breakdown."""
    name = board.get("name", "?")
    overlaps = board.get("overlaps", 0)
    status_class = "pass" if overlaps == 0 else "fail"
    parts = [
        f'<div class="board-card {status_class}" id="board-{html.escape(name)}">',
        f'<h3>{html.escape(name)} '
        f'<span class="source-tag">{html.escape(board.get("source", "?"))}</span>'
        f'<span class="status-tag {status_class}">'
        f'{"✓ 0 overlaps" if overlaps == 0 else f"⚠ {overlaps} overlaps"}'
        f"</span></h3>",
    ]

    # Inline SVG visualization
    model = board.get("model")
    if model is not None:
        try:
            svg = render_board_svg(model, target_width=800, title=name)
            parts.append(f'<div class="board-svg">{svg}</div>')
        except Exception as e:
            parts.append(f'<p class="muted">SVG render failed: {html.escape(str(e))}</p>')
    else:
        parts.append('<p class="muted">No model available for visualization.</p>')

    # Metric chips
    parts.append('<div class="metric-chips">')
    chips = [
        ("Components", str(board.get("n_components", "?"))),
        ("Density", f"{board.get('density', 0):.2f}" if isinstance(board.get("density"), (int, float)) else "?"),
        ("HPWL", f"{board.get('hpwl', 0):.1f}" if isinstance(board.get("hpwl"), (int, float)) else "?"),
        ("RUDY peak", f"{board.get('rudy_peak', 0):.3f}" if isinstance(board.get("rudy_peak"), (int, float)) else "?"),
        ("Edge.Cuts", "✓" if board.get("edge_cuts_present") else "✗"),
        ("Total time", f"{board.get('timing', {}).get('total', 0):.2f}s"),
    ]
    for label, value in chips:
        parts.append(
            f'<div class="chip"><span class="chip-label">{html.escape(label)}</span>'
            f'<span class="chip-value">{html.escape(str(value))}</span></div>'
        )
    parts.append("</div>")

    # Overlap pairs list (if any)
    overlap_pairs = board.get("overlap_pairs", [])
    if overlap_pairs:
        parts.append('<details class="overlap-details">')
        parts.append(f"<summary>Overlap pairs ({len(overlap_pairs)})</summary>")
        parts.append("<ul>")
        for pair in overlap_pairs[:20]:
            ref_a = pair.get("a", "?")
            ref_b = pair.get("b", "?")
            area = pair.get("area_mm2", 0.0)
            parts.append(
                f"<li><code>{html.escape(ref_a)}</code> &harr; "
                f"<code>{html.escape(ref_b)}</code> "
                f"<span class='muted'>({area:.3f} mm²)</span></li>"
            )
        if len(overlap_pairs) > 20:
            parts.append(f"<li><em>... and {len(overlap_pairs) - 20} more</em></li>")
        parts.append("</ul></details>")

    # Cap-IC distances (mini stats)
    cap_dists = board.get("cap_ic_distances", [])
    if cap_dists:
        n_caps = len(cap_dists)
        n_within_8 = sum(1 for d in cap_dists if d <= 8.0)
        worst = max(cap_dists)
        parts.append(
            f'<div class="cap-ic-stats">'
            f"<strong>Cap → IC distance:</strong> "
            f"{n_within_8}/{n_caps} within 8mm, "
            f"worst = {worst:.2f}mm"
            "</div>"
        )

    parts.append("</div>")
    return "".join(parts)


# ─── CSS ─────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  padding: 0;
  background: #f5f5f5;
  color: #222;
  line-height: 1.5;
}
.hero {
  background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
  color: white;
  padding: 24px 32px;
}
.hero h1 { margin: 0 0 8px 0; font-size: 24px; font-weight: 600; }
.hero .meta { font-size: 13px; opacity: 0.9; }
.hero .meta code { background: rgba(255,255,255,0.15); padding: 1px 6px; border-radius: 3px; }
.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.summary-card {
  background: rgba(255,255,255,0.1);
  border-left: 4px solid;
  padding: 12px 16px;
  border-radius: 4px;
}
.summary-card .value { font-size: 24px; font-weight: 700; }
.summary-card .label { font-size: 12px; opacity: 0.85; margin-top: 2px; }

section {
  background: white;
  margin: 24px 32px;
  padding: 24px;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
section h2 {
  margin: 0 0 16px 0;
  font-size: 18px;
  font-weight: 600;
  border-bottom: 2px solid #e5e7eb;
  padding-bottom: 8px;
}

.phases .phase-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}
.phase-card {
  padding: 12px;
  border-radius: 6px;
  border-left: 4px solid;
  background: #f9fafb;
}
.phase-card.ok { border-color: #28a745; }
.phase-card.bad { border-color: #dc3545; }
.phase-card.skip { border-color: #9ca3af; }
.phase-card h3 { margin: 0 0 4px 0; font-size: 14px; font-weight: 600; }
.phase-status { font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
.phase-card.ok .phase-status { color: #28a745; }
.phase-card.bad .phase-status { color: #dc3545; }
.phase-card.skip .phase-status { color: #6b7280; }
.phase-detail { font-size: 12px; color: #4b5563; margin-top: 2px; }

.kpi-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
.kpi-table th {
  background: #f3f4f6;
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #d1d5db;
}
.kpi-table td {
  padding: 6px 10px;
  border-bottom: 1px solid #e5e7eb;
}
.kpi-table tr.pass { background: #f0fdf4; }
.kpi-table tr.fail { background: #fef2f2; }
.kpi-table tr:hover { background: #eff6ff; }

.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 16px;
}
.chart-card {
  padding: 16px;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: #fafafa;
}
.chart-card h3 {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
}
.bar-chart { display: flex; flex-direction: column; gap: 4px; }
.bar-row {
  display: grid;
  grid-template-columns: 120px 1fr;
  align-items: center;
  gap: 8px;
}
.bar-label { font-size: 12px; font-family: monospace; text-align: right; }
.bar-track {
  position: relative;
  height: 18px;
  background: #e5e7eb;
  border-radius: 3px;
  display: flex;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  transition: width 0.2s;
}
.bar-value {
  position: absolute;
  right: 6px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 11px;
  font-weight: 600;
  color: #222;
  font-family: monospace;
}
.chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
  font-size: 11px;
}
.chart-legend span { display: inline-flex; align-items: center; gap: 4px; }
.chart-legend .dot {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}

.histogram {
  display: flex;
  align-items: flex-end;
  gap: 6px;
  height: 140px;
  padding: 8px 0;
}
.hist-bar {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  justify-content: flex-end;
}
.hist-fill {
  width: 100%;
  min-height: 2px;
  border-radius: 2px 2px 0 0;
  transition: height 0.2s;
}
.hist-label { font-size: 10px; margin-top: 4px; font-family: monospace; color: #4b5563; }
.hist-count { font-size: 11px; font-weight: 600; font-family: monospace; }

.boards { }
.board-card {
  border: 1px solid #e5e7eb;
  border-left: 4px solid;
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 16px;
  background: white;
}
.board-card.pass { border-left-color: #28a745; }
.board-card.fail { border-left-color: #dc3545; }
.board-card h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.source-tag {
  font-size: 11px;
  background: #e5e7eb;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: normal;
}
.status-tag {
  font-size: 12px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 3px;
  margin-left: auto;
}
.status-tag.pass { background: #d1fae5; color: #065f46; }
.status-tag.fail { background: #fee2e2; color: #991b1b; }
.board-svg {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  padding: 8px;
  margin-bottom: 12px;
  overflow-x: auto;
}
.board-svg svg { max-width: 100%; height: auto; }
.metric-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}
.chip {
  background: #f3f4f6;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.chip-label { color: #6b7280; font-size: 11px; }
.chip-value { font-weight: 600; font-family: monospace; }
.overlap-details {
  margin-bottom: 12px;
  font-size: 13px;
}
.overlap-details summary {
  cursor: pointer;
  font-weight: 600;
  color: #dc3545;
}
.overlap-details ul {
  margin: 8px 0 0 0;
  padding-left: 20px;
  font-family: monospace;
  font-size: 12px;
}
.cap-ic-stats {
  font-size: 13px;
  color: #4b5563;
  background: #f9fafb;
  padding: 8px 12px;
  border-radius: 4px;
}
.muted { color: #6b7280; }

footer {
  background: #1f2937;
  color: #d1d5db;
  padding: 16px 32px;
  font-size: 12px;
}
footer code {
  background: rgba(255,255,255,0.1);
  padding: 1px 6px;
  border-radius: 3px;
  color: white;
}
footer p { margin: 4px 0; }

@media (max-width: 768px) {
  section { margin: 16px; padding: 16px; }
  .hero { padding: 16px; }
  .bar-row { grid-template-columns: 80px 1fr; }
}
"""
