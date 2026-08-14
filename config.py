"""Configuration loader for KiCad Smart Auto-Placer.

Loads tunable parameters from config.json with built-in defaults.
CLI arguments override config file values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LegalizationConfig:
    grid_mm: float = 0.50        # 0.5mm grid — fine enough for most components
    max_iterations: int = 400
    push_strength: float = 1.2   # v9: slightly stronger pushes for faster convergence
    stall_threshold: int = 10
    adaptive_strength_max: float = 1.8
    adaptive_strength_multiplier: float = 1.3
    severity_threshold: float = 0.7
    severe_push_factor: float = 0.5
    # --- Adaptive push-apart (plan.md §2-3) ---
    max_bbox_expansions: int = 3          # cap on bbox-growth rounds
    push_apart_hard_cap: int = 1000       # absolute ceiling for _resolve_overlaps
    spread_pass_enabled: bool = True      # toggle anti-centroid spread pass
    bbox_expansion_factor: float = 0.05   # +5% per round toward board outline
    bbox_expansion_density_threshold: float = 0.80  # only expand if used/board < 0.80
    gradient_plateau_threshold: float = -0.5       # stop when gradient >= this (plateau)
    gradient_history_window: int = 20    # rolling history size
    gradient_split: int = 10             # split history into last-N vs prev-N
    density_push_min: float = 0.5        # push_strength scale for low-overlap zones
    density_push_max: float = 1.5        # push_strength scale for high-overlap zones


@dataclass
class AnnealerConfig:
    max_iterations: int = 300          # v11: density-adaptive SA adjusts at runtime
    reheat_count: int = 3              # v11: overridden by density-adaptive logic (1-3)
    reheat_ratio: float = 0.40
    calibration_samples: int = 500     # v11: more samples for robust T0
    initial_accept_rate: float = 0.92  # v11: target accept at start
    penalty_scale_min: float = 0.50    # v11: overridden by density-adaptive logic (0.50-0.95)
    min_temperature: float = 1e-8
    freeze_threshold: float = 0.005
    greedy_nudge_distances: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    greedy_rotations: tuple[float, ...] = (90.0, 180.0, 270.0)
    greedy_improve_threshold: float = 0.5  # v11: accept smaller improvements
    overlap_cap_factor: float = 2.0    # v11: reject moves exceeding this * initial overlaps
    rudy_weight: float = 1.0           # RUDY wire-density congestion penalty weight (0 = disabled).
                                       # 1.0 — bumped from 0.3 after test6 ablation showed 0.3 was
                                       # too weak to overcome HPWL on dense boards. At 1.0, test6
                                       # RUDY peak drops 0.43 -> 0.28 (-35%) and HPWL also drops
                                       # 1686 -> 1598 (-5%); multi-seed confirms 0 residual
                                       # overlaps and no regression on the other 5 boards. The
                                       # sweet spot is 0.5-1.0; above 1.5 RUDY overwhelms HPWL
                                       # and SA thrashes.
    rudy_grid_resolution: float = 2.0  # RUDY grid cell size in mm
    pin_density_weight: float = 0.2    # Pin-density congestion penalty weight (0 = disabled).
                                       # Complementary to rudy_weight: catches pin-escape
                                       # congestion (dense pin clusters) that RUDY misses.
                                       # Default-on so the placer produces a routable
                                       # result out of the box.
    sa_auto_disable_min_components: int = 6   # auto-disable SA on tiny boards
    sa_auto_disable_max_components: int = 50  # auto-disable SA on large boards
    spread_floor_fraction: float = 0.10       # reject moves that collapse spread


@dataclass
class CostConfig:
    overlap_weight: float = 10.0      # moderate — soft penalty; SA explores, legalizer resolves
    boundary_weight: float = 2.0       # low — HPWL dominates; legalizer handles OOB
    constraint_weight: float = 4.0     # delta — overridden by profile.delta at runtime
    exclude_nets: list[str] = field(default_factory=list)  # net names to drop from HPWL (e.g. global GND)


@dataclass
class PlacementConfig:
    margin: float = 5.0
    spacing_factor: float = 1.3
    inner_margin_extra: float = 2.0
    wiggle_factor: float = 0.12
    perimeter_margin_extra: float = 2.0
    repulsion_iterations: int = 150
    repulsion_min_distances: dict[str, float] = field(default_factory=lambda: {
        "resistor": 10.0, "capacitor": 8.0, "ic": 12.0,
    })
    repulsion_push_factor: float = 2.0
    repulsion_push_constant: float = 1.0
    force_iterations: int = 100
    force_k_attract: float = 0.01
    force_k_repel: float = 500.0
    force_min_spacing: float = 2.0
    force_dt: float = 0.5
    force_dt_cooling: float = 0.8
    shelf_packing_spacing: float = 0.5
    # Extra edge keepout (mm) applied to specific component types so that
    # ICs / MCUs / regulators stay further from the board edge than
    # passives — a common-sense DFM rule a human PCB designer always
    # applies (routing room, panelization clearance, assembly clearance).
    # The base `margin` is applied to ALL components; this extra is added
    # on top for the listed types. Types not in the map get 0 extra.
    edge_keepout_extra: dict[str, float] = field(default_factory=lambda: {
        "ic": 5.0, "mcu": 5.0, "regulator": 5.0, "crystal": 3.0,
    })
    # Finding 6 fix: single shared target pack density. Used by:
    #   - kicad_parser._infer_board_from_components (outline inference)
    #   - place/legalizer.expand_bounds_to_fit (bounds expansion)
    #   - place/pipeline._keepout_cb (IC edge keepout density scaling)
    # Default 0.55 = safe 2D packing target for irregular rectangles.
    target_pack_density: float = 0.55


@dataclass
class ParserConfig:
    bbox_margin: float = 0.5
    default_pad_size: float = 0.5
    default_bbox_size: float = 2.0
    min_bbox_size: float = 0.5
    board_margin: float = 0.5
    default_board_size: float = 100.0


@dataclass
class ClusteringConfig:
    cluster_min: int = 2
    cluster_max: int = 10
    margin: float = 5.0
    default_spacing_w: float = 3.0
    default_spacing_h: float = 3.0
    spacing_multiplier: float = 1.2
    min_spacing_x: float = 2.0
    min_spacing_y: float = 2.0


@dataclass
class Config:
    legalization: LegalizationConfig = field(default_factory=LegalizationConfig)
    annealer: AnnealerConfig = field(default_factory=AnnealerConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _convert_lists(data: dict) -> dict:
    """Convert list values to tuples where the dataclass expects tuples."""
    if "annealer" in data:
        a = data["annealer"]
        if "greedy_nudge_distances" in a and isinstance(a["greedy_nudge_distances"], list):
            a["greedy_nudge_distances"] = tuple(a["greedy_nudge_distances"])
        if "greedy_rotations" in a and isinstance(a["greedy_rotations"], list):
            a["greedy_rotations"] = tuple(a["greedy_rotations"])
    return data


def _dataclass_from_dict(cls, data: dict):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(config_path: str | None = None) -> Config:
    """Load configuration from JSON file.

    Searches for config.json in:
    1. Explicit path (if config_path provided)
    2. Same directory as the script (gridghost.py)

    Falls back to built-in defaults if no file found.
    """
    search_paths = []

    if config_path:
        search_paths.append(config_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(script_dir, "config.json"))

    data: dict[str, Any] = {}

    for path in search_paths:
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
            break

    data = _convert_lists(data)

    cfg = Config()

    if "legalization" in data:
        cfg.legalization = _dataclass_from_dict(LegalizationConfig, data["legalization"])
    if "annealer" in data:
        cfg.annealer = _dataclass_from_dict(AnnealerConfig, data["annealer"])
    if "cost" in data:
        cfg.cost = _dataclass_from_dict(CostConfig, data["cost"])
    if "placement" in data:
        cfg.placement = _dataclass_from_dict(PlacementConfig, data["placement"])
    if "parser" in data:
        cfg.parser = _dataclass_from_dict(ParserConfig, data["parser"])
    if "clustering" in data:
        cfg.clustering = _dataclass_from_dict(ClusteringConfig, data["clustering"])

    return cfg
