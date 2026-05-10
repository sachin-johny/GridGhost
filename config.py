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
    grid_mm: float = 1.0
    max_iterations: int = 300
    push_strength: float = 1.0
    stall_threshold: int = 10
    adaptive_strength_max: float = 1.5
    adaptive_strength_multiplier: float = 1.2
    severity_threshold: float = 0.7
    severe_push_factor: float = 0.5


@dataclass
class AnnealerConfig:
    max_iterations: int = 200
    reheat_count: int = 2
    reheat_ratio: float = 0.35
    calibration_samples: int = 200
    initial_accept_rate: float = 0.90
    penalty_scale_min: float = 0.65  # overlap=6.5 at hot — limits overlaps while exploring
    min_temperature: float = 1e-6
    freeze_threshold: float = 0.01
    greedy_nudge_distances: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
    greedy_rotations: tuple[float, ...] = (90.0, 180.0, 270.0)
    greedy_improve_threshold: float = 1.0


@dataclass
class CostConfig:
    overlap_weight: float = 10.0      # moderate — soft penalty; SA explores, legalizer resolves
    boundary_weight: float = 2.0       # low — HPWL dominates; legalizer handles OOB
    constraint_weight: float = 4.0     # delta — overridden by profile.delta at runtime


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
