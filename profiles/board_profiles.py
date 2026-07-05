"""Board profile system — pre-configured cost weights and constraint rules.

Board profiles configure the optimizer based on the type of PCB being laid out.
Each profile sets:
- Cost function weights (alpha, beta, gamma, delta)
- Active constraint rules with per-rule weights
- Component-specific heuristics (e.g., decoupling caps near ICs)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


def _require_yaml():
    """Lazy-import PyYAML, raising a helpful error if it's not installed.

    PyYAML is an optional dependency — it's only needed when loading or
    saving custom board profiles from YAML files.  The default pipeline
    uses the built-in dict profiles, so most users never need it.
    """
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
        raise ImportError(
            "PyYAML is required for YAML board profile loading/saving. "
            "Install it with: pip install pyyaml"
        ) from exc


@dataclass
class ConstraintRule:
    """A single constraint rule with its weight and parameters."""
    name: str
    weight: float = 1.0
    enabled: bool = True
    params: dict = field(default_factory=dict)

    # Rule descriptions for user-facing display
    DESCRIPTIONS = {
        "decoupling_proximity": "Decoupling capacitors must be close to IC power pins",
        "crystal_mcu": "Crystals must be near their MCU",
        "connector_edge": "Connectors should be near board edges",
        "thermal_grouping": "Group thermally-related components together",
        "thermal_separation": "Keep hot parts on different nets apart (DFM — prevents thermal hotspots)",
        "high_current_path": "Minimize path length for high-current nets",
        "bulk_cap_input": "Bulk capacitors near power input",
        "antenna_keepout": "Keep components away from antenna area",
        "analog_digital_separation": "Separate analog and digital domains",
        "matched_length": "Matched trace length requirements",
        "ground_plane_clearance": "Keep components clear of ground plane splits",
        "routing_congestion": "RUDY routing congestion penalty — penalises choke points HPWL misses",
    }

    @property
    def description(self) -> str:
        return self.DESCRIPTIONS.get(self.name, f"Constraint: {self.name}")


@dataclass
class BoardProfile:
    """A complete board profile with cost weights and constraint rules."""
    name: str
    display_name: str
    description: str
    alpha: float = 1.0   # HPWL weight
    beta: float = 5.0    # Overlap penalty
    gamma: float = 3.0   # Boundary penalty
    delta: float = 4.0   # Constraint penalty base
    rules: list[ConstraintRule] = field(default_factory=list)

    def get_rule(self, name: str) -> Optional[ConstraintRule]:
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def active_rules(self) -> list[ConstraintRule]:
        return [r for r in self.rules if r.enabled]


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

# Small board profile - for dense boards where spacing is critical
_SMALL_BOARD = BoardProfile(
    name="small_board",
    display_name="Small/Dense Board",
    description="For small boards where spacing is critical. "
                "Higher overlap penalty and lower HPWL weight to prevent congestion.",
    alpha=0.5,    # Lower HPWL weight
    beta=15.0,     # High overlap penalty to prevent congestion
    gamma=3.0,      # Standard boundary penalty
    delta=2.0,       # No constraint rules
    rules=[],
)

BUILTIN_PROFILES: dict[str, BoardProfile] = {
    "mcu_peripheral": BoardProfile(
        name="mcu_peripheral",
        display_name="MCU / Peripheral Board",
        description="Typical MCU board with decoupling caps, crystal, and peripheral ICs. "
                    "Emphasizes connectivity (HPWL) and decoupling cap proximity.",
        alpha=1.0,
        beta=5.0,
        gamma=3.0,
        delta=4.0,
        rules=[
            ConstraintRule("decoupling_proximity", weight=4.0, params={"max_distance_mm": 5.0}),
            ConstraintRule("crystal_mcu", weight=3.0, params={"max_distance_mm": 10.0}),
            ConstraintRule("connector_edge", weight=2.0, params={"max_edge_distance_mm": 15.0}),
        ],
    ),
    "power_supply": BoardProfile(
        name="power_supply",
        display_name="Power Supply Board",
        description="Power supply with thermal management needs. Emphasizes thermal grouping "
                    "and high-current path optimization.",
        alpha=0.5,
        beta=5.0,
        gamma=2.0,
        delta=6.0,
        rules=[
            ConstraintRule("thermal_grouping", weight=5.0),
            # Phase 2.4: Complementary thermal separation — keeps hot
            # parts on DIFFERENT nets ≥5mm apart so two independent
            # regulators don't stack and create a thermal hotspot.
            # See AUDIT_FINAL_REPORT.md §2.4.
            ConstraintRule("thermal_separation", weight=4.0,
                           params={"min_distance_mm": 5.0}),
            ConstraintRule("high_current_path", weight=4.0),
            ConstraintRule("bulk_cap_input", weight=3.0),
        ],
    ),
    "rf_frontend": BoardProfile(
        name="rf_frontend",
        display_name="RF Frontend Board",
        description="RF circuit requiring symmetry, isolation, and matched lengths. "
                    "High constraint weight for analog/digital separation.",
        alpha=0.8,
        beta=5.0,
        gamma=4.0,
        delta=7.0,
        rules=[
            ConstraintRule("antenna_keepout", weight=6.0, params={"clearance_mm": 10.0}),
            ConstraintRule("analog_digital_separation", weight=5.0),
            ConstraintRule("matched_length", weight=3.0),
            # Phase 2.1: RUDY routing congestion — RF frontends have
            # routing choke points between LNA/mixer/filter stages that
            # HPWL alone misses.  See engine/congestion.py for the RUDY
            # derivation (Spindler & Johannes, DATE 2007).
            ConstraintRule("routing_congestion", weight=2.0,
                           params={"grid_resolution_mm": 2.0}),
        ],
    ),
    "mixed_signal": BoardProfile(
        name="mixed_signal",
        display_name="Mixed-Signal Board",
        description="Board with both analog and digital domains. Enforces partition "
                    "boundaries between ground domains.",
        alpha=0.7,
        beta=5.0,
        gamma=3.0,
        delta=6.0,
        rules=[
            ConstraintRule("analog_digital_separation", weight=7.0),
            ConstraintRule("decoupling_proximity", weight=3.0, params={"max_distance_mm": 5.0}),
            ConstraintRule("ground_plane_clearance", weight=4.0),
            # Phase 2.1: RUDY — mixed-signal partition boundaries
            # concentrate crossing nets (ADC clock, SPI, etc.), creating
            # choke points that HPWL misses.
            ConstraintRule("routing_congestion", weight=1.5,
                           params={"grid_resolution_mm": 2.0}),
        ],
    ),
    "generic": BoardProfile(
        name="generic",
        display_name="Generic Board",
        description="General-purpose placement with balanced weights. "
                    "No specific constraint rules — relies on HPWL minimization.",
        alpha=1.0,
        beta=5.0,
        gamma=3.0,
        delta=2.0,
        rules=[],
    ),
    "small_board": _SMALL_BOARD,
}


def get_profile(name: str) -> BoardProfile:
    """Get a board profile by name."""
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    raise ValueError(f"Unknown profile: {name}. Available: {list(BUILTIN_PROFILES.keys())}")


def list_profiles() -> list[BoardProfile]:
    """Return all available board profiles."""
    return list(BUILTIN_PROFILES.values())


def load_profile_from_yaml(path: str) -> BoardProfile:
    """Load a custom board profile from a YAML file."""
    yaml = _require_yaml()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules = [
        ConstraintRule(
            name=r.get("name", ""),
            weight=r.get("weight", 1.0),
            enabled=r.get("enabled", True),
            params=r.get("params", {}),
        )
        for r in data.get("rules", [])
    ]
    return BoardProfile(
        name=data.get("name", "custom"),
        display_name=data.get("display_name", "Custom Profile"),
        description=data.get("description", ""),
        alpha=data.get("alpha", 1.0),
        beta=data.get("beta", 5.0),
        gamma=data.get("gamma", 3.0),
        delta=data.get("delta", 4.0),
        rules=rules,
    )


def save_profile_to_yaml(profile: BoardProfile, path: str) -> None:
    """Save a board profile to a YAML file."""
    yaml = _require_yaml()
    data = {
        "name": profile.name,
        "display_name": profile.display_name,
        "description": profile.description,
        "alpha": profile.alpha,
        "beta": profile.beta,
        "gamma": profile.gamma,
        "delta": profile.delta,
        "rules": [
            {
                "name": r.name,
                "weight": r.weight,
                "enabled": r.enabled,
                "params": r.params,
            }
            for r in profile.rules
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
