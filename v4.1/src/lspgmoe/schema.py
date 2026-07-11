"""Public input and output types for the v4 prediction system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PredictionMode = Literal["zero_shot", "probe_assisted"]


@dataclass(frozen=True)
class SurfaceDescriptor:
    surface_group_id: str
    source_group_id: str
    solid_name: str
    solid_family: str
    solid_substrate: str = ""
    surface_treatment: str = ""
    surface_treatment_detail: str = ""
    coating_or_layer: str = ""
    surface_state: str = ""
    roughness_ra_nm: float | None = None
    roughness_rq_nm: float | None = None
    roughness_r_factor: float | None = None
    independent_sfe_dispersion: float | None = None
    independent_sfe_polar: float | None = None


@dataclass(frozen=True)
class LiquidDescriptor:
    liquid_id: str
    name: str
    total_surface_tension: float
    dispersion_component: float
    polar_component: float
    viscosity_mpa_s: float | None = None
    dipole_moment_d: float | None = None
    dielectric_constant: float | None = None


@dataclass(frozen=True)
class ProbeMeasurement:
    measurement_id: str
    liquid: LiquidDescriptor
    contact_angle_deg: float
    contact_angle_std_deg: float | None = None
    replicates_n: int | None = None


@dataclass(frozen=True)
class PredictionResult:
    theta_pred_deg: float
    interval_lower_deg: float
    interval_upper_deg: float
    confidence_level: float
    risk_level: Literal["low", "medium", "high"]
    abstain_flag: bool
    physics_prediction: float
    neural_prediction: float
    tree_prediction: float
    expert_weights: dict[str, float] = field(default_factory=dict)
    ood_score: float = 0.0
    model_version: str = "lspgmoe-v4.0"
