"""Leakage-safe surface-energy fitting and physical decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import nnls

from .schema import LiquidDescriptor, ProbeMeasurement


@dataclass(frozen=True)
class OWRKFitResult:
    status: str
    dispersion_mj_m2: float | None
    polar_mj_m2: float | None
    total_mj_m2: float | None
    residual_norm: float | None
    n_probes: int
    n_unique_liquids: int
    boundary_fit: bool


def _design_row(liquid: LiquidDescriptor) -> tuple[float, float]:
    if liquid.total_surface_tension <= 0:
        raise ValueError("Liquid total surface tension must be positive.")
    if liquid.dispersion_component < 0 or liquid.polar_component < 0:
        raise ValueError("Liquid surface-tension components must be non-negative.")
    return np.sqrt(liquid.dispersion_component), np.sqrt(liquid.polar_component)


def _response(probe: ProbeMeasurement) -> float:
    theta = np.deg2rad(float(probe.contact_angle_deg))
    return probe.liquid.total_surface_tension * (1.0 + np.cos(theta)) / 2.0


def fit_owrk_nnls(probes: Iterable[ProbeMeasurement], tolerance: float = 1e-10) -> OWRKFitResult:
    """Fit non-negative sqrt(SFE) terms from target-free probe measurements."""

    rows = list(probes)
    unique_liquids = {probe.liquid.liquid_id for probe in rows}
    if len(rows) < 2 or len(unique_liquids) < 2:
        return OWRKFitResult(
            status="insufficient_probes",
            dispersion_mj_m2=None,
            polar_mj_m2=None,
            total_mj_m2=None,
            residual_norm=None,
            n_probes=len(rows),
            n_unique_liquids=len(unique_liquids),
            boundary_fit=False,
        )

    matrix = np.asarray([_design_row(probe.liquid) for probe in rows], dtype=np.float64)
    target = np.asarray([_response(probe) for probe in rows], dtype=np.float64)
    if np.linalg.matrix_rank(matrix, tol=tolerance) < 2:
        return OWRKFitResult(
            status="singular_fit",
            dispersion_mj_m2=None,
            polar_mj_m2=None,
            total_mj_m2=None,
            residual_norm=None,
            n_probes=len(rows),
            n_unique_liquids=len(unique_liquids),
            boundary_fit=False,
        )

    coefficients, residual_norm = nnls(matrix, target)
    dispersion = float(coefficients[0] ** 2)
    polar = float(coefficients[1] ** 2)
    boundary = bool(np.any(coefficients <= tolerance))
    return OWRKFitResult(
        status="boundary_fit" if boundary else "interior_fit",
        dispersion_mj_m2=dispersion,
        polar_mj_m2=polar,
        total_mj_m2=dispersion + polar,
        residual_norm=float(residual_norm),
        n_probes=len(rows),
        n_unique_liquids=len(unique_liquids),
        boundary_fit=boundary,
    )


def owens_wendt_cos(
    dispersion_mj_m2: float,
    polar_mj_m2: float,
    liquid: LiquidDescriptor,
) -> float:
    if dispersion_mj_m2 < 0 or polar_mj_m2 < 0:
        raise ValueError("Solid SFE components must be non-negative.")
    ld, lp = _design_row(liquid)
    raw = 2.0 * (np.sqrt(dispersion_mj_m2) * ld + np.sqrt(polar_mj_m2) * lp)
    raw = raw / liquid.total_surface_tension - 1.0
    return float(np.clip(raw, -1.0, 1.0))


def owens_wendt_angle(
    dispersion_mj_m2: float,
    polar_mj_m2: float,
    liquid: LiquidDescriptor,
) -> float:
    return float(np.degrees(np.arccos(owens_wendt_cos(dispersion_mj_m2, polar_mj_m2, liquid))))


def legacy_unconstrained_coefficients(probes: Iterable[ProbeMeasurement]) -> tuple[float, float] | None:
    """Reproduce the legacy linear coefficients for audit only; never use for prediction."""

    rows = list(probes)
    if len(rows) < 2:
        return None
    matrix = np.asarray([_design_row(probe.liquid) for probe in rows], dtype=np.float64)
    if np.linalg.matrix_rank(matrix) < 2:
        return None
    target = np.asarray([_response(probe) for probe in rows], dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return float(coefficients[0]), float(coefficients[1])
