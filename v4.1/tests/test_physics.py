from __future__ import annotations

import math

import pytest

from lspgmoe.physics import fit_owrk_nnls, legacy_unconstrained_coefficients, owens_wendt_angle
from lspgmoe.schema import LiquidDescriptor, ProbeMeasurement


WATER = LiquidDescriptor("water", "water", 72.8, 21.8, 51.0)
DIM = LiquidDescriptor("dim", "diiodomethane", 50.8, 48.5, 2.3)
FORMAMIDE = LiquidDescriptor("formamide", "formamide", 58.0, 39.0, 19.0)


def probe(liquid: LiquidDescriptor, angle: float) -> ProbeMeasurement:
    return ProbeMeasurement(f"m_{liquid.liquid_id}", liquid, angle)


def test_nnls_recovers_nonnegative_surface_energy() -> None:
    sd, sp = 25.0, 10.0
    probes = [probe(WATER, owens_wendt_angle(sd, sp, WATER)), probe(DIM, owens_wendt_angle(sd, sp, DIM))]
    result = fit_owrk_nnls(probes)
    assert result.status == "interior_fit"
    assert result.dispersion_mj_m2 == pytest.approx(sd, rel=1e-6)
    assert result.polar_mj_m2 == pytest.approx(sp, rel=1e-6)


def test_negative_unconstrained_coefficient_becomes_boundary_not_squared() -> None:
    probes = [probe(WATER, 150.0), probe(DIM, 30.0)]
    legacy = legacy_unconstrained_coefficients(probes)
    assert legacy is not None and min(legacy) < 0
    result = fit_owrk_nnls(probes)
    assert result.status == "boundary_fit"
    assert result.dispersion_mj_m2 is not None and result.dispersion_mj_m2 >= 0
    assert result.polar_mj_m2 is not None and result.polar_mj_m2 >= 0
    assert min(result.dispersion_mj_m2, result.polar_mj_m2) == pytest.approx(0.0, abs=1e-10)


def test_probe_order_does_not_change_fit() -> None:
    probes = [probe(WATER, 75.0), probe(DIM, 40.0), probe(FORMAMIDE, 55.0)]
    a = fit_owrk_nnls(probes)
    b = fit_owrk_nnls(list(reversed(probes)))
    assert a.dispersion_mj_m2 == pytest.approx(b.dispersion_mj_m2)
    assert a.polar_mj_m2 == pytest.approx(b.polar_mj_m2)
    assert math.isfinite(a.residual_norm or 0.0)


def test_single_probe_is_explicitly_insufficient() -> None:
    result = fit_owrk_nnls([probe(WATER, 80.0)])
    assert result.status == "insufficient_probes"
    assert result.total_mj_m2 is None
