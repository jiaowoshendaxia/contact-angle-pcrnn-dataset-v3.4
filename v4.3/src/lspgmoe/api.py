"""Validated public prediction interface for research and reproducibility scripts."""

from __future__ import annotations

from typing import Literal, Protocol

from .schema import (
    LiquidDescriptor,
    PredictionMode,
    PredictionResult,
    ProbeMeasurement,
    SurfaceDescriptor,
)


class DescriptorBackend(Protocol):
    def predict_descriptors(
        self,
        surface: SurfaceDescriptor,
        target_liquid: LiquidDescriptor,
        probes: list[ProbeMeasurement],
        mode: PredictionMode,
    ) -> PredictionResult: ...


def _validate_inputs(
    target_liquid: LiquidDescriptor,
    probes: list[ProbeMeasurement] | None,
    mode: PredictionMode,
) -> list[ProbeMeasurement]:
    if mode not in ("zero_shot", "probe_assisted"):
        raise ValueError("mode must be 'zero_shot' or 'probe_assisted'")
    normalized = list(probes or [])
    if mode == "zero_shot" and normalized:
        raise ValueError("zero_shot mode cannot receive contact-angle probes")
    if mode == "probe_assisted" and not normalized:
        raise ValueError("probe_assisted mode requires at least one probe")
    target_id = str(target_liquid.liquid_id)
    leaked = [probe.measurement_id for probe in normalized if str(probe.liquid.liquid_id) == target_id]
    if leaked:
        raise ValueError(f"Target liquid is present in probes: {', '.join(leaked)}")
    for probe in normalized:
        if not 0.0 <= float(probe.contact_angle_deg) <= 180.0:
            raise ValueError(f"Probe angle out of range: {probe.measurement_id}")
    return normalized


def predict(
    surface: SurfaceDescriptor,
    target_liquid: LiquidDescriptor,
    probes: list[ProbeMeasurement] | None,
    mode: Literal["zero_shot", "probe_assisted"],
    backend: DescriptorBackend | None = None,
) -> PredictionResult:
    """Predict a target angle after enforcing target masking at the API boundary.

    A fitted descriptor backend is deliberately injected so research scripts can use the
    same guardrails without coupling this public interface to a particular checkpoint.
    """
    normalized = _validate_inputs(target_liquid, probes, mode)
    if backend is None:
        raise RuntimeError("A fitted LS-PGMoE descriptor backend is required for prediction")
    return backend.predict_descriptors(surface, target_liquid, normalized, mode)
