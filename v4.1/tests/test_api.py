import pytest

from lspgmoe.api import predict
from lspgmoe.schema import LiquidDescriptor, ProbeMeasurement, SurfaceDescriptor


def _surface() -> SurfaceDescriptor:
    return SurfaceDescriptor("S1", "SRC1", "PTFE", "polymer")


def _liquid(liquid_id: str) -> LiquidDescriptor:
    return LiquidDescriptor(liquid_id, liquid_id, 72.8, 21.8, 51.0)


class DummyBackend:
    def predict_descriptors(self, surface, target_liquid, probes, mode):
        from lspgmoe.schema import PredictionResult
        return PredictionResult(80, 70, 90, 0.9, "low", False, 80, 81, 79)


def test_api_rejects_target_liquid_leak():
    probe = ProbeMeasurement("P1", _liquid("target"), 60)
    with pytest.raises(ValueError, match="Target liquid"):
        predict(_surface(), _liquid("target"), [probe], "probe_assisted", DummyBackend())


def test_api_zero_shot_has_no_probes():
    with pytest.raises(ValueError, match="zero_shot"):
        predict(_surface(), _liquid("target"), [ProbeMeasurement("P1", _liquid("probe"), 60)], "zero_shot", DummyBackend())


def test_api_delegates_valid_probe_assisted_call():
    result = predict(
        _surface(), _liquid("target"), [ProbeMeasurement("P1", _liquid("probe"), 60)],
        "probe_assisted", DummyBackend()
    )
    assert result.theta_pred_deg == 80
