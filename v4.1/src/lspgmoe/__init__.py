"""LS-PGMoE v4.0 research package."""

from .physics import OWRKFitResult, fit_owrk_nnls, owens_wendt_angle
from .schema import LiquidDescriptor, PredictionResult, ProbeMeasurement, SurfaceDescriptor
from .api import predict

__all__ = [
    "LiquidDescriptor",
    "OWRKFitResult",
    "PredictionResult",
    "ProbeMeasurement",
    "SurfaceDescriptor",
    "predict",
    "fit_owrk_nnls",
    "owens_wendt_angle",
]

__version__ = "4.0.0.dev0"
