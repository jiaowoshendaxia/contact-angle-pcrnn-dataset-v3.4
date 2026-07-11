"""Physics-guided neural and gating modules for LS-PGMoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def mlp(dimensions: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            layers.append(nn.SiLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class CategoricalContextEncoder(nn.Module):
    def __init__(self, cardinalities: Sequence[int], embedding_dim: int = 8) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities
        ])
        self.output_dim = len(cardinalities) * embedding_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            embedding(values[:, index]) for index, embedding in enumerate(self.embeddings)
        ], dim=-1)


class DeepSetsProbeEncoder(nn.Module):
    """Permutation-invariant mean/max encoder for a variable probe set."""

    def __init__(self, item_dim: int = 18, hidden_dim: int = 64, output_dim: int = 64) -> None:
        super().__init__()
        self.item_encoder = mlp([item_dim, hidden_dim, output_dim])
        self.output_dim = output_dim * 2

    def forward(self, probes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.item_encoder(probes)
        float_mask = mask.unsqueeze(-1).to(encoded.dtype)
        count = float_mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * float_mask).sum(dim=1) / count
        min_value = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(~mask.unsqueeze(-1), min_value).max(dim=1).values
        empty = ~mask.any(dim=1)
        max_pool = torch.where(empty.unsqueeze(-1), torch.zeros_like(max_pool), max_pool)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)


@dataclass
class NeuralExpertOutput:
    theta_physics: torch.Tensor
    theta_neural: torch.Tensor
    cosine_physics: torch.Tensor
    cosine_neural: torch.Tensor
    residual_cosine: torch.Tensor
    sfe_dispersion: torch.Tensor
    sfe_polar: torch.Tensor
    log_variance: torch.Tensor


class PhysicsGuidedNeuralExpert(nn.Module):
    def __init__(
        self,
        surface_numeric_dim: int,
        liquid_numeric_dim: int,
        condition_numeric_dim: int,
        categorical_cardinalities: Sequence[int],
        categorical_embedding_dim: int = 8,
        dropout: float = 0.1,
        max_delta_cos: float = 0.5,
        use_probe_encoder: bool = True,
        use_physics_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.max_delta_cos = float(max_delta_cos)
        self.use_probe_encoder = bool(use_probe_encoder)
        self.use_physics_decoder = bool(use_physics_decoder)
        self.category_encoder = CategoricalContextEncoder(
            categorical_cardinalities, categorical_embedding_dim
        )
        surface_input = surface_numeric_dim + condition_numeric_dim + self.category_encoder.output_dim
        self.surface_encoder = mlp([surface_input, 128, 64], dropout)
        self.liquid_encoder = mlp([liquid_numeric_dim, 32, 32], dropout)
        self.probe_encoder = DeepSetsProbeEncoder(item_dim=18, hidden_dim=64, output_dim=64)
        sfe_input = 64 + self.probe_encoder.output_dim + 1
        self.sfe_head = mlp([sfe_input, 64, 2], dropout)
        residual_input = 64 + 32 + self.probe_encoder.output_dim + 1 + 1
        self.residual_head = mlp([residual_input, 128, 64, 1], dropout)
        self.direct_cosine_head = mlp([residual_input - 1, 128, 64, 1], dropout)
        self.variance_head = mlp([residual_input, 64, 1], dropout)

    @staticmethod
    def physical_cosine(
        sfe_dispersion: torch.Tensor,
        sfe_polar: torch.Tensor,
        liquid_numeric: torch.Tensor,
    ) -> torch.Tensor:
        # The first three standardized liquid values cannot be used in the equation.
        # Raw physical components are reconstructed by the caller and passed separately.
        gamma_total = liquid_numeric[:, 0].clamp_min(1e-6)
        gamma_dispersion = liquid_numeric[:, 1].clamp_min(0.0)
        gamma_polar = liquid_numeric[:, 2].clamp_min(0.0)
        epsilon = 1e-8
        dispersion_product = (sfe_dispersion * gamma_dispersion).clamp_min(0.0)
        polar_product = (sfe_polar * gamma_polar).clamp_min(0.0)
        dispersion_root = torch.sqrt(dispersion_product + epsilon) - epsilon**0.5
        polar_root = torch.sqrt(polar_product + epsilon) - epsilon**0.5
        numerator = 2.0 * (
            dispersion_root + polar_root
        )
        # acos has an infinite derivative at exactly +/-1; keep training inside the open domain.
        return (numerator / gamma_total - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    def forward(
        self,
        surface_numeric: torch.Tensor,
        categorical: torch.Tensor,
        target_liquid_numeric: torch.Tensor,
        target_liquid_physical: torch.Tensor,
        condition_numeric: torch.Tensor,
        probes: torch.Tensor,
        probe_mask: torch.Tensor,
        independent_sfe: torch.Tensor,
        independent_sfe_mask: torch.Tensor,
        nnls_sfe: torch.Tensor | None = None,
        nnls_sfe_mask: torch.Tensor | None = None,
    ) -> NeuralExpertOutput:
        category_embedding = self.category_encoder(categorical)
        surface_context = self.surface_encoder(torch.cat([
            surface_numeric, condition_numeric, category_embedding
        ], dim=-1))
        liquid_context = self.liquid_encoder(target_liquid_numeric)
        probe_context = self.probe_encoder(probes, probe_mask)
        probe_available = probe_mask.any(dim=1, keepdim=True).to(surface_numeric.dtype)
        if not self.use_probe_encoder:
            probe_context = torch.zeros_like(probe_context)
            probe_available = torch.zeros_like(probe_available)

        latent_sfe = F.softplus(self.sfe_head(torch.cat([
            surface_context, probe_context, probe_available
        ], dim=-1)))
        sfe_mask = independent_sfe_mask.unsqueeze(-1)
        if nnls_sfe is None:
            nnls_sfe = torch.zeros_like(independent_sfe)
        if nnls_sfe_mask is None:
            nnls_sfe_mask = torch.zeros_like(independent_sfe_mask)
        nnls_mask = (1.0 - sfe_mask) * nnls_sfe_mask.unsqueeze(-1)
        effective_sfe = (
            sfe_mask * independent_sfe
            + nnls_mask * nnls_sfe
            + (1.0 - sfe_mask - nnls_mask) * latent_sfe
        )
        physics_cos = self.physical_cosine(
            effective_sfe[:, 0], effective_sfe[:, 1], target_liquid_physical
        )
        common_context = torch.cat([
            surface_context, liquid_context, probe_context, probe_available,
        ], dim=-1)
        residual_input = torch.cat([common_context, physics_cos.unsqueeze(-1)], dim=-1)
        if self.use_physics_decoder:
            residual = self.max_delta_cos * torch.tanh(self.residual_head(residual_input).squeeze(-1))
            neural_cos = (physics_cos + residual).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        else:
            neural_cos = torch.tanh(self.direct_cosine_head(common_context).squeeze(-1))
            neural_cos = neural_cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            residual = neural_cos
        theta_physics = torch.rad2deg(torch.acos(physics_cos))
        theta_neural = torch.rad2deg(torch.acos(neural_cos))
        log_variance = self.variance_head(residual_input).squeeze(-1).clamp(-4.0, 8.0)
        return NeuralExpertOutput(
            theta_physics=theta_physics,
            theta_neural=theta_neural,
            cosine_physics=physics_cos,
            cosine_neural=neural_cos,
            residual_cosine=residual,
            sfe_dispersion=latent_sfe[:, 0],
            sfe_polar=latent_sfe[:, 1],
            log_variance=log_variance,
        )


class ExpertGate(nn.Module):
    """Leakage-safe global convex gate trained only from OOF expert predictions.

    A high-capacity context gate can memorize the small OOF training table and
    fail on a new source. A single global simplex weight is intentionally less
    expressive, but its behavior is stable, auditable, and still allows the
    physics, neural, and tree experts to contribute to every prediction.
    """

    def __init__(self, context_dim: int, hidden_dim: int = 32, n_experts: int = 3) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.context_dim = int(context_dim)
        self.logits = nn.Parameter(torch.zeros(self.n_experts))

    def forward(self, expert_predictions: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del context
        weights = torch.softmax(self.logits, dim=-1).expand(expert_predictions.shape[0], -1)
        prediction = (weights * expert_predictions).sum(dim=-1)
        return prediction, weights


@dataclass
class PhysicsSummaryOutput:
    theta_physics: torch.Tensor
    theta_neural: torch.Tensor
    cosine_physics: torch.Tensor
    cosine_neural: torch.Tensor
    residual_cosine: torch.Tensor
    log_variance: torch.Tensor


class PhysicsSummaryResidualExpert(nn.Module):
    """Compact residual expert anchored to target-masked NNLS-OWRK physics."""

    def __init__(
        self,
        surface_numeric_dim: int,
        liquid_numeric_dim: int,
        condition_numeric_dim: int,
        physics_summary_dim: int,
        categorical_cardinalities: Sequence[int],
        categorical_embedding_dim: int = 8,
        dropout: float = 0.1,
        max_delta_cos: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_delta_cos = float(max_delta_cos)
        self.category_encoder = CategoricalContextEncoder(
            categorical_cardinalities, categorical_embedding_dim
        )
        surface_input = (
            surface_numeric_dim + condition_numeric_dim + self.category_encoder.output_dim
        )
        self.surface_encoder = mlp([surface_input, 64, 32], dropout)
        self.liquid_encoder = mlp([liquid_numeric_dim, 32, 16], dropout)
        self.physics_encoder = mlp([physics_summary_dim, 32, 16], dropout)
        self.fusion_encoder = mlp([64, 32], dropout)
        self.residual_head = nn.Linear(32, 1)
        self.variance_head = nn.Linear(32, 1)

    def forward(
        self,
        surface_numeric: torch.Tensor,
        categorical: torch.Tensor,
        target_liquid_numeric: torch.Tensor,
        condition_numeric: torch.Tensor,
        physics_summary: torch.Tensor,
        physics_cosine: torch.Tensor,
    ) -> PhysicsSummaryOutput:
        surface = self.surface_encoder(torch.cat([
            surface_numeric,
            condition_numeric,
            self.category_encoder(categorical),
        ], dim=-1))
        liquid = self.liquid_encoder(target_liquid_numeric)
        physics = self.physics_encoder(physics_summary)
        hidden = self.fusion_encoder(torch.cat([surface, liquid, physics], dim=-1))
        residual = self.max_delta_cos * torch.tanh(self.residual_head(hidden).squeeze(-1))
        neural_cosine = (physics_cosine + residual).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return PhysicsSummaryOutput(
            theta_physics=torch.rad2deg(torch.acos(physics_cosine)),
            theta_neural=torch.rad2deg(torch.acos(neural_cosine)),
            cosine_physics=physics_cosine,
            cosine_neural=neural_cosine,
            residual_cosine=residual,
            log_variance=self.variance_head(hidden).squeeze(-1).clamp(-4.0, 8.0),
        )
