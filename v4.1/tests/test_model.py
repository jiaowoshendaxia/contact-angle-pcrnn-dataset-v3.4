import torch

from lspgmoe.model import DeepSetsProbeEncoder, PhysicsGuidedNeuralExpert


def test_deepsets_is_probe_order_invariant():
    torch.manual_seed(3)
    encoder = DeepSetsProbeEncoder(item_dim=18, hidden_dim=16, output_dim=8).eval()
    probes = torch.randn(2, 4, 18)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = encoder(probes, mask)
    permuted = encoder(probes[:, permutation], mask[:, permutation])
    torch.testing.assert_close(original, permuted)


def test_zero_probe_representation_is_exactly_zero():
    encoder = DeepSetsProbeEncoder(item_dim=18, hidden_dim=16, output_dim=8).eval()
    output = encoder(torch.randn(3, 4, 18), torch.zeros(3, 4, dtype=torch.bool))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_latent_sfe_head_is_nonnegative():
    torch.manual_seed(4)
    model = PhysicsGuidedNeuralExpert(
        surface_numeric_dim=6,
        liquid_numeric_dim=12,
        condition_numeric_dim=8,
        categorical_cardinalities=[4, 4, 4, 4, 4, 4, 4],
    ).eval()
    output = model(
        torch.randn(5, 6),
        torch.zeros(5, 7, dtype=torch.long),
        torch.randn(5, 12),
        torch.tensor([[72.8, 21.8, 51.0]]).repeat(5, 1),
        torch.randn(5, 8),
        torch.randn(5, 3, 18),
        torch.zeros(5, 3, dtype=torch.bool),
        torch.zeros(5, 2),
        torch.zeros(5),
    )
    assert torch.all(output.sfe_dispersion >= 0)
    assert torch.all(output.sfe_polar >= 0)
    assert torch.all(output.residual_cosine.abs() <= 0.5)


def test_physical_decoder_avoids_acos_gradient_singularity():
    sfe_d = torch.tensor([1e6], requires_grad=True)
    sfe_p = torch.tensor([1e6], requires_grad=True)
    liquid = torch.tensor([[72.8, 21.8, 51.0]])
    cosine = PhysicsGuidedNeuralExpert.physical_cosine(sfe_d, sfe_p, liquid)
    angle = torch.acos(cosine).sum()
    angle.backward()
    assert torch.isfinite(angle)
    assert torch.isfinite(sfe_d.grad).all()
    assert torch.isfinite(sfe_p.grad).all()


def test_physical_decoder_has_finite_gradient_for_zero_component():
    sfe_d = torch.tensor([20.0], requires_grad=True)
    sfe_p = torch.tensor([0.0], requires_grad=True)
    liquid = torch.tensor([[44.4, 44.4, 0.0]])
    angle = torch.acos(PhysicsGuidedNeuralExpert.physical_cosine(sfe_d, sfe_p, liquid)).sum()
    angle.backward()
    assert torch.isfinite(sfe_d.grad).all()
    assert torch.isfinite(sfe_p.grad).all()
