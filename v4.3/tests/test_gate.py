import numpy as np

from lspgmoe.training import predict_gate, train_gate


def test_global_gate_is_invariant_to_context_and_returns_simplex_weights():
    experts = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]], dtype=np.float32)
    context_a = np.zeros((2, 7), dtype=np.float32)
    context_b = np.ones((2, 7), dtype=np.float32)
    gate = train_gate(experts, context_a, np.array([18.0, 22.0]), seed=7, max_epochs=100, patience=20)
    prediction_a, weights_a = predict_gate(gate, experts, context_a)
    prediction_b, weights_b = predict_gate(gate, experts, context_b)
    assert np.allclose(prediction_a, prediction_b)
    assert np.allclose(weights_a, weights_b)
    assert np.allclose(weights_a.sum(axis=1), 1.0)
    assert np.all(weights_a >= 0.0)
