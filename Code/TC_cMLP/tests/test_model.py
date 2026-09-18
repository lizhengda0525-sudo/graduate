import numpy as np
import torch

from tc_cmlp.config import ModelConfig, TrainingConfig
from tc_cmlp.models.tc_cmlp import TemporallyCoupledCMLP
from tc_cmlp.models.training import fit_tc_cmlp


def test_model_output_and_causal_score_shapes() -> None:
    model = TemporallyCoupledCMLP(num_windows=3, num_series=5, lag=2, hidden_dim=4)
    predictors = torch.randn(7, 5, 2)

    assert model.forward_window(predictors, window_index=1).shape == (7, 5)
    assert model.causal_scores().shape == (3, 5, 5)


def test_temporal_penalty_is_zero_for_equal_window_weights() -> None:
    model = TemporallyCoupledCMLP(num_windows=2, num_series=5, lag=2, hidden_dim=3)
    with torch.no_grad():
        model.first_weight[1].copy_(model.first_weight[0])
    assert model.temporal_penalty().item() == 0.0


def test_group_proximal_can_zero_weight_groups() -> None:
    model = TemporallyCoupledCMLP(num_windows=1, num_series=5, lag=2, hidden_dim=3)
    model.apply_group_proximal(threshold=100.0)
    assert torch.count_nonzero(model.first_weight).item() == 0


def test_joint_training_smoke() -> None:
    rng = np.random.default_rng(2)
    windows = rng.normal(size=(2, 20, 5)).astype(np.float32)
    result = fit_tc_cmlp(
        windows=windows,
        lag=2,
        model_config=ModelConfig(hidden_dim=3),
        training_config=TrainingConfig(
            epochs=2,
            learning_rate=0.01,
            lambda_group=0.001,
            lambda_temporal=0.01,
            lambda_weight=0.0001,
            seed=3,
            log_every=1,
            device="cpu",
            sample_batch_size=8,
        ),
    )

    assert result.causal_matrices.shape == (2, 5, 5)
    assert len(result.history) == 2
    assert np.isfinite(result.causal_matrices).all()
