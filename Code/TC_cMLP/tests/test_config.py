import pytest

from tc_cmlp.config import ModelConfig, TrainingConfig


def test_model_config_rejects_zero_hidden_dimension() -> None:
    with pytest.raises(ValueError):
        ModelConfig(hidden_dim=0)


def test_training_config_rejects_unknown_device() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            epochs=1,
            learning_rate=0.01,
            lambda_group=0.0,
            lambda_temporal=0.0,
            lambda_weight=0.0,
            seed=1,
            log_every=1,
            device="auto",
            sample_batch_size=8,
        )
