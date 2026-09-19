import numpy as np

from tc_cmlp.analysis import edge_metrics
from tc_cmlp.data.synthetic import generate_synthetic_data
from tc_cmlp.experiment import _fit_timed
from tc_cmlp.training import TrainingConfig, fit_original_cmlp, fit_temporal_cmlp


def test_training_and_metrics_on_synthetic_data() -> None:
    dataset = generate_synthetic_data(5, 2, 30, 2, 0.15, 1, 3)
    config = TrainingConfig(
        max_iter=2,
        learning_rate=0.01,
        lambda_group=0.001,
        lambda_ridge=0.001,
        lambda_temporal=0.01,
        check_every=1,
        lookback=2,
        seed=3,
    )
    for fit in (fit_original_cmlp, fit_temporal_cmlp):
        result = fit(dataset.windows, 2, [4], "relu", config)
        assert result.causal_strengths.shape == (2, 5, 5)
        assert result.causal_binary.shape == (2, 5, 5)
        assert np.isfinite(result.causal_strengths).all()
        assert np.array_equal(result.causal_binary > 0, result.causal_strengths > 0)
        assert result.history
        metrics = edge_metrics(
            dataset.causal_matrices, result.causal_strengths, result.causal_binary
        )
        assert 0 <= metrics["edge_auprc"] <= 1


def test_perfect_edge_scores() -> None:
    dataset = generate_synthetic_data(5, 2, 30, 2, 0.15, 1, 3)
    strengths = dataset.causal_matrices
    binary = (strengths > 0).astype(np.uint8)
    metrics = edge_metrics(dataset.causal_matrices, strengths, binary)
    assert metrics["edge_auprc"] == 1.0
    assert metrics["binary_f1"] == 1.0


def test_training_timing_on_real_models(capsys) -> None:
    dataset = generate_synthetic_data(5, 2, 30, 2, 0.15, 1, 3)
    config = TrainingConfig(
        max_iter=2,
        learning_rate=0.01,
        lambda_group=0.001,
        lambda_ridge=0.001,
        lambda_temporal=0.01,
        check_every=1,
        lookback=2,
        seed=3,
    )
    for method in ("cmlp", "tc_no_temporal", "tc"):
        result, timing = _fit_timed(method, dataset.windows, 2, [4], "relu", config)
        assert result.causal_strengths.shape == (2, 5, 5)
        assert timing["training_seconds"] > 0
        assert timing["started_at"] <= timing["finished_at"]
    output = capsys.readouterr().out
    assert "[cmlp] 窗口 2/2 已完成" in output
    assert "[tc_no_temporal] 迭代 2/2" in output
    assert "[tc] 训练完成" in output
