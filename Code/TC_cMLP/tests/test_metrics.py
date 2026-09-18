import numpy as np

from tc_cmlp.analysis.metrics import edge_metrics, soz_metrics


def test_edge_metrics_are_perfect_for_identical_matrices() -> None:
    matrices = np.asarray(
        [
            [
                [0.0, 0.7, 0.0],
                [0.0, 0.0, 0.4],
                [0.2, 0.0, 0.0],
            ]
        ]
    )
    metrics = edge_metrics(matrices, matrices)

    assert metrics["edge_auprc"] == 1.0
    assert metrics["edge_f1"] == 1.0
    assert metrics["causal_strength_error"] == 0.0


def test_soz_metrics() -> None:
    metrics = soz_metrics(
        true_soz=np.asarray([True, True, False, False]),
        predicted_soz=np.asarray([True, False, True, False]),
    )

    assert metrics == {"precision": 0.5, "sensitivity": 0.5, "f1": 0.5}
