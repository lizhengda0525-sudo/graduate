import numpy as np
import pytest

from tc_cmlp.data.windows import build_lagged_samples, sliding_window_bounds


def test_build_lagged_samples_orders_recent_value_first() -> None:
    values = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    predictors, targets = build_lagged_samples(values, lag=2)

    assert predictors.shape == (2, 2, 2)
    np.testing.assert_array_equal(predictors[0], [[2.0, 1.0], [20.0, 10.0]])
    np.testing.assert_array_equal(targets, [[3.0, 30.0], [4.0, 40.0]])


def test_sliding_window_bounds_includes_last_complete_window() -> None:
    assert sliding_window_bounds(num_points=11, window_size=5, step_size=3) == [
        (0, 5),
        (3, 8),
        (6, 11),
    ]


def test_build_lagged_samples_rejects_invalid_lag() -> None:
    with pytest.raises(ValueError):
        build_lagged_samples(np.ones((3, 2)), lag=3)
