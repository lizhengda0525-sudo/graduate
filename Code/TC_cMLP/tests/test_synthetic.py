import numpy as np

from tc_cmlp.data.synthetic import generate_dynamic_nonlinear_data


def test_synthetic_data_is_reproducible_and_changes_network() -> None:
    arguments = {
        "num_series": 5,
        "num_windows": 4,
        "points_per_window": 30,
        "lag": 2,
        "noise_std": 0.1,
        "change_window": 2,
        "seed": 7,
    }
    first = generate_dynamic_nonlinear_data(**arguments)
    second = generate_dynamic_nonlinear_data(**arguments)

    np.testing.assert_array_equal(first.windows, second.windows)
    np.testing.assert_array_equal(first.causal_matrices, second.causal_matrices)
    assert first.windows.shape == (4, 30, 5)
    assert not np.array_equal(first.causal_matrices[1], first.causal_matrices[2])
