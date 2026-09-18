from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticDataset:
    windows: NDArray[np.float32]
    causal_matrices: NDArray[np.float32]
    change_window: int


def _base_network(num_series: int) -> NDArray[np.float64]:
    matrix = np.zeros((num_series, num_series), dtype=np.float64)
    for source in range(num_series):
        target = (source + 1) % num_series
        matrix[target, source] = 0.55
    for source in range(0, num_series, 2):
        target = (source + 3) % num_series
        matrix[target, source] = -0.35
    np.fill_diagonal(matrix, 0.25)
    return matrix


def _changed_network(base: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = base.copy()
    num_series = matrix.shape[0]
    matrix[1 % num_series, 0] = 0.0
    matrix[3 % num_series, 2 % num_series] = 0.0
    matrix[0, (num_series - 1) % num_series] = 0.65
    matrix[2 % num_series, 0] = -0.45
    matrix[4 % num_series, 1 % num_series] = 0.40
    return matrix


def generate_dynamic_nonlinear_data(
    num_series: int,
    num_windows: int,
    points_per_window: int,
    lag: int,
    noise_std: float,
    change_window: int,
    seed: int,
) -> SyntheticDataset:
    if num_series < 5:
        raise ValueError("num_series must be at least 5")
    if num_windows < 2:
        raise ValueError("num_windows must be at least 2")
    if points_per_window <= lag:
        raise ValueError("points_per_window must be greater than lag")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 1 <= change_window < num_windows:
        raise ValueError("change_window must be inside the window sequence")

    rng = np.random.default_rng(seed)
    network_a = _base_network(num_series)
    network_b = _changed_network(network_a)
    total_points = num_windows * points_per_window
    values = np.zeros((total_points + lag, num_series), dtype=np.float64)
    values[:lag] = rng.normal(scale=0.1, size=(lag, num_series))

    causal_matrices = np.empty((num_windows, num_series, num_series), dtype=np.float32)
    for window_index in range(num_windows):
        network = network_a if window_index < change_window else network_b
        causal_matrices[window_index] = np.abs(network).astype(np.float32)
        start = lag + window_index * points_per_window
        stop = start + points_per_window
        for time_index in range(start, stop):
            lag_one = np.tanh(network @ values[time_index - 1])
            lag_two = 0.15 * values[time_index - lag]
            noise = rng.normal(scale=noise_std, size=num_series)
            values[time_index] = lag_one + lag_two + noise

    windows = values[lag:].reshape(num_windows, points_per_window, num_series)
    return SyntheticDataset(
        windows=windows.astype(np.float32),
        causal_matrices=causal_matrices,
        change_window=change_window,
    )
