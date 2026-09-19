from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticDataset:
    windows: NDArray[np.float32]
    causal_matrices: NDArray[np.float32]
    change_window: int


def generate_synthetic_data(
    num_series: int,
    num_windows: int,
    points_per_window: int,
    lag: int,
    noise_std: float,
    change_window: int,
    seed: int,
) -> SyntheticDataset:
    if num_series < 5 or num_windows < 2:
        raise ValueError("at least five series and two windows are required")
    if lag < 1 or points_per_window <= lag:
        raise ValueError("lag must be positive and smaller than points_per_window")
    if noise_std < 0 or not 1 <= change_window < num_windows:
        raise ValueError("noise_std or change_window is invalid")

    network_before = np.zeros((num_series, num_series), dtype=np.float64)
    for source in range(num_series):
        network_before[(source + 1) % num_series, source] = 0.55
    for source in range(0, num_series, 2):
        network_before[(source + 3) % num_series, source] = -0.35
    np.fill_diagonal(network_before, 0.25)

    network_after = network_before.copy()
    network_after[1 % num_series, 0] = 0.0
    network_after[3 % num_series, 2 % num_series] = 0.0
    network_after[0, (num_series - 1) % num_series] = 0.65
    network_after[2 % num_series, 0] = -0.45
    network_after[4 % num_series, 1 % num_series] = 0.40

    rng = np.random.default_rng(seed)
    total_points = num_windows * points_per_window
    values = np.zeros((total_points + lag, num_series), dtype=np.float64)
    values[:lag] = rng.normal(scale=0.1, size=(lag, num_series))
    true_matrices = np.empty((num_windows, num_series, num_series), dtype=np.float32)

    for window_index in range(num_windows):
        network = network_before if window_index < change_window else network_after
        true_matrices[window_index] = np.abs(network).astype(np.float32)
        start = lag + window_index * points_per_window
        for time_index in range(start, start + points_per_window):
            lag_one = np.tanh(network @ values[time_index - 1])
            lag_two = 0.15 * values[time_index - lag]
            noise = rng.normal(scale=noise_std, size=num_series)
            values[time_index] = lag_one + lag_two + noise

    windows = values[lag:].reshape(num_windows, points_per_window, num_series)
    return SyntheticDataset(
        windows=windows.astype(np.float32),
        causal_matrices=true_matrices,
        change_window=change_window,
    )
