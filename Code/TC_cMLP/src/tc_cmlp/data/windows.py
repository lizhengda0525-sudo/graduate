import numpy as np
from numpy.typing import NDArray


def build_lagged_samples(
    values: NDArray[np.floating],
    lag: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if values.ndim != 2:
        raise ValueError("values must have shape [time, series]")
    if lag < 1 or lag >= values.shape[0]:
        raise ValueError("lag must be between 1 and number of time points - 1")

    num_samples = values.shape[0] - lag
    num_series = values.shape[1]
    predictors = np.empty((num_samples, num_series, lag), dtype=np.float32)
    targets = values[lag:].astype(np.float32, copy=True)

    for sample_index in range(num_samples):
        current_index = sample_index + lag
        history = values[current_index - lag : current_index]
        predictors[sample_index] = history[::-1].T

    return predictors, targets


def sliding_window_bounds(
    num_points: int,
    window_size: int,
    step_size: int,
) -> list[tuple[int, int]]:
    if num_points < 1:
        raise ValueError("num_points must be positive")
    if window_size < 1 or window_size > num_points:
        raise ValueError("window_size must be within the signal length")
    if step_size < 1:
        raise ValueError("step_size must be positive")

    return [
        (start, start + window_size) for start in range(0, num_points - window_size + 1, step_size)
    ]
