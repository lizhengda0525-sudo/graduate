import math

import numpy as np
from numpy.typing import NDArray


def _without_diagonal(causal_matrices: NDArray[np.floating]) -> NDArray[np.float64]:
    if causal_matrices.ndim != 3:
        raise ValueError("causal_matrices must have shape [window, target, source]")
    if causal_matrices.shape[1] != causal_matrices.shape[2]:
        raise ValueError("causal matrices must be square")
    values = causal_matrices.astype(np.float64, copy=True)
    diagonal = np.arange(values.shape[1])
    values[:, diagonal, diagonal] = 0.0
    return values


def network_flows(
    causal_matrices: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    values = _without_diagonal(causal_matrices)
    outflow = values.sum(axis=1)
    inflow = values.sum(axis=2)
    netflow = outflow - inflow
    return outflow, inflow, netflow


def change_curve(causal_matrices: NDArray[np.floating]) -> NDArray[np.float64]:
    values = _without_diagonal(causal_matrices)
    if values.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return np.abs(values[1:] - values[:-1]).sum(axis=(1, 2))


def predict_soz(
    outflow: NDArray[np.floating],
    ictal_mask: NDArray[np.bool_],
    top_fraction: float,
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64]]:
    if outflow.ndim != 2:
        raise ValueError("outflow must have shape [window, channel]")
    if ictal_mask.shape != (outflow.shape[0],):
        raise ValueError("ictal_mask must match the number of windows")
    if not np.any(ictal_mask):
        raise ValueError("ictal_mask must select at least one window")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    num_channels = outflow.shape[1]
    top_count = max(1, math.ceil(num_channels * top_fraction))
    ictal_outflow = np.asarray(outflow[ictal_mask], dtype=np.float64)
    window_top_indices = np.argpartition(ictal_outflow, -top_count, axis=1)[:, -top_count:]
    abnormal_counts = np.zeros(num_channels, dtype=np.int64)
    for indices in window_top_indices:
        abnormal_counts[indices] += 1

    mean_outflow = ictal_outflow.mean(axis=0)
    ranking = np.lexsort((-mean_outflow, -abnormal_counts))
    prediction = np.zeros(num_channels, dtype=np.bool_)
    prediction[ranking[:top_count]] = True
    return prediction, abnormal_counts, mean_outflow
