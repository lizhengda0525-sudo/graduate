import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, f1_score


def _off_diagonal(values: NDArray[np.floating]) -> NDArray[np.float64]:
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("values must have shape [window, series, series]")
    mask = ~np.eye(values.shape[1], dtype=bool)
    return np.asarray(values[:, mask], dtype=np.float64)


def edge_metrics(
    true_matrices: NDArray[np.floating],
    predicted_matrices: NDArray[np.floating],
) -> dict[str, float]:
    if true_matrices.shape != predicted_matrices.shape:
        raise ValueError("true and predicted matrices must have identical shapes")
    true_values = _off_diagonal(true_matrices)
    predicted_values = _off_diagonal(predicted_matrices)
    true_edges = true_values > 0

    average_precision = average_precision_score(true_edges.ravel(), predicted_values.ravel())
    predicted_edges = np.zeros_like(true_edges)
    for window_index in range(true_edges.shape[0]):
        edge_count = int(true_edges[window_index].sum())
        if edge_count > 0:
            indices = np.argpartition(predicted_values[window_index], -edge_count)[-edge_count:]
            predicted_edges[window_index, indices] = True

    edge_f1 = f1_score(true_edges.ravel(), predicted_edges.ravel(), zero_division=0)
    scale = np.mean(true_values[true_values > 0])
    strength_error = np.sqrt(np.mean((true_values - predicted_values) ** 2)) / scale
    return {
        "edge_auprc": float(average_precision),
        "edge_f1": float(edge_f1),
        "causal_strength_error": float(strength_error),
    }


def soz_metrics(
    true_soz: NDArray[np.bool_],
    predicted_soz: NDArray[np.bool_],
) -> dict[str, float]:
    if true_soz.shape != predicted_soz.shape:
        raise ValueError("true_soz and predicted_soz must have identical shapes")
    true_values = np.asarray(true_soz, dtype=bool)
    predicted_values = np.asarray(predicted_soz, dtype=bool)
    true_positive = int(np.sum(true_values & predicted_values))
    false_positive = int(np.sum(~true_values & predicted_values))
    false_negative = int(np.sum(true_values & ~predicted_values))
    precision = true_positive / (true_positive + false_positive) if predicted_values.any() else 0.0
    sensitivity = true_positive / (true_positive + false_negative) if true_values.any() else 0.0
    denominator = precision + sensitivity
    f1_value = 2 * precision * sensitivity / denominator if denominator else 0.0
    return {
        "precision": precision,
        "sensitivity": sensitivity,
        "f1": f1_value,
    }
