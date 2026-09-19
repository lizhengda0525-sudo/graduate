import math

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def _off_diagonal(values: NDArray[np.generic]) -> NDArray[np.generic]:
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("values must have shape [window, target, source]")
    mask = ~np.eye(values.shape[1], dtype=bool)
    return values[:, mask]


def edge_metrics(
    true_matrices: NDArray[np.floating],
    strengths: NDArray[np.floating],
    binary: NDArray[np.integer],
) -> dict[str, float | int]:
    if true_matrices.shape != strengths.shape or true_matrices.shape != binary.shape:
        raise ValueError("all causal matrices must have identical shapes")
    true_edges = (_off_diagonal(true_matrices) > 0).ravel()
    score_values = _off_diagonal(strengths).ravel()
    predicted_edges = (_off_diagonal(binary) > 0).ravel()
    if not np.all(np.isfinite(score_values)):
        raise ValueError("causal strengths contain non-finite values")

    true_strengths = np.asarray(_off_diagonal(true_matrices), dtype=np.float64)
    predicted_strengths = np.asarray(_off_diagonal(strengths), dtype=np.float64)
    true_norm = np.linalg.norm(true_strengths, axis=1, keepdims=True)
    predicted_norm = np.linalg.norm(predicted_strengths, axis=1, keepdims=True)
    normalized_true = true_strengths / np.maximum(true_norm, 1e-12)
    normalized_predicted = predicted_strengths / np.maximum(predicted_norm, 1e-12)
    strength_error = np.sqrt(np.mean((normalized_true - normalized_predicted) ** 2))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_edges,
        predicted_edges,
        average="binary",
        zero_division=0,
    )
    return {
        "edge_auprc": float(average_precision_score(true_edges, score_values)),
        "causal_strength_error": float(strength_error),
        "binary_precision": float(precision),
        "binary_recall": float(recall),
        "binary_f1": float(f1),
        "true_edges": int(true_edges.sum()),
        "predicted_edges": int(predicted_edges.sum()),
        "true_positives": int(np.sum(true_edges & predicted_edges)),
    }


def change_point_summary(
    causal_matrices: NDArray[np.floating],
    change_window: int,
) -> dict[str, float | int]:
    matrices = np.asarray(causal_matrices, dtype=np.float64).copy()
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("causal_matrices must have shape [window, target, source]")
    if not 1 <= change_window < matrices.shape[0]:
        raise ValueError("change_window must identify an existing transition")
    diagonal = np.arange(matrices.shape[1])
    matrices[:, diagonal, diagonal] = 0.0
    change_values = np.abs(matrices[1:] - matrices[:-1]).sum(axis=(1, 2))
    peak_window = int(np.argmax(change_values) + 1)
    ranking = np.argsort(change_values)[::-1]
    change_rank = int(np.flatnonzero(ranking == change_window - 1)[0] + 1)
    return {
        "peak_transition_window": peak_window,
        "true_transition_window": change_window,
        "true_transition_rank": change_rank,
        "true_transition_value": float(change_values[change_window - 1]),
    }


def network_flows(
    causal_matrices: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    matrices = np.asarray(causal_matrices, dtype=np.float64).copy()
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("causal_matrices must have shape [window, target, source]")
    diagonal = np.arange(matrices.shape[1])
    matrices[:, diagonal, diagonal] = 0.0
    outflow = matrices.sum(axis=1)
    inflow = matrices.sum(axis=2)
    return outflow, inflow, outflow - inflow


def predict_soz(
    outflow: NDArray[np.floating],
    ictal_mask: NDArray[np.bool_],
    top_fraction: float,
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.float64]]:
    if outflow.ndim != 2 or ictal_mask.shape != (outflow.shape[0],):
        raise ValueError("outflow and ictal_mask have incompatible shapes")
    if not np.any(ictal_mask) or not 0 < top_fraction <= 1:
        raise ValueError("ictal_mask must select windows and top_fraction must be in (0, 1]")
    count = max(1, math.ceil(outflow.shape[1] * top_fraction))
    ictal_outflow = np.asarray(outflow[ictal_mask], dtype=np.float64)
    top_indices = np.argpartition(ictal_outflow, -count, axis=1)[:, -count:]
    abnormal_counts = np.zeros(outflow.shape[1], dtype=np.int64)
    for indices in top_indices:
        abnormal_counts[indices] += 1
    mean_outflow = ictal_outflow.mean(axis=0)
    ranking = np.lexsort((-mean_outflow, -abnormal_counts))
    prediction = np.zeros(outflow.shape[1], dtype=np.bool_)
    prediction[ranking[:count]] = True
    return prediction, abnormal_counts, mean_outflow


def soz_metrics(
    true_soz: NDArray[np.bool_],
    predicted_soz: NDArray[np.bool_],
) -> dict[str, float]:
    if true_soz.shape != predicted_soz.shape:
        raise ValueError("true_soz and predicted_soz must have identical shapes")
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.asarray(true_soz, dtype=bool),
        np.asarray(predicted_soz, dtype=bool),
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "sensitivity": float(recall),
        "f1": float(f1),
    }
