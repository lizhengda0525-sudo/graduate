from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from tc_cmlp.analysis.network import change_curve


def plot_synthetic_results(
    true_matrices: NDArray[np.floating],
    predicted_by_method: dict[str, NDArray[np.floating]],
    change_window: int,
    output_path: str | Path,
) -> None:
    methods = list(predicted_by_method)
    figure, axes = plt.subplots(2, len(methods) + 1, figsize=(4 * (len(methods) + 1), 7))
    before_index = max(0, change_window - 1)
    axes[0, 0].imshow(true_matrices[before_index], cmap="viridis")
    axes[0, 0].set_title("True network before change")
    axes[1, 0].imshow(true_matrices[change_window], cmap="viridis")
    axes[1, 0].set_title("True network after change")

    for column, method in enumerate(methods, start=1):
        matrices = predicted_by_method[method]
        axes[0, column].imshow(matrices[before_index], cmap="viridis")
        axes[0, column].set_title(f"{method}: before")
        axes[1, column].plot(np.arange(1, matrices.shape[0]), change_curve(matrices))
        axes[1, column].axvline(change_window, color="red", linestyle="--")
        axes[1, column].set_title(f"{method}: change curve")
        axes[1, column].set_xlabel("Window")

    for axis in axes.ravel():
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_hup_results(
    causal_matrices: NDArray[np.floating],
    outflow: NDArray[np.floating],
    window_starts: NDArray[np.floating],
    channel_names: list[str],
    soz_mask: NDArray[np.bool_],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    preictal = causal_matrices[window_starts < 0].mean(axis=0)
    ictal = causal_matrices[window_starts >= 0].mean(axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(preictal, cmap="viridis")
    axes[0].set_title("Preictal causal matrix")
    axes[1].imshow(ictal, cmap="viridis")
    axes[1].set_title("Ictal causal matrix")
    for axis in axes:
        axis.set_xlabel("Source channel")
        axis.set_ylabel("Target channel")
    figure.tight_layout()
    figure.savefig(destination / "causal_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    for channel_index, channel_name in enumerate(channel_names):
        color = "red" if soz_mask[channel_index] else "steelblue"
        alpha = 0.9 if soz_mask[channel_index] else 0.3
        axis.plot(
            window_starts,
            outflow[:, channel_index],
            color=color,
            alpha=alpha,
            linewidth=1.5 if soz_mask[channel_index] else 0.8,
            label=channel_name if soz_mask[channel_index] else None,
        )
    axis.axvspan(0, window_starts[-1], color="gray", alpha=0.15)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Time relative to seizure onset (s)")
    axis.set_ylabel("Dynamic causal outflow")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(destination / "dynamic_outflow.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
