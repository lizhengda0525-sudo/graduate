from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from tc_cmlp.config import ModelConfig, TrainingConfig
from tc_cmlp.data.windows import build_lagged_samples
from tc_cmlp.models.tc_cmlp import TemporallyCoupledCMLP


@dataclass
class FitResult:
    causal_matrices: NDArray[np.float32]
    history: list[dict[str, float]]
    models: list[TemporallyCoupledCMLP]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(device_name)


def _prepare_windows(
    windows: NDArray[np.floating],
    lag: int,
    device: torch.device,
) -> list[tuple[Tensor, Tensor]]:
    if windows.ndim != 3:
        raise ValueError("windows must have shape [window, time, series]")

    prepared: list[tuple[Tensor, Tensor]] = []
    for window in windows:
        predictors, targets = build_lagged_samples(window, lag)
        prepared.append(
            (
                torch.as_tensor(predictors, device=device),
                torch.as_tensor(targets, device=device),
            )
        )
    return prepared


def _sample_batch(
    predictors: Tensor,
    targets: Tensor,
    sample_batch_size: int,
) -> tuple[Tensor, Tensor]:
    if predictors.shape[0] <= sample_batch_size:
        return predictors, targets
    indices = torch.randperm(predictors.shape[0], device=predictors.device)[:sample_batch_size]
    return predictors[indices], targets[indices]


def _fit_joint_model(
    windows: NDArray[np.floating],
    lag: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> tuple[TemporallyCoupledCMLP, list[dict[str, float]]]:
    _set_seed(training_config.seed)
    device = _validate_device(training_config.device)
    num_windows, _, num_series = windows.shape
    prepared = _prepare_windows(windows, lag, device)

    model = TemporallyCoupledCMLP(
        num_windows=num_windows,
        num_series=num_series,
        lag=lag,
        hidden_dim=model_config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.epochs + 1):
        optimizer.zero_grad()
        window_losses = []
        for index, (predictors, targets) in enumerate(prepared):
            x_values, y_values = _sample_batch(
                predictors,
                targets,
                training_config.sample_batch_size,
            )
            predictions = model.forward_window(x_values, index)
            window_losses.append(torch.mean((predictions - y_values).square()))
        prediction_loss = torch.stack(window_losses).mean()
        temporal_loss = model.temporal_penalty()
        weight_loss = model.shared_weight_penalty()
        objective = (
            prediction_loss
            + training_config.lambda_temporal * temporal_loss
            + training_config.lambda_weight * weight_loss
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite training objective at epoch {epoch}")

        objective.backward()
        optimizer.step()
        model.apply_group_proximal(training_config.learning_rate * training_config.lambda_group)

        if epoch == 1 or epoch % training_config.log_every == 0 or epoch == training_config.epochs:
            with torch.no_grad():
                group_loss = model.group_penalty()
                total_loss = objective + training_config.lambda_group * group_loss
            history.append(
                {
                    "epoch": float(epoch),
                    "total_loss": float(total_loss.item()),
                    "prediction_loss": float(prediction_loss.item()),
                    "group_penalty": float(group_loss.item()),
                    "temporal_penalty": float(temporal_loss.item()),
                    "weight_penalty": float(weight_loss.item()),
                }
            )

    return model, history


def fit_tc_cmlp(
    windows: NDArray[np.floating],
    lag: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> FitResult:
    model, history = _fit_joint_model(windows, lag, model_config, training_config)
    causal_matrices = model.causal_scores().detach().cpu().numpy().astype(np.float32)
    return FitResult(causal_matrices=causal_matrices, history=history, models=[model])


def fit_independent_cmlp(
    windows: NDArray[np.floating],
    lag: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> FitResult:
    causal_matrices: list[NDArray[np.float32]] = []
    history: list[dict[str, float]] = []
    models: list[TemporallyCoupledCMLP] = []

    independent_config = TrainingConfig(
        epochs=training_config.epochs,
        learning_rate=training_config.learning_rate,
        lambda_group=training_config.lambda_group,
        lambda_temporal=0.0,
        lambda_weight=training_config.lambda_weight,
        seed=training_config.seed,
        log_every=training_config.log_every,
        device=training_config.device,
        sample_batch_size=training_config.sample_batch_size,
    )
    for window_index, window in enumerate(windows):
        current_config = TrainingConfig(
            **{
                **independent_config.__dict__,
                "seed": independent_config.seed + window_index,
            }
        )
        model, current_history = _fit_joint_model(
            window[None, ...], lag, model_config, current_config
        )
        causal_matrices.append(model.causal_scores()[0].detach().cpu().numpy().astype(np.float32))
        models.append(model)
        for record in current_history:
            history.append({"window": float(window_index), **record})

    return FitResult(
        causal_matrices=np.stack(causal_matrices),
        history=history,
        models=models,
    )
