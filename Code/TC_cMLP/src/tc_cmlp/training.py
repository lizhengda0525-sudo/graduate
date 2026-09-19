from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tc_cmlp.model import TemporallyCoupledCMLP
from tc_cmlp.official import cMLP, prox_update, train_model_ista

ProgressCallback = Callable[[dict[str, float | int | str]], None]


@dataclass(frozen=True)
class TrainingConfig:
    max_iter: int
    learning_rate: float
    lambda_group: float
    lambda_ridge: float
    lambda_temporal: float
    check_every: int
    lookback: int
    seed: int
    device: str = "cpu"
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.max_iter < 1 or self.check_every < 1 or self.max_iter < self.check_every:
            raise ValueError("max_iter must be at least check_every, and both must be positive")
        if self.lookback < 1 or self.torch_threads < 1:
            raise ValueError("lookback and torch_threads must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if min(self.lambda_group, self.lambda_ridge, self.lambda_temporal) < 0:
            raise ValueError("regularization strengths must be non-negative")


@dataclass
class FitResult:
    causal_strengths: NDArray[np.float32]
    causal_binary: NDArray[np.uint8]
    history: list[dict[str, float | int]]
    models: list[nn.Module]


def _prepare_windows(windows: NDArray[np.floating], config: TrainingConfig) -> Tensor:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("windows must have shape [window, time, series] with positive sizes")
    if not np.all(np.isfinite(values)):
        raise ValueError("windows contain non-finite values")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(config.torch_threads)
    return torch.as_tensor(values, device=device)


def fit_original_cmlp(
    windows: NDArray[np.floating],
    lag: int,
    hidden: list[int],
    activation: str,
    config: TrainingConfig,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    values = _prepare_windows(windows, config)
    if not 1 <= lag < values.shape[1]:
        raise ValueError("lag must be smaller than the window length")

    strengths = []
    binary = []
    history: list[dict[str, float | int]] = []
    models: list[nn.Module] = []
    for window_index, window in enumerate(values):
        if progress_callback is not None:
            progress_callback(
                {"event": "window_start", "window": window_index + 1, "total": len(values)}
            )
        torch.manual_seed(config.seed + window_index)
        model = cMLP(values.shape[2], lag, hidden, activation).to(values.device)
        losses = train_model_ista(
            model,
            window.unsqueeze(0),
            lr=config.learning_rate,
            max_iter=config.max_iter,
            lam=config.lambda_group,
            lam_ridge=config.lambda_ridge,
            penalty="GL",
            lookback=config.lookback,
            check_every=config.check_every,
            verbose=0,
        )
        with torch.no_grad():
            strengths.append(model.GC(threshold=False).cpu().numpy().astype(np.float32))
            binary.append(model.GC(threshold=True).cpu().numpy().astype(np.uint8))
        history.extend(
            {
                "window": window_index,
                "iteration": step * config.check_every,
                "objective_per_target": float(loss.detach().cpu()),
            }
            for step, loss in enumerate(losses, start=1)
        )
        models.append(model)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "window_complete",
                    "window": window_index + 1,
                    "total": len(values),
                    "iteration": len(losses) * config.check_every,
                    "objective": float(losses[-1]),
                }
            )

    return FitResult(np.stack(strengths), np.stack(binary), history, models)


def _smooth_terms(
    model: TemporallyCoupledCMLP,
    predictors: Tensor,
    targets: Tensor,
    config: TrainingConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    predictions = model(predictors)
    prediction_loss = (predictions - targets).square().mean(dim=(1, 2)).sum()
    ridge_loss = config.lambda_ridge * model.ridge_penalty()
    temporal_loss = config.lambda_temporal * model.temporal_penalty()
    return prediction_loss, ridge_loss, temporal_loss


def fit_temporal_cmlp(
    windows: NDArray[np.floating],
    lag: int,
    hidden: list[int],
    activation: str,
    config: TrainingConfig,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    values = _prepare_windows(windows, config)
    if not 1 <= lag < values.shape[1]:
        raise ValueError("lag must be smaller than the window length")
    torch.manual_seed(config.seed)
    model = TemporallyCoupledCMLP(
        num_windows=values.shape[0],
        num_series=values.shape[2],
        lag=lag,
        hidden=hidden,
        activation=activation,
    ).to(values.device)
    predictors = values[:, None, :-1, :]
    targets = values[:, None, lag:, :]
    history: list[dict[str, float | int]] = []
    best_value = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_checks = 0

    for iteration in range(1, config.max_iter + 1):
        model.zero_grad(set_to_none=True)
        prediction_loss, ridge_loss, temporal_loss = _smooth_terms(
            model, predictors, targets, config
        )
        smooth_loss = prediction_loss + ridge_loss + temporal_loss
        if not torch.isfinite(smooth_loss):
            raise FloatingPointError(f"non-finite training loss at iteration {iteration}")
        smooth_loss.backward()

        with torch.no_grad():
            for parameter in model.first_parameters():
                parameter.add_(parameter.grad, alpha=-config.learning_rate)
            for parameter in model.shared_parameters():
                parameter.add_(
                    parameter.grad,
                    alpha=-config.learning_rate / model.num_windows,
                )
            if config.lambda_group > 0:
                for window_model in model.window_models:
                    for network in window_model.networks:
                        prox_update(network, config.lambda_group, config.learning_rate, "GL")

        if iteration % config.check_every == 0 or iteration == config.max_iter:
            with torch.no_grad():
                prediction_loss, ridge_loss, temporal_loss = _smooth_terms(
                    model, predictors, targets, config
                )
                group_loss = config.lambda_group * model.group_penalty()
                objective = prediction_loss + ridge_loss + temporal_loss + group_loss
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"non-finite objective at iteration {iteration}")
                objective_value = float(objective)
                history.append(
                    {
                        "iteration": iteration,
                        "objective_per_target_window": objective_value
                        / (model.p * model.num_windows),
                        "prediction_loss": float(prediction_loss),
                        "ridge_loss": float(ridge_loss),
                        "temporal_loss": float(temporal_loss),
                        "group_loss": float(group_loss),
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "iteration",
                            "iteration": iteration,
                            "total": config.max_iter,
                            "objective": history[-1]["objective_per_target_window"],
                        }
                    )
                if objective_value < best_value:
                    best_value = objective_value
                    best_state = {
                        name: parameter.detach().clone()
                        for name, parameter in model.state_dict().items()
                    }
                    stale_checks = 0
                else:
                    stale_checks += 1
            if stale_checks >= config.lookback:
                break

    if best_state is None:
        raise RuntimeError("training did not record a model state")
    model.load_state_dict(best_state)
    with torch.no_grad():
        strengths = model.GC(threshold=False).cpu().numpy().astype(np.float32)
        binary = model.GC(threshold=True).cpu().numpy().astype(np.uint8)
    return FitResult(strengths, binary, history, [model])
