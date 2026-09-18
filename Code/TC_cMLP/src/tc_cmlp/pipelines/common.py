from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from tc_cmlp.config import ModelConfig, TrainingConfig
from tc_cmlp.models.training import FitResult, fit_independent_cmlp, fit_tc_cmlp


def train_method(
    method: str,
    windows: NDArray[np.floating],
    lag: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> FitResult:
    if method == "independent":
        return fit_independent_cmlp(windows, lag, model_config, training_config)
    if method == "tc_no_temporal":
        return fit_tc_cmlp(
            windows,
            lag,
            model_config,
            replace(training_config, lambda_temporal=0.0),
        )
    if method == "tc":
        return fit_tc_cmlp(windows, lag, model_config, training_config)
    raise ValueError(f"Unknown method: {method}")
