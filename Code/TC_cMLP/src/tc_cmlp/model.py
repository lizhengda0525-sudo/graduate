from collections.abc import Iterator

import torch
from torch import Tensor, nn

from tc_cmlp.official import cMLP, regularize, ridge_regularize


class TemporallyCoupledCMLP(nn.Module):
    def __init__(
        self,
        num_windows: int,
        num_series: int,
        lag: int,
        hidden: list[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if min(num_windows, num_series, lag) < 1:
            raise ValueError("num_windows, num_series and lag must be positive")
        if not hidden or min(hidden) < 1:
            raise ValueError("hidden must contain positive layer widths")

        self.num_windows = num_windows
        self.p = num_series
        self.lag = lag
        self.window_models = nn.ModuleList(
            [cMLP(num_series, lag, hidden, activation) for _ in range(num_windows)]
        )

        reference = self.window_models[0]
        for window_model in self.window_models[1:]:
            for target_index in range(num_series):
                reference_network = reference.networks[target_index]
                current_network = window_model.networks[target_index]
                current_network.layers[0].load_state_dict(reference_network.layers[0].state_dict())
                for layer_index in range(1, len(reference_network.layers)):
                    current_network.layers[layer_index] = reference_network.layers[layer_index]

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 4:
            raise ValueError("values must have shape [window, batch, time, series]")
        if values.shape[0] != self.num_windows or values.shape[-1] != self.p:
            raise ValueError("values do not match model dimensions")
        if values.shape[2] < self.lag:
            raise ValueError("time length must be at least lag")
        return torch.stack(
            [model(values[index]) for index, model in enumerate(self.window_models)]
        )

    def GC(self, threshold: bool = True, ignore_lag: bool = True) -> Tensor:
        return torch.stack(
            [
                model.GC(threshold=threshold, ignore_lag=ignore_lag)
                for model in self.window_models
            ]
        )

    def first_parameters(self) -> Iterator[nn.Parameter]:
        for model in self.window_models:
            for network in model.networks:
                yield from network.layers[0].parameters()

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        for network in self.window_models[0].networks:
            for layer in network.layers[1:]:
                yield from layer.parameters()

    def group_penalty(self) -> Tensor:
        return sum(
            regularize(network, 1.0, "GL")
            for model in self.window_models
            for network in model.networks
        )

    def ridge_penalty(self) -> Tensor:
        return sum(
            ridge_regularize(network, 1.0)
            for network in self.window_models[0].networks
        )

    def temporal_penalty(self) -> Tensor:
        scores = self.GC(threshold=False)
        if self.num_windows == 1:
            return scores.new_zeros(())
        return torch.abs(scores[1:] - scores[:-1]).sum()
