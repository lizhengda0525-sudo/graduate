import math

import torch
from torch import Tensor, nn


class TemporallyCoupledCMLP(nn.Module):
    def __init__(
        self,
        num_windows: int,
        num_series: int,
        lag: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(num_windows, num_series, lag, hidden_dim) < 1:
            raise ValueError("all model dimensions must be positive")

        self.num_windows = num_windows
        self.num_series = num_series
        self.lag = lag
        self.hidden_dim = hidden_dim

        self.first_weight = nn.Parameter(
            torch.empty(num_windows, num_series, hidden_dim, num_series, lag)
        )
        self.first_bias = nn.Parameter(torch.zeros(num_windows, num_series, hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(num_series, hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(num_series))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        first_bound = 1.0 / math.sqrt(self.num_series * self.lag)
        output_bound = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.first_weight, -first_bound, first_bound)
        nn.init.uniform_(self.output_weight, -output_bound, output_bound)
        nn.init.zeros_(self.first_bias)
        nn.init.zeros_(self.output_bias)

    def forward_window(self, predictors: Tensor, window_index: int) -> Tensor:
        if predictors.ndim != 3:
            raise ValueError("predictors must have shape [sample, source, lag]")
        if predictors.shape[1:] != (self.num_series, self.lag):
            raise ValueError("predictor dimensions do not match the model")
        if not 0 <= window_index < self.num_windows:
            raise IndexError("window_index is outside the model range")

        first_weight = self.first_weight[window_index]
        first_bias = self.first_bias[window_index]
        hidden = torch.relu(torch.einsum("bsk,thsk->bth", predictors, first_weight) + first_bias)
        return torch.einsum("bth,th->bt", hidden, self.output_weight) + self.output_bias

    def causal_scores(self) -> Tensor:
        return torch.linalg.vector_norm(self.first_weight, dim=(2, 4))

    def group_penalty(self) -> Tensor:
        return self.causal_scores().mean()

    def temporal_penalty(self) -> Tensor:
        scores = self.causal_scores()
        if self.num_windows == 1:
            return scores.new_zeros(())
        return torch.abs(scores[1:] - scores[:-1]).mean()

    def shared_weight_penalty(self) -> Tensor:
        return self.output_weight.square().mean()

    @torch.no_grad()
    def apply_group_proximal(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if threshold == 0:
            return

        norms = torch.linalg.vector_norm(self.first_weight, dim=(2, 4), keepdim=True)
        scale = torch.clamp(1.0 - threshold / norms.clamp_min(1e-12), min=0.0)
        self.first_weight.mul_(scale)
