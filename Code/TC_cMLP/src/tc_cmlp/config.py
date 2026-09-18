from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    lambda_group: float
    lambda_temporal: float
    lambda_weight: float
    seed: int
    log_every: int
    device: str
    sample_batch_size: int

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if min(self.lambda_group, self.lambda_temporal, self.lambda_weight) < 0:
            raise ValueError("regularization coefficients must be non-negative")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.sample_batch_size < 1:
            raise ValueError("sample_batch_size must be positive")


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    config["_config_dir"] = str(config_path.parent)
    return config


def parse_model_config(config: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**config["model"])


def parse_training_config(config: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**config["training"])


def resolve_config_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config["_config_dir"]) / path).resolve()
