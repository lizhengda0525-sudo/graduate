from pathlib import Path
from typing import Any

from tc_cmlp.analysis.metrics import edge_metrics
from tc_cmlp.config import (
    load_yaml,
    parse_model_config,
    parse_training_config,
    resolve_config_path,
)
from tc_cmlp.data.synthetic import generate_dynamic_nonlinear_data
from tc_cmlp.io import prepare_output_dir, save_fit_result, save_json
from tc_cmlp.pipelines.common import train_method
from tc_cmlp.visualization import plot_synthetic_results


def run_synthetic_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_config = config["data"]
    dataset = generate_dynamic_nonlinear_data(seed=config["seed"], **data_config)
    model_config = parse_model_config(config)
    training_config = parse_training_config(config)
    output_dir = prepare_output_dir(resolve_config_path(config, config["output_dir"]))

    metrics_by_method: dict[str, dict[str, float]] = {}
    matrices_by_method = {}
    for method in config["methods"]:
        result = train_method(
            method=method,
            windows=dataset.windows,
            lag=data_config["lag"],
            model_config=model_config,
            training_config=training_config,
        )
        save_fit_result(output_dir, method, result)
        metrics_by_method[method] = edge_metrics(dataset.causal_matrices, result.causal_matrices)
        matrices_by_method[method] = result.causal_matrices

    save_json(output_dir / "metrics.json", metrics_by_method)
    plot_synthetic_results(
        true_matrices=dataset.causal_matrices,
        predicted_by_method=matrices_by_method,
        change_window=dataset.change_window,
        output_path=output_dir / "synthetic_results.png",
    )
    return metrics_by_method
