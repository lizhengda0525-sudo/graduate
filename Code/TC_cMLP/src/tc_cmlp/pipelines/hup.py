from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tc_cmlp.analysis.metrics import soz_metrics
from tc_cmlp.analysis.network import network_flows, predict_soz
from tc_cmlp.config import (
    load_yaml,
    parse_model_config,
    parse_training_config,
    resolve_config_path,
)
from tc_cmlp.data.hup import load_hup_recording
from tc_cmlp.io import prepare_output_dir, save_fit_result, save_json
from tc_cmlp.pipelines.common import train_method
from tc_cmlp.visualization import plot_hup_results


def run_hup_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_config = dict(config["data"])
    dataset_root = resolve_config_path(config, data_config.pop("dataset_root"))
    lag = int(data_config.pop("lag"))
    recording = load_hup_recording(dataset_root=dataset_root, **data_config)

    result = train_method(
        method=config["method"],
        windows=recording.windows,
        lag=lag,
        model_config=parse_model_config(config),
        training_config=parse_training_config(config),
    )
    output_dir = prepare_output_dir(resolve_config_path(config, config["output_dir"]))
    save_fit_result(output_dir, config["method"], result)

    outflow, inflow, netflow = network_flows(result.causal_matrices)
    predicted_soz, abnormal_counts, mean_outflow = predict_soz(
        outflow=outflow,
        ictal_mask=recording.ictal_mask,
        top_fraction=float(config["soz_top_fraction"]),
    )
    metrics = soz_metrics(recording.soz_mask, predicted_soz)
    np.savez_compressed(
        output_dir / "network_features.npz",
        outflow=outflow,
        inflow=inflow,
        netflow=netflow,
        window_start_seconds=recording.window_start_seconds,
        ictal_mask=recording.ictal_mask,
    )
    pd.DataFrame(
        {
            "channel": recording.channel_names,
            "clinical_soz": recording.soz_mask,
            "predicted_soz": predicted_soz,
            "abnormal_window_count": abnormal_counts,
            "mean_ictal_outflow": mean_outflow,
        }
    ).sort_values(["abnormal_window_count", "mean_ictal_outflow"], ascending=False).to_csv(
        output_dir / "soz_ranking.csv", index=False
    )
    save_json(output_dir / "metrics.json", metrics)
    plot_hup_results(
        causal_matrices=result.causal_matrices,
        outflow=outflow,
        window_starts=recording.window_start_seconds,
        channel_names=recording.channel_names,
        soz_mask=recording.soz_mask,
        output_dir=output_dir,
    )
    return metrics
