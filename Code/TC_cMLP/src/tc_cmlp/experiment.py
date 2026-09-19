import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from tc_cmlp.analysis import (
    change_point_summary,
    edge_metrics,
    network_flows,
    predict_soz,
    soz_metrics,
)
from tc_cmlp.data.hup_recording import HUPRecording
from tc_cmlp.data.synthetic import generate_synthetic_data
from tc_cmlp.official import REPOSITORY_ROOT
from tc_cmlp.training import (
    FitResult,
    ProgressCallback,
    TrainingConfig,
    fit_original_cmlp,
    fit_temporal_cmlp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def _output_dir(config: dict[str, Any]) -> Path:
    results_root = (PROJECT_ROOT / "results").resolve()
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    if not output_dir.is_relative_to(results_root):
        raise ValueError("output_dir must be inside the project results directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _preprocessed_hup_path(config: dict[str, Any]) -> Path:
    results_root = (PROJECT_ROOT / "results").resolve()
    path = (PROJECT_ROOT / config["preprocessed_path"]).resolve()
    if not path.is_relative_to(results_root):
        raise ValueError("preprocessed_path must be inside the project results directory")
    return path


def _hup_data_signature(config: dict[str, Any]) -> str:
    return json.dumps(config["data"], ensure_ascii=False, sort_keys=True)


def preprocess_hup(config_path: str | Path) -> Path:
    from tc_cmlp.data.hup import load_hup_recording

    config = _load_config(config_path)
    path = _preprocessed_hup_path(config)
    if path.exists():
        raise FileExistsError(f"preprocessed HUP data already exist: {path}")
    data_config = dict(config["data"])
    data_config.pop("lag")
    dataset_root = (PROJECT_ROOT / data_config.pop("dataset_root")).resolve()
    segment_path = data_config.pop("signal_segment_path", None)
    if segment_path is not None:
        data_config["signal_segment_path"] = (PROJECT_ROOT / segment_path).resolve()
    recording = load_hup_recording(dataset_root=dataset_root, progress=True, **data_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        windows=recording.windows,
        channel_names=np.asarray(recording.channel_names),
        soz_mask=recording.soz_mask,
        window_start_seconds=recording.window_start_seconds,
        ictal_mask=recording.ictal_mask,
        sampling_rate=np.asarray(recording.sampling_rate),
        data_signature=np.asarray(_hup_data_signature(config)),
    )
    path.with_suffix(".md").write_text(
        "\n".join(
            [
                "# HUP116 预处理记录",
                "",
                f"- Subject：{config['data']['subject']}，run：{config['data']['run']}。",
                f"- 时间窗口：{recording.windows.shape[0]}。",
                f"- 有效通道：{len(recording.channel_names)}。",
                f"- 临床 SOZ 通道：{int(recording.soz_mask.sum())}。",
                f"- 采样率：{recording.sampling_rate} Hz。",
                (
                    f"- 陷波滤波：{config['data']['notch_hz']} Hz，"
                    f"quality factor {config['data']['notch_quality_factor']}。"
                ),
                (
                    f"- 低通滤波：{config['data']['lowpass_hz']} Hz，"
                    f"Butterworth 阶数 {config['data']['lowpass_order']}。"
                ),
                f"- 数据形状：{recording.windows.shape}。",
                "- 处理参数保存在 HUP 配置文件及 NPZ 内的 data_signature 字段。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _load_preprocessed_hup(config: dict[str, Any]) -> HUPRecording:
    path = _preprocessed_hup_path(config)
    with np.load(path, allow_pickle=False) as data:
        if str(data["data_signature"]) != _hup_data_signature(config):
            raise ValueError("preprocessed HUP data do not match the data configuration")
        recording = HUPRecording(
            windows=data["windows"],
            channel_names=data["channel_names"].tolist(),
            soz_mask=data["soz_mask"],
            window_start_seconds=data["window_start_seconds"],
            ictal_mask=data["ictal_mask"],
            sampling_rate=float(data["sampling_rate"]),
        )
    if recording.windows.ndim != 3 or recording.windows.shape[2] != len(recording.channel_names):
        raise ValueError("preprocessed HUP data have incompatible dimensions")
    return recording


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _log(message: str) -> None:
    print(f"[{datetime.now().astimezone():%H:%M:%S}] {message}", flush=True)


def _duration(seconds: float) -> str:
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _save_run_metadata(output_dir: Path, config: dict[str, Any], revision: str) -> None:
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    _save_json(
        output_dir / "run_metadata.json",
        {
            "status": "running",
            "source_revision": revision,
            "source_directory": str(REPOSITORY_ROOT),
            "started_at": datetime.now().astimezone().isoformat(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    )


def _fit_method(
    method: str,
    windows: np.ndarray,
    lag: int,
    hidden: list[int],
    activation: str,
    training: TrainingConfig,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    if method == "cmlp":
        return fit_original_cmlp(
            windows, lag, hidden, activation, training, progress_callback
        )
    if method == "tc_no_temporal":
        return fit_temporal_cmlp(
            windows,
            lag,
            hidden,
            activation,
            replace(training, lambda_temporal=0.0),
            progress_callback,
        )
    if method == "tc":
        return fit_temporal_cmlp(
            windows, lag, hidden, activation, training, progress_callback
        )
    raise ValueError(f"unknown method: {method}")


def _fit_timed(
    method: str,
    windows: np.ndarray,
    lag: int,
    hidden: list[int],
    activation: str,
    training: TrainingConfig,
    progress_every: int = 10,
    heartbeat_seconds: float = 30.0,
) -> tuple[FitResult, dict[str, float | str]]:
    if progress_every < 1 or heartbeat_seconds <= 0:
        raise ValueError("progress_every and heartbeat_seconds must be positive")
    device = torch.device(training.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = datetime.now().astimezone().isoformat()
    start = perf_counter()
    stop_heartbeat = Event()
    print_lock = Lock()
    progress_state = {"value": f"窗口 0/{len(windows)}" if method == "cmlp" else
                      f"迭代 0/{training.max_iter}"}

    def log_progress(message: str) -> None:
        with print_lock:
            _log(f"[{method}] {message} | 已用 {_duration(perf_counter() - start)}")

    def progress_callback(progress: dict[str, float | int | str]) -> None:
        event = progress["event"]
        if event == "window_start":
            progress_state["value"] = f"窗口 {progress['window']}/{progress['total']} 训练中"
            if progress["window"] == 1:
                log_progress(progress_state["value"])
        elif event == "window_complete":
            progress_state["value"] = f"窗口 {progress['window']}/{progress['total']} 已完成"
            if (progress["window"] == 1 or progress["window"] % progress_every == 0
                    or progress["window"] == progress["total"]):
                log_progress(
                    f"{progress_state['value']} | 迭代 {progress['iteration']}"
                    f" | 目标函数/target {progress['objective']:.6f}"
                )
        elif event == "iteration":
            progress_state["value"] = f"迭代 {progress['iteration']}/{progress['total']}"
            log_progress(
                f"{progress_state['value']}"
                f" | 目标函数/target/window {progress['objective']:.6f}"
            )
        else:
            raise ValueError(f"unknown progress event: {event}")

    def heartbeat() -> None:
        while not stop_heartbeat.wait(heartbeat_seconds):
            log_progress(f"训练中 | {progress_state['value']}")

    _log(
        f"[{method}] 开始训练 | 窗口 {len(windows)} | 通道 {windows.shape[2]}"
        f" | max_iter {training.max_iter} | device {training.device}"
    )
    temporal_strength = training.lambda_temporal if method == "tc" else 0.0
    _log(
        f"[{method}] 参数 | lag={lag} hidden={hidden} lr={training.learning_rate}"
        f" lambda_group={training.lambda_group} lambda_ridge={training.lambda_ridge}"
        f" lambda_temporal={temporal_strength} check_every={training.check_every}"
        f" seed={training.seed}"
    )
    heartbeat_thread = Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = _fit_method(
            method, windows, lag, hidden, activation, training, progress_callback
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join()
    timing = {
        "training_seconds": perf_counter() - start,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    log_progress("训练完成")
    return result, timing


def _save_fit(output_dir: Path, method: str, result: FitResult) -> Path:
    method_dir = output_dir / method
    method_dir.mkdir(exist_ok=False)
    np.save(method_dir / "causal_strengths.npy", result.causal_strengths)
    np.save(method_dir / "causal_binary.npy", result.causal_binary)
    pd.DataFrame(result.history).to_csv(method_dir / "training_history.csv", index=False)
    torch.save([model.state_dict() for model in result.models], method_dir / "models.pt")
    return method_dir


def _complete_run(output_dir: Path) -> None:
    metadata_path = output_dir / "run_metadata.json"
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["status"] = "completed"
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    _save_json(metadata_path, metadata)


def run_synthetic(
    config_path: str | Path,
    output_dir_override: str | Path | None = None,
    progress_every: int = 10,
    heartbeat_seconds: float = 30.0,
) -> dict[str, dict[str, float | int]]:
    experiment_start = perf_counter()
    config = _load_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    training = TrainingConfig(**config["training"])
    dataset = generate_synthetic_data(seed=config["seed"], **config["data"])
    output_dir = _output_dir(config)
    revision = _source_revision()
    _save_run_metadata(output_dir, config, revision)
    _log(f"Synthetic 数据：{dataset.windows.shape} | 结果目录：{output_dir}")
    np.savez_compressed(
        output_dir / "synthetic_dataset.npz",
        windows=dataset.windows,
        true_causal_matrices=dataset.causal_matrices,
        change_window=np.asarray(dataset.change_window),
    )

    metrics_by_method: dict[str, dict[str, float | int]] = {}
    changes_by_method: dict[str, dict[str, float | int]] = {}
    training_times: dict[str, dict[str, float | str]] = {}
    for method in config["methods"]:
        result, timing = _fit_timed(
            method,
            dataset.windows,
            config["data"]["lag"],
            config["model"]["hidden"],
            config["model"]["activation"],
            training,
            progress_every,
            heartbeat_seconds,
        )
        training_times[method] = timing
        _save_json(output_dir / "training_times.json", training_times)
        _log(f"[{method}] 保存模型并计算评价指标")
        method_dir = _save_fit(output_dir, method, result)
        metrics = edge_metrics(
            dataset.causal_matrices, result.causal_strengths, result.causal_binary
        )
        changes = change_point_summary(result.causal_strengths, dataset.change_window)
        metrics_by_method[method] = metrics
        changes_by_method[method] = changes
        _save_json(method_dir / "metrics.json", metrics)
        _save_json(method_dir / "change_summary.json", changes)
        _log(
            f"[{method}] Edge AUPRC {metrics['edge_auprc']:.4f}"
            f" | Binary F1 {metrics['binary_f1']:.4f}"
            f" | 训练时间 {timing['training_seconds']:.2f} 秒"
        )

    _save_json(output_dir / "metrics.json", metrics_by_method)
    _save_json(output_dir / "change_summary.json", changes_by_method)
    notes = [
        "# Synthetic 实验记录",
        "",
        f"- Neural-GC 原仓库 commit：`{revision}`。",
        f"- 数据随机种子：{config['seed']}。",
        f"- 比较方法：{', '.join(config['methods'])}。",
        "- cmlp 直接使用原仓库 `cMLP`、`train_model_ista` 和 `GC`。",
        "- TC-cMLP 使用原仓库 cMLP 预测网络与 GL 近端更新。",
        "- Edge AUPRC 使用全部窗口的非对角边强度。",
        "- binary precision、recall 和 F1 使用 `GC(threshold=True)`。",
        "- Causal strength error 使用每个窗口归一化后的非对角边强度。",
        "- 因果矩阵方向：target × source。",
        (
            "- 训练耗时包含数据准备、模型构建、模型训练和因果关系提取；"
            "不包含结果保存与指标计算。CUDA 训练包含同步等待。"
        ),
        "",
    ]
    for method in config["methods"]:
        metrics = metrics_by_method[method]
        changes = changes_by_method[method]
        notes.extend(
            [
                f"## {method}",
                "",
                f"- Edge AUPRC：{metrics['edge_auprc']:.6f}。",
                f"- Binary precision：{metrics['binary_precision']:.6f}。",
                f"- Binary recall：{metrics['binary_recall']:.6f}。",
                f"- Binary F1：{metrics['binary_f1']:.6f}。",
                f"- Causal strength error：{metrics['causal_strength_error']:.6f}。",
                f"- 训练耗时：{training_times[method]['training_seconds']:.3f} 秒。",
                f"- 二值预测边数：{metrics['predicted_edges']}。",
                f"- 最大变化对应窗口：{changes['peak_transition_window']}。",
                "",
            ]
        )
    (output_dir / "实验备注.md").write_text("\n".join(notes), encoding="utf-8")
    _complete_run(output_dir)
    _log(
        f"Synthetic 实验完成 | 总耗时 {_duration(perf_counter() - experiment_start)}"
        f" | 结果目录：{output_dir}"
    )
    return metrics_by_method


def run_hup(
    config_path: str | Path,
    output_dir_override: str | Path | None = None,
    progress_every: int = 10,
    heartbeat_seconds: float = 30.0,
) -> dict[str, dict[str, float]]:
    experiment_start = perf_counter()
    config = _load_config(config_path)
    if output_dir_override is not None:
        config["output_dir"] = str(output_dir_override)
    training = TrainingConfig(**config["training"])
    lag = config["data"]["lag"]
    recording = _load_preprocessed_hup(config)
    output_dir = _output_dir(config)
    revision = _source_revision()
    _save_run_metadata(output_dir, config, revision)
    _log(
        f"HUP 数据：{recording.windows.shape[0]} 个窗口，"
        f"{recording.windows.shape[2]} 个通道，临床 SOZ {int(recording.soz_mask.sum())} 个"
    )
    _log(f"方法：{', '.join(config['methods'])} | 结果目录：{output_dir}")
    pd.DataFrame(
        {
            "window": np.arange(recording.windows.shape[0]),
            "start_seconds": recording.window_start_seconds,
            "ictal": recording.ictal_mask,
        }
    ).to_csv(output_dir / "window_metadata.csv", index=False)

    metrics_by_method: dict[str, dict[str, float]] = {}
    training_times: dict[str, dict[str, float | str]] = {}
    for method in config["methods"]:
        result, timing = _fit_timed(
            method,
            recording.windows,
            lag,
            config["model"]["hidden"],
            config["model"]["activation"],
            training,
            progress_every,
            heartbeat_seconds,
        )
        training_times[method] = timing
        _save_json(output_dir / "training_times.json", training_times)
        _log(f"[{method}] 保存模型并计算 SOZ 指标")
        method_dir = _save_fit(output_dir, method, result)
        outflow, inflow, netflow = network_flows(result.causal_strengths)
        predicted_soz, abnormal_counts, mean_outflow = predict_soz(
            outflow, recording.ictal_mask, config["soz_top_fraction"]
        )
        metrics = soz_metrics(recording.soz_mask, predicted_soz)
        metrics_by_method[method] = metrics
        np.savez_compressed(
            method_dir / "network_features.npz",
            outflow=outflow,
            inflow=inflow,
            netflow=netflow,
        )
        pd.DataFrame(
            {
                "channel": recording.channel_names,
                "clinical_soz": recording.soz_mask,
                "predicted_soz": predicted_soz,
                "abnormal_window_count": abnormal_counts,
                "mean_ictal_outflow": mean_outflow,
            }
        ).sort_values(
            ["abnormal_window_count", "mean_ictal_outflow"], ascending=False
        ).to_csv(method_dir / "soz_ranking.csv", index=False)
        _save_json(method_dir / "metrics.json", metrics)
        _log(
            f"[{method}] SOZ Precision {metrics['precision']:.4f}"
            f" | Sensitivity {metrics['sensitivity']:.4f}"
            f" | F1 {metrics['f1']:.4f}"
            f" | 训练时间 {timing['training_seconds']:.2f} 秒"
        )

    _save_json(output_dir / "metrics.json", metrics_by_method)
    (output_dir / "实验备注.md").write_text(
        "\n".join(
            [
                "# HUP iEEG 实验记录",
                "",
                f"- Neural-GC 原仓库 commit：`{revision}`。",
                f"- Subject：{config['data']['subject']}，run：{config['data']['run']}。",
                f"- 有效窗口：{recording.windows.shape[0]}。",
                f"- 有效 channel：{len(recording.channel_names)}。",
                "- Outflow 排名忽略因果矩阵对角线。",
                "- 临床 SOZ 标记只用于结果评价。",
                (
                    "- 训练耗时包含数据准备、模型构建、模型训练和因果关系提取；"
                    "不包含结果保存与指标计算。CUDA 训练包含同步等待。"
                ),
                *(f"- {method} 训练耗时：{training_times[method]['training_seconds']:.3f} 秒。"
                  for method in config["methods"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    _complete_run(output_dir)
    _log(
        f"HUP 实验完成 | 总耗时 {_duration(perf_counter() - experiment_start)}"
        f" | 结果目录：{output_dir}"
    )
    return metrics_by_method
