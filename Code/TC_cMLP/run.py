import argparse
import sys
from pathlib import Path

from tc_cmlp.experiment import preprocess_hup, run_hup, run_synthetic

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行 cMLP 与 TC-cMLP 实验")
    parser.add_argument(
        "--task",
        choices=("hup", "synthetic", "preprocess-hup"),
        default="hup",
        help="实验任务，默认运行 HUP 实验",
    )
    parser.add_argument("--config", type=Path, help="YAML 配置文件")
    parser.add_argument("--output-dir", type=Path, help="本次训练的结果目录，位于 results 内")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="原始 cMLP 每完成多少个窗口显示一次进度，默认 10",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="训练期间的最长状态提示间隔，默认 30 秒",
    )
    args = parser.parse_args()

    if args.progress_every < 1:
        parser.error("--progress-every 必须大于零")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds 必须大于零")
    config = args.config or PROJECT_ROOT / "configs" / (
        "synthetic.yaml" if args.task == "synthetic" else "hup116.yaml"
    )
    if args.task == "preprocess-hup":
        if args.output_dir is not None:
            parser.error("预处理任务不使用 --output-dir")
        path = preprocess_hup(config)
        print(f"预处理完成：{path}", flush=True)
    elif args.task == "synthetic":
        run_synthetic(config, args.output_dir, args.progress_every, args.heartbeat_seconds)
    else:
        run_hup(config, args.output_dir, args.progress_every, args.heartbeat_seconds)


if __name__ == "__main__":
    main()
