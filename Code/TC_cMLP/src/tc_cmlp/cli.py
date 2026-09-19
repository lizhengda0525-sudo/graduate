import argparse
from pathlib import Path

from tc_cmlp.experiment import preprocess_hup, run_hup, run_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description="Original cMLP and TC-cMLP experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("synthetic", "preprocess-hup", "hup"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        run_synthetic(args.config)
    elif args.command == "preprocess-hup":
        path = preprocess_hup(args.config)
        print(f"预处理完成：{path}")
    elif args.command == "hup":
        run_hup(args.config)


if __name__ == "__main__":
    main()
