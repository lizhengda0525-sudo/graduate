import argparse
import json
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tc-cmlp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser("synthetic")
    synthetic_parser.add_argument("--config", required=True)

    hup_parser = subparsers.add_parser("hup")
    hup_parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "synthetic":
        from tc_cmlp.pipelines.synthetic import run_synthetic_experiment

        result = run_synthetic_experiment(arguments.config)
    else:
        from tc_cmlp.pipelines.hup import run_hup_experiment

        result = run_hup_experiment(arguments.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
