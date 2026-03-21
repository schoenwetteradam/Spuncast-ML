from __future__ import annotations

import argparse
import json
from pathlib import Path

from spuncast_ml.dataset import export_snapshot
from spuncast_ml.modeling import evaluate_latest_model, train_model


def command_export() -> None:
    frame, path = export_snapshot()
    print(json.dumps({"rows": int(len(frame)), "export_path": str(path)}, indent=2))


def command_train() -> None:
    outputs = train_model()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


def command_evaluate() -> None:
    path = evaluate_latest_model()
    print(json.dumps({"report_path": str(path)}, indent=2))


def command_pipeline() -> None:
    command_export()
    command_train()
    command_evaluate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spuncast ML workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("export", help="Export the upstream ML dataset view")
    subparsers.add_parser("train", help="Train the first-pass baseline model")
    subparsers.add_parser("evaluate", help="Evaluate the latest trained model")
    subparsers.add_parser("pipeline", help="Run export, train, and evaluate")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        command_export()
    elif args.command == "train":
        command_train()
    elif args.command == "evaluate":
        command_evaluate()
    elif args.command == "pipeline":
        command_pipeline()
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
