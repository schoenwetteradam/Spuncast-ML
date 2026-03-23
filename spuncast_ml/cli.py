from __future__ import annotations

import argparse
import json

from spuncast_ml.dataset import DEFAULT_FEATURE_SET, FEATURE_SET_EXCLUSIONS, export_snapshot
from spuncast_ml.modeling import evaluate_latest_model, train_model


def command_export() -> None:
    frame, path, metadata_path = export_snapshot()
    print(json.dumps({"rows": int(len(frame)), "export_path": str(path), "metadata_path": str(metadata_path)}, indent=2))


def command_train(feature_set: str, threshold: float) -> None:
    outputs = train_model(feature_set=feature_set, threshold=threshold)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


def command_evaluate(feature_set: str, threshold: float) -> None:
    path = evaluate_latest_model(feature_set=feature_set, threshold=threshold)
    print(json.dumps({"report_path": str(path)}, indent=2))


def command_pipeline(feature_set: str, threshold: float) -> None:
    command_export()
    command_train(feature_set=feature_set, threshold=threshold)
    command_evaluate(feature_set=feature_set, threshold=threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spuncast ML workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("export", help="Export the upstream ML dataset view")

    train_parser = subparsers.add_parser("train", help="Train baseline candidate models")
    train_parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_EXCLUSIONS), default=DEFAULT_FEATURE_SET)
    train_parser.add_argument("--threshold", type=float, default=0.5)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the latest trained model")
    evaluate_parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_EXCLUSIONS), default=DEFAULT_FEATURE_SET)
    evaluate_parser.add_argument("--threshold", type=float, default=0.5)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run export, train, and evaluate")
    pipeline_parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_EXCLUSIONS), default=DEFAULT_FEATURE_SET)
    pipeline_parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        command_export()
    elif args.command == "train":
        command_train(feature_set=args.feature_set, threshold=args.threshold)
    elif args.command == "evaluate":
        command_evaluate(feature_set=args.feature_set, threshold=args.threshold)
    elif args.command == "pipeline":
        command_pipeline(feature_set=args.feature_set, threshold=args.threshold)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
