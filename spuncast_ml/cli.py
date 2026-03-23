from __future__ import annotations

import argparse
import json

from spuncast_ml.dataset import DEFAULT_FEATURE_SET, FEATURE_SET_EXCLUSIONS, export_snapshot
from spuncast_ml.feedback import record_operator_feedback
from spuncast_ml.inference import score_dataset
from spuncast_ml.modeling import evaluate_latest_model, train_model
from spuncast_ml.monitoring import generate_drift_report


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


def command_score(feature_set: str, threshold: float, input_path: str | None, output_path: str | None) -> None:
    outputs = score_dataset(feature_set=feature_set, threshold=threshold, input_path=input_path, output_path=output_path)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


def command_monitor_drift(
    feature_set: str,
    psi_threshold: float,
    categorical_tvd_threshold: float,
    unseen_category_threshold: float,
) -> None:
    path = generate_drift_report(
        feature_set=feature_set,
        psi_threshold=psi_threshold,
        categorical_tvd_threshold=categorical_tvd_threshold,
        unseen_category_threshold=unseen_category_threshold,
    )
    print(json.dumps({"report_path": str(path)}, indent=2))


def command_feedback(
    feature_set: str,
    heat_number: str,
    recommendation: str,
    accepted: bool,
    score: float | None,
    operator_id: str | None,
    note: str | None,
    actual_scrap_flag: int | None,
) -> None:
    path, entry = record_operator_feedback(
        feature_set=feature_set,
        heat_number=heat_number,
        recommendation=recommendation,
        accepted=accepted,
        score=score,
        operator_id=operator_id,
        note=note,
        actual_scrap_flag=actual_scrap_flag,
    )
    print(json.dumps({"feedback_path": str(path), "entry": entry}, indent=2))


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

    score_parser = subparsers.add_parser("score", help="Score a dataset with the latest trained model")
    score_parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_EXCLUSIONS), default=DEFAULT_FEATURE_SET)
    score_parser.add_argument("--threshold", type=float, default=0.5)
    score_parser.add_argument("--input-path", help="Optional parquet input path. Defaults to latest export snapshot.")
    score_parser.add_argument("--output-path", help="Optional parquet output path for scored rows.")

    monitor_parser = subparsers.add_parser("monitor-drift", help="Compare current data to model training snapshot")
    monitor_parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_EXCLUSIONS), default=DEFAULT_FEATURE_SET)
    monitor_parser.add_argument("--psi-threshold", type=float, default=0.2)
    monitor_parser.add_argument("--categorical-tvd-threshold", type=float, default=0.2)
    monitor_parser.add_argument("--unseen-category-threshold", type=float, default=0.05)

    feedback_parser = subparsers.add_parser("feedback", help="Capture operator response to a recommendation")
    feedback_parser.add_argument("--feature-set", choices=sorted(FEATURE_SET_EXCLUSIONS), default=DEFAULT_FEATURE_SET)
    feedback_parser.add_argument("--heat-number", required=True)
    feedback_parser.add_argument("--recommendation", required=True)
    feedback_parser.add_argument("--score", type=float)
    feedback_parser.add_argument("--operator-id")
    feedback_parser.add_argument("--note")
    feedback_parser.add_argument("--actual-scrap-flag", type=int, choices=[0, 1])
    outcome_group = feedback_parser.add_mutually_exclusive_group(required=True)
    outcome_group.add_argument("--accepted", action="store_true")
    outcome_group.add_argument("--rejected", action="store_true")
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
    elif args.command == "score":
        command_score(
            feature_set=args.feature_set,
            threshold=args.threshold,
            input_path=args.input_path,
            output_path=args.output_path,
        )
    elif args.command == "monitor-drift":
        command_monitor_drift(
            feature_set=args.feature_set,
            psi_threshold=args.psi_threshold,
            categorical_tvd_threshold=args.categorical_tvd_threshold,
            unseen_category_threshold=args.unseen_category_threshold,
        )
    elif args.command == "feedback":
        command_feedback(
            feature_set=args.feature_set,
            heat_number=args.heat_number,
            recommendation=args.recommendation,
            accepted=bool(args.accepted and not args.rejected),
            score=args.score,
            operator_id=args.operator_id,
            note=args.note,
            actual_scrap_flag=args.actual_scrap_flag,
        )
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
