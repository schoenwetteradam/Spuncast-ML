from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spuncast_ml.dataset import create_splits, load_latest_export
from spuncast_ml.db import ensure_dir


def model_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_MODEL_DIR", "./artifacts/models"))


def report_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_REPORT_DIR", "./reports/generated"))


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = list(frame.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_rules_baseline(frame: pd.DataFrame) -> pd.Series:
    alert_columns = [
        "chem_not_ok_flag",
        "has_any_chem_alert",
        "tap_temp_missing",
        "pour_temp_missing",
        "die_temp_missing",
        "die_rpm_missing",
    ]
    available = [column for column in alert_columns if column in frame.columns]
    if not available:
        return pd.Series(0, index=frame.index, dtype=int)
    return (frame[available].fillna(0).sum(axis=1) > 0).astype(int)


def collect_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_score is not None and y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def train_model() -> dict[str, Path]:
    frame, export_path = load_latest_export()
    split_paths = create_splits(frame)

    x_train = pd.read_parquet(split_paths["x_train"])
    x_valid = pd.read_parquet(split_paths["x_valid"])
    y_train = pd.read_parquet(split_paths["y_train"])["scrap_flag"]
    y_valid = pd.read_parquet(split_paths["y_valid"])["scrap_flag"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    validation_scores = pd.Series(pipeline.predict_proba(x_valid)[:, 1], index=x_valid.index)
    validation_predictions = (validation_scores >= 0.5).astype(int)
    baseline_predictions = build_rules_baseline(x_valid)

    trained_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir() / f"scrap_baseline_{trained_at}.joblib"
    metadata_path = model_dir() / f"scrap_baseline_{trained_at}.json"

    model_metrics = collect_metrics(y_valid, validation_predictions, validation_scores)
    baseline_metrics = collect_metrics(y_valid, baseline_predictions)

    metadata = {
        "trained_at_utc": trained_at,
        "source_export": str(export_path),
        "model_type": "LogisticRegression",
        "target_column": "scrap_flag",
        "validation_metrics": model_metrics,
        "rules_baseline_metrics": baseline_metrics,
        "promotion_gate": {
            "baseline_recall": baseline_metrics["recall"],
            "model_recall": model_metrics["recall"],
            "baseline_f1": baseline_metrics["f1"],
            "model_f1": model_metrics["f1"],
            "passes": model_metrics["recall"] >= baseline_metrics["recall"] and model_metrics["f1"] > baseline_metrics["f1"],
        },
    }

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model_path,
        "metadata": metadata_path,
    }


def evaluate_latest_model() -> Path:
    frame, export_path = load_latest_export()
    split_paths = create_splits(frame)

    model_candidates = sorted(model_dir().glob("scrap_baseline_*.joblib"))
    if not model_candidates:
        raise FileNotFoundError("No trained model found. Run `spuncast-ml train` first.")
    latest_model = model_candidates[-1]
    pipeline: Pipeline = joblib.load(latest_model)

    x_test = pd.read_parquet(split_paths["x_test"])
    y_test = pd.read_parquet(split_paths["y_test"])["scrap_flag"]

    scores = pd.Series(pipeline.predict_proba(x_test)[:, 1], index=x_test.index)
    predictions = (scores >= 0.5).astype(int)
    baseline_predictions = build_rules_baseline(x_test)

    evaluation = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_export": str(export_path),
        "model_path": str(latest_model),
        "test_metrics": collect_metrics(y_test, predictions, scores),
        "rules_baseline_test_metrics": collect_metrics(y_test, baseline_predictions),
    }

    output_path = report_dir() / f"evaluation_{evaluation['evaluated_at_utc']}.json"
    output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return output_path

