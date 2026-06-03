from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

try:
    from lightgbm import LGBMClassifier
    import lightgbm as _lgbm_module
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from spuncast_ml.dataset import DEFAULT_FEATURE_SET, create_splits, load_latest_export
from spuncast_ml.db import ensure_dir


DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0


def find_cost_optimal_threshold(
    y_valid: pd.Series,
    valid_scores: pd.Series,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> tuple[float, float]:
    """Search validation set for the threshold that minimises FN*fn_cost + FP*fp_cost.

    A missing scrap (false negative) is much costlier than an unnecessary hold
    (false positive), so the default 5:1 ratio skews the threshold lower than 0.5.
    Returns (best_threshold, best_cost).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_cost = float(DEFAULT_DECISION_THRESHOLD), float("inf")
    for t in thresholds:
        preds = (valid_scores >= t).astype(int)
        fn = int(((y_valid == 1) & (preds == 0)).sum())
        fp = int(((y_valid == 0) & (preds == 1)).sum())
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t, best_cost


def compute_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_MODEL_DIR", "./artifacts/models"))


def report_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_REPORT_DIR", "./reports/generated"))


def build_preprocessor(frame: pd.DataFrame, dense_output: bool = False) -> ColumnTransformer:
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
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output),
                        ),
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
        "has_open_data_quality_violation",
        "open_data_quality_violation_count",
    ]
    available = [column for column in alert_columns if column in frame.columns]
    if not available:
        return pd.Series(0, index=frame.index, dtype=int)
    alerts = frame[available].fillna(0)
    return (alerts.sum(axis=1) > 0).astype(int)


def collect_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_score is not None and y_true.nunique() > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def build_candidate_pipelines(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    class_counts = y_train.value_counts()
    calibration_folds = min(3, int(class_counts.min())) if not class_counts.empty else 0

    candidates: dict[str, Pipeline] = {
        "logistic_regression_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train, dense_output=False)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    if calibration_folds >= 2:
        candidates["calibrated_hist_gradient_boosting"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train, dense_output=True)),
                (
                    "model",
                    CalibratedClassifierCV(
                        estimator=HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_depth=6,
                            max_iter=300,
                            random_state=42,
                        ),
                        cv=calibration_folds,
                        method="sigmoid",
                    ),
                ),
            ]
        )

    if _LGBM_AVAILABLE and calibration_folds >= 2:
        candidates["lightgbm_balanced"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train, dense_output=True)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=20,
                        class_weight="balanced",
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )

    return candidates


def promotion_gate(model_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    passes = (
        model_metrics["recall"] >= baseline_metrics["recall"]
        and model_metrics["false_negatives"] <= baseline_metrics["false_negatives"]
        and model_metrics.get("pr_auc", 0.0) >= baseline_metrics.get("pr_auc", 0.0)
    )
    return {
        "baseline_recall": baseline_metrics["recall"],
        "model_recall": model_metrics["recall"],
        "baseline_false_negatives": baseline_metrics["false_negatives"],
        "model_false_negatives": model_metrics["false_negatives"],
        "baseline_pr_auc": baseline_metrics.get("pr_auc"),
        "model_pr_auc": model_metrics.get("pr_auc"),
        "passes": passes,
        "rule": "Model must match or beat rules baseline on recall, not increase false negatives, and match or improve PR-AUC before promotion.",
    }


def train_model(
    feature_set: str = DEFAULT_FEATURE_SET,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> dict[str, Path]:
    frame, export_path, export_metadata = load_latest_export()
    split_paths = create_splits(frame, feature_set=feature_set)

    x_train = pd.read_parquet(split_paths["x_train"])
    x_valid = pd.read_parquet(split_paths["x_valid"])
    y_train = pd.read_parquet(split_paths["y_train"])["scrap_flag"]
    y_valid = pd.read_parquet(split_paths["y_valid"])["scrap_flag"]

    if y_train.nunique() < 2:
        raise ValueError("Training split must contain both scrap and non-scrap classes to fit a classifier.")

    candidates = build_candidate_pipelines(x_train, y_train)
    candidate_results: dict[str, dict[str, Any]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    baseline_predictions = build_rules_baseline(x_valid)
    baseline_metrics = collect_metrics(y_valid, baseline_predictions, baseline_predictions)

    for candidate_name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        validation_scores = pd.Series(pipeline.predict_proba(x_valid)[:, 1], index=x_valid.index)
        validation_predictions = (validation_scores >= threshold).astype(int)
        metrics = collect_metrics(y_valid, validation_predictions, validation_scores)
        candidate_results[candidate_name] = {
            "validation_metrics": metrics,
            "promotion_gate": promotion_gate(metrics, baseline_metrics),
        }
        trained_pipelines[candidate_name] = pipeline

    best_candidate_name = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["validation_metrics"]["recall"],
            -candidate_results[name]["validation_metrics"]["false_negatives"],
            candidate_results[name]["validation_metrics"].get("pr_auc", 0.0),
        ),
    )
    best_pipeline = trained_pipelines[best_candidate_name]

    # Optimise decision threshold on validation set using the cost ratio.
    # A missed scrap (FN) costs fn_cost times more than an unnecessary hold (FP).
    optimized_threshold = threshold
    optimized_cost: float | None = None
    if y_valid.nunique() >= 2:
        best_valid_scores = pd.Series(best_pipeline.predict_proba(x_valid)[:, 1], index=x_valid.index)
        optimized_threshold, optimized_cost = find_cost_optimal_threshold(
            y_valid, best_valid_scores, fn_cost=fn_cost, fp_cost=fp_cost
        )

    trained_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir() / f"scrap_baseline_{feature_set}_{trained_at}.joblib"
    metadata_path = model_dir() / f"scrap_baseline_{feature_set}_{trained_at}.json"

    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "runtime_executable": sys.executable,
    }
    if _LGBM_AVAILABLE:
        env["lightgbm_version"] = _lgbm_module.__version__

    metadata = {
        "trained_at_utc": trained_at,
        "source_export": str(export_path),
        "source_export_metadata": export_metadata,
        "feature_set": feature_set,
        "decision_threshold": optimized_threshold,
        "decision_threshold_fallback": threshold,
        "threshold_optimization": {
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "optimized_threshold": optimized_threshold,
            "optimized_validation_cost": optimized_cost,
        },
        "selected_model": best_candidate_name,
        "candidate_models": candidate_results,
        "target_column": "scrap_flag",
        "rules_baseline_metrics": baseline_metrics,
        "promotion_gate": candidate_results[best_candidate_name]["promotion_gate"],
        "model_sha256": None,
        "training_environment": env,
    }

    joblib.dump(best_pipeline, model_path)
    metadata["model_sha256"] = compute_file_sha256(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model": model_path,
        "metadata": metadata_path,
    }


def evaluate_latest_model(feature_set: str = DEFAULT_FEATURE_SET, threshold: float = DEFAULT_DECISION_THRESHOLD) -> Path:
    frame, export_path, export_metadata = load_latest_export()
    split_paths = create_splits(frame, feature_set=feature_set)

    model_candidates = sorted(model_dir().glob(f"scrap_baseline_{feature_set}_*.joblib"))
    if not model_candidates:
        raise FileNotFoundError(f"No trained model found for feature set '{feature_set}'. Run `spuncast-ml train --feature-set {feature_set}` first.")
    latest_model = model_candidates[-1]
    pipeline: Pipeline = joblib.load(latest_model)

    x_test = pd.read_parquet(split_paths["x_test"])
    y_test = pd.read_parquet(split_paths["y_test"])["scrap_flag"]

    scores = pd.Series(pipeline.predict_proba(x_test)[:, 1], index=x_test.index)
    predictions = (scores >= threshold).astype(int)
    baseline_predictions = build_rules_baseline(x_test)

    model_metrics = collect_metrics(y_test, predictions, scores)
    baseline_metrics = collect_metrics(y_test, baseline_predictions, baseline_predictions)

    evaluation = {
        "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_export": str(export_path),
        "source_export_metadata": export_metadata,
        "feature_set": feature_set,
        "decision_threshold": threshold,
        "model_path": str(latest_model),
        "test_metrics": model_metrics,
        "rules_baseline_test_metrics": baseline_metrics,
        "promotion_gate": promotion_gate(model_metrics, baseline_metrics),
    }

    output_path = report_dir() / f"evaluation_{feature_set}_{evaluation['evaluated_at_utc']}.json"
    output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    return output_path
