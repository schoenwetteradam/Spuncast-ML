from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from spuncast_ml.dataset import (
    DEFAULT_FEATURE_SET,
    FEATURE_SET_EXCLUSIONS,
    HIGH_CARDINALITY_TEXT_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    load_latest_export,
)
from spuncast_ml.db import ensure_dir
from spuncast_ml.modeling import DEFAULT_DECISION_THRESHOLD, compute_file_sha256, model_dir


def score_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_SCORE_DIR", "./reports/generated"))


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_latest_model_and_metadata(feature_set: str) -> tuple[Path, Pipeline, dict[str, Any] | None]:
    model_candidates = sorted(model_dir().glob(f"scrap_baseline_{feature_set}_*.joblib"))
    if not model_candidates:
        raise FileNotFoundError(
            f"No trained model found for feature set '{feature_set}'. Run `spuncast-ml train --feature-set {feature_set}` first."
        )
    latest_model = model_candidates[-1]
    pipeline: Pipeline = joblib.load(latest_model)

    metadata_path = latest_model.with_suffix(".json")
    metadata: dict[str, Any] | None = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_sha256 = metadata.get("model_sha256")
        allow_unsigned = os.environ.get("SPUNCAST_ML_ALLOW_UNSIGNED_MODEL", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not expected_sha256:
            if not allow_unsigned:
                raise ValueError(
                    f"Model metadata {metadata_path} is missing 'model_sha256'. "
                    "Retrain the model or set SPUNCAST_ML_ALLOW_UNSIGNED_MODEL=1 for temporary compatibility."
                )
        else:
            actual_sha256 = compute_file_sha256(latest_model)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for model {latest_model}. "
                    "Model artifact integrity check failed; refusing to load."
                )
    return latest_model, pipeline, metadata


def _prepare_inference_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_EXCLUSIONS:
        raise ValueError(f"Unsupported feature set: {feature_set}. Expected one of {sorted(FEATURE_SET_EXCLUSIONS)}")

    excluded_columns = FEATURE_SET_EXCLUSIONS[feature_set]
    drop_columns = sorted(
        {
            column
            for column in frame.columns
            if column in excluded_columns
            or column in IDENTIFIER_COLUMNS
            or column in HIGH_CARDINALITY_TEXT_COLUMNS
            or column == TARGET_COLUMN
        }
    )
    x = frame.drop(columns=drop_columns, errors="ignore").copy()
    for column in x.select_dtypes(include=["datetime", "datetimetz"]).columns:
        x[column] = pd.to_datetime(x[column], errors="coerce", utc=True).astype("int64") / 1_000_000_000
        x.loc[x[column] < 0, column] = pd.NA
    return x


def _recommend_action(score: float, threshold: float) -> str:
    if score >= max(threshold, 0.8):
        return "hold_for_operator_review"
    if score >= threshold:
        return "increase_monitoring"
    return "continue_standard_run"


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def score_dataset(
    feature_set: str = DEFAULT_FEATURE_SET,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    input_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Path]:
    if input_path:
        frame = pd.read_parquet(input_path)
        source_path = Path(input_path).resolve()
    else:
        frame, source_path, _ = load_latest_export()

    latest_model_path, pipeline, model_metadata = _load_latest_model_and_metadata(feature_set=feature_set)
    x = _prepare_inference_features(frame, feature_set=feature_set)

    scores = pd.Series(pipeline.predict_proba(x)[:, 1], index=frame.index, name="scrap_probability")
    predictions = (scores >= threshold).astype(int).rename("predicted_scrap_flag")
    recommendations = scores.apply(lambda value: _recommend_action(float(value), threshold)).rename("recommended_action")

    output = pd.DataFrame(index=frame.index)
    for column in ("heat_number", "analysis_date"):
        if column in frame.columns:
            output[column] = frame[column]
    output["scrap_probability"] = scores
    output["predicted_scrap_flag"] = predictions
    output["recommended_action"] = recommendations

    stamp = _timestamp_now()
    scored_path = Path(output_path).resolve() if output_path else score_dir() / f"scored_{feature_set}_{stamp}.parquet"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(scored_path, index=False)

    summary = {
        "scored_at_utc": stamp,
        "feature_set": feature_set,
        "decision_threshold": threshold,
        "rows_scored": int(len(output)),
        "source_path": str(source_path),
        "model_path": str(latest_model_path),
        "model_selection": model_metadata.get("selected_model") if model_metadata else None,
        "action_counts": output["recommended_action"].value_counts(dropna=False).to_dict(),
        "average_scrap_probability": _safe_float(output["scrap_probability"].mean()) if len(output) else None,
        "max_scrap_probability": _safe_float(output["scrap_probability"].max()) if len(output) else None,
        "score_output_path": str(scored_path),
    }
    summary_path = score_dir() / f"score_summary_{feature_set}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"scores": scored_path, "summary": summary_path}
