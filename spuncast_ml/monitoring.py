from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from spuncast_ml.dataset import DEFAULT_FEATURE_SET, load_latest_export
from spuncast_ml.db import ensure_dir
from spuncast_ml.inference import _prepare_inference_features
from spuncast_ml.modeling import model_dir


def monitor_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_MONITOR_DIR", "./reports/generated"))


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if reference_values.empty or current_values.empty:
        return 0.0

    quantiles = [index / bins for index in range(1, bins)]
    edges = sorted(set(reference_values.quantile(quantiles).tolist()))
    if not edges:
        return 0.0

    reference_bins = pd.cut(reference_values, bins=[-float("inf"), *edges, float("inf")], include_lowest=True)
    current_bins = pd.cut(current_values, bins=[-float("inf"), *edges, float("inf")], include_lowest=True)
    reference_dist = reference_bins.value_counts(normalize=True, sort=False)
    current_dist = current_bins.value_counts(normalize=True, sort=False).reindex(reference_dist.index, fill_value=0.0)

    epsilon = 1e-6
    reference_dist = reference_dist.clip(lower=epsilon)
    current_dist = current_dist.clip(lower=epsilon)
    return float(((current_dist - reference_dist) * (current_dist / reference_dist).apply(lambda value: math.log(value))).sum())


def _categorical_tvd(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    ref = reference.fillna("__missing__").astype(str)
    cur = current.fillna("__missing__").astype(str)
    if ref.empty or cur.empty:
        return 0.0, 0.0

    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)
    categories = ref_dist.index.union(cur_dist.index)
    ref_aligned = ref_dist.reindex(categories, fill_value=0.0)
    cur_aligned = cur_dist.reindex(categories, fill_value=0.0)
    tvd = float((cur_aligned - ref_aligned).abs().sum() * 0.5)
    unseen_rate = float(cur_aligned.loc[~cur_aligned.index.isin(ref_dist.index)].sum())
    return tvd, unseen_rate


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _load_latest_model_metadata(feature_set: str) -> tuple[Path, dict[str, Any]]:
    model_candidates = sorted(model_dir().glob(f"scrap_baseline_{feature_set}_*.joblib"))
    if not model_candidates:
        raise FileNotFoundError(
            f"No trained model found for feature set '{feature_set}'. Run `spuncast-ml train --feature-set {feature_set}` first."
        )
    latest_model = model_candidates[-1]
    metadata_path = latest_model.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model metadata for {latest_model.name}: expected {metadata_path.name}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return latest_model, metadata


def generate_drift_report(
    feature_set: str = DEFAULT_FEATURE_SET,
    psi_threshold: float = 0.2,
    categorical_tvd_threshold: float = 0.2,
    unseen_category_threshold: float = 0.05,
) -> Path:
    latest_model, model_metadata = _load_latest_model_metadata(feature_set=feature_set)
    reference_export = Path(model_metadata["source_export"]).resolve()
    if not reference_export.exists():
        raise FileNotFoundError(f"Reference export from model metadata not found: {reference_export}")

    reference_frame = pd.read_parquet(reference_export)
    current_frame, current_export, current_export_metadata = load_latest_export()
    reference_x = _prepare_inference_features(reference_frame, feature_set=feature_set)
    current_x = _prepare_inference_features(current_frame, feature_set=feature_set)

    shared_columns = sorted(set(reference_x.columns).intersection(current_x.columns))
    numeric_metrics: dict[str, Any] = {}
    categorical_metrics: dict[str, Any] = {}
    high_drift_features: list[dict[str, Any]] = []

    for column in shared_columns:
        reference_series = reference_x[column]
        current_series = current_x[column]
        if pd.api.types.is_numeric_dtype(reference_series):
            psi_value = _psi(reference_series, current_series)
            metric = {
                "psi": psi_value,
                "reference_mean": _safe_float(pd.to_numeric(reference_series, errors="coerce").mean(skipna=True)),
                "current_mean": _safe_float(pd.to_numeric(current_series, errors="coerce").mean(skipna=True)),
            }
            numeric_metrics[column] = metric
            if psi_value >= psi_threshold:
                high_drift_features.append({"feature": column, "kind": "numeric", **metric})
        else:
            tvd, unseen_rate = _categorical_tvd(reference_series, current_series)
            metric = {"tvd": tvd, "unseen_category_rate": unseen_rate}
            categorical_metrics[column] = metric
            if tvd >= categorical_tvd_threshold or unseen_rate >= unseen_category_threshold:
                high_drift_features.append({"feature": column, "kind": "categorical", **metric})

    evaluated_at = _timestamp_now()
    report = {
        "evaluated_at_utc": evaluated_at,
        "feature_set": feature_set,
        "model_path": str(latest_model),
        "reference_export": str(reference_export),
        "current_export": str(current_export),
        "current_export_metadata": current_export_metadata,
        "row_counts": {
            "reference_rows": int(len(reference_x)),
            "current_rows": int(len(current_x)),
        },
        "thresholds": {
            "psi_threshold": psi_threshold,
            "categorical_tvd_threshold": categorical_tvd_threshold,
            "unseen_category_threshold": unseen_category_threshold,
        },
        "numeric_feature_metrics": numeric_metrics,
        "categorical_feature_metrics": categorical_metrics,
        "high_drift_features": high_drift_features,
        "high_drift_feature_count": int(len(high_drift_features)),
    }

    path = monitor_dir() / f"drift_report_{feature_set}_{evaluated_at}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
