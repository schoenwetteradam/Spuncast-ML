from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from spuncast_ml.db import ensure_dir, fetch_dataframe, source_view

TARGET_COLUMN = "scrap_flag"

LEAKAGE_COLUMNS = {
    "scrap_flag",
    "has_scrap",
    "scrap_event_count",
    "scrap_event_quantity",
    "scrap_weight_lbs",
    "scrap_estimated_cost",
    "reason_code",
    "defect_type",
    "department",
    "total_recorded_scrap_qty",
    "scrap_rate_pct",
    "latest_scrap_ts",
    "quantity_scrapped",
}

IDENTIFIER_COLUMNS = {
    "heat_id",
    "heat_number",
}

HIGH_CARDINALITY_TEXT_COLUMNS = {
    "job_number",
    "operator_id",
    "heat_treat_operator_id",
    "melter",
}


def export_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_EXPORT_DIR", "./data/exports"))


def processed_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_PROCESSED_DIR", "./data/processed"))


def export_snapshot() -> tuple[pd.DataFrame, Path]:
    frame = fetch_dataframe(f"SELECT * FROM {source_view()}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = export_dir() / f"ml_heat_dataset_{stamp}.parquet"
    frame.to_parquet(output, index=False)
    return frame, output


def load_latest_export() -> tuple[pd.DataFrame, Path]:
    candidates = sorted(export_dir().glob("ml_heat_dataset_*.parquet"))
    if not candidates:
        raise FileNotFoundError("No exported dataset snapshot found. Run `spuncast-ml export` first.")
    latest = candidates[-1]
    return pd.read_parquet(latest), latest


def build_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = {TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise KeyError(f"Required target column missing from dataset: {sorted(missing)}")

    y = frame[TARGET_COLUMN].astype(int)

    drop_columns = sorted(
        {column for column in frame.columns if column in LEAKAGE_COLUMNS or column in IDENTIFIER_COLUMNS or column in HIGH_CARDINALITY_TEXT_COLUMNS}
    )
    x = frame.drop(columns=drop_columns, errors="ignore").copy()
    return x, y


def create_splits(frame: pd.DataFrame) -> dict[str, Path]:
    x, y = build_feature_frame(frame)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp if y_temp.nunique() > 1 else None,
    )

    output_dir = processed_dir()
    outputs = {
        "x_train": output_dir / "x_train.parquet",
        "x_valid": output_dir / "x_valid.parquet",
        "x_test": output_dir / "x_test.parquet",
        "y_train": output_dir / "y_train.parquet",
        "y_valid": output_dir / "y_valid.parquet",
        "y_test": output_dir / "y_test.parquet",
        "metadata": output_dir / "split_metadata.json",
    }

    x_train.to_parquet(outputs["x_train"], index=False)
    x_valid.to_parquet(outputs["x_valid"], index=False)
    x_test.to_parquet(outputs["x_test"], index=False)
    y_train.to_frame(TARGET_COLUMN).to_parquet(outputs["y_train"], index=False)
    y_valid.to_frame(TARGET_COLUMN).to_parquet(outputs["y_valid"], index=False)
    y_test.to_frame(TARGET_COLUMN).to_parquet(outputs["y_test"], index=False)

    metadata = {
        "source_rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "test_rows": int(len(x_test)),
        "feature_columns": list(x.columns),
        "dropped_columns": sorted(
            [column for column in frame.columns if column not in x.columns and column != TARGET_COLUMN]
        ),
        "target_column": TARGET_COLUMN,
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs

