from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from spuncast_ml.contract import DEFAULT_CONTRACT_VERSION, load_contract, validate_contract_columns
from spuncast_ml.db import ensure_dir, fetch_dataframe, source_view

TARGET_COLUMN = "scrap_flag"
TIME_COLUMN = "analysis_date"
DEFAULT_FEATURE_SET = "pre_pour_in_process"
EARLY_REMELT_SCORE_CONTRACT = "v_ml_heat_early_score_v1"

LEAKAGE_REVIEW_COLUMNS = {
    "quantity_scrapped",
    "scrap_event_count",
    "scrap_event_quantity",
    "scrap_weight_lbs",
    "scrap_estimated_cost",
    "total_recorded_scrap_qty",
    "reason_code",
    "defect_type",
    "department",
    "scrap_events_row_count",
    "scrap_events_qty_sum",
    "scrap_events_weight_lbs_sum",
    "scrap_events_last_ts",
    "scrap_distinct_reason_code_count",
    "scrap_distinct_defect_type_count",
    "scrap_reason_codes_csv",
    "scrap_defect_types_csv",
    "latest_decision_code",
    "latest_risk_score",
}

IDENTIFIER_COLUMNS = {
    "heat_id",
    "heat_number",
    "product_number",
}

HIGH_CARDINALITY_TEXT_COLUMNS = {
    "job_number",
    "operator_id",
    "heat_treat_operator_id",
    "melter",
}

PRE_POUR_IN_PROCESS_EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "has_scrap",
    "scrap_rate_pct",
    "latest_scrap_ts",
    "reason_code_bucket",
    *LEAKAGE_REVIEW_COLUMNS,
    "feature_cutoff_ts",
    # analysis_date is the chronological split key; not present in v_ml_heat_early_score_v1
    "analysis_date",
    "quantity_produced",
    "quantity_shipped",
    "quantity_on_hand",
    "quantity_on_hold",
    "latest_lot_date",
    "lot_count",
    "heat_end_ts",
    "heat_treat_start_ts",
    "heat_treat_end_ts",
    "equipment_code",
    "cycle_name",
    "setpoint_temp_f",
    "actual_temp_f",
    "heat_treat_temp_delta",
    # instruction_source is data-entry metadata, not a process signal
    "instruction_source",
}

POST_RUN_DIAGNOSTIC_EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "has_scrap",
    "feature_cutoff_ts",
    "latest_decision_code",
    "latest_risk_score",
}

EARLY_REMELT_DECISION_EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "has_scrap",
    "scrap_rate_pct",
    "feature_cutoff_ts",
    "quantity_scrapped",
    "scrap_event_count",
    "scrap_event_quantity",
    "scrap_weight_lbs",
    "scrap_estimated_cost",
    "total_recorded_scrap_qty",
    "scrap_events_row_count",
    "scrap_events_qty_sum",
    "scrap_events_weight_lbs_sum",
    "scrap_events_last_ts",
    "scrap_distinct_reason_code_count",
    "scrap_distinct_defect_type_count",
    "scrap_reason_codes_csv",
    "scrap_defect_types_csv",
    "reason_code",
    "defect_type",
    "department",
    "heat_treat_temp_delta",
    "setpoint_temp_f",
    "actual_temp_f",
    "cycle_name",
    "equipment_code",
    "quantity_shipped",
    "quantity_on_hold",
    "lot_count",
    "latest_lot_date",
    "latest_decision_code",
    "latest_risk_score",
    # instruction_source is data-entry metadata, not a process signal
    "instruction_source",
    # fps_die_number / fps_material_name duplicate die_no / alloy_grade — exclude to avoid
    # redundant one-hot dimensions; keep numeric fps_* targets and deviation columns
    "fps_die_number",
    "fps_material_name",
}

FEATURE_SET_EXCLUSIONS = {
    "pre_pour_in_process": PRE_POUR_IN_PROCESS_EXCLUDED_COLUMNS,
    "post_run_diagnostic": POST_RUN_DIAGNOSTIC_EXCLUDED_COLUMNS,
    "early_remelt_decision": EARLY_REMELT_DECISION_EXCLUDED_COLUMNS,
}


@dataclass(frozen=True)
class SnapshotMetadata:
    extraction_timestamp_utc: str
    source_query_hash: str
    source_view: str
    source_query: str
    contract_version: str
    schema_version_note: str
    row_count: int
    columns: list[str]
    data_path: str
    metadata_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_timestamp_utc": self.extraction_timestamp_utc,
            "source_query_hash": self.source_query_hash,
            "source_view": self.source_view,
            "source_query": self.source_query,
            "contract_version": self.contract_version,
            "schema_version_note": self.schema_version_note,
            "row_count": self.row_count,
            "columns": self.columns,
            "data_path": self.data_path,
            "metadata_path": self.metadata_path,
        }


def export_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_EXPORT_DIR", "./data/exports"))


def processed_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_PROCESSED_DIR", "./data/processed"))


def build_extraction_query() -> str:
    return (
        f"SELECT * FROM {source_view()} "
        f"WHERE {TIME_COLUMN} IS NOT NULL "
        f"ORDER BY {TIME_COLUMN}, heat_number"
    )


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def export_snapshot() -> tuple[pd.DataFrame, Path, Path]:
    query = build_extraction_query()
    frame = fetch_dataframe(query)
    contract = validate_contract_columns(frame.columns.tolist(), DEFAULT_CONTRACT_VERSION)

    stamp = _timestamp_now()
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    base_name = f"ml_heat_dataset_{stamp}_{query_hash}"
    parquet_path = export_dir() / f"{base_name}.parquet"
    metadata_path = export_dir() / f"{base_name}.json"

    frame.to_parquet(parquet_path, index=False)

    metadata = SnapshotMetadata(
        extraction_timestamp_utc=stamp,
        source_query_hash=query_hash,
        source_view=source_view(),
        source_query=query,
        contract_version=contract["contract_version"],
        schema_version_note=contract["schema_version_note"],
        row_count=int(len(frame)),
        columns=frame.columns.tolist(),
        data_path=str(parquet_path),
        metadata_path=str(metadata_path),
    )
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return frame, parquet_path, metadata_path


def load_latest_export() -> tuple[pd.DataFrame, Path, dict[str, Any] | None]:
    candidates = sorted(export_dir().glob("ml_heat_dataset_*.parquet"))
    if not candidates:
        raise FileNotFoundError("No exported dataset snapshot found. Run `spuncast-ml export` first.")
    latest = candidates[-1]
    frame = pd.read_parquet(latest)
    validate_contract_columns(frame.columns.tolist(), DEFAULT_CONTRACT_VERSION)

    metadata_candidate = latest.with_suffix(".json")
    metadata = None
    if metadata_candidate.exists():
        metadata = json.loads(metadata_candidate.read_text(encoding="utf-8"))
    return frame, latest, metadata


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if TIME_COLUMN not in frame.columns:
        raise KeyError(f"Required chronological split column missing from dataset: {TIME_COLUMN}")

    normalized = frame.copy()
    normalized[TIME_COLUMN] = pd.to_datetime(normalized[TIME_COLUMN], errors="coerce", utc=True)
    normalized = normalized.dropna(subset=[TIME_COLUMN]).sort_values([TIME_COLUMN, "heat_number"]).reset_index(drop=True)
    return normalized


def build_feature_frame(frame: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> tuple[pd.DataFrame, pd.Series]:
    if feature_set not in FEATURE_SET_EXCLUSIONS:
        raise ValueError(f"Unsupported feature set: {feature_set}. Expected one of {sorted(FEATURE_SET_EXCLUSIONS)}")

    missing = {TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise KeyError(f"Required target column missing from dataset: {sorted(missing)}")

    y = frame[TARGET_COLUMN].astype(int)

    excluded_columns = FEATURE_SET_EXCLUSIONS[feature_set]
    drop_columns = sorted(
        {
            column
            for column in frame.columns
            if column in excluded_columns
            or column in IDENTIFIER_COLUMNS
            or column in HIGH_CARDINALITY_TEXT_COLUMNS
        }
    )
    x = frame.drop(columns=drop_columns, errors="ignore").copy()
    if feature_set == "early_remelt_decision":
        early_allowed = set(load_contract(EARLY_REMELT_SCORE_CONTRACT)["expected_columns"])
        early_allowed.discard(TARGET_COLUMN)
        x = x[[column for column in x.columns if column in early_allowed]].copy()
    for column in x.select_dtypes(include=["datetime", "datetimetz"]).columns:
        x[column] = pd.to_datetime(x[column], errors="coerce", utc=True).astype("int64") / 1_000_000_000
        x.loc[x[column] < 0, column] = pd.NA
    for column in x.select_dtypes(include=["object"]).columns:
        sample = x[column].dropna().head(20)
        if sample.empty:
            continue
        if sample.apply(lambda v: isinstance(v, (_dt.date, _dt.datetime))).all():
            x[column] = pd.to_datetime(x[column], errors="coerce", utc=True).astype("int64") / 1_000_000_000
            x.loc[x[column] < 0, column] = pd.NA
    return x, y


def create_splits(frame: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> dict[str, Path]:
    normalized = _normalize_dates(frame)
    x, y = build_feature_frame(normalized, feature_set=feature_set)

    total_rows = len(normalized)
    if total_rows < 3:
        raise ValueError("Need at least 3 rows to create chronological train/validation/test splits.")

    test_size = max(1, round(total_rows * 0.15))
    valid_size = max(1, round(total_rows * 0.15))
    train_size = total_rows - valid_size - test_size
    if train_size < 1:
        train_size = 1
        valid_size = 1
        test_size = total_rows - train_size - valid_size

    train_end = train_size
    valid_end = train_end + valid_size

    x_train, x_valid, x_test = x.iloc[:train_end], x.iloc[train_end:valid_end], x.iloc[valid_end:]
    y_train, y_valid, y_test = y.iloc[:train_end], y.iloc[train_end:valid_end], y.iloc[valid_end:]

    output_dir = processed_dir()
    outputs = {
        "x_train": output_dir / f"x_train_{feature_set}.parquet",
        "x_valid": output_dir / f"x_valid_{feature_set}.parquet",
        "x_test": output_dir / f"x_test_{feature_set}.parquet",
        "y_train": output_dir / f"y_train_{feature_set}.parquet",
        "y_valid": output_dir / f"y_valid_{feature_set}.parquet",
        "y_test": output_dir / f"y_test_{feature_set}.parquet",
        "metadata": output_dir / f"split_metadata_{feature_set}.json",
    }

    x_train.to_parquet(outputs["x_train"], index=False)
    x_valid.to_parquet(outputs["x_valid"], index=False)
    x_test.to_parquet(outputs["x_test"], index=False)
    y_train.to_frame(TARGET_COLUMN).to_parquet(outputs["y_train"], index=False)
    y_valid.to_frame(TARGET_COLUMN).to_parquet(outputs["y_valid"], index=False)
    y_test.to_frame(TARGET_COLUMN).to_parquet(outputs["y_test"], index=False)

    metadata = {
        "feature_set": feature_set,
        "source_rows": int(len(frame)),
        "eligible_rows": int(total_rows),
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "test_rows": int(len(x_test)),
        "train_date_range": [str(normalized.iloc[0][TIME_COLUMN]), str(normalized.iloc[len(x_train) - 1][TIME_COLUMN])],
        "validation_date_range": [str(normalized.iloc[len(x_train)][TIME_COLUMN]), str(normalized.iloc[len(x_train) + len(x_valid) - 1][TIME_COLUMN])],
        "test_date_range": [str(normalized.iloc[len(x_train) + len(x_valid)][TIME_COLUMN]), str(normalized.iloc[-1][TIME_COLUMN])],
        "feature_columns": list(x.columns),
        "dropped_columns": sorted(
            [column for column in normalized.columns if column not in x.columns and column != TARGET_COLUMN]
        ),
        "target_column": TARGET_COLUMN,
        "time_column": TIME_COLUMN,
        "chronological_split_policy": "70/15/15 ordered by analysis_date then heat_number",
        "leakage_review_columns": sorted(LEAKAGE_REVIEW_COLUMNS),
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs
