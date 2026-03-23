from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spuncast_ml.dataset import DEFAULT_FEATURE_SET
from spuncast_ml.db import ensure_dir
from spuncast_ml.modeling import model_dir


def feedback_dir() -> Path:
    return ensure_dir(os.environ.get("SPUNCAST_ML_FEEDBACK_DIR", "./data/feedback"))


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_model_path(feature_set: str) -> Path | None:
    candidates = sorted(model_dir().glob(f"scrap_baseline_{feature_set}_*.joblib"))
    return candidates[-1] if candidates else None


def record_operator_feedback(
    heat_number: str,
    recommendation: str,
    accepted: bool,
    feature_set: str = DEFAULT_FEATURE_SET,
    score: float | None = None,
    operator_id: str | None = None,
    note: str | None = None,
    actual_scrap_flag: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    if actual_scrap_flag is not None and actual_scrap_flag not in (0, 1):
        raise ValueError("actual_scrap_flag must be 0, 1, or omitted.")

    latest_model = _latest_model_path(feature_set=feature_set)
    entry = {
        "feedback_timestamp_utc": _timestamp_now(),
        "feature_set": feature_set,
        "heat_number": heat_number,
        "model_path": str(latest_model) if latest_model else None,
        "recommendation": recommendation,
        "accepted": bool(accepted),
        "score": float(score) if score is not None else None,
        "operator_id": operator_id,
        "note": note,
        "actual_scrap_flag": actual_scrap_flag,
    }

    file_path = feedback_dir() / "operator_feedback.jsonl"
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return file_path, entry
