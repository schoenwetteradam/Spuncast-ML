#!/usr/bin/env python3
"""Near-real-time scoring daemon for early re-melt decisions.

Polls the ``v_ml_heat_early_score_v1`` view for heats that have not yet been
scored, runs inference with the ``early_remelt_decision`` feature set, and
writes results to the ``ml_heat_scores`` table in the operations database.

Heats scoring above ``REMELT_THRESHOLD`` are additionally inserted into
``heat_recommendations`` as hold candidates.

Usage
-----
Run as a long-lived background process::

    python scripts/score_heat_live.py

Environment variables (all optional, with defaults):

    SCORE_POLL_INTERVAL_SEC   – seconds between poll cycles  (default 180)
    SCORE_HORIZON_HOURS       – look-back window for new pours (default 8)
    SCORE_REMELT_THRESHOLD    – probability above which a heat is flagged
                                (default 0.65)
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure the repo root is on ``sys.path`` so that ``spuncast_ml`` is
# importable even when the package has not been installed globally.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spuncast_ml.db import get_conn  # noqa: E402
from spuncast_ml.inference import (  # noqa: E402
    _load_latest_model_and_metadata,
    _prepare_inference_features,
    _recommend_action,
)

POLL_INTERVAL_SEC: int = int(os.environ.get("SCORE_POLL_INTERVAL_SEC", "180"))
SCORE_HORIZON_HOURS: int = int(os.environ.get("SCORE_HORIZON_HOURS", "8"))
REMELT_THRESHOLD: float = float(os.environ.get("SCORE_REMELT_THRESHOLD", "0.65"))
FEATURE_SET: str = "early_remelt_decision"

logger = logging.getLogger("score_heat_live")


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def get_unscored_heats(conn: Any) -> pd.DataFrame:
    """Return rows from the early-score view that have no entry in ``ml_heat_scores``."""
    sql = (
        "SELECT v.* "
        "FROM v_ml_heat_early_score_v1 v "
        "LEFT JOIN ml_heat_scores s ON s.heat_number = v.heat_number "
        "WHERE s.heat_number IS NULL "
        "  AND v.pour_date >= NOW() - INTERVAL '%s hours'"
    ) % SCORE_HORIZON_HOURS
    return pd.read_sql_query(sql, conn)


def _top_contributing_feature(row: pd.Series, feature_columns: list[str]) -> str | None:
    """Heuristic: return the feature with the highest absolute z-value."""
    best_col: str | None = None
    best_val: float = -1.0
    for col in feature_columns:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            abs_val = abs(float(val))
            if abs_val > best_val:
                best_val = abs_val
                best_col = col
    return best_col


def insert_remelt_recommendation(cursor: Any, heat_number: str, probability: float, primary_driver: str | None) -> None:
    """Insert a hold recommendation for a high-risk heat."""
    cursor.execute(
        "INSERT INTO heat_recommendations "
        "(heat_number, decision_code, recommendation_text, primary_driver, scrap_probability, feature_set, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, NOW()) "
        "ON CONFLICT (heat_number) DO UPDATE "
        "SET scrap_probability = EXCLUDED.scrap_probability, "
        "    primary_driver = EXCLUDED.primary_driver, "
        "    created_at = NOW()",
        (
            heat_number,
            "HOLD",
            "Re-melt candidate \u2014 high scrap probability before blast",
            primary_driver,
            probability,
            FEATURE_SET,
        ),
    )


def write_scores(conn: Any, raw_frame: pd.DataFrame, scored: pd.DataFrame, feature_columns: list[str], model_version: str | None) -> None:
    """Persist scores to ``ml_heat_scores`` and optionally ``heat_recommendations``."""
    with conn.cursor() as cur:
        for idx, row in scored.iterrows():
            probability = _safe_float(row["scrap_probability"])
            if probability is None:
                continue
            cur.execute(
                "INSERT INTO ml_heat_scores "
                "(heat_number, scrap_probability, predicted_flag, "
                " recommended_action, feature_set, model_version, scored_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, NOW()) "
                "ON CONFLICT (heat_number) DO UPDATE "
                "SET scrap_probability = EXCLUDED.scrap_probability, "
                "    predicted_flag = EXCLUDED.predicted_flag, "
                "    recommended_action = EXCLUDED.recommended_action, "
                "    model_version = EXCLUDED.model_version, "
                "    scored_at = NOW()",
                (
                    row["heat_number"],
                    probability,
                    int(row["predicted_scrap_flag"]),
                    row["recommended_action"],
                    FEATURE_SET,
                    model_version,
                ),
            )
            if probability >= REMELT_THRESHOLD:
                raw_row = raw_frame.loc[idx] if idx in raw_frame.index else row
                driver = _top_contributing_feature(raw_row, feature_columns)
                insert_remelt_recommendation(cur, row["heat_number"], probability, driver)
    conn.commit()


def scoring_loop() -> None:
    """Single poll-score-write cycle."""
    model_path, pipeline, model_metadata = _load_latest_model_and_metadata(FEATURE_SET)
    model_version = model_path.stem

    with get_conn() as conn:
        df = get_unscored_heats(conn)
        if df.empty:
            logger.debug("No unscored heats found.")
            return

        x = _prepare_inference_features(df, FEATURE_SET)
        feature_columns = list(x.columns)

        scores = pd.Series(pipeline.predict_proba(x)[:, 1], index=df.index, name="scrap_probability")
        predictions = (scores >= REMELT_THRESHOLD).astype(int).rename("predicted_scrap_flag")
        recommendations = scores.apply(lambda v: _recommend_action(float(v), REMELT_THRESHOLD)).rename("recommended_action")

        scored = pd.DataFrame(
            {
                "heat_number": df["heat_number"],
                "scrap_probability": scores,
                "predicted_scrap_flag": predictions,
                "recommended_action": recommendations,
            },
            index=df.index,
        )

        write_scores(conn, df, scored, feature_columns, model_version)
        logger.info("Scored %d heats (%d flagged)", len(scored), int(predictions.sum()))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info(
        "Starting live scoring daemon  poll=%ds  horizon=%dh  threshold=%.2f",
        POLL_INTERVAL_SEC,
        SCORE_HORIZON_HOURS,
        REMELT_THRESHOLD,
    )

    while True:
        try:
            scoring_loop()
        except FileNotFoundError as exc:
            logger.error("Model not found — train first: %s", exc)
        except Exception:
            logger.exception("Scoring loop error")
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
