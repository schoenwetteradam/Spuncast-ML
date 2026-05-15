#!/usr/bin/env python3
"""Near-real-time scoring daemon for early re-melt decisions.

Polls the ``v_ml_heat_early_score_v1`` view for heats that have not yet been
scored, runs inference with the ``early_remelt_decision`` feature set, and
writes results to the ``ml_heat_scores`` table in the operations database.

Heats scoring above ``REMELT_THRESHOLD`` are additionally inserted into
``heat_recommendations`` as hold candidates.

Optional SHAP summaries (``SCORE_ENABLE_SHAP=1``) are written to
``ml_heat_scores.explanation_json`` when that column exists (see
``sql/073_ml_heat_scores_explanation.sql``).

Usage
-----
Run as a long-lived background process::

    python scripts/score_heat_live.py

Environment variables (all optional, with defaults):

    SCORE_POLL_INTERVAL_SEC   – seconds between poll cycles  (default 180)
    SCORE_HORIZON_HOURS       – look-back window for new pours (default 8)
    SCORE_REMELT_THRESHOLD    – probability above which a heat is flagged
                                (default 0.65)
    SCORE_ENABLE_SHAP         – set to 1/true to compute SHAP summaries
    SCORE_SHAP_MAX_HEATS      – max rows per cycle to explain (default 10)
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg2.extras import Json

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
ENABLE_SHAP: bool = os.environ.get("SCORE_ENABLE_SHAP", "").strip().lower() in {"1", "true", "yes"}
SHAP_MAX_HEATS: int = int(os.environ.get("SCORE_SHAP_MAX_HEATS", "10"))

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


def _build_shap_explainer(pipeline: Any, x: pd.DataFrame) -> Any | None:
    if not ENABLE_SHAP or len(x) < 2:
        return None
    try:
        import shap  # type: ignore[import-untyped]

        n_bg = min(64, len(x))
        background = x.sample(n=n_bg, random_state=42)

        def _positive_class_proba(frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(pipeline.predict_proba(frame)[:, 1], dtype=float)

        return shap.Explainer(_positive_class_proba, background)
    except Exception:
        logger.exception("Failed to initialise SHAP explainer; continuing without explanations")
        return None


def _shap_top_features(explainer: Any, row_frame: pd.DataFrame, top_k: int = 5) -> dict[str, float] | None:
    try:
        values_obj = explainer(row_frame)
        vals = np.asarray(values_obj.values)
        if vals.ndim > 1:
            vals = vals[0]
        cols = list(row_frame.columns)
        if len(cols) != len(vals.ravel()):
            return None
        flat = vals.ravel()
        pairs = sorted(zip(cols, flat), key=lambda item: abs(float(item[1])), reverse=True)[:top_k]
        return {str(name): float(val) for name, val in pairs}
    except Exception:
        logger.debug("SHAP row explanation failed", exc_info=True)
        return None


def _format_shap_note(payload: dict[str, float] | None) -> str | None:
    if not payload:
        return None
    bits = [f"{name} ({value:+.3f})" for name, value in list(payload.items())[:3]]
    return "SHAP drivers: " + ", ".join(bits)


def insert_remelt_recommendation(
    cursor: Any,
    heat_number: str,
    probability: float,
    primary_driver: str | None,
    extra_note: str | None = None,
) -> None:
    """Insert a hold recommendation for a high-risk heat."""
    text = "Re-melt candidate \u2014 high scrap probability before blast"
    if extra_note:
        text = f"{text} \u2014 {extra_note}"[:4000]
    cursor.execute(
        "INSERT INTO heat_recommendations "
        "(heat_number, decision_code, recommendation_text, primary_driver, scrap_probability, feature_set, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, NOW()) "
        "ON CONFLICT (heat_number) DO UPDATE "
        "SET scrap_probability = EXCLUDED.scrap_probability, "
        "    primary_driver = EXCLUDED.primary_driver, "
        "    recommendation_text = EXCLUDED.recommendation_text, "
        "    created_at = NOW()",
        (
            heat_number,
            "HOLD",
            text,
            primary_driver,
            probability,
            FEATURE_SET,
        ),
    )


def _ml_scores_has_explanation_column(conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = ANY (current_schemas(false)) "
            "  AND table_name = 'ml_heat_scores' "
            "  AND column_name = 'explanation_json' "
            "LIMIT 1"
        )
        return cur.fetchone() is not None


def write_scores(
    conn: Any,
    raw_frame: pd.DataFrame,
    scored: pd.DataFrame,
    feature_columns: list[str],
    model_version: str | None,
    x_features: pd.DataFrame,
    explainer: Any | None,
    shap_index_allowlist: set[Any],
) -> None:
    """Persist scores to ``ml_heat_scores`` and optionally ``heat_recommendations``."""
    include_json = _ml_scores_has_explanation_column(conn)
    if ENABLE_SHAP and not include_json:
        logger.warning("SCORE_ENABLE_SHAP is set but explanation_json column is missing; apply sql/073_ml_heat_scores_explanation.sql")

    with conn.cursor() as cur:
        for idx, row in scored.iterrows():
            probability = _safe_float(row["scrap_probability"])
            if probability is None:
                continue

            shap_payload: dict[str, Any] | None = None
            if include_json and explainer is not None and idx in shap_index_allowlist:
                top_map = _shap_top_features(explainer, x_features.loc[[idx]])
                if top_map:
                    shap_payload = {
                        "top_features": top_map,
                        "model_version": model_version,
                    }

            if include_json:
                cur.execute(
                    "INSERT INTO ml_heat_scores "
                    "(heat_number, scrap_probability, predicted_flag, "
                    " recommended_action, feature_set, model_version, scored_at, explanation_json) "
                    "VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s) "
                    "ON CONFLICT (heat_number) DO UPDATE "
                    "SET scrap_probability = EXCLUDED.scrap_probability, "
                    "    predicted_flag = EXCLUDED.predicted_flag, "
                    "    recommended_action = EXCLUDED.recommended_action, "
                    "    model_version = EXCLUDED.model_version, "
                    "    scored_at = NOW(), "
                    "    explanation_json = COALESCE(EXCLUDED.explanation_json, ml_heat_scores.explanation_json)",
                    (
                        row["heat_number"],
                        probability,
                        int(row["predicted_scrap_flag"]),
                        row["recommended_action"],
                        FEATURE_SET,
                        model_version,
                        Json(shap_payload) if shap_payload is not None else None,
                    ),
                )
            else:
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
                tops = shap_payload.get("top_features") if isinstance(shap_payload, dict) else None
                note = _format_shap_note(tops) if isinstance(tops, dict) else None
                insert_remelt_recommendation(cur, row["heat_number"], probability, driver, note)
    conn.commit()


def scoring_loop() -> None:
    """Single poll-score-write cycle."""
    model_path, pipeline, _model_metadata = _load_latest_model_and_metadata(FEATURE_SET)
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

        explainer = _build_shap_explainer(pipeline, x) if ENABLE_SHAP else None
        order = scored["scrap_probability"].sort_values(ascending=False).index.tolist()
        shap_allow = set(order[: max(0, SHAP_MAX_HEATS)])

        write_scores(conn, df, scored, feature_columns, model_version, x, explainer, shap_allow)
        logger.info("Scored %d heats (%d flagged)", len(scored), int(predictions.sum()))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info(
        "Starting live scoring daemon  poll=%ds  horizon=%dh  threshold=%.2f  shap=%s",
        POLL_INTERVAL_SEC,
        SCORE_HORIZON_HOURS,
        REMELT_THRESHOLD,
        ENABLE_SHAP,
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
