#!/usr/bin/env python3
"""Near-real-time scoring daemon for early re-melt decisions.

Polls the ``v_ml_heat_early_score_v1`` view for heats that have not yet been
scored, runs inference with the ``early_remelt_decision`` feature set, and
writes results to the ``ml_heat_scores`` table in the operations database.

Heats at or above ``SCORE_REMELT_THRESHOLD`` are treated as positive scrap-risk
flags for ``ml_heat_scores``.  Rows can optionally be mirrored into
``heat_recommendations`` with Grafana-friendly ``decision_code`` values
(``HOLD`` / ``CAUTION`` / ``ADVISORY``) using ``SCORE_HOLD_THRESHOLD`` and
``SCORE_CAUTION_THRESHOLD``.  Set ``SCORE_WRITE_ADVISORY_ROWS=1`` to also insert
lower-probability advisory rows down to ``SCORE_ADVISORY_THRESHOLD``.

Optional SHAP summaries (``SCORE_ENABLE_SHAP=1``) are written to
``ml_heat_scores.explanation_json`` when that column exists (see
``sql/073_ml_heat_scores_explanation.sql``).

Usage
-----
Run as a long-lived background process::

    python scripts/score_heat_live.py

Read-only smoke test (no writes to ``ml_heat_scores`` or ``heat_recommendations``)::

    python scripts/score_heat_live.py --dry-run --once --test-heats 100

Environment variables (all optional, with defaults):

    SCORE_POLL_INTERVAL_SEC   – seconds between poll cycles  (default 180)
    SCORE_HORIZON_HOURS       – look-back window for new pours (default 8)
    SCORE_REMELT_THRESHOLD    – probability at/above which predicted_flag=1
                                (default 0.65)
    SCORE_HOLD_THRESHOLD      – decision_code HOLD when prob >= this (default 0.80)
    SCORE_CAUTION_THRESHOLD   – CAUTION band floor (default 0.65)
    SCORE_ADVISORY_THRESHOLD  – minimum prob for advisory inserts when
                                SCORE_WRITE_ADVISORY_ROWS=1 (default 0.50)
    SCORE_WRITE_ADVISORY_ROWS – set to 1 to insert CAUTION/ADVISORY rows below
                                the remelt threshold down to advisory threshold
    SCORE_ENABLE_SHAP         – set to 1/true to compute SHAP summaries
    SCORE_SHAP_MAX_HEATS      – max rows per cycle to explain (default 10)
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import warnings
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
HOLD_THRESHOLD: float = float(os.environ.get("SCORE_HOLD_THRESHOLD", "0.80"))
CAUTION_THRESHOLD: float = float(os.environ.get("SCORE_CAUTION_THRESHOLD", "0.65"))
ADVISORY_THRESHOLD: float = float(os.environ.get("SCORE_ADVISORY_THRESHOLD", "0.50"))
WRITE_ADVISORY_ROWS: bool = os.environ.get("SCORE_WRITE_ADVISORY_ROWS", "").strip().lower() in {"1", "true", "yes"}
FEATURE_SET: str = "early_remelt_decision"
ENABLE_SHAP: bool = os.environ.get("SCORE_ENABLE_SHAP", "").strip().lower() in {"1", "true", "yes"}
SHAP_MAX_HEATS: int = int(os.environ.get("SCORE_SHAP_MAX_HEATS", "10"))

logger = logging.getLogger("score_heat_live")


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def get_unscored_heats(conn: Any, limit: int | None = None) -> pd.DataFrame:
    """Return rows from the early-score view that have no entry in ``ml_heat_scores``."""
    sql = (
        "SELECT v.* "
        "FROM v_ml_heat_early_score_v1 v "
        "LEFT JOIN ml_heat_scores s ON s.heat_number = v.heat_number "
        "WHERE s.heat_number IS NULL "
        "  AND v.pour_date >= NOW() - INTERVAL '%s hours'"
    ) % SCORE_HORIZON_HOURS
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1 when provided")
        sql = f"{sql} LIMIT {int(limit)}"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable.*",
            category=UserWarning,
        )
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


def _recommendation_insert_min_probability() -> float:
    return ADVISORY_THRESHOLD if WRITE_ADVISORY_ROWS else REMELT_THRESHOLD


def decision_code_for_probability(probability: float) -> str:
    """Map scrap probability to operator-facing decision tiers (see configuration guide)."""
    if probability >= HOLD_THRESHOLD:
        return "HOLD"
    if probability >= CAUTION_THRESHOLD:
        return "CAUTION"
    return "ADVISORY"


def _format_shap_note(payload: dict[str, float] | None) -> str | None:
    if not payload:
        return None
    bits = [f"{name} ({value:+.3f})" for name, value in list(payload.items())[:3]]
    return "SHAP drivers: " + ", ".join(bits)


def upsert_heat_recommendation(
    cursor: Any,
    heat_number: str,
    probability: float,
    decision_code: str,
    primary_driver: str | None,
    extra_note: str | None = None,
) -> None:
    """Upsert a Grafana-facing recommendation row for the current heat."""
    templates = {
        "HOLD": "Re-melt candidate \u2014 high scrap probability before blast",
        "CAUTION": "Elevated scrap probability \u2014 increase monitoring before blast",
        "ADVISORY": "Advisory scrap signal \u2014 review process indicators before next operations",
    }
    text = templates.get(decision_code, templates["ADVISORY"])
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
        "    decision_code = EXCLUDED.decision_code, "
        "    created_at = NOW()",
        (
            heat_number,
            decision_code,
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
            insert_min = _recommendation_insert_min_probability()
            if probability >= insert_min:
                raw_row = raw_frame.loc[idx] if idx in raw_frame.index else row
                driver = _top_contributing_feature(raw_row, feature_columns)
                tops = shap_payload.get("top_features") if isinstance(shap_payload, dict) else None
                note = _format_shap_note(tops) if isinstance(tops, dict) else None
                tier = decision_code_for_probability(probability)
                upsert_heat_recommendation(cur, row["heat_number"], probability, tier, driver, note)
    conn.commit()


def scoring_loop(*, dry_run: bool = False, limit: int | None = None) -> int:
    """Single poll-score-write cycle. Returns the number of heats scored this cycle."""
    model_path, pipeline, _model_metadata = _load_latest_model_and_metadata(FEATURE_SET)
    model_version = model_path.stem

    with get_conn() as conn:
        df = get_unscored_heats(conn, limit=limit)
        if df.empty:
            logger.debug("No unscored heats found.")
            return 0

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

        if dry_run:
            logger.info(
                "Dry-run: loaded model %s  unscored_heats=%d  (no database writes)",
                model_version,
                len(scored),
            )
            preview = scored.sort_values("scrap_probability", ascending=False).head(15)
            for _, r in preview.iterrows():
                tier = decision_code_for_probability(float(r["scrap_probability"]))
                logger.info(
                    "  %s  P(scrap)=%.3f  tier=%s  remelt_flag=%s",
                    r["heat_number"],
                    float(r["scrap_probability"]),
                    tier,
                    int(r["predicted_scrap_flag"]),
                )
            logger.info("Dry-run completed; no changes written to the database")
            return len(scored)

        explainer = _build_shap_explainer(pipeline, x) if ENABLE_SHAP else None
        order = scored["scrap_probability"].sort_values(ascending=False).index.tolist()
        shap_allow = set(order[: max(0, SHAP_MAX_HEATS)])

        write_scores(conn, df, scored, feature_columns, model_version, x, explainer, shap_allow)
        logger.info("Scored %d heats (%d flagged)", len(scored), int(predictions.sum()))
        return len(scored)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Near-real-time heat scoring daemon (see module docstring).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score in memory and log a sample; do not write to ml_heat_scores or heat_recommendations.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scoring cycle then exit (default is to poll forever).",
    )
    parser.add_argument(
        "--test-heats",
        "--limit",
        type=int,
        default=None,
        dest="limit",
        metavar="N",
        help="Max unscored heats to fetch from v_ml_heat_early_score_v1 this cycle (testing).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if not (REMELT_THRESHOLD <= CAUTION_THRESHOLD <= HOLD_THRESHOLD):
        logger.warning(
            "Score thresholds not monotonic (expected remelt <= caution <= hold): "
            "remelt=%.3f caution=%.3f hold=%.3f",
            REMELT_THRESHOLD,
            CAUTION_THRESHOLD,
            HOLD_THRESHOLD,
        )
    logger.info(
        "Starting live scoring daemon  poll=%ds  horizon=%dh  remelt=%.2f hold=%.2f caution=%.2f "
        "advisory_min=%.2f advisory_rows=%s shap=%s  dry_run=%s  once=%s  limit=%s",
        POLL_INTERVAL_SEC,
        SCORE_HORIZON_HOURS,
        REMELT_THRESHOLD,
        HOLD_THRESHOLD,
        CAUTION_THRESHOLD,
        ADVISORY_THRESHOLD,
        WRITE_ADVISORY_ROWS,
        ENABLE_SHAP,
        args.dry_run,
        args.once,
        args.limit,
    )
    if args.dry_run and not args.once:
        logger.warning(
            "--dry-run without --once will re-score the same unscored heats on every poll interval "
            "because nothing is written to ml_heat_scores; prefer --dry-run --once for tests."
        )

    while True:
        try:
            scoring_loop(dry_run=args.dry_run, limit=args.limit)
        except FileNotFoundError as exc:
            logger.error("Model not found — train first: %s", exc)
        except Exception:
            logger.exception("Scoring loop error")
        if args.once:
            break
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
