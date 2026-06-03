"""Spuncast ML – Operator Dashboard

Run with:
    streamlit run app/dashboard.py

Reads from the ml_heat_scores and heat_recommendations tables (same DB the
scoring daemon writes to). Falls back to demo data when the DB is unavailable
so floor workers can still orient themselves while connectivity is restored.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── env / path ────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spuncast_ml.feedback import record_operator_feedback  # noqa: E402

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spuncast ML – Heat Board",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "hold_for_operator_review": "#FF4B4B",
    "increase_monitoring":      "#FFA500",
    "continue_standard_run":    "#21C55D",
}
ACTION_LABELS = {
    "hold_for_operator_review": "🔴 HOLD",
    "increase_monitoring":      "🟡 MONITOR",
    "continue_standard_run":    "🟢 CLEAR",
}
REPORT_DIR = Path(os.environ.get("SPUNCAST_ML_REPORT_DIR", "./reports/generated"))


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_connect():
    """Open a fresh psycopg2 connection using env vars. Returns None on failure."""
    try:
        import psycopg2
        return psycopg2.connect(
            dbname=os.environ.get("PGDATABASE", "spuncast"),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD"),
            host=os.environ.get("PGHOST", os.environ.get("PG_HOST", "localhost")),
            port=int(os.environ.get("PG_PORT", 5432)),
        )
    except Exception:
        return None


def _run_query(sql: str, params: tuple = ()) -> pd.DataFrame | None:
    conn = _db_connect()
    if conn is None:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy.*")
            result = pd.read_sql_query(sql, conn, params=params if params else None)
        return result
    except Exception:
        return None
    finally:
        conn.close()


def _write_operator_action(heat_number: str, action: str) -> bool:
    conn = _db_connect()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ml_heat_scores SET operator_action = %s WHERE heat_number = %s",
                (action, heat_number),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


# ── demo data ─────────────────────────────────────────────────────────────────

def _demo_heats() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(int(time.time()) // 300)  # refreshes every 5 min
    n = 18
    now = datetime.now(timezone.utc)
    probs = rng.beta(2, 5, n).astype(float)
    actions = [
        "hold_for_operator_review" if p >= 0.80
        else "increase_monitoring" if p >= 0.50
        else "continue_standard_run"
        for p in probs
    ]
    drivers = [
        rng.choice(["tap_temp_deviation", "chem_not_ok_flag", "die_rpm_deviation",
                    "operator_rolling_scrap_rate", "charge_scrap_pct", "shift_rolling_scrap_rate"])
        for _ in range(n)
    ]
    return pd.DataFrame({
        "heat_number":        [f"H-{4000 + i}" for i in range(n)],
        "scrap_probability":  probs,
        "recommended_action": actions,
        "primary_driver":     drivers,
        "scored_at":          [now - timedelta(minutes=int(rng.integers(1, 120))) for _ in range(n)],
        "operator_action":    [None] * n,
        "decision_code":      ["HOLD" if a == "hold_for_operator_review" else "WATCH" if a == "increase_monitoring" else None for a in actions],
        "explanation_json":   [None] * n,
    })


def _demo_explanation(heat_number: str) -> dict[str, float]:
    import numpy as np
    rng = np.random.default_rng(hash(heat_number) % (2**31))
    keys = ["tap_deviation_from_fps_pct", "operator_rolling_scrap_rate",
            "chem_not_ok_flag", "charge_scrap_pct", "die_rolling_scrap_rate",
            "pour_deviation_from_fps_pct", "shift_rolling_scrap_rate",
            "has_open_data_quality_violation"]
    vals = rng.uniform(-0.4, 0.8, len(keys)).tolist()
    return dict(zip(keys, vals))


# ── data loaders ──────────────────────────────────────────────────────────────

def load_heats(hours: int = 24) -> tuple[pd.DataFrame, bool]:
    """Return (heats_df, is_live). Falls back to demo data if DB unavailable."""
    # Use make_interval so the integer parameter binds cleanly without
    # string-interpolating inside an INTERVAL literal (which psycopg2 rejects).
    sql = """
        SELECT
            s.heat_number,
            s.scrap_probability,
            s.recommended_action,
            s.scored_at,
            s.operator_action,
            s.explanation_json,
            r.decision_code,
            r.primary_driver
        FROM ml_heat_scores s
        LEFT JOIN heat_recommendations r ON r.heat_number = s.heat_number
        WHERE s.scored_at >= NOW() - make_interval(hours => %s)
        ORDER BY s.scored_at DESC
        LIMIT 500
    """
    result = _run_query(sql, (int(hours),))
    if result is not None and len(result) > 0:
        return result, True
    return _demo_heats(), False


def load_explanation(heat_number: str) -> dict[str, float] | None:
    sql = "SELECT explanation_json FROM ml_heat_scores WHERE heat_number = %s LIMIT 1"
    result = _run_query(sql, (heat_number,))
    if result is not None and len(result) > 0 and result["explanation_json"].iloc[0]:
        raw = result["explanation_json"].iloc[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw
    return None


def load_model_metrics() -> dict[str, Any] | None:
    candidates = sorted(REPORT_DIR.glob("evaluation_*.json"))
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


# ── styling helpers ───────────────────────────────────────────────────────────

def _style_row(row: pd.Series) -> list[str]:
    action = row.get("recommended_action", "")
    color = ACTION_COLORS.get(action, "#FFFFFF")
    alpha = "33"  # light background tint
    bg = f"background-color: {color}{alpha}"
    return [bg] * len(row)


def _prob_bar_html(prob: float) -> str:
    color = ACTION_COLORS.get(
        "hold_for_operator_review" if prob >= 0.8 else
        "increase_monitoring" if prob >= 0.5 else
        "continue_standard_run"
    )
    pct = int(prob * 100)
    return (
        f'<div style="background:#eee;border-radius:4px;height:18px;width:120px;display:inline-block">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div></div>'
        f' <span style="font-size:0.85em">{pct}%</span>'
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[int, bool]:
    st.sidebar.title("🏭 Spuncast ML")
    st.sidebar.caption("Scrap Prediction Dashboard")
    st.sidebar.divider()

    hours = st.sidebar.selectbox("Show heats from last…", [8, 24, 48, 72], index=1, format_func=lambda h: f"{h} hours")
    auto_refresh = st.sidebar.toggle("Auto-refresh (30 s)", value=False)

    st.sidebar.divider()
    metrics = load_model_metrics()
    if metrics:
        st.sidebar.subheader("Model")
        m = metrics.get("test_metrics", {})
        st.sidebar.metric("Recall", f"{m.get('recall', 0):.0%}")
        st.sidebar.metric("PR-AUC", f"{m.get('pr_auc', 0):.3f}")
        st.sidebar.metric("False Negatives", m.get("false_negatives", "—"))
        st.sidebar.caption(f"Threshold: {metrics.get('decision_threshold', '—')}")
        st.sidebar.caption(f"Model: {metrics.get('feature_set', '—')}")
    else:
        st.sidebar.info("No evaluation report found.\nRun `spuncast-ml evaluate` first.")

    st.sidebar.divider()
    st.sidebar.caption("© Spuncast Industries")
    return int(hours), bool(auto_refresh)


# ── tab 1: heat board ─────────────────────────────────────────────────────────

def render_heat_board(heats: pd.DataFrame, is_live: bool) -> None:
    if not is_live:
        st.warning("⚠️ Database unavailable — showing demo data. Connect to the Spuncast Operations DB to see live heats.")

    # KPI strip
    total = len(heats)
    holds = (heats["recommended_action"] == "hold_for_operator_review").sum()
    monitors = (heats["recommended_action"] == "increase_monitoring").sum()
    clears = (heats["recommended_action"] == "continue_standard_run").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Heats", total)
    c2.metric("🔴 Hold", int(holds))
    c3.metric("🟡 Monitor", int(monitors))
    c4.metric("🟢 Clear", int(clears))
    st.divider()

    # Build display frame
    display = heats.copy()
    display["Risk"] = display["recommended_action"].map(ACTION_LABELS).fillna("—")
    display["Probability"] = (display["scrap_probability"] * 100).round(1).astype(str) + "%"
    display["Scored At"] = pd.to_datetime(display["scored_at"]).dt.strftime("%m/%d %H:%M")
    display["Operator Action"] = display["operator_action"].fillna("—")
    display["Primary Driver"] = display["primary_driver"].fillna("—")

    show_cols = ["heat_number", "Risk", "Probability", "Primary Driver", "Operator Action", "Scored At"]
    styled = (
        display[show_cols]
        .rename(columns={"heat_number": "Heat #"})
        .style.apply(_style_row, axis=1)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv = heats.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Export CSV", data=csv, file_name="heat_scores.csv", mime="text/csv")


# ── tab 2: heat detail ────────────────────────────────────────────────────────

def render_heat_detail(heats: pd.DataFrame, is_live: bool) -> None:
    heat_numbers = heats["heat_number"].tolist()
    if not heat_numbers:
        st.info("No heats available for the selected time window.")
        return

    selected = st.selectbox("Select a heat to inspect", heat_numbers)
    row = heats[heats["heat_number"] == selected].iloc[0]

    # Risk badge
    action = row.get("recommended_action", "")
    color = ACTION_COLORS.get(action, "#888")
    label = ACTION_LABELS.get(action, action)
    prob = float(row.get("scrap_probability", 0))

    col_a, col_b, col_c = st.columns([2, 2, 3])
    with col_a:
        st.markdown(
            f'<div style="background:{color};color:white;padding:12px 18px;border-radius:8px;font-size:1.3em;font-weight:bold;text-align:center">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.metric("Scrap Probability", f"{prob:.0%}")
    with col_c:
        if row.get("primary_driver"):
            st.info(f"**Top driver:** {row['primary_driver']}")
        if row.get("decision_code"):
            st.caption(f"Decision code: {row['decision_code']}")

    st.divider()

    # SHAP / explanation section
    st.subheader("Why is this heat flagged?")
    explanation: dict[str, float] | None = None
    if is_live:
        raw = row.get("explanation_json")
        if raw:
            explanation = raw if isinstance(raw, dict) else json.loads(raw)
        else:
            explanation = load_explanation(selected)
    if explanation is None:
        explanation = _demo_explanation(selected) if not is_live else None

    if explanation:
        import plotly.express as px
        items = sorted(explanation.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        feat_names, shap_vals = zip(*items)
        colors = ["#FF4B4B" if v > 0 else "#21C55D" for v in shap_vals]
        fig = px.bar(
            x=list(shap_vals),
            y=list(feat_names),
            orientation="h",
            color=colors,
            color_discrete_map="identity",
            labels={"x": "SHAP contribution (→ higher risk)", "y": "Feature"},
            title=f"Top factors for Heat {selected}",
        )
        fig.update_layout(showlegend=False, height=400, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red bars increase scrap probability. Green bars reduce it.")
    else:
        st.info("SHAP explanations not available. Set SCORE_ENABLE_SHAP=1 in the daemon environment to enable them.")

    st.divider()

    # Feedback form
    st.subheader("Operator Feedback")
    with st.form(key=f"feedback_{selected}"):
        col1, col2 = st.columns(2)
        with col1:
            outcome = st.radio("What happened with this heat?", ["Did NOT scrap", "DID scrap", "Unknown"])
        with col2:
            decision = st.radio("Did you act on the recommendation?", ["Accepted (held/monitored)", "Rejected (ran anyway)", "N/A"])

        operator_id = st.text_input("Your initials or operator ID (optional)")
        note = st.text_area("Note (optional)", placeholder="e.g. chemistry was borderline but met spec")

        submitted = st.form_submit_button("Submit Feedback", type="primary")
        if submitted:
            accepted = "accepted" in decision.lower()
            actual = 1 if "did scrap" in outcome.lower() else (0 if "did not" in outcome.lower() else None)
            recommendation = row.get("recommended_action", "unknown")

            try:
                record_operator_feedback(
                    heat_number=str(selected),
                    recommendation=recommendation,
                    accepted=accepted,
                    score=float(prob),
                    operator_id=operator_id or None,
                    note=note or None,
                    actual_scrap_flag=actual,
                )
                if is_live and actual is not None:
                    op_action = "remelt" if actual == 1 else "proceed"
                    _write_operator_action(str(selected), op_action)
                st.success(f"Feedback recorded for Heat {selected}. Thank you!")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")


# ── tab 3: model performance ──────────────────────────────────────────────────

def render_model_performance() -> None:
    metrics = load_model_metrics()
    if not metrics:
        st.info("No evaluation report found. Run `spuncast-ml evaluate` to generate one.")
        return

    test = metrics.get("test_metrics", {})
    baseline = metrics.get("rules_baseline_test_metrics", {})
    gate = metrics.get("promotion_gate", {})

    st.subheader("Test Set Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recall",    f"{test.get('recall', 0):.0%}",    delta=f"{(test.get('recall',0)-baseline.get('recall',0)):+.0%} vs baseline")
    c2.metric("Precision", f"{test.get('precision', 0):.0%}")
    c3.metric("F1",        f"{test.get('f1', 0):.0%}")
    c4.metric("PR-AUC",    f"{test.get('pr_auc', 0):.3f}",   delta=f"{(test.get('pr_auc',0)-baseline.get('pr_auc',0)):+.3f} vs baseline")
    c5.metric("False Negatives", test.get("false_negatives", "—"), delta=f"{test.get('false_negatives',0)-baseline.get('false_negatives',0):+d} vs baseline", delta_color="inverse")

    passes = gate.get("passes")
    if passes is True:
        st.success("✅ Promotion gate PASSED — model beats the rules baseline on recall, false negatives, and PR-AUC.")
    elif passes is False:
        st.error("❌ Promotion gate FAILED — model does not meet the rules baseline. Review training before deploying.")

    st.divider()
    st.subheader("Confusion Matrix")
    cm = test.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, index=["Actual: No Scrap", "Actual: Scrap"], columns=["Predicted: No Scrap", "Predicted: Scrap"])
        st.dataframe(cm_df.style.background_gradient(cmap="Reds"), use_container_width=False)

    st.divider()
    st.subheader("Metadata")
    info_cols = {
        "Feature Set": metrics.get("feature_set"),
        "Decision Threshold": metrics.get("decision_threshold"),
        "Evaluated At": metrics.get("evaluated_at_utc"),
        "Model Path": Path(metrics.get("model_path", "")).name if metrics.get("model_path") else "—",
    }
    st.table(pd.DataFrame.from_dict(info_cols, orient="index", columns=["Value"]))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    hours, auto_refresh = render_sidebar()

    st.title("🏭 Spuncast ML — Heat Scrap Prediction")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    heats, is_live = load_heats(hours=hours)

    tab1, tab2, tab3 = st.tabs(["Heat Board", "Heat Detail", "Model Performance"])
    with tab1:
        render_heat_board(heats, is_live)
    with tab2:
        render_heat_detail(heats, is_live)
    with tab3:
        render_model_performance()


if __name__ == "__main__":
    main()
