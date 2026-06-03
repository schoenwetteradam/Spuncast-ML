from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spuncast_ml.db import fetch_dataframe
from spuncast_ml.modeling import model_dir, report_dir


def _latest_json(pattern: str) -> Path | None:
    matches = sorted(report_dir().glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _latest_model_metadata(feature_set: str) -> tuple[Path | None, dict[str, Any] | None]:
    models = sorted(model_dir().glob(f"scrap_baseline_{feature_set}_*.json"))
    if not models:
        return None, None
    latest = models[-1]
    return latest, json.loads(latest.read_text(encoding="utf-8"))


def generate_weekly_report(output_path: str | Path) -> Path:
    """Write a lightweight HTML summary from the latest evaluation JSON and model metadata."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    eval_path = _latest_json("evaluation_*.json")
    eval_blob: dict[str, Any] | None = None
    if eval_path and eval_path.exists():
        eval_blob = json.loads(eval_path.read_text(encoding="utf-8"))

    meta_path, meta = _latest_model_metadata("early_remelt_decision")
    if meta is None:
        meta_path, meta = _latest_model_metadata("pre_pour_in_process")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Weekly ML report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:0.4rem 0.75rem;} code{background:#f4f4f4;padding:0.1rem 0.3rem;}</style>",
        "</head><body>",
        f"<h1>Spuncast-ML weekly summary</h1><p>Generated {html.escape(stamp)} UTC.</p>",
    ]

    if eval_blob:
        tm = eval_blob.get("test_metrics") or {}
        parts.append("<h2>Latest evaluation (hold-out)</h2><table>")
        for key in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"):
            if key in tm:
                parts.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(tm[key]))}</td></tr>")
        parts.append("</table>")
        parts.append(f"<p>Source: <code>{html.escape(str(eval_path))}</code></p>")
    else:
        parts.append("<p>No evaluation JSON found yet. Run <code>spuncast-ml evaluate</code> after training.</p>")

    if meta:
        parts.append("<h2>Latest trained model</h2><ul>")
        parts.append(f"<li>Selected: <code>{html.escape(str(meta.get('selected_model', '')))}</code></li>")
        parts.append(f"<li>Feature set: <code>{html.escape(str(meta.get('feature_set', '')))}</code></li>")
        gate = meta.get("promotion_gate") or {}
        parts.append(f"<li>Promotion gate passes: <strong>{html.escape(str(gate.get('passes')))}</strong></li>")
        parts.append("</ul>")
        parts.append(f"<p>Metadata: <code>{html.escape(str(meta_path))}</code></p>")
    else:
        parts.append("<p>No model metadata found yet.</p>")

    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


# ── per-heat report card ──────────────────────────────────────────────────────

_DEVIATION_COLS = [
    ("tap_deviation_from_fps_pct",       "Tap Temp vs FPS"),
    ("pour_deviation_from_fps_pct",      "Pour Temp vs FPS"),
    ("die_deviation_from_fps_pct",       "Die Temp vs FPS"),
    ("rpm_deviation_from_fps_pct",       "Die RPM vs FPS"),
    ("spin_time_deviation_from_fps_pct", "Spin Time vs FPS"),
    ("pour_time_deviation_from_fps_pct", "Pour Time vs FPS"),
    ("funnel_deviation_from_fps_pct",    "Funnel Size vs FPS"),
]

_PARAM_COLS = [
    ("tap_temp",             "Tap Temp (°F)"),
    ("pour_temp",            "Pour Temp (°F)"),
    ("die_temp_before_pour", "Die Temp (°F)"),
    ("die_rpm",              "Die RPM"),
    ("spin_time_min",        "Spin Time (min)"),
    ("pour_time_sec",        "Pour Time (sec)"),
    ("charge_scrap_pct",     "Charge Scrap %"),
    ("chem_not_ok_flag",     "Chemistry OK"),
]


def _pct_badge(val: float) -> str:
    color = "#FF4B4B" if abs(val) >= 5 else "#FFA500" if abs(val) >= 2 else "#21C55D"
    sign = "+" if val > 0 else ""
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold">{sign}{val:.1f}%</span>'


def _risk_badge(prob: float) -> str:
    if prob >= 0.80:
        color, label = "#FF4B4B", "HIGH RISK — HOLD"
    elif prob >= 0.50:
        color, label = "#FFA500", "ELEVATED RISK — MONITOR"
    else:
        color, label = "#21C55D", "LOW RISK — CLEAR"
    return (
        f'<div style="background:{color};color:#fff;display:inline-block;'
        f'padding:10px 24px;border-radius:8px;font-size:1.4em;font-weight:bold">'
        f'{label} &nbsp; {prob:.0%}</div>'
    )


def generate_heat_report(heat_number: str, output_path: str | Path | None = None) -> Path:
    """Generate a single-heat HTML report card from ml_heat_scores and the dataset view."""
    out = Path(output_path).resolve() if output_path else report_dir() / f"heat_report_{heat_number}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    # Fetch score row
    score_df = fetch_dataframe(
        f"SELECT * FROM ml_heat_scores WHERE heat_number = {repr(heat_number)} LIMIT 1"
    )
    # Fetch full heat parameters
    param_df = fetch_dataframe(
        f"SELECT * FROM v_ml_heat_dataset_v1 WHERE heat_number = {repr(heat_number)} LIMIT 1"
    )
    # Fetch recommendation if any
    rec_df = fetch_dataframe(
        f"SELECT * FROM heat_recommendations WHERE heat_number = {repr(heat_number)} LIMIT 1"
    )

    score_row = score_df.iloc[0].to_dict() if len(score_df) else {}
    param_row = param_df.iloc[0].to_dict() if len(param_df) else {}
    rec_row   = rec_df.iloc[0].to_dict()   if len(rec_df)   else {}

    prob      = float(score_row.get("scrap_probability", 0))
    actual    = param_row.get("scrap_flag")
    pour_date = str(param_row.get("pour_date", "—"))[:10]
    melter    = str(param_row.get("melter") or "—")
    die_no    = str(param_row.get("die_no") or "—")
    alloy     = str(param_row.get("alloy_grade") or "—")

    css = (
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:900px}"
        "h1,h2{color:#1a1a2e} table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ddd;padding:0.45rem 0.8rem}"
        "th{background:#f0f0f0;text-align:left} .meta{color:#666;font-size:0.9em}"
    )

    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Heat Report — {html.escape(heat_number)}</title>"
        f"<style>{css}</style></head><body>",
        f"<h1>Heat Report &mdash; {html.escape(heat_number)}</h1>",
        f"<p class='meta'>Generated {html.escape(stamp)} UTC &nbsp;|&nbsp; "
        f"Pour date: {html.escape(pour_date)} &nbsp;|&nbsp; "
        f"Melter: {html.escape(melter)} &nbsp;|&nbsp; "
        f"Die: {html.escape(die_no)} &nbsp;|&nbsp; "
        f"Alloy: {html.escape(alloy)}</p>",
    ]

    # Risk badge
    parts.append("<h2>Scrap Risk</h2>")
    parts.append(_risk_badge(prob))
    if actual is not None:
        outcome_label = "✅ Did NOT scrap" if int(actual) == 0 else "❌ SCRAPPED"
        parts.append(f"&nbsp;&nbsp;<strong>Actual outcome:</strong> {outcome_label}")
    if rec_row.get("recommendation_text"):
        parts.append(f"<p><strong>Recommendation:</strong> {html.escape(str(rec_row['recommendation_text']))}</p>")

    # FPS deviations
    if param_row:
        parts.append("<h2>Deviations from FPS Spec</h2><table><tr><th>Parameter</th><th>Deviation</th><th>Status</th></tr>")
        for col, label in _DEVIATION_COLS:
            val = param_row.get(col)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            status = "⚠️ Out of tolerance" if abs(fval) >= 5 else ("⚡ Watch" if abs(fval) >= 2 else "✅ OK")
            parts.append(f"<tr><td>{html.escape(label)}</td><td>{_pct_badge(fval)}</td><td>{status}</td></tr>")
        parts.append("</table>")

    # Key parameters
    if param_row:
        parts.append("<h2>Process Parameters</h2><table><tr><th>Parameter</th><th>Value</th></tr>")
        for col, label in _PARAM_COLS:
            val = param_row.get(col)
            if val is None:
                continue
            if col == "chem_not_ok_flag":
                display = "❌ Failed" if val else "✅ Passed"
            elif col == "charge_scrap_pct":
                try:
                    display = f"{float(val):.1%}"
                except (TypeError, ValueError):
                    display = str(val)
            else:
                try:
                    display = f"{float(val):,.1f}"
                except (TypeError, ValueError):
                    display = html.escape(str(val))
            parts.append(f"<tr><td>{html.escape(label)}</td><td>{display}</td></tr>")
        parts.append("</table>")

    # SHAP / explanation
    explanation = score_row.get("explanation_json")
    if explanation:
        if isinstance(explanation, str):
            try:
                explanation = json.loads(explanation)
            except Exception:
                explanation = None
        if isinstance(explanation, dict):
            tops = explanation.get("top_features", explanation)
            if isinstance(tops, dict):
                parts.append("<h2>Top Contributing Factors (SHAP)</h2><table><tr><th>Feature</th><th>Contribution</th><th>Direction</th></tr>")
                for feat, shap_val in list(tops.items())[:8]:
                    try:
                        sv = float(shap_val)
                    except (TypeError, ValueError):
                        continue
                    direction = "↑ Increases risk" if sv > 0 else "↓ Decreases risk"
                    color = "#FF4B4B" if sv > 0 else "#21C55D"
                    parts.append(
                        f"<tr><td>{html.escape(str(feat).replace('_',' '))}</td>"
                        f"<td style='color:{color};font-weight:bold'>{sv:+.4f}</td>"
                        f"<td>{direction}</td></tr>"
                    )
                parts.append("</table>")

    parts.append(f"<hr><p class='meta'>Model: {html.escape(str(score_row.get('model_version') or '—'))} &nbsp;|&nbsp; Feature set: {html.escape(str(score_row.get('feature_set') or '—'))}</p>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
