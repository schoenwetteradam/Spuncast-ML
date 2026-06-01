#!/usr/bin/env python3
"""
causal_analysis.py — DoWhy causal analysis of foundry scrap drivers.

For each key process variable (tap/pour/die temp deviation, RPM deviation,
charge composition, chemistry compliance), estimates the causal effect on
scrap_flag using backdoor identification + linear regression, then validates
each finding with a placebo refutation test.

Results are written to:
  - reports/causal_<scope>_<timestamp>.html  (human-readable)
  - reports/causal_<scope>_<timestamp>.json  (machine-readable)
  - PostgreSQL causal_analysis_results table (for web UI)

Usage:
    python scripts/causal_analysis.py                       # all heats
    python scripts/causal_analysis.py --product CAT536-6768
    python scripts/causal_analysis.py --grade HU
    python scripts/causal_analysis.py --product CAT536-6768 --simulations 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env so DATABASE_URL is available.
# Check REPO_ROOT/.env first, then sibling Spuncast-Operations-main/.env
# (used when ML repo lives alongside the operations repo on the VM).
try:
    from dotenv import load_dotenv
    _env_candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "Spuncast-Operations-main" / ".env",
        REPO_ROOT.parent / ".env",
    ]
    for _env_path in _env_candidates:
        if _env_path.exists():
            load_dotenv(_env_path)
            break
except ImportError:
    pass

from spuncast_ml.causal.dag import (
    COMMON_CONFOUNDERS,
    OUTCOME,
    SCRAP_DAG_GML,
    TREATMENTS,
)

log = logging.getLogger("causal_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORTS_DIR = REPO_ROOT / "reports"
EXPORTS_DIR = REPO_ROOT / "data" / "exports"
MIN_ROWS_DEFAULT = 100
REFUTATION_SIMULATIONS = 50


# ── Data loading ──────────────────────────────────────────────────────────────

def load_latest_export() -> pd.DataFrame:
    parquets = sorted(EXPORTS_DIR.glob("ml_heat_dataset_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No ML export parquets found in {EXPORTS_DIR}")
    path = parquets[-1]
    log.info("Loading export: %s", path.name)
    return pd.read_parquet(path)


def prepare_frame(df: pd.DataFrame, product: str | None, grade: str | None) -> pd.DataFrame:
    if product:
        df = df[df["product_number"].astype(str).str.upper() == product.upper()].copy()
        log.info("Filtered to product %s: %d heats", product, len(df))
    if grade:
        df = df[df["alloy_grade"].astype(str).str.upper() == grade.upper()].copy()
        log.info("Filtered to grade %s: %d heats", grade, len(df))

    required = [OUTCOME] + COMMON_CONFOUNDERS + [t["name"] for t in TREATMENTS]
    available = [c for c in required if c in df.columns]
    extra = [c for c in ["heat_number", "product_number", "alloy_grade", "reason_code_bucket"] if c in df.columns]
    df = df[list(set(available + extra))].copy()

    for col in COMMON_CONFOUNDERS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes

    df[OUTCOME] = df[OUTCOME].astype(float)
    return df


# ── Single-treatment causal analysis ─────────────────────────────────────────

def _run_one_treatment(df: pd.DataFrame, treatment: dict) -> dict:
    try:
        from dowhy import CausalModel
    except ImportError:
        raise RuntimeError("dowhy is not installed. Run: pip install dowhy")

    tname = treatment["name"]
    cols_needed = [tname, OUTCOME] + [c for c in COMMON_CONFOUNDERS if c in df.columns]
    sub = df[cols_needed].dropna()

    if len(sub) < 30:
        return {
            "treatment": tname,
            "label": treatment["label"],
            "status": "skipped",
            "reason": f"Only {len(sub)} observations after dropping nulls (need ≥30)",
            "n_obs": len(sub),
        }

    # Skip zero-variance columns (e.g. all-zero charge_scrap_pct before .$$L ingestion)
    if sub[tname].std() < 1e-9:
        return {
            "treatment": tname,
            "label": treatment["label"],
            "status": "skipped",
            "reason": "No variation in this variable for this scope (all values identical — check data ingestion)",
            "n_obs": len(sub),
        }

    result: dict = {
        "treatment": tname,
        "label": treatment["label"],
        "unit": treatment.get("unit", ""),
        "type": treatment["type"],
        "interpretation_direction": treatment.get("interpretation_direction", ""),
        "n_obs": len(sub),
        "scrap_rate": round(float(sub[OUTCOME].mean()), 4),
        "status": "ok",
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CausalModel(
                data=sub,
                treatment=tname,
                outcome=OUTCOME,
                graph=SCRAP_DAG_GML,
                proceed_when_unidentifiable=True,
            )
            estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                estimand,
                method_name="backdoor.linear_regression",
                control_value=0,
                treatment_value=1,
                confidence_intervals=True,
                method_params={"need_conditional_estimates": False},
            )

            raw_ate = estimate.value
            if raw_ate is None:
                raise ValueError(
                    "DoWhy could not estimate this effect — the causal path could not be identified. "
                    "This usually means a required confounder is unobserved for this scope."
                )
            ate = float(raw_ate)
            try:
                ci = estimate.get_confidence_intervals()
                ci_lower = float(ci[0]) if ci is not None else None
                ci_upper = float(ci[1]) if ci is not None else None
            except Exception:
                ci_lower = ci_upper = None

            result["effect_estimate"] = round(ate, 5)
            result["ci_lower"] = round(ci_lower, 5) if ci_lower is not None else None
            result["ci_upper"] = round(ci_upper, 5) if ci_upper is not None else None

            refutation = model.refute_estimate(
                estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=REFUTATION_SIMULATIONS,
            )
            result["refutation_p_value"] = round(float(refutation.refutation_result.get("p_value", 1.0)), 4)
            result["passes_refutation"] = result["refutation_p_value"] < 0.10

            sign = "increases" if ate > 0 else "decreases"
            magnitude = abs(ate) * 100
            tname_local = treatment["name"]
            unit = treatment.get("unit", "")
            if "from_instruction_pct" in tname_local or "from_fps_pct" in tname_local:
                result["interpretation"] = (
                    f"Running 1 full band-width outside the target {sign} scrap probability by "
                    f"{magnitude:.1f} pp (e.g. ±0.5 = at the allowed limit; ±1.0 = one full range "
                    f"outside spec). Controlling for alloy grade and furnace."
                )
            elif treatment.get("type") == "binary":
                result["interpretation"] = (
                    f"When {treatment['label']} = 1 (vs 0), scrap probability {sign} by "
                    f"{magnitude:.1f} pp. Controlling for alloy grade and furnace."
                )
            elif "°F" in unit:
                result["interpretation"] = (
                    f"Each 1°F change in {treatment['label']} {sign} scrap probability by "
                    f"{magnitude:.2f} pp. A 10°F shift changes risk by {magnitude * 10:.1f} pp. "
                    f"Controlling for alloy grade and furnace."
                )
            elif "count" in unit:
                result["interpretation"] = (
                    f"Each additional element outside spec {sign} scrap probability by "
                    f"{magnitude:.1f} pp. Controlling for alloy grade and furnace."
                )
            elif "%" in unit:
                result["interpretation"] = (
                    f"Each 1 percentage-point increase in {treatment['label']} {sign} scrap "
                    f"probability by {magnitude:.2f} pp. Controlling for alloy grade and furnace."
                )
            else:
                result["interpretation"] = (
                    f"A 1-unit increase in {treatment['label']} ({unit}) {sign} scrap probability "
                    f"by {magnitude:.1f} pp. Controlling for alloy grade and furnace."
                )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        log.warning("Failed analysis for %s: %s", tname, exc)

    return result


# ── Per-reason signal analysis ────────────────────────────────────────────────

def reason_split_analysis(df: pd.DataFrame) -> list[dict]:
    if "reason_code_bucket" not in df.columns:
        return []

    results = []
    for reason in df["reason_code_bucket"].dropna().unique():
        sub = df.copy()
        sub["is_reason"] = (sub["reason_code_bucket"] == reason).astype(int)
        n_heats = int(sub["is_reason"].sum())
        if n_heats < 10:
            continue

        row: dict = {"reason_code": str(reason), "n_heats": n_heats, "signals": []}
        for treatment in TREATMENTS:
            tname = treatment["name"]
            if tname not in sub.columns:
                continue
            good = sub.loc[sub["is_reason"] == 0, tname].dropna()
            scrap = sub.loc[sub["is_reason"] == 1, tname].dropna()
            if len(good) < 5 or len(scrap) < 5:
                continue
            delta = float(scrap.mean() - good.mean())
            base = float(good.mean())
            delta_pct = round(delta / base * 100, 1) if abs(base) > 1e-9 else None
            row["signals"].append({
                "treatment": tname,
                "label": treatment["label"],
                "unit": treatment.get("unit", ""),
                "avg_no_reason": round(base, 3),
                "avg_with_reason": round(float(scrap.mean()), 3),
                "delta": round(delta, 3),
                "delta_pct": delta_pct,
            })
        row["signals"].sort(key=lambda x: abs(x["delta"]), reverse=True)
        results.append(row)

    results.sort(key=lambda x: x["n_heats"], reverse=True)
    return results


# ── Database write ────────────────────────────────────────────────────────────

def write_results_to_db(report: dict) -> None:
    try:
        import psycopg2
    except ImportError:
        log.warning("psycopg2 not available — skipping DB write")
        return

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        log.warning("DATABASE_URL not set — skipping DB write")
        return

    scope = report["scope"]
    scope_type = (
        "product" if scope.get("product_number")
        else "grade" if scope.get("alloy_grade")
        else "global"
    )
    scope_value = scope.get("product_number") or scope.get("alloy_grade")
    run_at = report["run_at"]

    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            # Replace existing rows for this scope
            cur.execute(
                "DELETE FROM causal_analysis_results WHERE scope_type=%s AND scope_value IS NOT DISTINCT FROM %s",
                (scope_type, scope_value),
            )
            for r in report["results"]:
                cur.execute(
                    """
                    INSERT INTO causal_analysis_results
                        (scope_type, scope_value, treatment, label, treatment_type, unit,
                         n_obs, effect_estimate, ci_lower, ci_upper, refutation_p_value,
                         passes_refutation, interpretation, status, error_msg,
                         run_at, n_heats, scrap_rate_pct)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        scope_type, scope_value,
                        r.get("treatment"), r.get("label"), r.get("type"), r.get("unit"),
                        r.get("n_obs"),
                        r.get("effect_estimate"), r.get("ci_lower"), r.get("ci_upper"),
                        r.get("refutation_p_value"), r.get("passes_refutation"),
                        r.get("interpretation"), r.get("status", "ok"), r.get("error"),
                        run_at, scope.get("n_heats"), scope.get("scrap_rate_pct"),
                    ),
                )

            cur.execute(
                "DELETE FROM causal_reason_signals WHERE scope_type=%s AND scope_value IS NOT DISTINCT FROM %s",
                (scope_type, scope_value),
            )
            for rr in report.get("reason_analysis", []):
                for sig in rr.get("signals", []):
                    cur.execute(
                        """
                        INSERT INTO causal_reason_signals
                            (scope_type, scope_value, reason_code, treatment, label,
                             avg_no_reason, avg_with_reason, delta, n_heats, run_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            scope_type, scope_value,
                            rr["reason_code"], sig["treatment"], sig.get("label"),
                            sig["avg_no_reason"], sig["avg_with_reason"], sig["delta"],
                            rr["n_heats"], run_at,
                        ),
                    )
        conn.commit()
        log.info("Results written to DB (%s=%s)", scope_type, scope_value)
    except Exception as exc:
        log.warning("DB write failed: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── HTML report ───────────────────────────────────────────────────────────────

def build_html_report(report: dict) -> str:
    scope = report.get("scope", {})
    results = report.get("results", [])
    reason_rows = report.get("reason_analysis", [])

    def badge(passes):
        if passes is True:
            return "<span style='color:green;font-weight:bold'>✓ confirmed</span>"
        if passes is False:
            return "<span style='color:orange'>⚠ not confirmed</span>"
        return "<span style='color:gray'>—</span>"

    def effect_bar(ate):
        if ate is None:
            return "—"
        color = "#c0392b" if ate > 0 else "#27ae60"
        width = min(abs(ate) * 2000, 100)
        return (
            f"<span style='color:{color};font-weight:bold'>{ate:+.4f}</span>"
            f"<div style='background:{color};height:6px;width:{width:.0f}px;"
            f"border-radius:3px;margin-top:3px'></div>"
        )

    rows = ""
    for r in results:
        if r.get("status") == "skipped":
            rows += f"<tr><td>{r['label']}</td><td colspan=5 style='color:#999'>{r.get('reason','skipped')}</td></tr>"
            continue
        if r.get("status") == "error":
            rows += f"<tr><td>{r['label']}</td><td colspan=5 style='color:red'>{r.get('error','error')}</td></tr>"
            continue
        rows += (
            f"<tr>"
            f"<td>{r['label']}<br><small style='color:#888'>{r.get('unit','')}</small></td>"
            f"<td>{r.get('n_obs','—')}</td>"
            f"<td>{effect_bar(r.get('effect_estimate'))}</td>"
            f"<td>[{r.get('ci_lower','?')}, {r.get('ci_upper','?')}]</td>"
            f"<td>{r.get('refutation_p_value','—')}</td>"
            f"<td>{badge(r.get('passes_refutation'))}</td>"
            f"</tr>"
        )

    reason_html = ""
    if reason_rows:
        reason_html = "<h2>Scrap Reason Signal Analysis</h2>"
        reason_html += "<p style='color:#555;font-size:13px'>Mean difference in process variables between heats with vs without each scrap reason. Not fully causal — use as a screen to prioritize.</p>"
        for rr in reason_rows[:10]:
            reason_html += f"<h3>{rr['reason_code']} ({rr['n_heats']} heats)</h3><ul>"
            for s in rr["signals"][:5]:
                direction = "↑ higher" if s["delta"] > 0 else "↓ lower"
                reason_html += (
                    f"<li><b>{s['label']}</b>: {direction} on heats with this reason "
                    f"(avg {s['avg_with_reason']:.3f} vs {s['avg_no_reason']:.3f}, Δ={s['delta']:+.3f})</li>"
                )
            reason_html += "</ul>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Causal Scrap Analysis — {scope.get('label','All Heats')}</title>
<style>
body{{font-family:Arial,sans-serif;margin:30px;color:#222;max-width:1100px}}
h1{{color:#1F4E79}}h2{{color:#1F4E79;border-bottom:2px solid #1F4E79;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{background:#1F4E79;color:white;padding:8px 10px;text-align:left}}
td{{padding:8px 10px;border-bottom:1px solid #ddd}}
tr:hover td{{background:#f0f6ff}}
.meta{{background:#f8f9fa;border-radius:6px;padding:14px;margin-bottom:20px;font-size:13px}}
</style></head><body>
<h1>Causal Scrap Analysis — {scope.get('label','All Heats')}</h1>
<div class="meta">
  <b>Scope:</b> {scope.get('label','All heats')} &nbsp;|&nbsp;
  <b>Heats:</b> {scope.get('n_heats','?')} &nbsp;|&nbsp;
  <b>Overall scrap rate:</b> {scope.get('scrap_rate_pct','?')}% &nbsp;|&nbsp;
  <b>Run at:</b> {report.get('run_at','')}
</div>
<h2>Causal Effect Estimates on Scrap Probability</h2>
<p style="color:#555;font-size:13px">
Effect = change in scrap probability per 1-unit increase in treatment,
controlling for alloy grade and furnace. Positive = higher value → more scrap.
"Confirmed" = estimate survives placebo refutation test (p &lt; 0.10).
</p>
<table>
<thead><tr>
<th>Treatment Variable</th><th>N Obs</th><th>Causal Effect (ATE)</th>
<th>95% CI</th><th>Placebo p-value</th><th>Confirmed</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
{reason_html}
<hr><p style="color:#aaa;font-size:11px">
Generated by causal_analysis.py using DoWhy backdoor linear regression.
Refutation: placebo permutation ({REFUTATION_SIMULATIONS} simulations).
</p></body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_scope(args, df_raw: "pd.DataFrame", product: str | None, grade: str | None) -> None:
    """Run a single scope (global / product / grade) and write results."""
    df = prepare_frame(df_raw, product=product, grade=grade)
    n = len(df)
    if n < args.min_rows:
        log.warning("Skipping scope product=%s grade=%s — only %d heats (need ≥ %d)",
                    product, grade, n, args.min_rows)
        return

    scope_label = (
        f"Product {product}" if product
        else f"Grade {grade}" if grade
        else "All Heats"
    )
    scrap_rate = round(float(df[OUTCOME].mean()) * 100, 1) if OUTCOME in df.columns else None
    log.info("Running analysis — scope: %s, n=%d, scrap_rate=%s%%", scope_label, n, scrap_rate)

    results = []
    for treatment in TREATMENTS:
        tname = treatment["name"]
        if tname not in df.columns:
            log.debug("Skipping %s — column not in export", tname)
            continue
        non_null = df[tname].notna().sum()
        if non_null < args.min_rows:
            log.info("Skipping %s — only %d non-null rows (need ≥ %d)", tname, non_null, args.min_rows)
            continue
        log.info("Analyzing: %s (%d non-null rows)", tname, non_null)
        res = _run_one_treatment(df, treatment)
        results.append(res)

    reason_analysis = reason_split_analysis(df)

    results.sort(key=lambda r: (
        0 if r.get("passes_refutation") else 1,
        -abs(r.get("effect_estimate") or 0),
    ))

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "label": scope_label,
            "product_number": product,
            "alloy_grade": grade,
            "n_heats": n,
            "scrap_rate_pct": scrap_rate,
        },
        "results": results,
        "reason_analysis": reason_analysis,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = (product or grade or "all").lower().replace("-", "").replace(" ", "_")[:40]
    json_path = REPORTS_DIR / f"causal_{slug}_{ts}.json"
    html_path = REPORTS_DIR / f"causal_{slug}_{ts}.html"

    json_path.write_text(json.dumps(report, indent=2, default=str))
    html_path.write_text(build_html_report(report))
    log.info("Report: %s", html_path)

    if not args.no_db:
        write_results_to_db(report)

    confirmed = [r for r in results if r.get("passes_refutation") and r.get("status") == "ok"]
    if confirmed:
        print(f"\n=== CONFIRMED CAUSAL FINDINGS ({scope_label}) ===")
        for r in confirmed:
            print(f"  {r['label']:50s}  ATE={r['effect_estimate']:+.4f}  "
                  f"CI=[{r.get('ci_lower','?')}, {r.get('ci_upper','?')}]  "
                  f"p={r['refutation_p_value']:.3f}")
    else:
        print(f"\nNo confirmed findings for {scope_label}.")


def main() -> None:
    global REFUTATION_SIMULATIONS
    parser = argparse.ArgumentParser(description="Causal scrap analysis via DoWhy")
    parser.add_argument("--product", default=None, help="Filter to one product number")
    parser.add_argument("--grade", default=None, help="Filter to one alloy grade")
    parser.add_argument("--all-scopes", action="store_true",
                        help="Run global + all products + all grades that meet --min-rows")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT)
    parser.add_argument("--simulations", type=int, default=REFUTATION_SIMULATIONS)
    parser.add_argument("--no-db", action="store_true", help="Skip DB write")
    args = parser.parse_args()

    REFUTATION_SIMULATIONS = args.simulations

    df_raw = load_latest_export()

    if args.all_scopes:
        # Global
        _run_scope(args, df_raw, product=None, grade=None)
        # All products with enough heats
        for product in sorted(df_raw["product_number"].dropna().unique()):
            _run_scope(args, df_raw, product=str(product), grade=None)
        # All grades with enough heats
        for grade in sorted(df_raw["alloy_grade"].dropna().unique()):
            _run_scope(args, df_raw, product=None, grade=str(grade))
        return

    _run_scope(args, df_raw, product=args.product, grade=args.grade)


if __name__ == "__main__":
    main()
