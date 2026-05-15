# Spuncast ML Configuration Guide

**Date:** May 15, 2026  
**Objective:** Run scrap-risk scoring in near real time, explain drivers for operators, keep training on a weekly cadence, and surface everything in Grafana—while leaving explicit room to align numeric thresholds with any legacy **CART / Minitab** baseline once that artifact is available.

This document is the **operator + engineering configuration guide** for the **Spuncast-ML** repository. Database views, Operations `docker-compose`, and Minitab files live outside this repo; items that belong in **Spuncast-Operations** are called out explicitly.

---

## Part 1: ML Scoring Strategy (Design Targets)

| Question | Target answer | Implementation in this repo |
|----------|---------------|------------------------------|
| **Scoring mode** | Near–real-time 24/7 | `scripts/score_heat_live.py` polls `v_ml_heat_early_score_v1` on `SCORE_POLL_INTERVAL_SEC` (default 180s). |
| **Training frequency** | Weekly (Monday 02:00) | `scripts/run_pipeline.sh` (and `.ps1`) run `export → train → evaluate → promote → weekly-report`. Schedule via cron or Airflow; see `deploy/crontab-ml.example`. |
| **Causation discovery** | SHAP (+ partial dependence in offline analysis) | Optional `SCORE_ENABLE_SHAP=1` writes structured summaries to `ml_heat_scores.explanation_json` when column exists (`sql/073_ml_heat_scores_explanation.sql`). Partial dependence remains a **training / notebook** workflow, not the hot path. |
| **Operator notification** | Grafana live heats | SQL panel queries under `grafana/ml_active_heat_predictions.sql`, `grafana/ml_model_performance.sql`, `grafana/ml_causation_insights.sql`—import into Operations Grafana. |
| **Archive recommendations** | After 60 days | `sql/074_heat_recommendations_archive.sql` + `spuncast-ml archive-recommendations --days 60` (see cron example). |
| **Confidence thresholds** | TBD vs CART | **Configurable today** via `SCORE_REMELT_THRESHOLD`, `SCORE_HOLD_THRESHOLD`, `SCORE_CAUTION_THRESHOLD`, `SCORE_ADVISORY_THRESHOLD`, and optional `SCORE_WRITE_ADVISORY_ROWS` (see Part 5). |

---

## Part 2: Legacy CART (Minitab) Baseline

**CART** (Classification And Regression Trees) is a transparent, rule-like model: easy for operators to reason about, but typically less flexible than gradient boosting for rare scrap events.

### Questions the business still needs to answer

1. **Where is the CART model stored?** (`.mpx`, spreadsheet, internal memo, or rules-only?)  
2. **Exact splits and drivers** (die temp, tap temp, CR, pour time, etc.)  
3. **Measured performance** on historical heats (accuracy, precision, recall at the chosen alert cut).

Until those answers exist, Spuncast-ML continues to ship the **trained sklearn pipeline** (`spuncast-ml train`, artifacts under `artifacts/models/`) as the production scorer, with **promotion** (`spuncast-ml promote`) comparing successive training runs—not CART parity.

**Optional future work (not automated here):** digitize agreed CART rules into a small Python module or the Operations rules engine, run it **in parallel** for a calibration period, and tune `SCORE_*` thresholds to match historical CART alert rates.

---

## Part 3: Model Evolution Roadmap

| Phase | Scope | Repo status |
|-------|--------|---------------|
| **1** | Rules / CART parity + instrumentation | Capture CART externally; tune `SCORE_*` env vars to mirror alert mix. |
| **2** | ML ensemble vs baseline | `train` / `evaluate` / `promote` implement weekly retrain with a relative improvement gate (default 5% on validation accuracy or PR-AUC fallback). XGBoost/LightGBM are **installed** for future extensions; default trainers remain sklearn pipelines in `spuncast_ml/modeling.py`. |
| **3** | Full loop | Live scorer + weekly automation + drift CLI (`monitor-drift`) + operator feedback JSONL (`feedback` command). |

---

## Part 4: Variable Inventory & Gap Analysis

### Already represented in ML contracts

Feature contracts live in `spuncast_ml/contracts/` (for example `v_ml_heat_dataset_v1.json`) with companion SQL notes under `sql/` (for example `068_ml_heat_early_score.sql`).

### Verifying columns in PostgreSQL (Operations)

Run against the live `spuncast` database (host/path will differ by site):

```bash
docker exec spuncast_db psql -U postgres -d spuncast <<'EOF'
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('pour_logs', 'chem_readings', 'heat_treat_log', 'scrap_events')
ORDER BY table_name, ordinal_position;
EOF
```

Post the output when scoping new features; anything absent from contracts needs a **contract bump** plus Operations view/migration work.

### Common gaps called out in planning

Examples often requested in foundry ML: shift patterns, ambient conditions, raw-material lot traceability, die maintenance recency, ladle additions, Baume readings, etc. Each requires **Operations ingestion + view columns** before Spuncast-ML can consume it.

---

## Part 5: Confidence Threshold Calibration

### Current ML mapping (configurable)

The live daemon maps **model scrap probability** (positive class) to:

| Grafana-style band | `decision_code` | Default probability gate |
|----------------------|-----------------|---------------------------|
| Red / HOLD | `HOLD` | `p ≥ SCORE_HOLD_THRESHOLD` (default **0.80**) |
| Yellow / CAUTION | `CAUTION` | `SCORE_CAUTION_THRESHOLD ≤ p < SCORE_HOLD_THRESHOLD` (default caution **0.65**) |
| Blue / ADVISORY | `ADVISORY` | `p < SCORE_CAUTION_THRESHOLD` but still written when policy allows |

**`ml_heat_scores.predicted_flag`** remains `1` when `p ≥ SCORE_REMELT_THRESHOLD` (default **0.65**) so the classifier decision stays stable even if alert colours change.

**`heat_recommendations` insert policy**

- Default (`SCORE_WRITE_ADVISORY_ROWS` unset / 0): upsert recommendations only when `p ≥ SCORE_REMELT_THRESHOLD`, but `decision_code` still reflects HOLD vs CAUTION vs ADVISORY using the hold/caution thresholds.  
- Optional noise for dashboards (`SCORE_WRITE_ADVISORY_ROWS=1`): upsert rows down to `SCORE_ADVISORY_THRESHOLD` (default **0.50**).

### Example calibration once CART rules are known

If Minitab historically fired **HOLD** at an equivalent posterior of ~0.85, set `SCORE_HOLD_THRESHOLD=0.85`. If **CAUTION** began around 0.70, set `SCORE_CAUTION_THRESHOLD=0.70`. Keep `SCORE_REMELT_THRESHOLD` aligned with the minimum probability you want **counted as a positive scrap prediction** for metrics.

Document the final numbers in your runbook and mirror them in `.env` / compose for each environment.

---

## Part 6: Grafana Dashboards

### Panel queries shipped in this repo

| Panel intent | SQL file |
|--------------|----------|
| Active heat predictions | `grafana/ml_active_heat_predictions.sql` |
| Model performance vs outcomes | `grafana/ml_model_performance.sql` (needs `ml_heat_scores.actual_scrap_flag` backfill) |
| Causation / SHAP payloads | `grafana/ml_causation_insights.sql` (needs `explanation_json` column) |

Wire them as PostgreSQL panels in Operations Grafana. The mock ASCII dashboards in the original brief are **UX targets**—translate labels/colours in Grafana thresholds to match `decision_code` and `scrap_probability`.

---

## Part 7: Implementation Checklist

### Stakeholder / Operations prerequisites

- [ ] Provide CART / Minitab artifact or written rules + historic performance.  
- [ ] Run the column inventory query (Part 4) and close schema gaps in Operations.  
- [ ] Agree on `SCORE_*` thresholds per environment (dev/stage/prod).  
- [ ] Define operator workflow: who sees Grafana, how recommendations are acknowledged (`resolved_at` / `resolved_by` today are manual columns—portal integration is Operations-side).

### Spuncast-ML repository (done unless noted)

- [x] Live scorer with SHAP option + JSON persistence (`scripts/score_heat_live.py`, `sql/073_ml_heat_scores_explanation.sql`).  
- [x] Weekly automation + promotion + HTML summary (`scripts/run_pipeline.sh`, `spuncast_ml/promotion.py`, `spuncast_ml/reporting.py`).  
- [x] Grafana-friendly SQL snippets (`grafana/ml_*.sql`).  
- [x] Tiered `decision_code` + optional advisory inserts (`SCORE_WRITE_ADVISORY_ROWS`).  
- [x] Recommendation archival migration + CLI (`sql/074_heat_recommendations_archive.sql`, `spuncast-ml archive-recommendations`).  
- [ ] **Operations** apply SQL migrations `073` + `074` in prod.  
- [ ] **Operations** compose service for `ml_score_daemon` (build context pointing at this repo).  
- [ ] Digitize CART into code (only after artifact exists—tracked above).

### Post-deployment cadence

- Daily: drift report (`monitor-drift`) + archive job (`archive-recommendations`).  
- Weekly: `run_pipeline.sh` + review `reports/training_*.log` / `weekly_*.html`.  
- Monthly: review SHAP / feature drift narratives with metallurgy + engineering.  
- Quarterly: operator refresher tied to Grafana UX changes.

---

## Part 8: Immediate Next Steps

1. **Attach the CART reference** (file or prose rules) so thresholds can be matched quantitatively.  
2. **Share the information_schema output** for core tables to close variable gaps.  
3. **Pick production thresholds** (`SCORE_HOLD_THRESHOLD`, `SCORE_CAUTION_THRESHOLD`, `SCORE_REMELT_THRESHOLD`, optional advisory policy).  
4. **Confirm acknowledgement workflow** with Operations (who resolves rows, audit expectations).

---

## Quick reference: key commands

| Goal | Command |
|------|---------|
| Export training snapshot | `spuncast-ml export` |
| Train latest feature set | `spuncast-ml train --feature-set early_remelt_decision` |
| Evaluate | `spuncast-ml evaluate --feature-set early_remelt_decision` |
| Weekly bundle | `bash scripts/run_pipeline.sh` |
| Live scorer | `python3 scripts/score_heat_live.py` |
| Archive old recs | `spuncast-ml archive-recommendations --days 60` |
| Drift monitoring | `spuncast-ml monitor-drift --feature-set early_remelt_decision` |

Environment variables are documented in `.env.example`.

---

## Summary configuration snapshot

| Item | Value / status |
|------|----------------|
| Scoring | Near–real-time daemon (`score_heat_live.py`) |
| Training | Weekly (`run_pipeline.sh` + cron template) |
| Models on disk | `artifacts/models/scrap_baseline_<feature_set>_*.joblib` |
| Alert tiers | `HOLD` / `CAUTION` / `ADVISORY` via `SCORE_*_THRESHOLD` |
| Explanations | Optional SHAP JSON on `ml_heat_scores` |
| Archival | 60-day default via `archive-recommendations` |
| CART alignment | Pending stakeholder inputs (Part 2 / Part 8) |

When CART rules and performance numbers land, update **Part 5** with the agreed literal thresholds and link to the change ticket / runbook entry.
