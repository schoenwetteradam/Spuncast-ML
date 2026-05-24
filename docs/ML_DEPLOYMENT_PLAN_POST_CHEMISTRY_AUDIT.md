# ML deployment plan — post-chemistry audit

**Status:** Data ready for production (program narrative, May 2026)  
**Owner:** Industry 4.0 Program  

This document captures the operational deployment narrative after the chemistry spec mapping audit (Grade C), and maps each phase to **this repository** (`Spuncast-ML`) so engineering work stays aligned with what is already implemented versus what still lives in Spuncast-Operations (database views, VM paths, package mirrors).

---

## How this repo fits the plan

| Plan area | In `Spuncast-ML` today | Notes |
|-----------|------------------------|--------|
| Training snapshots | `spuncast-ml export` | Reads `SPUNCAST_ML_SOURCE_VIEW` (default `v_ml_heat_dataset_v1`). Pin a new contract JSON if Operations ships `v_ml_heat_dataset_v2`. |
| Baseline / XGBoost training | `spuncast-ml train`, `evaluate`, `promote` | Python sklearn pipelines in `spuncast_ml/modeling.py`; not Minitab/R by default. External R/CART workflows remain optional on the VM. |
| Live 24/7 scoring | `scripts/score_heat_live.py` | Polls `v_ml_heat_early_score_v1`, feature set `early_remelt_decision`, writes `ml_heat_scores` and optionally `heat_recommendations`. |
| Dry-run scoring | `python scripts/score_heat_live.py --dry-run --once --test-heats N` | Connects and scores in memory; **no** inserts/updates. |
| Weekly automation | `scripts/run_pipeline.sh` | Export → train → evaluate → promote → weekly HTML report. Cron example: `deploy/crontab-ml.example` (if present). |
| Recommendation archival | `spuncast-ml archive-recommendations --days 60` | Uses `heat_recommendations_archive` when migration `sql/074_heat_recommendations_archive.sql` is applied. |
| SHAP in production scorer | `SCORE_ENABLE_SHAP=1` | Optional; requires `explanation_json` on `ml_heat_scores` per `sql/073_ml_heat_scores_explanation.sql`. |

**Path reminder:** Plans that reference `/opt/Spuncast-Operations/Spuncast-ML-main/` describe the **on-prem clone** of this repo. Prefer the same relative paths (`scripts/`, `sql/`) after `git pull`.

---

## Phase 1 — Retrain baseline (CART / alternative)

**Goal:** Refresh the interpretable baseline on ~15.5k clean operational heats.

**Data extraction (Operations + this repo):**

- Narrative plans often cite `v_ml_heat_dataset_v2` with ad-hoc `psql` CSV exports. In-repo, repeatable exports use the configured upstream view and contract validation.
- After Operations publishes `v_ml_heat_dataset_v2`, add `spuncast_ml/contracts/v_ml_heat_dataset_v2.json`, set `SPUNCAST_ML_SOURCE_VIEW=v_ml_heat_dataset_v2`, then run `spuncast-ml export`.

**Optional external CART (Minitab / R):** If you keep a parallel R workflow, store scripts and `.rds` artifacts on the deployment host; the live daemon in this repo expects **Python joblib** models under `artifacts/models/` for `early_remelt_decision`, not `.rds` files.

**Targets (from program plan):** Accuracy, AUC-ROC, precision/recall on scrap — track in evaluation reports and promotion gates.

---

## Phase 2 — Deploy live scoring

**Configuration:** Thresholds and poll interval are **environment variables**, not a `score_heat_live.yaml` file. See `docs/SPUNCAST_ML_CONFIGURATION_GUIDE.md` and `.env.example` (`SCORE_*`).

**Smoke test before writes:**

```bash
cd /path/to/Spuncast-ML
python3 scripts/score_heat_live.py --dry-run --once --test-heats 100
```

**Production loop:**

```bash
python3 scripts/score_heat_live.py
```

**systemd:** Use `WorkingDirectory` pointing at the repo root and `ExecStart=/usr/bin/python3 scripts/score_heat_live.py` (or the venv interpreter). The plan’s Docker wording applies if you wrap the same command in a container; `docker-compose.yml` in this repo already uses `score_heat_live.py` as the scorer image command.

---

## Phase 3 — XGBoost + SHAP (offline analysis)

Train and evaluate through the CLI for the chosen feature set (`pre_pour_in_process`, `post_run_diagnostic`, or `early_remelt_decision`). Dedicated `scripts/train_xgboost_shap.py` is **not** required unless you add it; batch training logic belongs in `spuncast_ml/modeling.py` / CLI for consistency.

Persist SHAP or driver narratives for Grafana using the live scorer’s `explanation_json` path when appropriate.

---

## Phase 4 — Operator alerts (Grafana)

The program draft used column names such as `risk_score`, `generated_at`, and `recommendation_status`. The migration shipped from this repo (`sql/071_heat_recommendations.sql`) uses:

| Draft name | Actual column |
|------------|----------------|
| `risk_score` | `scrap_probability` |
| `generated_at` | `created_at` |
| (status) | Treat unresolved rows as active: `resolved_at IS NULL` |

**Example — live table (top risks):**

```sql
SELECT heat_number,
       scrap_probability AS risk_score,
       decision_code,
       primary_driver,
       created_at
FROM heat_recommendations
WHERE resolved_at IS NULL
ORDER BY scrap_probability DESC NULLS LAST, created_at DESC
LIMIT 50;
```

**Example — high-risk count in the last hour:**

```sql
SELECT COUNT(*) AS high_risk_heats
FROM heat_recommendations
WHERE scrap_probability > 0.75
  AND created_at > NOW() - INTERVAL '1 hour'
  AND resolved_at IS NULL;
```

Wire Grafana alert thresholds to `SCORE_HOLD_THRESHOLD` / `SCORE_REMELT_THRESHOLD` policy on the plant floor.

---

## Phase 5 — Weekly retraining

Use `scripts/run_pipeline.sh` with cron or an orchestrator. Slack webhooks and “auto deploy best model” are **not** built into that shell script by default; add them in your environment if required, or extend the script minimally so secrets stay out of git.

---

## Phase 6 — Recommendation archival

Run periodically (for example monthly):

```bash
spuncast-ml archive-recommendations --days 60
```

Ensure `sql/074_heat_recommendations_archive.sql` is applied in Operations before enabling this in production.

---

## Success criteria (condensed)

- Clean exports from the authoritative dataset view with contract checks passing.
- Live scorer stable with operator-visible recommendations and sane alert volume.
- Promotion path documents when XGBoost (or other) replaces the baseline for a given feature set.

---

## Risks (unchanged intent)

- Offline Python wheels / air-gapped installs remain an IT coordination item on the plant VM.
- Threshold tuning reduces false-positive fatigue on the floor.
- Weekly jobs need monitoring and log retention independent of this repository.

---

## References

- `AGENTS.md` — Cloud VM constraints and command matrix  
- `docs/SPUNCAST_ML_CONFIGURATION_GUIDE.md` — Scorer and pipeline environment variables  
- `IT_DIRECTOR_HANDOVER.md` — Operations database dependencies  

**Source narrative date:** 2026-05-24  
