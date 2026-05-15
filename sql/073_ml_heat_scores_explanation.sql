-- Optional SHAP / driver payload persisted by scripts/score_heat_live.py
-- when SCORE_ENABLE_SHAP=1. Apply in Operations DB alongside 070_ml_scores.sql.

ALTER TABLE ml_heat_scores
    ADD COLUMN IF NOT EXISTS explanation_json JSONB;

COMMENT ON COLUMN ml_heat_scores.explanation_json IS
    'Top contributing features (SHAP) or other structured explainability payload.';
