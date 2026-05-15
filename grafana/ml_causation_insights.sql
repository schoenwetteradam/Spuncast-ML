-- Grafana panel: SHAP-style payloads persisted by score_heat_live when explanation_json exists.
-- Apply sql/073_ml_heat_scores_explanation.sql in Operations before this panel will return rows.

SELECT
    heat_number,
    scrap_probability,
    explanation_json -> 'top_features' AS top_features,
    scored_at
FROM ml_heat_scores
WHERE explanation_json IS NOT NULL
ORDER BY scored_at DESC
LIMIT 100;
