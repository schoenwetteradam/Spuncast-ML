-- Grafana panel: recent live scores with optional operator recommendation row.
-- Expects PostgreSQL datasource pointed at the Spuncast operations database.

SELECT
    s.heat_number,
    s.scrap_probability,
    s.recommended_action,
    s.model_version,
    s.scored_at,
    r.decision_code,
    r.primary_driver
FROM ml_heat_scores s
LEFT JOIN heat_recommendations r ON r.heat_number = s.heat_number
WHERE s.scored_at >= NOW() - INTERVAL '48 hours'
ORDER BY s.scored_at DESC
LIMIT 200;
