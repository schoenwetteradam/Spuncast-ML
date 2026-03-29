-- Grafana panel: "Scoring History"
-- Time-series of all scores over the past 24 hours, useful for monitoring
-- model behaviour and threshold calibration.

SELECT
    s.scored_at       AS "time",
    s.heat_number,
    s.scrap_probability,
    s.recommended_action,
    s.operator_action
FROM ml_heat_scores s
WHERE s.scored_at >= NOW() - INTERVAL '24 hours'
ORDER BY s.scored_at;
