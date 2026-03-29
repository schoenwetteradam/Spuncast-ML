-- Grafana panel: "Operator Response Rate"
-- Tracks how many flagged heats have been acted on vs. ignored,
-- broken down by recommended action.

SELECT
    s.recommended_action,
    COUNT(*)                                              AS total,
    COUNT(s.operator_action)                              AS acted,
    COUNT(*) - COUNT(s.operator_action)                   AS pending,
    ROUND(COUNT(s.operator_action)::numeric / NULLIF(COUNT(*), 0) * 100, 1) AS response_pct
FROM ml_heat_scores s
WHERE s.scored_at >= NOW() - INTERVAL '7 days'
GROUP BY s.recommended_action
ORDER BY s.recommended_action;
