-- Grafana panel: "Re-melt Candidates"
-- Shows heats that exceed the re-melt threshold and have not yet been
-- acted on by an operator.  Join ml_heat_scores with heat_recommendations
-- to surface the primary contributing feature alongside the probability.

SELECT
    s.heat_number,
    s.scrap_probability,
    s.recommended_action,
    r.recommendation_text,
    r.primary_driver,
    s.scored_at
FROM ml_heat_scores s
LEFT JOIN heat_recommendations r
    ON r.heat_number = s.heat_number
WHERE s.scrap_probability >= 0.65
  AND s.operator_action IS NULL
  AND s.scored_at >= NOW() - INTERVAL '8 hours'
ORDER BY s.scrap_probability DESC;
