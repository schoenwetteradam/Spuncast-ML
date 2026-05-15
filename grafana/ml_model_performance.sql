-- Grafana panel: approximate classifier performance once outcomes are backfilled.
-- Requires ml_heat_scores.actual_scrap_flag to be populated post-run.

WITH labeled AS (
    SELECT
        predicted_flag,
        actual_scrap_flag
    FROM ml_heat_scores
    WHERE actual_scrap_flag IS NOT NULL
      AND scored_at >= NOW() - INTERVAL '90 days'
)
SELECT
    COUNT(*) FILTER (WHERE predicted_flag = 1 AND actual_scrap_flag = 1) AS true_positive,
    COUNT(*) FILTER (WHERE predicted_flag = 1 AND actual_scrap_flag = 0) AS false_positive,
    COUNT(*) FILTER (WHERE predicted_flag = 0 AND actual_scrap_flag = 1) AS false_negative,
    COUNT(*) FILTER (WHERE predicted_flag = 0 AND actual_scrap_flag = 0) AS true_negative,
    COUNT(*) AS labeled_rows,
    CASE WHEN COUNT(*) > 0
         THEN (
             COUNT(*) FILTER (WHERE predicted_flag = actual_scrap_flag)::double precision
             / COUNT(*)::double precision
         )
    END AS accuracy
FROM labeled;
