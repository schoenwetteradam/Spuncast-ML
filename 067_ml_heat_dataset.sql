-- ML export surface: one row per operational heat with pour/chem/HT/lot/scrap context.
-- Depends: v_heat_complete_operational, data_quality_violations, heat_recommendations, scrap_events.
--
-- Column set aligned with the downstream ML app pinned dataset contract (identity, timestamps, engineered flags).
-- Post-outcome scrap diagnostics remain on this view for labels/evaluation — strip from model inputs downstream.
--
-- Column renames or reordering require dropping first: CREATE OR REPLACE VIEW cannot change
-- existing column names/order. Drop dependent view first, then this view.

DROP VIEW IF EXISTS v_ml_heat_early_score_v1 CASCADE;
DROP VIEW IF EXISTS v_ml_heat_dataset_v1 CASCADE;

-- Helper: resolve the best-scoped process_control_profile for a heat without
-- cross-product contamination. Stock-specific profiles beat spec-level which beat
-- grade-level; within each tier, instruction-sheet sources beat historical baselines.
-- Used both in this view and in the early-score view via v_ml_heat_dataset_v1.
CREATE OR REPLACE FUNCTION fn_resolve_pcp_for_heat(
    p_stock_code TEXT,
    p_spec_code  TEXT,
    p_grade      TEXT
)
RETURNS SETOF process_control_profiles
LANGUAGE SQL STABLE AS
$$
    SELECT p.*
    FROM process_control_profiles p
    WHERE p.is_active = TRUE
      AND (
            (p_stock_code IS NOT NULL AND p.stock_code = p_stock_code)
         OR (p_spec_code IS NOT NULL AND p.spec_code = p_spec_code AND p.stock_code IS NULL)
         OR (p_grade     IS NOT NULL AND p.grade_name = p_grade    AND p.stock_code IS NULL AND p.spec_code IS NULL)
      )
    ORDER BY
        CASE
            WHEN p.stock_code = p_stock_code THEN 1
            WHEN p.spec_code  = p_spec_code AND p.stock_code IS NULL THEN 2
            WHEN p.grade_name = p_grade     AND p.stock_code IS NULL AND p.spec_code IS NULL THEN 3
            ELSE 9
        END,
        CASE WHEN p.source = 'historical_good_heat_baseline' THEN 2 ELSE 1 END,
        p.updated_at DESC NULLS LAST,
        p.id DESC
    LIMIT 1
$$;

CREATE VIEW v_ml_heat_dataset_v1 AS
WITH scrap_agg AS (
    SELECT
        se.heat_number,
        COUNT(*)::integer AS scrap_events_row_count,
        SUM(COALESCE(se.quantity, 0))::numeric AS scrap_events_qty_sum,
        SUM(COALESCE(se.weight_lbs, 0))::numeric AS scrap_events_weight_lbs_sum,
        MAX(se.ts) AS scrap_events_last_ts,
        COUNT(DISTINCT NULLIF(TRIM(se.reason_code), ''))::integer AS scrap_distinct_reason_code_count,
        COUNT(DISTINCT NULLIF(TRIM(se.defect_type), ''))::integer AS scrap_distinct_defect_type_count
    FROM scrap_events se
    WHERE NULLIF(TRIM(se.heat_number), '') IS NOT NULL
    GROUP BY se.heat_number
),
scrap_reasons AS (
    SELECT
        se.heat_number,
        string_agg(DISTINCT NULLIF(TRIM(se.reason_code), ''), ' | ' ORDER BY NULLIF(TRIM(se.reason_code), '')) AS scrap_reason_codes_csv
    FROM scrap_events se
    WHERE NULLIF(TRIM(se.heat_number), '') IS NOT NULL
      AND NULLIF(TRIM(se.reason_code), '') IS NOT NULL
    GROUP BY se.heat_number
),
scrap_defects AS (
    SELECT
        se.heat_number,
        string_agg(DISTINCT NULLIF(TRIM(se.defect_type), ''), ' | ' ORDER BY NULLIF(TRIM(se.defect_type), '')) AS scrap_defect_types_csv
    FROM scrap_events se
    WHERE NULLIF(TRIM(se.heat_number), '') IS NOT NULL
      AND NULLIF(TRIM(se.defect_type), '') IS NOT NULL
    GROUP BY se.heat_number
)
SELECT
    vhc.heat_id,
    vhc.heat_number,
    vhc.product_number,
    COALESCE(vhc.pour_entered_at, vhc.pour_date::timestamptz) AS feature_cutoff_ts,
    COALESCE(vhc.alloy_grade, vhc.grade_name, vhc.chemistry_grade, 'UNKNOWN') AS alloy_grade,
    vhc.grade_name,
    vhc.chemistry_grade,
    vhc.analysis_date,
    vhc.pour_date,
    vhc.shift,
    vhc.job_number,
    vhc.stock_code,
    vhc.spec_code,
    vhc.furn_no,
    vhc.die_no,
    vhc.melter,
    vhc.cmop,
    vhc.analysis_equipment_code,
    vhc.operator_id,
    vhc.heat_start_ts,
    vhc.heat_end_ts,
    vhc.heat_created_at,
    vhc.pour_entered_at,
    vhc.heat_treat_start_ts,
    vhc.heat_treat_end_ts,
    vhc.chem_last_tested_at,
    vhc.latest_scrap_ts,
    CASE WHEN vhc.tap_temp IS NULL THEN 1 ELSE 0 END AS tap_temp_missing,
    CASE WHEN vhc.pour_temp IS NULL THEN 1 ELSE 0 END AS pour_temp_missing,
    CASE WHEN vhc.die_temp_before_pour IS NULL THEN 1 ELSE 0 END AS die_temp_missing,
    CASE WHEN vhc.die_rpm IS NULL THEN 1 ELSE 0 END AS die_rpm_missing,
    CASE WHEN NULLIF(BTRIM(vhc.spec_code), '') IS NULL THEN 1 ELSE 0 END AS spec_code_missing,
    CASE WHEN NULLIF(BTRIM(vhc.stock_code), '') IS NULL THEN 1 ELSE 0 END AS stock_code_missing,
    CASE WHEN NULLIF(BTRIM(vhc.die_no), '') IS NULL THEN 1 ELSE 0 END AS die_no_missing,
    vhc.tap_temp - vhc.pour_temp AS tap_minus_pour_temp,
    vhc.tap_temp - vhc.die_temp_before_pour AS tap_minus_die_temp,
    vhc.pour_temp - vhc.die_temp_before_pour AS pour_minus_die_temp,
    CASE
        WHEN COALESCE(vhc.chem_low_count, 0) > 0 OR COALESCE(vhc.chem_high_count, 0) > 0 THEN 1
        ELSE 0
    END AS chem_not_ok_flag,
    CASE
        WHEN COALESCE(vhc.chem_low_count, 0) > 0
          OR COALESCE(vhc.chem_high_count, 0) > 0
          OR COALESCE(vhc.chem_missing_spec_count, 0) > 0
          OR COALESCE(vhc.chem_no_spec_count, 0) > 0
        THEN 1
        ELSE 0
    END AS has_any_chem_alert,
    vhc.heat_treat_operator_id,
    CASE
        WHEN vhc.setpoint_temp_f IS NOT NULL AND vhc.actual_temp_f IS NOT NULL
        THEN vhc.actual_temp_f - vhc.setpoint_temp_f
        ELSE NULL::numeric
    END AS heat_treat_temp_delta,
    vhc.lot_count,
    vhc.latest_lot_date,
    vhc.has_scrap,
    CASE
        WHEN NULLIF(TRIM(vhc.reason_code), '') IS NULL
          OR UPPER(TRIM(vhc.reason_code)) = 'UNSPECIFIED'
        THEN 'UNKNOWN'
        ELSE SPLIT_PART(TRIM(vhc.reason_code), ' ', 1)
    END AS reason_code_bucket,
    vhc.tap_temp,
    vhc.pour_temp,
    vhc.die_temp_before_pour,
    vhc.die_rpm,
    vhc.spin_time_min,
    vhc.pour_time_sec,
    (
        SELECT cr.measured_value
        FROM chem_readings cr
        WHERE cr.heat_number = vhc.heat_number
          AND UPPER(TRIM(cr.element)) = 'CR'
        ORDER BY COALESCE(cr.tested_at, cr.imported_at) DESC NULLS LAST
        LIMIT 1
    ) AS cr_pct,
    vhc.tap_pct_of_band,
    vhc.pour_pct_of_band,
    vhc.die_pct_of_band,
    vhc.rpm_deviation_pct,
    vhc.chem_element_count,
    vhc.chem_ok_count,
    vhc.chem_low_count,
    vhc.chem_high_count,
    vhc.chem_missing_spec_count,
    vhc.chem_no_spec_count,
    vhc.chem_heat_status,
    vhc.pour_heat_status,
    vhc.pour_rpm_status,
    vhc.equipment_code,
    vhc.cycle_name,
    vhc.setpoint_temp_f,
    vhc.actual_temp_f,
    vhc.quantity_produced,
    vhc.quantity_scrapped,
    vhc.quantity_shipped,
    vhc.quantity_on_hold,
    vhc.scrap_event_count,
    vhc.scrap_event_quantity,
    vhc.scrap_weight_lbs,
    vhc.scrap_estimated_cost,
    vhc.reason_code,
    vhc.defect_type,
    vhc.department,
    vhc.total_recorded_scrap_qty,
    vhc.scrap_rate_pct,
    CASE WHEN vhc.has_scrap THEN 1 ELSE 0 END AS scrap_flag,
    COALESCE(sa.scrap_events_row_count, 0) AS scrap_events_row_count,
    COALESCE(sa.scrap_events_qty_sum, 0) AS scrap_events_qty_sum,
    COALESCE(sa.scrap_events_weight_lbs_sum, 0) AS scrap_events_weight_lbs_sum,
    sa.scrap_events_last_ts,
    COALESCE(sa.scrap_distinct_reason_code_count, 0) AS scrap_distinct_reason_code_count,
    COALESCE(sa.scrap_distinct_defect_type_count, 0) AS scrap_distinct_defect_type_count,
    sr.scrap_reason_codes_csv,
    sd.scrap_defect_types_csv,
    CASE WHEN dq.open_violation_count > 0 THEN 1 ELSE 0 END AS has_open_data_quality_violation,
    COALESCE(dq.open_violation_count, 0) AS open_data_quality_violation_count,
    reco.latest_decision_code,
    reco.latest_risk_score,
    -- Instruction-target deviation features (from the correctly-scoped process_control_profile).
    -- Signed fraction of the instruction band half-width: 0 = on target, ±1 = at limit, >|1| = outside band.
    -- NULL when no instruction profile matches or the band width is zero.
    pcp.source AS instruction_source,
    CASE
        WHEN pcp.tap_temp_target IS NOT NULL AND vhc.tap_temp IS NOT NULL
             AND (pcp.tap_temp_max - pcp.tap_temp_min) > 0
        THEN (vhc.tap_temp - pcp.tap_temp_target)
             / ((pcp.tap_temp_max - pcp.tap_temp_min) / 2.0)
        ELSE NULL
    END AS tap_deviation_from_instruction_pct,
    CASE
        WHEN pcp.pour_temp_target IS NOT NULL AND vhc.pour_temp IS NOT NULL
             AND (pcp.pour_temp_max - pcp.pour_temp_min) > 0
        THEN (vhc.pour_temp - pcp.pour_temp_target)
             / ((pcp.pour_temp_max - pcp.pour_temp_min) / 2.0)
        ELSE NULL
    END AS pour_deviation_from_instruction_pct,
    CASE
        WHEN pcp.die_temp_target IS NOT NULL AND vhc.die_temp_before_pour IS NOT NULL
             AND (pcp.die_temp_max - pcp.die_temp_min) > 0
        THEN (vhc.die_temp_before_pour - pcp.die_temp_target)
             / ((pcp.die_temp_max - pcp.die_temp_min) / 2.0)
        ELSE NULL
    END AS die_deviation_from_instruction_pct,
    CASE
        WHEN pcp.die_rpm_target IS NOT NULL AND vhc.die_rpm IS NOT NULL
             AND (pcp.die_rpm_max - pcp.die_rpm_min) > 0
        THEN (vhc.die_rpm - pcp.die_rpm_target)
             / ((pcp.die_rpm_max - pcp.die_rpm_min) / 2.0)
        ELSE NULL
    END AS rpm_deviation_from_instruction_pct
FROM v_heat_complete_operational vhc
LEFT JOIN scrap_agg sa ON sa.heat_number = vhc.heat_number
LEFT JOIN scrap_reasons sr ON sr.heat_number = vhc.heat_number
LEFT JOIN scrap_defects sd ON sd.heat_number = vhc.heat_number
LEFT JOIN (
    SELECT
        heat_number,
        COUNT(*) AS open_violation_count
    FROM data_quality_violations
    WHERE violation_status = 'open'
    GROUP BY heat_number
) dq
    ON dq.heat_number = vhc.heat_number
LEFT JOIN (
    SELECT DISTINCT ON (heat_number)
        heat_number,
        decision_code AS latest_decision_code,
        risk_score AS latest_risk_score
    FROM heat_recommendations
    ORDER BY heat_number, generated_at DESC, id DESC
) reco
    ON reco.heat_number = vhc.heat_number
LEFT JOIN LATERAL fn_resolve_pcp_for_heat(vhc.stock_code, vhc.spec_code, vhc.alloy_grade) pcp ON TRUE;

COMMENT ON VIEW v_ml_heat_dataset_v1 IS
    'One row per operational heat for ML export: pour/chem/HT/lot/scrap + scrap_events aggregates, '
    'product_number, feature_cutoff_ts (pour time for leakage control), latest Cr (cr_pct) from chem_readings. '
    'Instruction deviation columns (tap/pour/die/rpm_deviation_from_instruction_pct, instruction_source) '
    'use the correctly-scoped process_control_profile — stock-specific > spec-level > grade-level, '
    'instruction-sheet sources preferred over historical_good_heat_baseline. '
    'Post-outcome scrap diagnostics (quantity_scrapped, scrap_event_count, scrap_event_quantity, '
    'scrap_weight_lbs, total_recorded_scrap_qty, scrap_rate_pct, scrap_events_row_count, '
    'scrap_events_qty_sum, scrap_events_weight_lbs_sum, scrap_events_last_ts, '
    'scrap_distinct_reason_code_count, scrap_distinct_defect_type_count, scrap_reason_codes_csv, '
    'scrap_defect_types_csv) are for labels/evaluation only — strip from model inputs.';
