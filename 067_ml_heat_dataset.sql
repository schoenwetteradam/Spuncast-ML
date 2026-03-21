-- ============================================================================
-- 067_ml_heat_dataset.sql
-- First-pass ML-ready heat dataset for scrap prediction.
--
-- Design goals:
-- - one row per heat_number
-- - stable, interpretable features sourced from v_heat_complete
-- - direct support for a future ML repo via SELECT * FROM v_ml_heat_dataset_v1
-- ============================================================================

CREATE OR REPLACE VIEW v_ml_heat_dataset_v1 AS
SELECT
    vhc.heat_id,
    vhc.heat_number,
    vhc.analysis_date,

    -- Identity / grouping
    vhc.alloy_grade,
    vhc.grade_name,
    vhc.chemistry_grade,
    vhc.spec_code,
    vhc.stock_code,
    vhc.job_number,
    vhc.furn_no,
    vhc.die_no,
    vhc.shift,
    vhc.melter,
    vhc.cmop,
    vhc.analysis_equipment_code,
    vhc.operator_id,

    -- Time anchors
    vhc.heat_start_ts,
    vhc.heat_end_ts,
    vhc.heat_created_at,
    vhc.pour_date,
    vhc.pour_entered_at,
    vhc.heat_treat_start_ts,
    vhc.heat_treat_end_ts,
    vhc.chem_last_tested_at,
    vhc.latest_scrap_ts,

    -- Pour / process variables
    vhc.tap_temp,
    vhc.pour_temp,
    vhc.die_temp_before_pour,
    vhc.die_rpm,
    vhc.pour_heat_status,
    vhc.pour_rpm_status,

    -- Missingness flags
    CASE WHEN vhc.tap_temp IS NULL THEN 1 ELSE 0 END AS tap_temp_missing,
    CASE WHEN vhc.pour_temp IS NULL THEN 1 ELSE 0 END AS pour_temp_missing,
    CASE WHEN vhc.die_temp_before_pour IS NULL THEN 1 ELSE 0 END AS die_temp_missing,
    CASE WHEN vhc.die_rpm IS NULL THEN 1 ELSE 0 END AS die_rpm_missing,
    CASE WHEN vhc.spec_code IS NULL OR BTRIM(vhc.spec_code) = '' THEN 1 ELSE 0 END AS spec_code_missing,
    CASE WHEN vhc.stock_code IS NULL OR BTRIM(vhc.stock_code) = '' THEN 1 ELSE 0 END AS stock_code_missing,
    CASE WHEN vhc.die_no IS NULL OR BTRIM(vhc.die_no) = '' THEN 1 ELSE 0 END AS die_no_missing,

    -- Simple process deltas
    CASE
        WHEN vhc.tap_temp IS NOT NULL AND vhc.pour_temp IS NOT NULL
        THEN vhc.tap_temp - vhc.pour_temp
    END AS tap_minus_pour_temp,
    CASE
        WHEN vhc.tap_temp IS NOT NULL AND vhc.die_temp_before_pour IS NOT NULL
        THEN vhc.tap_temp - vhc.die_temp_before_pour
    END AS tap_minus_die_temp,
    CASE
        WHEN vhc.pour_temp IS NOT NULL AND vhc.die_temp_before_pour IS NOT NULL
        THEN vhc.pour_temp - vhc.die_temp_before_pour
    END AS pour_minus_die_temp,

    -- Chemistry summary
    vhc.chem_element_count,
    vhc.chem_ok_count,
    vhc.chem_low_count,
    vhc.chem_high_count,
    vhc.chem_missing_spec_count,
    vhc.chem_no_spec_count,
    vhc.chem_heat_status,
    CASE
        WHEN vhc.chem_heat_status IS NOT NULL AND vhc.chem_heat_status NOT IN ('OK', 'NO_READING')
        THEN 1 ELSE 0
    END AS chem_not_ok_flag,
    CASE
        WHEN COALESCE(vhc.chem_low_count, 0) + COALESCE(vhc.chem_high_count, 0) > 0
        THEN 1 ELSE 0
    END AS has_any_chem_alert,

    -- Heat-treat summary
    vhc.equipment_code,
    vhc.cycle_name,
    vhc.setpoint_temp_f,
    vhc.actual_temp_f,
    vhc.heat_treat_operator_id,
    CASE
        WHEN vhc.setpoint_temp_f IS NOT NULL AND vhc.actual_temp_f IS NOT NULL
        THEN vhc.actual_temp_f - vhc.setpoint_temp_f
    END AS heat_treat_temp_delta,

    -- Lot / output summary
    vhc.lot_count,
    vhc.latest_lot_date,
    vhc.quantity_produced,
    vhc.quantity_scrapped,
    vhc.quantity_shipped,
    vhc.quantity_on_hold,

    -- Scrap summary and labels
    vhc.scrap_event_count,
    vhc.scrap_event_quantity,
    vhc.scrap_weight_lbs,
    vhc.scrap_estimated_cost,
    vhc.reason_code,
    vhc.defect_type,
    vhc.department,
    vhc.total_recorded_scrap_qty,
    vhc.scrap_rate_pct,
    vhc.has_scrap,
    CASE WHEN vhc.has_scrap THEN 1 ELSE 0 END AS scrap_flag,

    -- Lightweight reason-code grouping for first-pass classification
    CASE
        WHEN vhc.reason_code ~ '^[0-9]+$' THEN
            LPAD(((CAST(vhc.reason_code AS INTEGER) / 100) * 100)::TEXT, 3, '0') || '-' ||
            LPAD((((CAST(vhc.reason_code AS INTEGER) / 100) * 100) + 99)::TEXT, 3, '0')
        ELSE 'UNSPECIFIED'
    END AS reason_code_bucket
FROM v_heat_complete vhc;

COMMENT ON VIEW v_ml_heat_dataset_v1 IS
'ML-ready first-pass heat-level dataset for scrap prediction and explainable analytics.';
