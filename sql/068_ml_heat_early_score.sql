-- Early-score view for near-real-time re-melt decisions.
-- Available the moment a pour_log row exists, before heat treat or lot data.
-- Companion to 067_ml_heat_dataset.sql (Spuncast-Operations).

CREATE OR REPLACE VIEW v_ml_heat_early_score_v1 AS
SELECT
    pl.heat_number,
    pl.pour_date,
    pl.shift,
    pl.furn_no,
    pl.die_no,
    pl.spec_code,
    pl.alloy_grade,
    -- Pour process variables
    pl.tap_temp,
    pl.pour_temp,
    pl.die_temp_before_pour,
    pl.die_rpm,
    pl.spin_time_min,
    pl.pour_time_sec,
    -- pct_of_band deviations (already computed in v_heat_complete)
    vhc.tap_pct_of_band,
    vhc.pour_pct_of_band,
    vhc.die_pct_of_band,
    vhc.rpm_deviation_pct,
    -- Missingness flags
    CASE WHEN pl.tap_temp IS NULL THEN 1 ELSE 0 END AS tap_temp_missing,
    CASE WHEN pl.pour_temp IS NULL THEN 1 ELSE 0 END AS pour_temp_missing,
    CASE WHEN pl.die_temp_before_pour IS NULL THEN 1 ELSE 0 END AS die_temp_missing,
    CASE WHEN pl.die_rpm IS NULL THEN 1 ELSE 0 END AS die_rpm_missing,
    -- Process deltas
    pl.tap_temp - pl.pour_temp AS tap_minus_pour_temp,
    pl.tap_temp - pl.die_temp_before_pour AS tap_minus_die_temp,
    pl.pour_temp - pl.die_temp_before_pour AS pour_minus_die_temp,
    vhc.cr_pct,
    -- Per-element chemistry actuals (pivoted from chem_readings)
    MAX(CASE WHEN cr.element_code = 'CR' THEN cr.actual_value END) AS chem_cr,
    MAX(CASE WHEN cr.element_code = 'NI' THEN cr.actual_value END) AS chem_ni,
    MAX(CASE WHEN cr.element_code = 'MN' THEN cr.actual_value END) AS chem_mn,
    MAX(CASE WHEN cr.element_code = 'SI' THEN cr.actual_value END) AS chem_si,
    MAX(CASE WHEN cr.element_code = 'C'  THEN cr.actual_value END) AS chem_c,
    MAX(CASE WHEN cr.element_code = 'MO' THEN cr.actual_value END) AS chem_mo,
    -- Chemistry compliance summary
    vhc.chem_element_count,
    vhc.chem_ok_count,
    vhc.chem_low_count,
    vhc.chem_high_count,
    vhc.chem_heat_status,
    CASE WHEN vhc.chem_low_count + vhc.chem_high_count > 0
         THEN 1 ELSE 0
    END AS has_any_chem_alert,
    -- Target label (for training only — excluded from inference feature set)
    CASE WHEN vhc.has_scrap THEN 1 ELSE 0 END AS scrap_flag
FROM pour_logs pl
LEFT JOIN v_heat_complete vhc ON vhc.heat_number = pl.heat_number
LEFT JOIN chem_readings cr   ON cr.heat_number  = pl.heat_number
GROUP BY
    pl.heat_number, pl.pour_date, pl.shift, pl.furn_no, pl.die_no,
    pl.spec_code, pl.alloy_grade,
    pl.tap_temp, pl.pour_temp, pl.die_temp_before_pour,
    pl.die_rpm, pl.spin_time_min, pl.pour_time_sec,
    vhc.tap_pct_of_band, vhc.pour_pct_of_band,
    vhc.die_pct_of_band, vhc.rpm_deviation_pct,
    vhc.chem_element_count, vhc.chem_ok_count,
    vhc.chem_low_count, vhc.chem_high_count,
    vhc.chem_heat_status, vhc.cr_pct, vhc.has_scrap;
