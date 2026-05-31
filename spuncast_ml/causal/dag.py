"""
Causal DAG for foundry scrap analysis.

Confounders (exogenous):
  - alloy_grade: determines FPS targets, inherent metallurgical scrap risk
  - furn_no:     furnace-specific temperature behaviour

Treatments (all analyzed independently via backdoor identification):
  - FPS deviations (tap/pour/die/rpm/spin_time/pour_time from FPS sheet)
  - Instruction deviations (tap/pour/die/rpm from process_control_profiles)
  - Funnel: wrong_funnel_flag, funnel_deviation_from_fps_pct
  - Charge composition: charge_scrap_pct, charge_virgin_pct
  - Chemistry compliance: chem_not_ok_flag, chem_low_count, chem_high_count
  - Heat treat: heat_treat_temp_delta
  - Temp differentials: tap_minus_pour_temp, pour_minus_die_temp

All column names match v_ml_heat_dataset_v1 (contract schema_version 113+).
"""

# GML directed graph encoding the causal structure.
# charge_scrap_pct feeds chem_not_ok_flag as a mediator.
SCRAP_DAG_GML = """
graph [
  directed 1

  node [ id "alloy_grade"   label "alloy_grade" ]
  node [ id "furn_no"       label "furn_no" ]

  node [ id "tap_deviation_from_fps_pct"        label "tap_deviation_from_fps_pct" ]
  node [ id "pour_deviation_from_fps_pct"       label "pour_deviation_from_fps_pct" ]
  node [ id "die_deviation_from_fps_pct"        label "die_deviation_from_fps_pct" ]
  node [ id "rpm_deviation_from_fps_pct"        label "rpm_deviation_from_fps_pct" ]
  node [ id "spin_time_deviation_from_fps_pct"  label "spin_time_deviation_from_fps_pct" ]
  node [ id "pour_time_deviation_from_fps_pct"  label "pour_time_deviation_from_fps_pct" ]
  node [ id "funnel_deviation_from_fps_pct"     label "funnel_deviation_from_fps_pct" ]
  node [ id "wrong_funnel_flag"                 label "wrong_funnel_flag" ]

  node [ id "tap_deviation_from_instruction_pct"   label "tap_deviation_from_instruction_pct" ]
  node [ id "pour_deviation_from_instruction_pct"  label "pour_deviation_from_instruction_pct" ]
  node [ id "die_deviation_from_instruction_pct"   label "die_deviation_from_instruction_pct" ]
  node [ id "rpm_deviation_from_instruction_pct"   label "rpm_deviation_from_instruction_pct" ]

  node [ id "charge_scrap_pct"    label "charge_scrap_pct" ]
  node [ id "charge_virgin_pct"   label "charge_virgin_pct" ]
  node [ id "chem_not_ok_flag"    label "chem_not_ok_flag" ]
  node [ id "chem_low_count"      label "chem_low_count" ]
  node [ id "chem_high_count"     label "chem_high_count" ]

  node [ id "heat_treat_temp_delta"   label "heat_treat_temp_delta" ]
  node [ id "tap_minus_pour_temp"     label "tap_minus_pour_temp" ]
  node [ id "pour_minus_die_temp"     label "pour_minus_die_temp" ]

  node [ id "scrap_flag"   label "scrap_flag" ]

  edge [ source "alloy_grade"  target "tap_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "pour_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "die_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "rpm_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "spin_time_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "pour_time_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "funnel_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "tap_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "pour_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "die_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "rpm_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "chem_not_ok_flag" ]
  edge [ source "alloy_grade"  target "chem_low_count" ]
  edge [ source "alloy_grade"  target "chem_high_count" ]
  edge [ source "alloy_grade"  target "scrap_flag" ]

  edge [ source "furn_no"  target "tap_deviation_from_fps_pct" ]
  edge [ source "furn_no"  target "pour_deviation_from_fps_pct" ]
  edge [ source "furn_no"  target "tap_deviation_from_instruction_pct" ]
  edge [ source "furn_no"  target "pour_deviation_from_instruction_pct" ]
  edge [ source "furn_no"  target "scrap_flag" ]

  edge [ source "charge_scrap_pct"   target "chem_not_ok_flag" ]
  edge [ source "charge_scrap_pct"   target "chem_low_count" ]
  edge [ source "charge_scrap_pct"   target "chem_high_count" ]
  edge [ source "charge_scrap_pct"   target "scrap_flag" ]
  edge [ source "charge_virgin_pct"  target "chem_not_ok_flag" ]
  edge [ source "charge_virgin_pct"  target "scrap_flag" ]

  edge [ source "tap_deviation_from_fps_pct"        target "scrap_flag" ]
  edge [ source "pour_deviation_from_fps_pct"       target "scrap_flag" ]
  edge [ source "die_deviation_from_fps_pct"        target "scrap_flag" ]
  edge [ source "rpm_deviation_from_fps_pct"        target "scrap_flag" ]
  edge [ source "spin_time_deviation_from_fps_pct"  target "scrap_flag" ]
  edge [ source "pour_time_deviation_from_fps_pct"  target "scrap_flag" ]
  edge [ source "funnel_deviation_from_fps_pct"     target "scrap_flag" ]
  edge [ source "wrong_funnel_flag"                 target "scrap_flag" ]

  edge [ source "tap_deviation_from_instruction_pct"   target "scrap_flag" ]
  edge [ source "pour_deviation_from_instruction_pct"  target "scrap_flag" ]
  edge [ source "die_deviation_from_instruction_pct"   target "scrap_flag" ]
  edge [ source "rpm_deviation_from_instruction_pct"   target "scrap_flag" ]

  edge [ source "chem_not_ok_flag"   target "scrap_flag" ]
  edge [ source "chem_low_count"     target "scrap_flag" ]
  edge [ source "chem_high_count"    target "scrap_flag" ]

  edge [ source "heat_treat_temp_delta"  target "scrap_flag" ]
  edge [ source "tap_minus_pour_temp"    target "scrap_flag" ]
  edge [ source "pour_minus_die_temp"    target "scrap_flag" ]
]
"""

# Each entry is analyzed as one independent causal query.
# type: "continuous" or "binary"
# analysis_note: optional — explains why a treatment may have limited data
TREATMENTS = [
    # ── FPS sheet deviations ──────────────────────────────────────────────
    {
        "name": "tap_deviation_from_fps_pct",
        "label": "Tap Temp Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = hotter than FPS target",
    },
    {
        "name": "pour_deviation_from_fps_pct",
        "label": "Pour Temp Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = hotter than FPS target",
    },
    {
        "name": "die_deviation_from_fps_pct",
        "label": "Die Temp Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = hotter than FPS target",
    },
    {
        "name": "rpm_deviation_from_fps_pct",
        "label": "RPM Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = faster than FPS target",
    },
    {
        "name": "spin_time_deviation_from_fps_pct",
        "label": "Spin Time Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = longer spin than FPS target",
    },
    {
        "name": "pour_time_deviation_from_fps_pct",
        "label": "Pour Time Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = slower pour than FPS target",
    },
    {
        "name": "funnel_deviation_from_fps_pct",
        "label": "Funnel Size Deviation from FPS",
        "type": "continuous",
        "unit": "fraction of FPS band",
        "interpretation_direction": "positive = larger funnel than FPS target",
    },
    {
        "name": "wrong_funnel_flag",
        "label": "Wrong Funnel Used",
        "type": "binary",
        "unit": "0=correct, 1=wrong funnel vs FPS",
        "interpretation_direction": "1 = funnel size does not match FPS spec",
    },
    # ── Process control profile (instruction) deviations ─────────────────
    {
        "name": "tap_deviation_from_instruction_pct",
        "label": "Tap Temp Deviation from Instruction",
        "type": "continuous",
        "unit": "fraction of instruction band",
        "interpretation_direction": "positive = hotter than instruction target",
    },
    {
        "name": "pour_deviation_from_instruction_pct",
        "label": "Pour Temp Deviation from Instruction",
        "type": "continuous",
        "unit": "fraction of instruction band",
        "interpretation_direction": "positive = hotter than instruction target",
    },
    {
        "name": "die_deviation_from_instruction_pct",
        "label": "Die Temp Deviation from Instruction",
        "type": "continuous",
        "unit": "fraction of instruction band",
        "interpretation_direction": "positive = hotter than instruction target",
    },
    {
        "name": "rpm_deviation_from_instruction_pct",
        "label": "RPM Deviation from Instruction",
        "type": "continuous",
        "unit": "fraction of instruction band",
        "interpretation_direction": "positive = faster than instruction target",
    },
    # ── Charge composition ────────────────────────────────────────────────
    {
        "name": "charge_scrap_pct",
        "label": "Charge Scrap %",
        "type": "continuous",
        "unit": "% of charge weight from SCRAP material",
        "interpretation_direction": "higher = more recycled/scrap metal in charge",
        "analysis_note": "requires charge_materials data; zero if .$$L files not ingested",
    },
    {
        "name": "charge_virgin_pct",
        "label": "Charge Virgin Metal %",
        "type": "continuous",
        "unit": "% of charge weight from virgin/ingot material",
        "interpretation_direction": "higher = more virgin metal in charge",
        "analysis_note": "requires charge_materials data",
    },
    # ── Chemistry compliance ──────────────────────────────────────────────
    {
        "name": "chem_not_ok_flag",
        "label": "Chemistry Out of Spec (any element)",
        "type": "binary",
        "unit": "0=all in spec, 1=any element out of spec",
        "interpretation_direction": "1 = at least one element outside min/max",
    },
    {
        "name": "chem_low_count",
        "label": "Elements Below Spec (count)",
        "type": "continuous",
        "unit": "count of elements below minimum spec",
        "interpretation_direction": "higher = more elements under-alloyed",
    },
    {
        "name": "chem_high_count",
        "label": "Elements Above Spec (count)",
        "type": "continuous",
        "unit": "count of elements above maximum spec",
        "interpretation_direction": "higher = more elements over-alloyed",
    },
    # ── Heat treat ────────────────────────────────────────────────────────
    {
        "name": "heat_treat_temp_delta",
        "label": "Heat Treat Temp Delta (actual - setpoint)",
        "type": "continuous",
        "unit": "°F from setpoint",
        "interpretation_direction": "positive = ran hotter than setpoint",
    },
    # ── Temperature differentials (pour behavior) ─────────────────────────
    {
        "name": "tap_minus_pour_temp",
        "label": "Tap-to-Pour Temp Drop",
        "type": "continuous",
        "unit": "°F (tap_temp - pour_temp)",
        "interpretation_direction": "higher = more heat lost between tap and pour",
    },
    {
        "name": "pour_minus_die_temp",
        "label": "Pour-to-Die Temp Gap",
        "type": "continuous",
        "unit": "°F (pour_temp - die_temp_before_pour)",
        "interpretation_direction": "higher = hotter metal relative to die preheat",
    },
]

# Confounders controlled in every backdoor adjustment
COMMON_CONFOUNDERS = ["alloy_grade", "furn_no"]

# Outcome
OUTCOME = "scrap_flag"
