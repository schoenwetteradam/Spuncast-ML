"""
Causal DAG for foundry scrap analysis.

Structure encodes domain knowledge:
  - alloy_grade and furn_no are exogenous confounders that affect both
    process deviations and scrap independently.
  - Process deviations from the instruction (FPS) targets are the primary
    modifiable treatments.
  - chem_not_ok_flag is an observable chemistry compliance flag.

Column names match v_ml_heat_dataset_v1:
  tap/pour/die/rpm_deviation_from_instruction_pct
"""

# GML directed graph encoding the causal structure
SCRAP_DAG_GML = """
graph [
  directed 1
  node [ id "alloy_grade"                          label "alloy_grade" ]
  node [ id "furn_no"                              label "furn_no" ]
  node [ id "chem_not_ok_flag"                     label "chem_not_ok_flag" ]
  node [ id "tap_deviation_from_instruction_pct"   label "tap_deviation_from_instruction_pct" ]
  node [ id "pour_deviation_from_instruction_pct"  label "pour_deviation_from_instruction_pct" ]
  node [ id "die_deviation_from_instruction_pct"   label "die_deviation_from_instruction_pct" ]
  node [ id "rpm_deviation_from_instruction_pct"   label "rpm_deviation_from_instruction_pct" ]
  node [ id "scrap_flag"                           label "scrap_flag" ]
  edge [ source "alloy_grade"  target "chem_not_ok_flag" ]
  edge [ source "alloy_grade"  target "tap_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "pour_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "die_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "rpm_deviation_from_instruction_pct" ]
  edge [ source "alloy_grade"  target "scrap_flag" ]
  edge [ source "furn_no"      target "tap_deviation_from_instruction_pct" ]
  edge [ source "furn_no"      target "scrap_flag" ]
  edge [ source "chem_not_ok_flag"                    target "scrap_flag" ]
  edge [ source "tap_deviation_from_instruction_pct"  target "scrap_flag" ]
  edge [ source "pour_deviation_from_instruction_pct" target "scrap_flag" ]
  edge [ source "die_deviation_from_instruction_pct"  target "scrap_flag" ]
  edge [ source "rpm_deviation_from_instruction_pct"  target "scrap_flag" ]
]
"""

# Treatments to analyze — each is run as an independent causal query.
# Names must match columns in v_ml_heat_dataset_v1.
TREATMENTS = [
    {
        "name": "tap_deviation_from_instruction_pct",
        "label": "Tap Temp Deviation from Instruction",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means running hotter than target",
    },
    {
        "name": "pour_deviation_from_instruction_pct",
        "label": "Pour Temp Deviation from Instruction",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means running hotter than target",
    },
    {
        "name": "die_deviation_from_instruction_pct",
        "label": "Die Temp Deviation from Instruction",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means running hotter than target",
    },
    {
        "name": "rpm_deviation_from_instruction_pct",
        "label": "Die RPM Deviation from Instruction",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means higher RPM than target",
    },
    {
        "name": "chem_not_ok_flag",
        "label": "Chemistry Out of Spec",
        "type": "binary",
        "unit": "0=ok, 1=any element out of spec",
        "interpretation_direction": "1 means at least one element outside min/max",
    },
]

# Confounders that must be controlled for in every analysis
COMMON_CONFOUNDERS = ["alloy_grade", "furn_no"]

# Outcome
OUTCOME = "scrap_flag"
