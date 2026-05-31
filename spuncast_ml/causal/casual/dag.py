"""
Causal DAG for foundry scrap analysis.

Structure encodes domain knowledge:
  - alloy_grade and furn_no are exogenous confounders that affect both
    process deviations and scrap independently.
  - charge_scrap_pct (charge composition) affects chemistry compliance
    and ultimately scrap.
  - Process deviations from FPS targets are the primary modifiable treatments.
  - chem_not_ok_flag is a mediator between charge composition and scrap.
"""

# GML directed graph encoding the causal structure
SCRAP_DAG_GML = """
graph [
  directed 1
  node [ id "alloy_grade"                 label "alloy_grade" ]
  node [ id "furn_no"                     label "furn_no" ]
  node [ id "charge_scrap_pct"            label "charge_scrap_pct" ]
  node [ id "chem_not_ok_flag"            label "chem_not_ok_flag" ]
  node [ id "tap_deviation_from_fps_pct"  label "tap_deviation_from_fps_pct" ]
  node [ id "pour_deviation_from_fps_pct" label "pour_deviation_from_fps_pct" ]
  node [ id "die_deviation_from_fps_pct"  label "die_deviation_from_fps_pct" ]
  node [ id "rpm_deviation_from_fps_pct"  label "rpm_deviation_from_fps_pct" ]
  node [ id "scrap_flag"                  label "scrap_flag" ]
  edge [ source "alloy_grade"  target "chem_not_ok_flag" ]
  edge [ source "alloy_grade"  target "tap_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "pour_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "die_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "rpm_deviation_from_fps_pct" ]
  edge [ source "alloy_grade"  target "scrap_flag" ]
  edge [ source "furn_no"      target "tap_deviation_from_fps_pct" ]
  edge [ source "furn_no"      target "scrap_flag" ]
  edge [ source "charge_scrap_pct"            target "chem_not_ok_flag" ]
  edge [ source "charge_scrap_pct"            target "scrap_flag" ]
  edge [ source "chem_not_ok_flag"            target "scrap_flag" ]
  edge [ source "tap_deviation_from_fps_pct"  target "scrap_flag" ]
  edge [ source "pour_deviation_from_fps_pct" target "scrap_flag" ]
  edge [ source "die_deviation_from_fps_pct"  target "scrap_flag" ]
  edge [ source "rpm_deviation_from_fps_pct"  target "scrap_flag" ]
]
"""

# Treatments to analyze — each is run as an independent causal query
TREATMENTS = [
    {
        "name": "tap_deviation_from_fps_pct",
        "label": "Tap Temp Deviation from FPS",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means running hotter than target",
    },
    {
        "name": "pour_deviation_from_fps_pct",
        "label": "Pour Temp Deviation from FPS",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means running hotter than target",
    },
    {
        "name": "die_deviation_from_fps_pct",
        "label": "Die Temp Deviation from FPS",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means running hotter than target",
    },
    {
        "name": "rpm_deviation_from_fps_pct",
        "label": "Die RPM Deviation from FPS",
        "type": "continuous",
        "unit": "% from target",
        "interpretation_direction": "positive means higher RPM than target",
    },
    {
        "name": "charge_scrap_pct",
        "label": "Charge Scrap %",
        "type": "continuous",
        "unit": "% of charge weight from SCRAP material",
        "interpretation_direction": "higher = more recycled / scrap metal in charge",
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
