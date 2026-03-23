# ML Handoff

## Purpose

`Spuncast-Operations` is the operational system of record.

It should:
- ingest foundry data
- normalize that data into stable tables and views
- expose one repeatable ML-ready export surface

It should not become the long-term home for model training experiments,
notebooks, or production model artifacts.

## Canonical ML source

The first recommended export surface for a separate `Spuncast-ML` repo is:

```sql
SELECT * FROM v_ml_heat_dataset_v1;
```

This view is defined in:

- `db/init/067_ml_heat_dataset.sql`

## Modeling grain

- one row per `heat_number`

## Initial target

- `scrap_flag`

## Stable contract expectations in `Spuncast-ML`

- pin the expected schema in source control
- fail fast if columns are added, removed, or renamed unexpectedly
- coordinate breaking changes across both repos using PR notes and contract updates

## Snapshot policy

Preferred extraction query:

```sql
SELECT *
FROM v_ml_heat_dataset_v1
WHERE analysis_date IS NOT NULL
ORDER BY analysis_date, heat_number;
```

Each saved snapshot should include:

- extraction timestamp
- source query hash
- source contract version
- source schema/version note

## Leakage guardrails

Operational models should keep a dedicated `pre_pour_in_process` feature set
that excludes leakage-prone columns such as:

- `quantity_scrapped`
- `scrap_event_count`
- `scrap_event_quantity`
- `scrap_weight_lbs`
- `scrap_estimated_cost`
- `total_recorded_scrap_qty`
- `reason_code`
- `defect_type`
- `department`

A separate `post_run_diagnostic` feature set can be used for retrospective
analysis and explainability.

## Split policy

Use chronological splits rather than random row splits:

- train: oldest period
- validation: middle period
- test: newest period

## Baselines and metrics

Recommended baseline candidates:

- logistic regression with class weighting
- gradient boosting / tree ensemble
- calibrated probability model for thresholding

Track at minimum:

- recall for `scrap_flag=1`
- precision / PR-AUC
- ROC-AUC
- confusion matrix
- false negatives

## Promotion rule

Do not deploy an ML model into live operations unless it clearly beats the
current rules-based baseline on scrap detection and false-negative control.
