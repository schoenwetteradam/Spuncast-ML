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

## Feature groups in v1

- heat identity and grouping
- pour/process variables
- missing-data flags
- simple process deltas
- chemistry summary and alert flags
- heat-treat summary
- lot/output summary
- scrap summary and reason-code bucket

## Suggested ML repo responsibilities

- export dataset snapshots
- feature engineering beyond the base view
- train/validation/test splits
- model training and evaluation
- reporting
- model promotion decisions

## Promotion rule

Do not deploy an ML model into live operations unless it clearly beats the
current rules-based baseline on scrap detection and false-negative control.
