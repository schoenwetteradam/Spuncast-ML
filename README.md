# Spuncast ML

`Spuncast-ML` is the machine learning companion repository for
[`Spuncast-Operations`](https://github.com/schoenwetteradam/Spuncast-Operations.git).

`Spuncast-Operations` remains the operational system of record. This repository
owns ML-specific workflows built on top of the canonical upstream export:

```sql
SELECT * FROM v_ml_heat_dataset_v1;
```

## Responsibilities

- export repeatable dataset snapshots from the operations database
- pin and validate the upstream view contract before training
- keep separate operational and diagnostic feature sets
- generate chronological train / validation / test splits
- train and evaluate baseline scrap prediction models
- compare model results against a rules-based baseline before promotion

## Repository layout

```text
Spuncast-ML/
|-- artifacts/
|   `-- models/
|-- data/
|   |-- exports/
|   `-- processed/
|-- reports/
|   `-- generated/
|-- scripts/
`-- spuncast_ml/
    |-- cli.py
    |-- contract.py
    |-- contracts/
    |   `-- v_ml_heat_dataset_v1.json
    |-- dataset.py
    |-- db.py
    `-- modeling.py
```

## Data contract and snapshot policy

The stable upstream contract is `v_ml_heat_dataset_v1`, with the schema pinned in
`spuncast_ml/contracts/v_ml_heat_dataset_v1.json`. Exports now fail fast if the
view adds, removes, or renames columns unexpectedly.

The default extraction query follows the agreed snapshot policy:

```sql
SELECT *
FROM v_ml_heat_dataset_v1
WHERE analysis_date IS NOT NULL
ORDER BY analysis_date, heat_number;
```

Each export writes:

- an immutable Parquet snapshot under `data/exports/`
- a JSON sidecar containing the extraction timestamp
- the source query hash
- the pinned contract version
- the schema/version note

## Feature-set guardrails

Two explicit feature sets are supported:

- `pre_pour_in_process` for operational prediction
- `post_run_diagnostic` for retrospective analysis and explainability

The operational feature set excludes leakage-prone scrap outcome columns and
other likely post-outcome fields such as scrap quantities, scrap reason fields,
and downstream lot or heat-treat summaries.

## Time split policy

Dataset splits are chronological, not random:

- train: oldest 70%
- validation: middle 15%
- test: newest 15%

Rows are ordered by `analysis_date` and `heat_number` before splitting.

## Modeling and promotion gates

Baseline candidates currently trained in this repo:

- logistic regression with class weighting
- calibrated histogram gradient boosting
- rules-based alert baseline for comparison

Tracked metrics include:

- recall for `scrap_flag=1`
- precision
- PR-AUC
- ROC-AUC
- confusion matrix
- false negatives

Promotion remains blocked unless the selected model:

- matches or beats the rules baseline on recall
- does not increase false negatives
- matches or improves PR-AUC

## Quick start

### Option 1: Docker runtime

This is the recommended path on this machine because Docker is available and
Python is not guaranteed to match the project runtime requirements.

```powershell
Copy-Item .env.example .env
```

Set the database credentials in `.env`. If `Spuncast-Operations` is running on
your host machine and Postgres is published on port `5432`, leave:

```ini
PGHOST=host.docker.internal
PG_HOST=host.docker.internal
```

Then run:

```powershell
docker compose build spuncast-ml
docker compose run --rm spuncast-ml pipeline
```

Or use the helper script:

```powershell
.\scripts\run_pipeline.ps1
```

### Option 2: Local Python runtime

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

For a host-local Python run, set:

```ini
PGHOST=localhost
PG_HOST=localhost
```

## Commands

Export the canonical dataset view to an immutable timestamped snapshot with
provenance metadata:

```powershell
spuncast-ml export
```

Train the current baseline candidate set using the operational feature set:

```powershell
spuncast-ml train --feature-set pre_pour_in_process --threshold 0.5
```

Run a retrospective diagnostic evaluation instead:

```powershell
spuncast-ml train --feature-set post_run_diagnostic --threshold 0.5
```

Generate a fresh evaluation report:

```powershell
spuncast-ml evaluate --feature-set pre_pour_in_process --threshold 0.5
```

Run the full workflow end to end:

```powershell
spuncast-ml pipeline --feature-set pre_pour_in_process --threshold 0.5
```

Score a dataset with the latest trained model and generate operator-facing
recommendations (`continue_standard_run`, `increase_monitoring`,
`hold_for_operator_review`):

```powershell
spuncast-ml score --feature-set pre_pour_in_process --threshold 0.5
```

Generate a drift report by comparing the latest export to the snapshot used to
train the latest model:

```powershell
spuncast-ml monitor-drift --feature-set pre_pour_in_process
```

Capture operator feedback (accepted or rejected recommendation) for continuous
learning inputs:

```powershell
spuncast-ml feedback --heat-number 12345 --recommendation increase_monitoring --accepted --score 0.67 --operator-id shift-b
```

## Live operations loop (recommended rollout)

Use ML in three controlled stages before any automation:

1. **Shadow mode**: run `score` and track quality without changing operations.
2. **Advisor mode**: expose recommendations to operators and log responses with
   `feedback`.
3. **Guarded automation**: only allow bounded machine adjustments behind a
   policy layer with safety limits, approvals, and audit logs.

`monitor-drift` should run on a schedule (for example each shift or each day)
to identify retraining triggers before performance degrades.

## Relationship to Spuncast-Operations

- Upstream SQL contract reference: `067_ml_heat_dataset.sql`
- Handoff notes: `ML_HANDOFF.md`
- Breaking schema changes should be coordinated across both repos and reflected
  in the pinned contract file plus PR notes.
