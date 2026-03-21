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
- apply ML-safe feature selection and preprocessing
- generate train / validation / test splits
- train and evaluate a first-pass scrap prediction model
- compare model results against a rules-based baseline before promotion

## Repository layout

```text
Spuncast-ML/
|-- .env.example
|-- pyproject.toml
|-- README.md
|-- ML_HANDOFF.md
|-- data/
|   |-- exports/
|   `-- processed/
|-- artifacts/
|   `-- models/
|-- reports/
|   `-- generated/
`-- spuncast_ml/
    |-- cli.py
    |-- dataset.py
    |-- db.py
    `-- modeling.py
```

## Quick start

### Option 1: Docker runtime

This is the recommended path on this machine because Docker is available and
Python is not currently on `PATH`.

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

Set the rest of the database credentials in `.env` so this repo can connect to
the same Postgres / TimescaleDB instance used by `Spuncast-Operations`.

## Commands

Export the canonical dataset view to a timestamped parquet snapshot:

```powershell
spuncast-ml export
```

Train the first-pass baseline model and write outputs to `artifacts/models/`:

```powershell
spuncast-ml train
```

Generate a fresh evaluation report:

```powershell
spuncast-ml evaluate
```

Run the full workflow end to end:

```powershell
spuncast-ml pipeline
```

Docker equivalent:

```powershell
docker compose run --rm spuncast-ml pipeline
```

## Baselines and guardrails

- The modeling grain is one row per `heat_number`.
- The initial target is `scrap_flag`.
- Training intentionally excludes direct label columns and downstream scrap
  summary fields to avoid leakage.
- A simple rules baseline is included so model promotion can be gated on
  measurable improvement instead of intuition.

## Relationship to Spuncast-Operations

- Upstream SQL contract: `db/init/067_ml_heat_dataset.sql`
- Handoff notes: [`ML_HANDOFF.md`](/D:/Data/Spuncast-ML/ML_HANDOFF.md)
- Recommended promotion rule: do not deploy a model unless it clearly beats
  the current rules-based baseline on scrap detection and false-negative control.

## Runtime notes

- The Docker image includes the Python runtime and project dependencies.
- Output files are persisted to the local `data/`, `artifacts/`, and `reports/`
  folders through bind mounts.
- If the operations database is only reachable inside a Docker network and not
  on host port `5432`, update `PGHOST` accordingly before running the pipeline.
