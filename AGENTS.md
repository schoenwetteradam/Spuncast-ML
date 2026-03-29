# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Spuncast-ML is a CLI-based batch ML pipeline for scrap prediction in a spun-cast
metal foundry. It connects to an external PostgreSQL database owned by
**Spuncast-Operations** and has no web frontend of its own.

### Development environment

- Python ≥ 3.11 (the VM ships 3.12). Use a virtualenv at `.venv/`.
- Install in editable mode: `pip install -e .` from the repo root.
- The CLI entry point is `spuncast-ml`.

### Running without a database

Most CLI commands (`export`, `train`, `evaluate`, `pipeline`, `score`,
`monitor-drift`) require a live PostgreSQL connection to the `spuncast`
database. In the Cloud VM that database is **not available**.

Commands that work without a database:

- `spuncast-ml --help` and all `--help` subcommands
- `spuncast-ml feedback ...` (writes to a local JSONL file)

### Feature sets

Three feature sets are defined in `spuncast_ml/dataset.py`:

| Feature set               | Purpose                                       |
|---------------------------|-----------------------------------------------|
| `pre_pour_in_process`     | Operational prediction (original)             |
| `post_run_diagnostic`     | Retrospective analysis                        |
| `early_remelt_decision`   | Near-real-time re-melt go/no-go call (new)    |

### Key directories

| Path                       | Contents                            |
|----------------------------|-------------------------------------|
| `sql/`                     | SQL migrations for Operations DB    |
| `spuncast_ml/contracts/`   | Pinned view schema contracts (JSON) |
| `grafana/`                 | Dashboard panel queries             |
| `scripts/`                 | Helper and daemon scripts           |
| `data/`, `artifacts/`, `reports/` | Runtime output (gitignored)  |

### Testing notes

- There are no automated test suites (`tests/` does not exist).
- Validate changes by running `python -c "from spuncast_ml.<module> import ..."` for import checks.
- The `feedback` command is the simplest end-to-end check that does not need a DB.
- For a full pipeline test you need the Spuncast-Operations PostgreSQL instance running with the `v_ml_heat_dataset_v1` and `v_ml_heat_early_score_v1` views populated.

### Live scoring daemon

`scripts/score_heat_live.py` runs as a long-lived background process. It polls
`v_ml_heat_early_score_v1` every N seconds and writes to `ml_heat_scores` and
`heat_recommendations` tables. Configurable via environment variables
(`SCORE_POLL_INTERVAL_SEC`, `SCORE_HORIZON_HOURS`, `SCORE_REMELT_THRESHOLD`).
