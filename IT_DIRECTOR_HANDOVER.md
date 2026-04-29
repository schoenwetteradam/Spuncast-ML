# Spuncast-ML IT Director Handover Runbook

This runbook provides the exact installation sequence, required software, PuTTY/SSH commands, and first-run validation steps to stand up `Spuncast-ML` so it works in sync with `Spuncast-Operations`.

---

## 1) Scope and architecture (what this deployment does)

- `Spuncast-ML` is a **CLI-only** machine-learning pipeline (no web UI).
- It reads operational data from the `Spuncast-Operations` PostgreSQL database (`spuncast`) via the `v_ml_heat_dataset_v1` and `v_ml_heat_early_score_v1` views.
- It writes local ML artifacts/reports and can optionally write live recommendations back through the scoring daemon.

---

## 2) Prerequisites (install in this order)

## 2.1 Workstation prerequisites (Windows admin PC)

1. **PuTTY** (includes `putty.exe`, `plink.exe`, `pscp.exe`)  
   Download: https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
2. **Git for Windows**  
   Download: https://git-scm.com/download/win
3. (Optional but recommended) **VS Code**  
   Download: https://code.visualstudio.com/

## 2.2 Linux server/VM prerequisites (target runtime host)

1. Linux host with network route to `Spuncast-Operations` PostgreSQL.
2. Python **3.11+** (3.12 preferred).
3. `python3-venv`, `python3-pip`, `git`.
4. Firewall rules allowing outbound connection to PostgreSQL (`tcp/5432` unless Operations uses a different port).
5. Service account with least-privilege access to:
   - `v_ml_heat_dataset_v1`
   - `v_ml_heat_early_score_v1`
   - write targets used by live scoring (`ml_heat_scores`, `heat_recommendations`) if daemon mode is enabled.

---

## 3) Collect required values before install

Have these values confirmed by Operations/DBA before starting:

- `PGHOST` / `PG_HOST` (Operations DB hostname or IP)
- `PG_PORT` (usually `5432`)
- `PGDATABASE` (expected: `spuncast`)
- `PGUSER`
- `PGPASSWORD`
- Source view name (default `SPUNCAST_ML_SOURCE_VIEW=v_ml_heat_dataset_v1`)

---

## 4) PuTTY session setup (GUI)

1. Open **PuTTY**.
2. In **Session**:
   - Host Name: `<ml-server-host-or-ip>`
   - Port: `22`
   - Connection type: `SSH`
3. In **Connection > Data**, set auto-login username (service account).
4. In **Connection > SSH > Auth**, load `.ppk` key (if key-based auth is required).
5. Save as: `spuncast-ml-prod`.
6. Click **Open** and authenticate.

---

## 5) Exact command sequence on Linux host (copy/paste in order)

> Use these in the SSH shell after connecting with PuTTY.

```bash
# 1) OS package prep (Debian/Ubuntu example)
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip

# 2) Create application directory
sudo mkdir -p /opt/spuncast
sudo chown "$USER":"$USER" /opt/spuncast
cd /opt/spuncast

# 3) Clone repo
# Replace with your internal Git URL if different.
git clone https://github.com/schoenwetteradam/Spuncast-ML.git
cd Spuncast-ML

# 4) Create virtualenv and install package
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

# 5) Create environment file
cp .env.example .env
```

### 5.1 Edit `.env` with production connection values

```bash
nano .env
```

Set at minimum:

```ini
PGDATABASE=spuncast
PGUSER=<operations_ml_user>
PGPASSWORD=<strong_password>
PG_HOST=<operations_db_host>
PGHOST=<operations_db_host>
PG_PORT=5432
SPUNCAST_ML_SOURCE_VIEW=v_ml_heat_dataset_v1
```

---

## 6) Non-interactive PuTTY command variants (plink)

Use from a Windows command prompt/PowerShell on an admin machine.

```powershell
# Run one command remotely
plink -ssh <user>@<ml-server-host> -i C:\keys\spuncast-ml.ppk "hostname && whoami"

# Run initial install block remotely (example)
plink -ssh <user>@<ml-server-host> -i C:\keys\spuncast-ml.ppk "sudo apt-get update && sudo apt-get install -y git python3 python3-venv python3-pip"

# Copy local .env to server (optional approach)
pscp -i C:\keys\spuncast-ml.ppk .\.env <user>@<ml-server-host>:/opt/spuncast/Spuncast-ML/.env
```

---

## 7) First validation checks (required before handoff complete)

Run on server from repo root:

```bash
source .venv/bin/activate

# CLI is installed
spuncast-ml --help

# Local-only command path (no DB required)
spuncast-ml feedback --heat-number 12345 --recommendation increase_monitoring --accepted --score 0.67 --operator-id it-smoke-test

# DB-backed connectivity check (requires live Operations DB)
spuncast-ml export
```

Expected results:
- `--help` prints command list.
- `feedback` writes local JSONL under `./data/feedback` (or configured feedback dir).
- `export` produces timestamped dataset snapshot and JSON sidecar under `data/exports/` if credentials and upstream views are correct.

---

## 8) Operating commands after go-live

```bash
# Full ML workflow
spuncast-ml pipeline --feature-set pre_pour_in_process --threshold 0.5

# Score and create operator recommendations
spuncast-ml score --feature-set pre_pour_in_process --threshold 0.5

# Drift monitoring
spuncast-ml monitor-drift --feature-set pre_pour_in_process
```

Feature sets available:
- `pre_pour_in_process` (operational)
- `post_run_diagnostic` (retrospective)
- `early_remelt_decision` (near-real-time remelt decisioning)

---

## 9) Live scoring daemon (optional production automation)

Daemon script:

```bash
python scripts/score_heat_live.py
```

Optional environment overrides:

```ini
SCORE_POLL_INTERVAL_SEC=180
SCORE_HORIZON_HOURS=8
SCORE_REMELT_THRESHOLD=0.65
```

Recommended: run this script as a managed Linux service (systemd) with restart policy and centralized logs.

---

## 10) Coordination contract with Spuncast-Operations (must enforce)

1. `Spuncast-Operations` owns operational schema and views.
2. Any schema changes to ML views must be coordinated and versioned.
3. `Spuncast-ML` uses pinned contracts in `spuncast_ml/contracts/` and will fail fast if the upstream schema drifts unexpectedly.
4. Promote model updates only after review of recall, false negatives, and PR-AUC against rules baseline.

---

## 11) Troubleshooting quick map

- **`connection refused` / timeout**: host, port, routing, or firewall issue between ML host and Operations DB.
- **auth failed**: wrong `PGUSER` / `PGPASSWORD` or restricted DB role.
- **missing relation/view**: Operations migrations/views not deployed (`v_ml_heat_dataset_v1`, `v_ml_heat_early_score_v1`).
- **schema mismatch**: upstream columns changed; coordinate update and re-pin contract.
- **model signature mismatch while scoring**: retrain model or temporarily set `SPUNCAST_ML_ALLOW_UNSIGNED_MODEL=1` during controlled recovery.

---

## 12) Handover acceptance checklist

- [ ] SSH access to ML host validated via PuTTY profile.
- [ ] Repo cloned to `/opt/spuncast/Spuncast-ML`.
- [ ] `.venv` created and `pip install -e .` successful.
- [ ] `.env` populated with production DB values.
- [ ] `spuncast-ml --help` successful.
- [ ] `spuncast-ml feedback ...` successful.
- [ ] `spuncast-ml export` successful against Operations DB.
- [ ] Decision recorded for daemon mode (`scripts/score_heat_live.py`) and service management approach.
