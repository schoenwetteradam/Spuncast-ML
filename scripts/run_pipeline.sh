#!/usr/bin/env bash
# Weekly automated training: export → train → evaluate → optional promotion gate → HTML report.
# Intended for cron (see deploy/crontab-ml.example) or Airflow. Requires DB credentials in .env when using Docker.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.env" ]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  echo "Created .env from .env.example. Set PGHOST, PGPASSWORD, and related values before rerunning."
  exit 0
fi

LOG_ROOT="${SPUNCAST_ML_REPORT_DIR:-$REPO_ROOT/reports}"
mkdir -p "$LOG_ROOT"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG_FILE="$LOG_ROOT/training_${STAMP}.log"

FEATURE_SET="${SPUNCAST_ML_FEATURE_SET:-early_remelt_decision}"
THRESHOLD="${SPUNCAST_ML_THRESHOLD:-0.5}"
MIN_REL="${SPUNCAST_ML_PROMOTE_MIN_REL:-0.05}"
USE_DOCKER="${SPUNCAST_ML_USE_DOCKER:-1}"

_run_native() {
  local py="${SPUNCAST_ML_PYTHON:-python3}"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "[$(date -u)] Weekly ML pipeline (native) feature_set=${FEATURE_SET}"
  "$py" -m spuncast_ml export
  "$py" -m spuncast_ml train --feature-set "$FEATURE_SET" --threshold "$THRESHOLD"
  "$py" -m spuncast_ml evaluate --feature-set "$FEATURE_SET" --threshold "$THRESHOLD"
  "$py" -m spuncast_ml promote --feature-set "$FEATURE_SET" --min-relative-improvement "$MIN_REL"
  "$py" -m spuncast_ml weekly-report --output-path "$LOG_ROOT/weekly_${STAMP}.html"
}

_run_docker() {
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "[$(date -u)] Weekly ML pipeline (Docker) feature_set=${FEATURE_SET}"
  docker compose -f "$REPO_ROOT/docker-compose.yml" build spuncast-ml
  docker compose -f "$REPO_ROOT/docker-compose.yml" run --rm spuncast-ml export
  docker compose -f "$REPO_ROOT/docker-compose.yml" run --rm spuncast-ml train --feature-set "$FEATURE_SET" --threshold "$THRESHOLD"
  docker compose -f "$REPO_ROOT/docker-compose.yml" run --rm spuncast-ml evaluate --feature-set "$FEATURE_SET" --threshold "$THRESHOLD"
  docker compose -f "$REPO_ROOT/docker-compose.yml" run --rm spuncast-ml promote --feature-set "$FEATURE_SET" --min-relative-improvement "$MIN_REL"
  docker compose -f "$REPO_ROOT/docker-compose.yml" run --rm spuncast-ml weekly-report --output-path "/app/reports/weekly_${STAMP}.html"
}

if [ "$USE_DOCKER" = "1" ] && command -v docker >/dev/null 2>&1 && [ -f "$REPO_ROOT/docker-compose.yml" ]; then
  _run_docker
else
  _run_native
fi

echo "[$(date -u)] Weekly ML pipeline complete. Log: $LOG_FILE"
