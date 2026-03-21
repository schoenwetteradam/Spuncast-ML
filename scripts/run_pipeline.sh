#!/usr/bin/env sh
set -eu

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Set PGHOST, PGPASSWORD, and related values before rerunning."
  exit 0
fi

docker compose build spuncast-ml
docker compose run --rm spuncast-ml pipeline

