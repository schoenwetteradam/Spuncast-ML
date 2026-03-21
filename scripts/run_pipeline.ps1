$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Set PGHOST, PGPASSWORD, and related values before rerunning."
    exit 0
}

docker compose build spuncast-ml
docker compose run --rm spuncast-ml pipeline

