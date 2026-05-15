$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$LogRoot = if ($env:SPUNCAST_ML_REPORT_DIR) { $env:SPUNCAST_ML_REPORT_DIR } else { Join-Path $RepoRoot "reports" }
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$LogFile = Join-Path $LogRoot "training_$Stamp.log"

$FeatureSet = if ($env:SPUNCAST_ML_FEATURE_SET) { $env:SPUNCAST_ML_FEATURE_SET } else { "early_remelt_decision" }
$Threshold = if ($env:SPUNCAST_ML_THRESHOLD) { $env:SPUNCAST_ML_THRESHOLD } else { "0.5" }
$MinRel = if ($env:SPUNCAST_ML_PROMOTE_MIN_REL) { $env:SPUNCAST_ML_PROMOTE_MIN_REL } else { "0.05" }
$UseDocker = if ($null -ne $env:SPUNCAST_ML_USE_DOCKER) { $env:SPUNCAST_ML_USE_DOCKER } else { "1" }

function Invoke-NativePipeline {
    $py = if ($env:SPUNCAST_ML_PYTHON) { $env:SPUNCAST_ML_PYTHON } else { "python3" }
    Write-Host "[$(Get-Date).ToUniversalTime()] Weekly ML pipeline (native) feature_set=$FeatureSet"
    & $py -m spuncast_ml export
    & $py -m spuncast_ml train --feature-set $FeatureSet --threshold $Threshold
    & $py -m spuncast_ml evaluate --feature-set $FeatureSet --threshold $Threshold
    & $py -m spuncast_ml promote --feature-set $FeatureSet --min-relative-improvement $MinRel
    & $py -m spuncast_ml weekly-report --output-path (Join-Path $LogRoot "weekly_$Stamp.html")
}

function Invoke-DockerPipeline {
    Write-Host "[$(Get-Date).ToUniversalTime()] Weekly ML pipeline (Docker) feature_set=$FeatureSet"
    docker compose -f (Join-Path $RepoRoot "docker-compose.yml") build spuncast-ml
    docker compose -f (Join-Path $RepoRoot "docker-compose.yml") run --rm spuncast-ml export
    docker compose -f (Join-Path $RepoRoot "docker-compose.yml") run --rm spuncast-ml train --feature-set $FeatureSet --threshold $Threshold
    docker compose -f (Join-Path $RepoRoot "docker-compose.yml") run --rm spuncast-ml evaluate --feature-set $FeatureSet --threshold $Threshold
    docker compose -f (Join-Path $RepoRoot "docker-compose.yml") run --rm spuncast-ml promote --feature-set $FeatureSet --min-relative-improvement $MinRel
    docker compose -f (Join-Path $RepoRoot "docker-compose.yml") run --rm spuncast-ml weekly-report --output-path "/app/reports/weekly_$Stamp.html"
}

Start-Transcript -Path $LogFile
try {
    if ($UseDocker -eq "1" -and (Get-Command docker -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $RepoRoot "docker-compose.yml"))) {
        Invoke-DockerPipeline
    } else {
        Invoke-NativePipeline
    }
    Write-Host "[$(Get-Date).ToUniversalTime()] Weekly ML pipeline complete. Log: $LogFile"
} finally {
    Stop-Transcript | Out-Null
}
