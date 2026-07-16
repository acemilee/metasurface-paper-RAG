$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

docker compose down
if ($LASTEXITCODE -ne 0) {
    throw "Paper RAG services failed to stop cleanly."
}

Write-Output "Paper RAG stopped. Data and models were preserved."
