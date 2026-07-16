param(
    [switch]$NoBrowser,
    [switch]$BuildLocal,
    [string]$Image = $(if ($env:PAPER_RAG_IMAGE) { $env:PAPER_RAG_IMAGE } else { "ghcr.io/acemilee/metasurface-paper-rag:0.1.0" }),
    [string]$BuildProxy = $env:PAPER_RAG_BUILD_PROXY,
    [ValidateRange(30, 7200)]
    [int]$WaitTimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Show-StartupDiagnostics {
    Write-Output "`nCompose status:"
    docker compose ps 2>&1 | Write-Output
    Write-Output "`nRecent startup logs:"
    docker compose logs --tail 120 model-init migrate embedding worker api 2>&1 | Write-Output
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed. Install and start Docker Desktop, then retry."
}

docker compose version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is unavailable. Install or update Docker Desktop."
}

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon is unavailable. Start Docker Desktop, then retry."
}

if (-not $BuildProxy) {
    try {
        $proxySettings = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction Stop
        if ($proxySettings.ProxyServer -match "(?:^|;)127\.0\.0\.1:(\d+)(?:;|$)") {
            $proxyPort = [int]$Matches[1]
            $proxyListener = Get-NetTCPConnection -LocalPort $proxyPort -State Listen -ErrorAction SilentlyContinue
            if ($proxyListener) {
                $BuildProxy = "http://host.docker.internal:$proxyPort"
                Write-Output "Using detected local build proxy on port $proxyPort."
            }
        }
    } catch {
        # Proxy auto-detection is optional. Direct Docker networking remains valid.
    }
}
if ($BuildProxy) {
    $BuildProxy = $BuildProxy -replace "://(?:127\.0\.0\.1|localhost):", "://host.docker.internal:"
}

try {
    if ($BuildLocal) {
        Write-Output "Building Paper RAG image locally."
        if ($BuildProxy) {
            $buildArguments = @(
                "build", "--tag", "paper-rag:local",
                "--build-arg", "HTTP_PROXY=$BuildProxy",
                "--build-arg", "HTTPS_PROXY=$BuildProxy",
                "."
            )
            & docker @buildArguments
        } else {
            docker build --tag paper-rag:local .
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Paper RAG local image build failed."
        }
        $env:PAPER_RAG_IMAGE = "paper-rag:local"
    } else {
        $env:PAPER_RAG_IMAGE = $Image
        docker pull $Image
        if ($LASTEXITCODE -ne 0) {
            throw "Published image pull failed: $Image. Retry or use -BuildLocal."
        }
    }

    docker compose up --detach --no-build --wait --wait-timeout $WaitTimeoutSeconds
    if ($LASTEXITCODE -ne 0) {
        Show-StartupDiagnostics
        throw "Paper RAG failed to become ready."
    }

    $ready = Invoke-RestMethod "http://127.0.0.1:8010/ready" -TimeoutSec 10
    if (-not $ready.ready) {
        Show-StartupDiagnostics
        throw "Compose completed but API readiness is false."
    }
} catch {
    if ($_.Exception.Message -notmatch "Paper RAG failed|readiness is false") {
        Show-StartupDiagnostics
    }
    throw
}

Write-Output "GUI ready: http://127.0.0.1:8010"
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:8010" -ErrorAction SilentlyContinue
}
