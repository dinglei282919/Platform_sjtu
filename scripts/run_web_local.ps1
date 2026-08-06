[CmdletBinding()]
param(
    [switch]$Rebuild,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = 'D:\ana3\envs\Platform\python.exe'
$npm = 'C:\Program Files\nodejs\npm.cmd'
$frontend = Join-Path $repoRoot 'web_frontend'
$index = Join-Path $frontend 'dist\index.html'
$hostAddress = if ($env:PLATFORM_WEB_HOST) { $env:PLATFORM_WEB_HOST } else { '127.0.0.1' }
$port = if ($env:PLATFORM_WEB_PORT) { [int]$env:PLATFORM_WEB_PORT } else { 8000 }

if ($hostAddress -ne '127.0.0.1') {
    throw 'The local Web edition may only bind to 127.0.0.1.'
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Platform Python environment was not found: $python"
}

if ($Rebuild -or -not (Test-Path -LiteralPath $index)) {
    if (-not (Test-Path -LiteralPath $npm)) {
        throw 'Node.js/npm was not found. Install Node.js LTS and try again.'
    }
    Push-Location $frontend
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules'))) {
            & $npm install
            if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
        }
        & $npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    }
    finally {
        Pop-Location
    }
}

$url = "http://${hostAddress}:$port"
$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listener -and $Rebuild) {
    foreach ($owner in $listener) {
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.OwningProcess)" -ErrorAction SilentlyContinue
        if ($processInfo -and $processInfo.CommandLine -like '*uvicorn web_backend.app:app*') {
            Stop-Process -Id $owner.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 400
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
}
if ($listener) {
    try {
        $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "$url/api/health"
        if ($health.StatusCode -ne 200) { throw 'unexpected status' }
    }
    catch {
        throw "Port $port is occupied by a service other than the local platform Web edition."
    }
    Write-Host "Port $port is already serving the local platform Web edition."
}
else {
    $arguments = @('-m', 'uvicorn', 'web_backend.app:app', '--host', $hostAddress, '--port', $port)
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
    $ready = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "$url/api/health"
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        }
        catch { }
    }
    if (-not $ready) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw 'Local Web service failed to start. Confirm dependencies and try again.'
    }
}

Write-Host "Local Web edition is ready: $url"
if (-not $NoBrowser) {
    Start-Process $url
}
