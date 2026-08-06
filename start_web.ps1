param(
    [switch]$Rebuild,
    [switch]$NoBrowser
)
# One-click PowerShell launcher that calls scripts\run_web_local.ps1 from repo root.
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $scriptRoot
try {
    & .\scripts\run_web_local.ps1 @PSBoundParameters
}
finally {
    Pop-Location
}
