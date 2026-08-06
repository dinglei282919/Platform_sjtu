@echo off
REM One-click launcher for the local Web edition (Windows)
REM Usage: double-click or run in cmd: start_web.bat [-Rebuild] [-NoBrowser]
SETLOCAL ENABLEDELAYEDEXPANSION
pushd %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_web_local.ps1 %*
popd
n