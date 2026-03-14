# Sync with GitHub: fetch → rebase on main → push
# Run: powershell -ExecutionPolicy Bypass -File scripts\sync-github.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host ">>> git fetch origin"
git fetch origin

Write-Host ">>> git pull --rebase origin main"
git pull --rebase origin main

Write-Host ">>> git push origin main"
git push origin main

Write-Host ">>> Done."
