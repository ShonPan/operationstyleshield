# Free localhost port 5173 (e.g. stuck Vite/Node). Run: powershell -ExecutionPolicy Bypass -File scripts\free-port-5173.ps1
$port = 5173
$conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $conns) {
    Write-Host "Port $port is not in use (nothing to kill)."
    exit 0
}
$pids = $conns.OwningProcess | Sort-Object -Unique
foreach ($p in $pids) {
    $proc = Get-Process -Id $p -ErrorAction SilentlyContinue
    $name = if ($proc) { $proc.ProcessName } else { "?" }
    Write-Host "Stopping PID $p ($name)..."
    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
}
Write-Host "Port $port should be free now."
