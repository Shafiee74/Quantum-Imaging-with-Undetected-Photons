# Launch the QIUP Next.js UI.  Usage:  .\start-ui.ps1   [-Python C:\path\to\python.exe] [-Port 3000] [-Prod]
param(
    [string]$Python = "",
    [int]$Port = 3000,
    [switch]$Prod
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$portableNode = Join-Path $here ".tools\node"
if (Test-Path (Join-Path $portableNode "node.exe")) {
    $env:Path = "$portableNode;" + $env:Path
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js not found. Install Node 20+ or place a portable build in .tools\node"
}

if ($Python) { $env:QIUP_PYTHON = $Python }
if (-not $env:QIUP_ROOT) { $env:QIUP_ROOT = (Resolve-Path (Join-Path $here "..")).Path }

if (-not (Test-Path (Join-Path $here "node_modules"))) {
    Write-Host "Installing npm dependencies..." -ForegroundColor Cyan
    npm install --no-audit --no-fund
}

Write-Host "QIUP UI  ->  http://localhost:$Port" -ForegroundColor Green
Write-Host "Python   ->  $(if ($env:QIUP_PYTHON) { $env:QIUP_PYTHON } else { 'python (PATH)' })"
Write-Host "Package  ->  $env:QIUP_ROOT"

if ($Prod) {
    npm run build
    npx next start -p $Port
} else {
    npx next dev -p $Port
}
