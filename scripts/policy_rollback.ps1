<#
.SYNOPSIS
  Roll the NEXT policy back: copy ml/policy_net.prev.pt over ml/policy_net.pt.
  The checkpoint being replaced is kept as ml/policy_net.replaced-<timestamp>.pt.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\policy_rollback.ps1
  HSBG_NEXT_POLICY defaults to on. To switch the policy off entirely instead,
  set HSBG_NEXT_POLICY=0 (false and off also disable it).
#>
param([string]$MlDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'ml'))
$ErrorActionPreference = 'Stop'
$live = Join-Path $MlDir 'policy_net.pt'
$prev = Join-Path $MlDir 'policy_net.prev.pt'
if (-not (Test-Path -LiteralPath $prev)) {
    Write-Host "No previous checkpoint at $prev; nothing to roll back to."
    Write-Host "To disable the NEXT policy and use the eval-net advisor, set HSBG_NEXT_POLICY=0."
    exit 1
}
if (Test-Path -LiteralPath $live) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backup = Join-Path $MlDir "policy_net.replaced-$stamp.pt"
    Copy-Item -LiteralPath $live -Destination $backup
    Write-Host "Backed up current checkpoint to $backup"
}
Copy-Item -LiteralPath $prev -Destination $live -Force
Write-Host "Restored $prev -> $live"