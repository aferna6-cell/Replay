# Daily Firestone pipeline (Windows): fetch new games -> states -> labels -> train -> promote.
# Run it by hand from the Replay folder (output on screen):
#   .\scripts\daily_firestone_pipeline.ps1
#   .\scripts\daily_firestone_pipeline.ps1 -PipelineArgs '--no-install --max-new 50'
# (-PipelineArgs is one string of hsbg_coach.daily_pipeline flags.)
# The scheduled task (scripts/install_daily_pipeline_task.ps1) adds -Scheduled, which
# sends all output to data\firestone\logs\daily-YYYY-MM-DD.log instead.
param([string]$PipelineArgs = '', [switch]$Scheduled)

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { $Py = 'python' }
$Log = Join-Path $Repo ('data\firestone\logs\daily-' + (Get-Date -Format 'yyyy-MM-dd') + '.log')

$Extra = @()
if ($PipelineArgs.Trim()) { $Extra = $PipelineArgs.Trim() -split '\s+' }
if ($Scheduled) { $Extra = @('--log', $Log) + $Extra }
& $Py -m hsbg_coach.daily_pipeline @Extra
exit $LASTEXITCODE
