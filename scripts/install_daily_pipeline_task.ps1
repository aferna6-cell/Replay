# Register (or update) the Windows scheduled task that runs the daily Firestone pipeline.
#   .\scripts\install_daily_pipeline_task.ps1                      # every day at 04:30
#   .\scripts\install_daily_pipeline_task.ps1 -At 06:00 -PipelineArgs '--no-install'
#   Unregister-ScheduledTask -TaskName 'Replay daily Firestone pipeline'   # remove it
#
# The task runs as you, while you are logged on. StartWhenAvailable runs a missed
# day as soon as the PC is back. Firestone's list only covers the last ~4 days of
# games, so a PC that is off longer than that misses the games in between.
param(
    [string]$At = '04:30',
    [string]$TaskName = 'Replay daily Firestone pipeline',
    [string]$PipelineArgs = ''
)

$Script = Join-Path $PSScriptRoot 'daily_firestone_pipeline.ps1'
$Repo = Split-Path -Parent $PSScriptRoot
$TaskArgs = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Script`" -Scheduled"
if ($PipelineArgs.Trim()) { $TaskArgs += " -PipelineArgs `"$($PipelineArgs.Trim())`"" }
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $Repo -Argument $TaskArgs
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description 'Fetch new Firestone BG games, build states + labels, retrain the NEXT policy, install on gate PASS.' `
    -Force | Out-Null
Write-Host "Registered '$TaskName' daily at $At."
Write-Host "Run it now:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs:        $Repo\data\firestone\logs\"
