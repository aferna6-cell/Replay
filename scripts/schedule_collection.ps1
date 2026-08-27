# Install a Windows Scheduled Task that runs the Power.log collector.
#
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_collection.ps1            # every 48 hours
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_collection.ps1 -Weekly    # weekly (Sunday)
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_collection.ps1 -Remove    # uninstall
#
# The task collects new Power.log files into logs\, parses them into training
# trajectories, retrains the eval net, and pushes the logs to GitHub. Output
# goes to logs\collect.log (gitignored). Re-running replaces the existing task.
param(
    [switch]$Weekly,
    [switch]$Remove,
    [string]$Time = "21:00"
)

$TaskName = "HSBGCoach-CollectPowerLogs"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($Remove) {
    schtasks /Delete /TN $TaskName /F
    Write-Host "Removed scheduled task $TaskName."
    exit 0
}

$Cmd = "cmd /c cd /d ""$Repo"" && python scripts\collect_power_logs.py --train >> logs\collect.log 2>&1"

if ($Weekly) {
    schtasks /Create /F /TN $TaskName /TR $Cmd /SC WEEKLY /D SUN /ST $Time
} else {
    # DAILY with modifier 2 = every 2nd day = every 48 hours
    schtasks /Create /F /TN $TaskName /TR $Cmd /SC DAILY /MO 2 /ST $Time
}

Write-Host ""
Write-Host "Scheduled task '$TaskName' installed ($(if ($Weekly) {'weekly, Sunday'} else {'every 48 hours'}) at $Time)."
Write-Host "Run it now to test:  schtasks /Run /TN $TaskName"
Write-Host "Output lands in:     $Repo\logs\collect.log"
