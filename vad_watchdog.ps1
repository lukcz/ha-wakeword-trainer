param(
    [string]$WslRepo = "/home/czarn/ha-wakeword-trainer",
    [int]$PollSeconds = 60,
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Stop"

$windowsRepo = Split-Path -Parent $PSCommandPath
$logsDir = Join-Path $windowsRepo "output\_logs"
$logPath = Join-Path $logsDir "wsl-vad-watchdog.log"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $logPath -Value "[$timestamp] Starting vad_orchestrator.py"

    & wsl.exe bash -lc "cd $WslRepo && exec .venv/bin/python vad_orchestrator.py --poll-seconds $PollSeconds"
    $returnCode = $LASTEXITCODE

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $logPath -Value "[$timestamp] vad_orchestrator.py exited with rc=$returnCode; restarting in $RestartDelaySeconds seconds"

    Start-Sleep -Seconds $RestartDelaySeconds
}
