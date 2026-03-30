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
    Add-Content -Path $logPath -Value "[$timestamp] Starting auto_vad_research.py"

    & wsl.exe bash -lc "cd $WslRepo && .venv/bin/python auto_vad_research.py --max-launches 999 --poll-seconds $PollSeconds" >> $logPath 2>&1
    $returnCode = $LASTEXITCODE

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
    Add-Content -Path $logPath -Value "[$timestamp] auto_vad_research.py exited with rc=$returnCode; restarting in $RestartDelaySeconds seconds"

    Start-Sleep -Seconds $RestartDelaySeconds
}
