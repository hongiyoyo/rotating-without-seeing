# Waits for the training started by C:\YangJaehong\train_full.sh to finish, then uploads
# the best checkpoint (+ final checkpoint, config, tensorboard summaries, console log)
# as a GitHub Release of hongiyoyo/rotating-without-seeing. Also keeps a local copy in
# C:\YangJaehong\train_logs\backup_<tag>\ (outside the repo).
# Progress/errors: C:\YangJaehong\train_logs\backup_after_training.log
param(
    [string]$RunDir = "C:\YangJaehong\rotating-without-seeing\runs\z-axis-touch-only\z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810",
    [string]$TrainLog = "C:\YangJaehong\train_logs\train_20261028_152248.log",
    [string]$Tag = "ckpt-rws2-retrain-20261028",
    [string]$Name = "rws2-retrain",
    [string]$WandbUrl = "https://wandb.ai/jhyang321-seoul-national-university/isaacgymenvs/runs/lxgr93nn",
    [switch]$NoWait,  # testing: skip waiting for the training process
    [switch]$Draft    # testing: create a draft (non-public) release
)
$ErrorActionPreference = "Stop"
$Gh = "C:\Program Files\GitHub CLI\gh.exe"
$Repo = "hongiyoyo/rotating-without-seeing"
$Out = "C:\YangJaehong\train_logs\backup_after_training.log"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File $Out -Append -Encoding utf8 }

Log "watching for training (train_full.sh) to finish"
while (-not $NoWait -and (Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Where-Object { $_.CommandLine -like '*train_full.sh*' })) {
    Start-Sleep -Seconds 300
}
if (-not $NoWait) { Start-Sleep -Seconds 30 }  # let the final checkpoint and log flush
Log "training process ended"

try {
    $logText = Get-Content $TrainLog -Raw -Encoding utf8
    $finished = $logText -match "MAX EPOCHS NUM!"
    $exitLine = ([regex]::Matches($logText, "EXIT CODE: \d+") | Select-Object -Last 1).Value
    $bestLine = ([regex]::Matches($logText, "saving next best rewards:\s*\S+") | Select-Object -Last 1).Value
    $lastEpoch = ([regex]::Matches($logText, "epoch: \d+/\d+") | Select-Object -Last 1).Value

    $nn = Join-Path $RunDir "nn"
    $best = Get-ChildItem $nn -Filter *.pth | Where-Object { $_.Name -notlike "last_*" } | Select-Object -First 1
    $final = Get-ChildItem $nn -Filter "last_*.pth" | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $best) { throw "no best checkpoint in $nn" }

    $stage = "C:\YangJaehong\train_logs\backup_$Tag"
    New-Item -ItemType Directory -Force $stage | Out-Null
    $files = @()
    Copy-Item -LiteralPath $best.FullName "$stage\$($Name)_best.pth"; $files += "$stage\$($Name)_best.pth"
    if ($final) {
        $ep = if ($final.Name -match "ep_?(\d+)") { $Matches[1] } else { "x" }
        Copy-Item -LiteralPath $final.FullName "$stage\$($Name)_last_ep$ep.pth"; $files += "$stage\$($Name)_last_ep$ep.pth"
    }
    Copy-Item -LiteralPath (Join-Path $RunDir "config.yaml") "$stage\config.yaml"; $files += "$stage\config.yaml"
    Compress-Archive -LiteralPath (Join-Path $RunDir "summaries") -DestinationPath "$stage\tensorboard_summaries.zip" -Force
    $files += "$stage\tensorboard_summaries.zip"
    Copy-Item -LiteralPath $TrainLog "$stage\train_console.log"; $files += "$stage\train_console.log"
    Log "staged: $($files -join ', ')"

    $status = if ($finished) { "finished" } else { "INCOMPLETE (training stopped early)" }
    $commit = (& git -C C:\YangJaehong\rotating-without-seeing rev-parse --short HEAD)
    $notes = @"
Automatic backup after training ($status).

- wandb: $WandbUrl
- last epoch seen: $lastEpoch
- best: $bestLine
- $exitLine
- repo HEAD at upload: $commit

Files: $($Name)_best.pth = best checkpoint (rl_games nn/<name>.pth); last_ep*.pth = final checkpoint; config.yaml; tensorboard summaries; console log.
"@
    $notesFile = "$stage\release_notes.md"
    $notes | Out-File $notesFile -Encoding utf8

    $draftArg = @(); if ($Draft) { $draftArg = @("--draft") }
    $ErrorActionPreference = "Continue"  # gh writes progress to stderr; judge by exit code instead
    & $Gh release create $Tag @files @draftArg --repo $Repo --target main --title "Checkpoint: $Name ($status)" --notes-file $notesFile 2>&1 | ForEach-Object { Log "gh: $_" }
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed (exit $LASTEXITCODE). Is gh still logged in?" }
    Log "UPLOAD DONE: https://github.com/$Repo/releases/tag/$Tag"
} catch {
    Log "ERROR: $_"
}
