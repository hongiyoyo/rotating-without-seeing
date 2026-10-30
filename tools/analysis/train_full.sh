#!/bin/bash
# Full training run (launched 2026-10-28). Log: /mnt/c/YangJaehong/train_logs/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1  # write log lines immediately
cd /mnt/c/YangJaehong/rotating-without-seeing
LOG=/mnt/c/YangJaehong/train_logs/train_$(date +%Y%m%d_%H%M%S).log
echo "git: $(git rev-parse --short HEAD) dirty: [$(git status --short | tr '\n' ' ')]" > "$LOG"
bash scripts/train_z_axis.sh 0 wandb_name=rws2-retrain >> "$LOG" 2>&1
echo "EXIT CODE: $?" >> "$LOG"
