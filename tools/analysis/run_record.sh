#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/train_logs/viewer
python record_failed_episodes.py "$@"
