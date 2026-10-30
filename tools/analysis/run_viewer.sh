#!/bin/bash
# Opens the Isaac Gym viewer replaying block_4 failed episodes. Extra args go to replay_viewer.py
# (e.g. --sort len, --start 10). Log: /mnt/c/YangJaehong/train_logs/viewer/viewer.log
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/train_logs/viewer
python replay_viewer.py "$@" > viewer.log 2>&1
