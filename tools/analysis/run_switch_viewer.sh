#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
pkill -f replay_viewer.py
cd /mnt/c/YangJaehong/rotating-without-seeing/tools/analysis
python replay_viewer.py --file /mnt/c/YangJaehong/train_logs/viewer/block4_switch_episodes.pkl > /mnt/c/YangJaehong/train_logs/viewer/viewer_switch.log 2>&1
