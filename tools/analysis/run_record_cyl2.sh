#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/train_logs/viewer
pkill -f replay_viewer.py && echo "closed previous viewer"
TARGET=cylinder_2 PICK=5 python record_failed_episodes.py 1500
