#!/bin/bash
# usage: bash run_closed_loop.sh   (env vars PER_GROUP, KS, OUT pass through)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing/tools/analysis
python closed_loop_switch.py 2>&1 | grep -vE "^\s*$|Warning|warn"
