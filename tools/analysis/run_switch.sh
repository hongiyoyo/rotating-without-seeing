#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing/tools/analysis
python record_switch.py 2>&1 | grep -v "CFG DICT\|Hand QPos\|DOF_\|Warning"
