#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing/tools/analysis
python compare_switch.py 2>&1 | grep -E "configs:|generalist|G ->| only|COMPARE_DONE|Error|Traceback|error"
