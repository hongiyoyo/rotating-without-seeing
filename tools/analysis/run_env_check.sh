#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
cd /mnt/c/YangJaehong/rotating-without-seeing
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
N=${N:-512} python "$1"
