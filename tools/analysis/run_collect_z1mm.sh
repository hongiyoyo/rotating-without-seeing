#!/bin/bash
# Tactile dataset for the classifier, paper placement setting (objects 0-1 mm above nominal).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing
CK=runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth
python tools/collect_tactile_data.py checkpoint="'$CK'" task.env.objInitPosZHigh=0.001 \
  task.env.numEnvs=900 +target_episodes_per_class=300 +target_full_per_class=200 +max_control_steps=4000 \
  +out_path=tools/data/tactile_dataset_z1mm.npz
echo "EXIT CODE: $?"
