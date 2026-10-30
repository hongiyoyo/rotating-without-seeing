#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
cd /mnt/c/YangJaehong/rotating-without-seeing
CK=$(ls runs/smoke64/*/nn/last_*.pth | head -1)
echo "checkpoint: $CK"
python tools/evaluate_policy_per_object.py checkpoint="'$CK'" task.env.numEnvs=90 \
  +target_episodes_per_class=5 +max_control_steps=400 +out_path=/tmp/smoke_eval.npz
echo "EXIT CODE: $?"
