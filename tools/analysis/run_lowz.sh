#!/bin/bash
# Evaluation with objects placed only 0-1 mm above the nominal start (training uses 0-10 mm).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing
CK=runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth
echo "== eval"
python tools/evaluate_policy_per_object.py checkpoint="'$CK'" task.env.objInitPosZHigh=0.001 \
  task.env.numEnvs=900 +target_episodes_per_class=300 +max_control_steps=3000 \
  +out_path=tools/data/policy_eval_z1mm.npz
echo "EVAL EXIT: $?"
echo "== reasons"
cd /mnt/c/YangJaehong/train_logs/viewer
OUT=/mnt/c/YangJaehong/train_logs/viewer/reset_reasons_z1mm.json python record_reset_reasons.py task.env.objInitPosZHigh=0.001
echo "REASONS EXIT: $?"
