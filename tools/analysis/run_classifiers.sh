#!/bin/bash
# 6 windows x 3 channel sets x 3 seeds (model init; split fixed) on the 0-1 mm tactile dataset.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing
DATA=tools/data/tactile_dataset_z1mm.npz
OUT=tools/data/classifier_z1mm
mkdir -p $OUT
for K in 1 2 5 10 20 50; do
  for CH in tactile joint_pos tactile,joint_pos; do
    for SEED in 0 1 2; do
      TAG="k${K}_${CH/,/+}_s${SEED}"
      python tools/train_tactile_classifier.py --data $DATA --epochs 50 --window-sec $K --channels $CH \
        --seed $SEED --split-seed 42 --out-json $OUT/$TAG.json --save-model $OUT/$TAG.pt > $OUT/$TAG.log 2>&1
      echo "$TAG exit $? $(grep -o 'test loss [0-9.]* acc [0-9.]*' $OUT/$TAG.log)"
    done
  done
done
echo ALL_DONE
