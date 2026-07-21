#!/bin/bash
# Touch-only (no vision) z-axis in-hand rotation teacher policy training.
# Run from the repo root: bash scripts/train_z_axis.sh <GPU_ID> [extra hydra overrides...]
#
# All the scenario-specific settings (objSet=C, axis=z, observationType=partial_stack,
# sensor=thick, numEnvs=8192, minibatch_size=16384, and the paper-fidelity domain
# randomization/reward values) are already baked into the defaults of
# isaacgymenvs/cfg/task/AllegroArmMOAR.yaml and isaacgymenvs/cfg/train/AllegroArmMOARPPO.yaml,
# so this script only needs to set headless mode and an experiment name.

GPUS=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
experiment=z-axis-touch-only \
train.params.config.user_prefix=z-axis-touch-only \
wandb_activate=True \
${EXTRA_ARGS}
