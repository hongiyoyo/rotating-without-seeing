#!/bin/bash
# $1 = numEnvs, $2 = minibatch, $3 = max_iterations, $4 = experiment/wandb name, $5 = wandb on (True/False)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
cd /mnt/c/YangJaehong/rotating-without-seeing
echo "git: $(git rev-parse --short HEAD) dirty: [$(git status --short)]"
wandb status 2>&1 | head -5; grep -c api.wandb.ai ~/.netrc
bash scripts/train_z_axis.sh 0 task.env.numEnvs=$1 \
  train.params.config.minibatch_size=$2 \
  train.params.config.central_value_config.minibatch_size=$2 \
  max_iterations=$3 wandb_activate=$5 wandb_name=$4 experiment=$4 train.params.config.user_prefix=$4
echo "EXIT CODE: $?"
