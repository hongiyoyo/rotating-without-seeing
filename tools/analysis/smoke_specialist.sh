#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robosyn
export PYTHONUNBUFFERED=1
cd /mnt/c/YangJaehong/rotating-without-seeing
SRC=runs/z-axis-touch-only/z-axis-touch-onlyS1.0_C0.0_M0.02026-10-28_15-22-53-83810/nn/z-axis-touch-only.pth
mkdir -p runs/finetune_init
python tools/make_finetune_init.py $SRC runs/finetune_init/generalist_best_reset.pth
bash scripts/train_z_axis.sh 0 task.env.forceObject=block_4 checkpoint=runs/finetune_init/generalist_best_reset.pth \
  task.env.numEnvs=64 train.params.config.minibatch_size=512 train.params.config.central_value_config.minibatch_size=512 \
  max_iterations=3 wandb_activate=False experiment=smoke_block4 train.params.config.user_prefix=smoke_block4 2>&1 \
  | grep -E "saved:|source:|epoch [0-9]+, frame|=> loading|Loading|fps step|PerObject|Error|Traceback|error|saving|MAX EPOCHS"
echo "EXIT ${PIPESTATUS[0]}"
