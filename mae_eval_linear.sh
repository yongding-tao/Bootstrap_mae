#!/bin/bash

set -e

# choose GPU
export CUDA_VISIBLE_DEVICES=1

# define the network and dataset path
MODEL="mae_Deit_tiny_patch4"  # choose model
DATA_PATH="./CIFAR10"  # the path of CIFAR10
IMG_SIZE=32
NB_CLASSES=10

# define the path to save models and the log, and the save frequency
OUTPUT_DIR="./MAE-1-normPixelLoss/eval_linear/output_dir"
LOG_DIR="./MAE-1-normPixelLoss/eval_linear/log_dir"
# SAVE_FREQ=20

# hyperparameters
BATCH_SIZE=256
EPOCHS=100 # follow the requirement
LR=1e-4
WEIGHT_DECAY=0

# finetuning
CHECK_POINT="./MAE-1-normPixelLoss/pretrain/output_dir/checkpoint-199.pth"

python main_linprobe.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device cuda \
    --nb_classes $NB_CLASSES \
    --finetune $CHECK_POINT

# ---------------------------------------------------------------------

# # Run training with torch.distributed.launch
# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --master_port=12355 \
#     main_pretrain.py \
#     --model $MODEL \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --log_dir $LOG_DIR \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --weight_decay $WEIGHT_DECAY \
#     --save_freq $SAVE_FREQ \
#     --device cuda \
#     --input_size 32 \
#     --world_size 2 \
#     --dist_url 'tcp://127.0.0.1:12355'