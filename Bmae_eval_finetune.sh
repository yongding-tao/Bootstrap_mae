#!/bin/bash

set -e

# choose GPU
export CUDA_VISIBLE_DEVICES="${1:-1}"

# define the network and dataset path
MODEL="mae_Deit_tiny_patch4"  # choose model
DATA_PATH="./CIFAR10"  # the path of CIFAR10
IMG_SIZE=32
NB_CLASSES=10

# bootstrap
bootstrap_k=${3:-4}
# timestamp: $(date +"%Y%m%d-%H%M%S")
timestamp=$2
base_path="./Bootstrap_MAE/$timestamp"
load_epoch=$((200/bootstrap_k-1))

# define the path to save models and the log, and the save frequency
OUTPUT_DIR="$base_path/eval_finetune/output_dir"
LOG_DIR="$base_path/eval_finetune/log_dir"
# SAVE_FREQ=20

# hyperparameters
BATCH_SIZE=256
EPOCHS=100 # follow the requirement
LR=1e-4

# finetuning
CHECK_POINT="$base_path/MAE-$bootstrap_k/output_dir/checkpoint-$load_epoch.pth"

python main_finetune.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --device cuda \
    --nb_classes $NB_CLASSES \
    --finetune $CHECK_POINT \
    --input_size $IMG_SIZE