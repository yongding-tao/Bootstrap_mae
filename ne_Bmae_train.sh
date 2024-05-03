#!/bin/bash

set -e

# choose GPU
export CUDA_VISIBLE_DEVICES="${1:-1}"
# echo "gpu_id: $CUDA_VISIBLE_DEVICES"

# define the network, dataset path, the path to save model and results
MODEL="mae_Deit_tiny_patch4"  # choose model
DATA_PATH="./CIFAR10"  # the path of CIFAR10
img_size=32
base_path="./ne_Bootstrap_MAE"
# timestamp: $(date +"%Y%m%d-%H%M%S")
timestamp=$2
save_dir="$base_path/$timestamp"

# define the save frequency
SAVE_FREQ=1

# hyperparameters
BATCH_SIZE=$4
EPOCHS_SUM=200 # follow the requirement
LR=1e-4
WEIGHT_DECAY=0.05
bootstrap_k=${3:-4}
feature_depth=8
EPOCHS=150

k=1
OUTPUT_DIR="$save_dir/MAE-$k/output_dir"
LOG_DIR="$save_dir/MAE-$k/log_dir"

python main_pretrain.py \
--model $MODEL \
--data_path $DATA_PATH \
--output_dir $OUTPUT_DIR \
--log_dir $LOG_DIR \
--batch_size $BATCH_SIZE \
--epochs $EPOCHS \
--lr $LR \
--weight_decay $WEIGHT_DECAY \
--save_freq $SAVE_FREQ \
--device cuda \
--input_size $img_size \
--bootstrap_k $k \
--feature_depth $feature_depth \

k=2
OUTPUT_DIR="$save_dir/MAE-$k/output_dir"
LOG_DIR="$save_dir/MAE-$k/log_dir"
EPOCHS=50
pre_k=$((k-1))
check_point="$save_dir/MAE-$pre_k/output_dir/checkpoint-149.pth"

python main_pretrain.py \
--model $MODEL \
--data_path $DATA_PATH \
--output_dir $OUTPUT_DIR \
--log_dir $LOG_DIR \
--batch_size $BATCH_SIZE \
--epochs $EPOCHS \
--lr $LR \
--weight_decay $WEIGHT_DECAY \
--save_freq $SAVE_FREQ \
--device cuda \
--input_size $img_size \
--bootstrap_k $k \
--feature_depth $feature_depth \
--last_model_checkpoint $check_point