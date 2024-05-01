#!/bin/bash

set -e

# choose GPU
export CUDA_VISIBLE_DEVICES=1

# Define the network and dataset path
MODEL="mae_Deit_tiny_patch4"  # Choose model
DATA_PATH="./CIFAR10"  # The path of CIFAR10

# Define the path to save models and the log, and the save frequency
OUTPUT_DIR="./MAE-1-normPixelLoss/pretrain/output_dir"
LOG_DIR="./MAE-1-normPixelLoss/pretrain/log_dir"
SAVE_FREQ=20

# Hyperparameters
BATCH_SIZE=256
EPOCHS=200
LR=1e-4
WEIGHT_DECAY=0.05

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
    --input_size 32 \
    --norm_pix_loss

# --------------------------------------------------------------------------

# # Run training with torch.distributed.launch in the background using nohup
# nohup python -m torch.distributed.launch \
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
#     --dist_url 'tcp://127.0.0.1:12355' > output.log 2>&1 &

# # Output a message indicating that the script is running in the background
# echo "Training is running in the background. Check output.log for details."