#!/bin/bash

# Parse options using getopt
TEMP=$(getopt -o "" --long gpu_id:,batch_size:,base_path:,use_ema: -- "$@")
if [ $? != 0 ]; then
    echo "Error parsing options."
    exit 1
fi

# Set the positional parameters to the parsed options
eval set -- "$TEMP"

# Initialize variables with default values
gpu_id=0
batch_size=256
use_ema=false
epochs_sum=200 # follow the requirement
bootstrap_k=4

# Process options
while true; do
    case "$1" in
        --gpu_id)
            gpu_id="$2"
            echo "Option --gpu_id with value '$gpu_id'"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            echo "Option --batch_size with value '$batch_size'"
            shift 2
            ;;
        --bootstrap_k)
            bootstrap_k="$2"
            echo "Option --bootstrap_k with value '$bootstrap_k'"
            shift 2
            ;;
        --base_path)
            base_path="$2"
            echo "Option --base_path with value '$base_path'"
            shift 2
            ;;
        --use_ema)
            use_ema=true
            echo "Option --use_ema turn on"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Remaining arguments are available as positional parameters ($1, $2, etc.)
for arg in "$@"; do
    echo "Remaining arguments: $arg"
done

# Demonstration of using parsed options
echo "Using GPU ID: $gpu_id"
echo "Pretrain using batch size: $batch_size"
echo "base_path: $base_path"
echo "bootstrap_k: $boot_strap"

# choose GPU
export CUDA_VISIBLE_DEVICES=$gpu_id

# define the network and dataset path
MODEL="mae_Deit_tiny_patch4"  # choose model
DATA_PATH="./CIFAR10"  # the path of CIFAR10
IMG_SIZE=32
NB_CLASSES=10

# bootstrap
load_epoch=$((200/bootstrap_k-1))

OUTPUT_DIR="$base_path/eval_linear/output_dir"
LOG_DIR="$base_path/eval_linear/log_dir"
# SAVE_FREQ=20

# hyperparameters
EPOCHS=100 # follow the requirement
LR=1e-4
WEIGHT_DECAY=0

# finetuning
CHECK_POINT="$base_path/MAE-$bootstrap_k/output_dir/checkpoint-$load_epoch.pth"

python main_linprobe.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --batch_size $batch_size \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device cuda \
    --nb_classes $NB_CLASSES \
    --finetune $CHECK_POINT