#!/bin/bash

# run: sh Experient_settings/run_EMAed_MAE.sh

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
timestamp=$(date +"%Y%m%d-%H%M%S")
use_ema=false

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

# Run MAE training script
echo "Running EMAed_MAE training script..."
sh ema_mae_train.sh --base_path $base_path --gpu_id $gpu_id --batch_size $batch_size

# Check if MAE training script executed successfully
if [ $? -ne 0 ]; then
    echo "MAE training script failed. Exiting..."
    exit 1
fi

# Run MAE linear evaluation script
echo "Running MAE linear evaluation script..."
sh mae_eval_linear.sh --base_path $base_path --gpu_id $gpu_id --batch_size $batch_size

# Check if MAE linear evaluation script executed successfully
if [ $? -ne 0 ]; then
    echo "MAE linear evaluation script failed. Exiting..."
    exit 1
fi

# Run MAE fine-tuning evaluation script
echo "Running MAE fine-tuning evaluation script..."
sh mae_eval_finetune.sh --base_path $base_path --gpu_id $gpu_id --batch_size $batch_size

# Check if MAE fine-tuning evaluation script executed successfully
if [ $? -ne 0]; then
    echo "MAE fine-tuning evaluation script failed. Exiting..."
    exit 1
fi

echo "All scripts ran successfully."