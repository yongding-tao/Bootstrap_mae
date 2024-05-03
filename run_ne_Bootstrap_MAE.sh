#!/bin/bash

# Parse options using getopt
TEMP=$(getopt -o "" --long gpu_id:,batch_size:,bootstrap_k: -- "$@")
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
echo "Using bootstrap_k: $bootstrap_k"
echo "Pretrain using batch size: $batch_size"

# Run Bootstrap_MAE training script
echo "Running ne_Bootstrap_MAE training script..."
sh ne_Bmae_train.sh "$gpu_id" "$timestamp" "$bootstrap_k" "$batch_size"

# Check if Bootstrap_MAE training script executed successfully
if [ $? -ne 0 ]; then
    echo "training script failed. Exiting..."
    exit 1
fi

# Run Bootstrap_MAE linear evaluation script
echo "Running Bootstrap_MAE linear evaluation script..."
sh Bmae_eval_linear.sh "$gpu_id" "$timestamp" "$bootstrap_k"

# Check if Bootstrap_MAE linear evaluation script executed successfully
if [ $? -ne 0 ]; then
    echo "linear evaluation script failed. Exiting..."
    exit 1
fi

# Run Bootstrap_MAE fine-tuning evaluation script
echo "Running Bootstrap_MAE fine-tuning evaluation script..."
sh Bmae_eval_finetune.sh "$gpu_id" "$timestamp" "$bootstrap_k"

# Check if Bootstrap_MAE fine-tuning evaluation script executed successfully
if [ $? -ne 0]; then
    echo "fine-tuning evaluation script failed. Exiting..."
    exit 1
fi

echo "All scripts ran successfully."