#!/bin/bash

gpu_id=1 # agr1
timestamp=$(date +"%Y%m%d-%H%M%S") #arg2
bootstrap_k=4 # arg3

# Run Bootstrap_MAE training script
echo "Running Bootstrap_MAE training script..."
sh Bmae_train.sh "$gpu_id" "$timestamp" "$bootstrap_k"

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