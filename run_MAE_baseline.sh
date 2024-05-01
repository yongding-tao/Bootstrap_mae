#!/bin/bash

# Run MAE training script
echo "Running MAE training script..."
sh mae_train.sh

# Check if MAE training script executed successfully
if [ $? -ne 0 ]; then
    echo "MAE training script failed. Exiting..."
    exit 1
fi

# Run MAE linear evaluation script
echo "Running MAE linear evaluation script..."
sh mae_eval_linear.sh

# Check if MAE linear evaluation script executed successfully
if [ $? -ne 0 ]; then
    echo "MAE linear evaluation script failed. Exiting..."
    exit 1
fi

# Run MAE fine-tuning evaluation script
echo "Running MAE fine-tuning evaluation script..."
sh mae_eval_finetune.sh

# Check if MAE fine-tuning evaluation script executed successfully
if [ $? -ne 0]; then
    echo "MAE fine-tuning evaluation script failed. Exiting..."
    exit 1
fi

echo "All scripts ran successfully."