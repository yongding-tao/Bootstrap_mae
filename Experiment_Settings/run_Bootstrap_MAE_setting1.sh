#!/bin/bash

# run: nohup sh ./Experiment_Settings/run_Bootstrap_MAE_setting1.sh > ./Experiment_results/Bmae_k_2_pn/screen.log 2>&1 &

start_time=$SECONDS

# Settings
gpu_id=1
batch_size=256
bootstrap_k=2
base_path="./Experiment_results/Bmae_k_2_pn" # remember to turn on pixelnorm


# Run Bootstrap_MAE training script
echo "Running Bootstrap_MAE training script..."
sh ./Bmae_train.sh --gpu_id $gpu_id  --bootstrap_k $bootstrap_k --batch_size $batch_size --base_path $base_path

# Check if Bootstrap_MAE training script executed successfully
if [ $? -ne 0 ]; then
    echo "training script failed. Exiting..."
    exit 1
fi

# Run Bootstrap_MAE linear evaluation script
echo "Running Bootstrap_MAE linear evaluation script..."
sh ./Bmae_eval_linear.sh --gpu_id $gpu_id  --bootstrap_k $bootstrap_k --batch_size $batch_size --base_path $base_path

# Check if Bootstrap_MAE linear evaluation script executed successfully
if [ $? -ne 0 ]; then
    echo "linear evaluation script failed. Exiting..."
    exit 1
fi

# Run Bootstrap_MAE fine-tuning evaluation script
echo "Running Bootstrap_MAE fine-tuning evaluation script..."
sh ./Bmae_eval_finetune.sh --gpu_id $gpu_id  --bootstrap_k $bootstrap_k --batch_size $batch_size --base_path $base_path

# Check if Bootstrap_MAE fine-tuning evaluation script executed successfully
if [ $? -ne 0]; then
    echo "fine-tuning evaluation script failed. Exiting..."
    exit 1
fi

echo "All scripts ran successfully."

elapsed_time=$((SECONDS - start_time))

hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

printf "run script use: %02d:%02d:%02d\n" $hours $minutes $seconds