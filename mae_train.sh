# #!/bin/bash

# set -e

# # Define the network and dataset path
# MODEL="mae_Deit_tiny_patch4"  # Choose model
# DATA_PATH="../../Data/CIFAR10"  # The path of CIFAR10

# # Define the path to save models and the log, and the save frequency
# OUTPUT_DIR="./MAE-1/pretrain/output_dir"
# LOG_DIR="./MAE-1/pretrain/log_dir"
# SAVE_FREQ=20

# # Hyperparameters
# BATCH_SIZE=128
# EPOCHS=200
# LR=1e-4
# WEIGHT_DECAY=0.05

# # Run training with torch.distributed.launch in the background using nohup
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

# # Output a message indicating that the script is running in the background
# echo "Training is running in the background. Check output.log for details."

# -------------------------------------------------------------------------

#!/bin/bash

set -e

# choose GPU
export CUDA_VISIBLE_DEVICES=${1:-0}

# define the network and dataset path
MODEL="mae_Deit_tiny_patch4"  # choose model
DATA_PATH="./CIFAR10"  # the path of CIFAR10

# define the path to save models and the log, and the save frequency
OUTPUT_DIR="tmp" #"./MAE-1/pretrain/output_dir"
LOG_DIR="tmp" #"./MAE-1/pretrain/log_dir"
SAVE_FREQ=20

# hyperparameters
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
    --save_freq $SAVE_FREQ\
    --device cuda \
    --input_size 32  # the img_size of CIFAR10 is 32x32

# ---------------------------------------------------------------------

# #!/bin/bash

# set -e

# # define the network and dataset path
# MODEL="mae_Deit_tiny_patch4"  # choose model
# DATA_PATH="../../Data/CIFAR10"  # the path of CIFAR10

# # define the path to save models and the log, and the save frequency
# OUTPUT_DIR="./MAE-1/output_dir"
# LOG_DIR="./MAE-1/log_dir"
# SAVE_FREQ=20

# # hyperparameters
# BATCH_SIZE=128
# EPOCHS=200
# LR=1e-4
# WEIGHT_DECAY=0.05

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