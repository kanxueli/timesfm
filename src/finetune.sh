#!/bin/bash

# Script to finetune a model with specific configurations
# Adjust the parameters below as needed. For a full list of options and descriptions, run the script with the --help flag.


export TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
timesfm_path=/home/likx/time_series_forecasting/datasets_and_checkpoints/timesfm-1.0-200m/checkpoints/
dataset_path=/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained
adapter_save_checkpoint_dir=./checkpoints/run_finetune

# clear adapter checkpoint dir
rm $adapter_save_checkpoint_dir -rf

# Note: current version don't support multi-GPU finetune
CUDA_VISIBLE_DEVICES=4 python3 finetune.py \
    --model-name="google/timesfm-1.0-200m" \
    --checkpoint-path=$timesfm_path \
    --data-path=$dataset_path \
    --backend="gpu" \
    --horizon-len=128 \
    --context-len=448 \
    --batch-size=32 \
    --freq="15min" \
    --num-features 1 \
    --dataset-type "IOH" \
    --checkpoint-dir=$adapter_save_checkpoint_dir \
    --num-epochs=10 \
    --learning-rate=1e-1 \
    --adam-epsilon=1e-3 \
    --adam-clip-threshold=1e2 \
    --early-stop-patience=10 \
    --datetime-col="date" \
    --is-instance-finetune \
    --case-id 1 \
    --use-lora \
    --lora-rank=1 \
    --lora-target-modules="all" \
    --cos-initial-decay-value=1e-4 \
    --cos-decay-steps=40000 \
    --cos-final-decay-value=1e-5 \
    --ema-decay=0.9999 \

# To see all available options and their descriptions, use the --help flag
# python3 finetune.py --help

# zero
# eval_mae_timesfm: 23.3532923634847
# eval_mse_timesfm: 1531.1454660408597
# eval_pred_lable_timesfm: 0.0

# --learning-rate=1e-3 --adam-epsilon=1e-7 
# eval_mae_timesfm: 23.34867436726888
# eval_mse_timesfm: 1530.6243313585521
# eval_pred_lable_timesfm: 0.0

# --learning-rate=1e-2 --adam-epsilon=1e-5 
# eval_mae_timesfm: 23.310136625501844
# eval_mse_timesfm: 1526.9395033677017
# eval_pred_lable_timesfm: 0.0

# --learning-rate=1e-1 --adam-epsilon=1e-3 
# eval_mae_timesfm: 21.7416833919949
# eval_mse_timesfm: 1356.9805721641346
# eval_pred_lable_timesfm: 0.0

# --learning-rate=1 --adam-epsilon=1e-1 
