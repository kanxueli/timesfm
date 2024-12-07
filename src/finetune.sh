#!/bin/bash

# Script to finetune a model with specific configurations
# Adjust the parameters below as needed. For a full list of options and descriptions, run the script with the --help flag.


export TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false

# Note: current version don't support multi-GPU finetune
CUDA_VISIBLE_DEVICES=4 python3 finetune.py \
    --model-name="google/timesfm-1.0-200m" \
    --checkpoint-path="/home/likx/time_series_forecasting/datasets_and_checkpoints/timesfm-1.0-200m/checkpoints/" \
    --backend="gpu" \
    --horizon-len=128 \
    --context-len=448 \
    --batch-size=32 \
    --freq="15min" \
    --num-features 1 \
    --dataset-type "IOH" \
    --data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained" \
    --num-epochs=1 \
    --learning-rate=1e-3 \
    --adam-epsilon=1e-7 \
    --adam-clip-threshold=1e2 \
    --early-stop-patience=10 \
    --datetime-col="date" \
    --is-instance-finetune \
    --case-id 4 \
    --use-lora \
    --lora-rank=1 \
    --lora-target-modules="all" \
    --cos-initial-decay-value=1e-4 \
    --cos-decay-steps=40000 \
    --cos-final-decay-value=1e-5 \
    --ema-decay=0.9999

# To see all available options and their descriptions, use the --help flag
# python3 finetune.py --help
