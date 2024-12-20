#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
timesfm_path=/home/likx/time_series_forecasting/datasets_and_checkpoints/timesfm-1.0-200m/checkpoints/
dataset_path=/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained
 

CUDA_VISIBLE_DEVICES=4 python3 test_finetuned_timesfm.py \
 --checkpoint-path=$timesfm_path \
 --data-path=$dataset_path \
 --horizon-len=150 \
 --context-len=450 \
 --batch-size=1 \
 --use-lora \
 --lora-rank=2 \
 --lora-target-modules="all" \
 --adapter-path="./checkpoints/ft_lora" \

