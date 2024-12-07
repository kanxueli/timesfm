#!/bin/bash

# --data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/" \
# --data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained"
# for caseid in "${caseid_list[@]}"
#  do
#  echo "Testing caseid: $caseid"
CUDA_VISIBLE_DEVICES=4 python3 test_finetuned_timesfm.py \
    --model-name="google/timesfm-1.0-200m" \
    --checkpoint-path="/home/likx/time_series_forecasting/datasets_and_checkpoints/timesfm-1.0-200m/checkpoints/" \
    --data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained" \
    --horizon-len=150 \
    --context-len=450 \
    --batch-size=128 \
    --use-lora-adapter \
    --lora-rank=1 \
    --lora-target-modules="all" \
    --adapter-path="./checkpoints/run_20241205_231449_pg3jjpcg" \

# done
# To see all available options and their descriptions, use the --help flag
# python3 finetune.py --help
