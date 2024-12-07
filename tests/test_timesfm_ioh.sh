#!/bin/bash

# --data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/" \
# --data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained"
# for caseid in "${caseid_list[@]}"
#  do
#  echo "Testing caseid: $caseid"
CUDA_VISIBLE_DEVICES=4 python3 test_timesfm_ioh.py \
    --model-name="google/timesfm-1.0-200m" \
    --checkpoint-path="/home/data/times-forecasting/checkpoints/timesfm-1.0-200m-pytorch/torch_model.ckpt" \
    --horizon-len=150 \
    --context-len=450 \
    --batch-size=128 \
--data-path="/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained" \

# done
# To see all available options and their descriptions, use the --help flag
# python3 finetune.py --help
