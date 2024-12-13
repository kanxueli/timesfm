#!/bin/bash
# caseid_list=(304 5597 3558 501 4367 1942 5634 1793 206 5242 5080 5795 533 1286 2182 530 5439 673 4673 6085 5871 5451 4949 3271 3149 1687 5194 932 6273 1695 3186 4886 3895 1182 3712 3589 4408 1222 3798 1880 852 283 6160 1520 10 4991 3835 4648 5568 4187 272 4730 4540 4474 5822 1170 5641 5593 5475 3736 584 1040 3196 4929 791 5977 4808 3059 911 3479 70 6209 6260 1035 1633 2671 1708 5446 3106 1386 419 2762 3399 4402 3034 4742 230 2455 970 5234 4802 5965 2312 1932 1759 1172 4122 2883 3137 1668)
caseid_list=(304 5597 3558 501 4367 1942 5634 1793 206 5242)

export TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
timesfm_path=/home/likx/time_series_forecasting/datasets_and_checkpoints/timesfm-1.0-200m/checkpoints/
dataset_path=/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/Instance_Level_Dataset_FineGrained
 

for caseid in "${caseid_list[@]}"
 do
 echo "Testing caseid: $caseid"
 CUDA_VISIBLE_DEVICES=4 python3 test_finetuned_timesfm.py \
    --checkpoint-path=$timesfm_path \
    --data-path=$dataset_path \
    --horizon-len=150 \
    --context-len=450 \
    --batch-size=1 \
    --case-id $caseid \
    --is-instance-setting \

done
