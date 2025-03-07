# from numba import njit
import numpy as np
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve

import timesfm
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import typer
import wandb
import os
from typing_extensions import Annotated
from adapter.utils import load_adapter_checkpoint
import json

# Define metrics
def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

# @njit
def Check_If_IOH(time_series, srate, IOH_value, duration):
    """
    Check if there is a period of intraoperative hypotension (IOH) in the time series.

    Parameters:
    - time_series (1D array-like): The blood pressure time series.
    - srate: Sampling rate of the time series (samples per second).
    - IOH_value: Threshold value for IOH (blood pressure below this is considered hypotensive).
    - duration: duration in seconds that defines IOH (must stay below IOH_value for this period).

    Returns:
    - bool: True if IOH is detected, otherwise False.
    """
    if isinstance(time_series, list):
        time_series = np.array(time_series)

    duration_samples = int(duration * srate)
    
    if len(time_series) < duration_samples:
        return False
    
    below_threshold = time_series < IOH_value
    
    for i in range(len(below_threshold) - duration_samples + 1):
        if np.all(below_threshold[i:i + duration_samples]):
            return True
    
    return False

def sliding_window_average(time_series, slide_len):
    if slide_len <= 0:
        raise ValueError("slide_len must be greater than 0")
    
    window_averages = []
    
    for i in range(0, len(time_series), slide_len):
        window = time_series[i:i + slide_len]
        window_avg = round(np.nanmean(window), 2)
        window_averages.append(window_avg)
    
    return window_averages

def Check_If_IOH_Combined_S(time_series, stime, IOH_value, Duration, slide_len):
    duration_samples = int(Duration / slide_len)
    slide_samples = slide_len
    
    if slide_samples == 1:
        smoothed_series = time_series
    else:
        smoothed_series = sliding_window_average(time_series, slide_samples)

    if duration_samples == 1:
        evt = np.nanmax(smoothed_series) < IOH_value
    else:
        evt = Check_If_IOH(smoothed_series, 1, IOH_value, duration_samples)

    return evt

def round_data_to_two_decimals(data):
    return np.where(np.isnan(data), data, np.round(data, 2))

def user_definable_IOH(time_series):
    predcition_lables = []
    for i in time_series:
        predcition_lables.append(Check_If_IOH_Combined_S(i, 1, 65, 30, 1))
    return predcition_lables

def get_batched_data_fn(
    batch_size: int = 128, 
    data_path: str = None,
    context_len: int = 120, 
    horizon_len: int = 24,
    is_instance: bool = False,
    data_flag: str = "test",
    case_id: int = 0,
    sr=0,
):
    # 读取CSV文件
    if is_instance :
        if data_flag == 'train':
            data = pd.read_csv(os.path.join(data_path, str(case_id) + '_train.csv'))
        elif data_flag == 'val':
            data = pd.read_csv(os.path.join(data_path, str(case_id)  + '_val.csv'))
        elif data_flag == 'test':
            data = pd.read_csv(os.path.join(data_path, str(case_id)  + '_test.csv'))
    else:
        if data_flag == 'train':
            data = pd.read_csv(os.path.join(data_path, 'vitaldb_train_data.csv'))
        elif data_flag == 'val':
            data = pd.read_csv(os.path.join(data_path, 'vitaldb_val_data.csv'))
        elif data_flag == 'test':
            data = pd.read_csv(os.path.join(data_path, 'vitaldb_test_data.csv'))

    # 数据预处理前总数据
    print("源数据长度：", len(data))
    label_counts = data['label'].value_counts(normalize=True) * 100
    print("处理前的Label分布 (%):")
    print(label_counts)

    def parse_sequence(sequence_str, skip_rate=0, sample_type='avg_sample'):
        try:
            sequence_list = sequence_str.split()
            sequence_array = np.array([np.nan if x == 'nan' else float(x) for x in sequence_list])
            mean_value = round(np.nanmean(sequence_array), 2)
            sequence_array_filled = np.where(np.isnan(sequence_array), mean_value, sequence_array)
            if np.any(np.isnan(sequence_array_filled)):
                return [] 
                    
            if skip_rate > 0: # 如果需要重采样
                if sample_type == 'skip_sample':
                    sequence_array_filled = sequence_array_filled[::skip_rate]
                elif sample_type == 'avg_sample': #默认按平均值进行采样
                    sequence_array_filled = sliding_window_average(sequence_array_filled, skip_rate)

            return sequence_array_filled
        except ValueError:
            return [] 
    # 初始化 defaultdict
    examples = defaultdict(list)

    for index, row in data.iterrows():
        bts = parse_sequence(row['bts'][1:-1], skip_rate=sr, sample_type='avg_sample') #采样周期是：2*skip_rate
        hrs = parse_sequence(row['hrs'][1:-1], skip_rate=sr, sample_type='avg_sample')
        dbp = parse_sequence(row['dbp'][1:-1], skip_rate=sr, sample_type='avg_sample')
        mbp = parse_sequence(row['mbp'][1:-1], skip_rate=sr, sample_type='avg_sample')
        prediction_mbp = parse_sequence(row['prediction_mbp'][1:-1], skip_rate=sr, sample_type='avg_sample')
        # print(len(bts), len(hrs), len(dbp), len(mbp), len(prediction_mbp))
        if len(bts) != context_len or len(hrs) != context_len or len(dbp) != context_len or\
            len(mbp) != context_len or len(prediction_mbp) != horizon_len:
            continue
        
        if (np.abs(np.diff(mbp)) > 30).any() or (np.abs(np.diff(prediction_mbp)) > 30).any():
            continue

        # if np.mean(mbp) > 100 or np.mean(prediction_mbp) > 100:
        #     continue

        examples['caseid'].append(row['caseid'])
        examples['stime'].append(row['stime'])
        examples['ioh_stime'].append(row['ioh_stime'])
        examples['ioh_dtime'].append(row['ioh_dtime'])
        examples['age'].append(row['age']) # np.full(len(bts), row['age'])
        examples['sex'].append(row['sex'])
        examples['bmi'].append(row['bmi'])
        examples['label'].append(Check_If_IOH_Combined_S(prediction_mbp, 1, 65, 30, 1))
        examples['bts'].append(bts)
        examples['hrs'].append(hrs)
        examples['dbp'].append(dbp)
        examples['inputs'].append(mbp)
        examples['outputs'].append(prediction_mbp)

    # 修正统计处理后的样本数量
    print("处理后的测试样本数量:", len(examples['caseid']))

    # 统计处理后 examples 中 label 列的分布
    label_counts = pd.Series(examples['label']).value_counts(normalize=True) * 100
    print("处理后的Label分布 (%):")
    print(label_counts)

    def data_fn(): # 批次生成器函数
        for i in range(1 + (len(examples['caseid']) - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
    
    return data_fn

def save_predictions(preds, trues, file_path="predictions_and_trues.json"):
    """
    Save the predictions and true values to a file for later evaluation.
    If the file already exists, append the new predictions and true values.

    Parameters:
    - preds (ndarray): The predicted values from the model.
    - trues (ndarray): The ground truth values.
    - file_path (str): The file path where the data will be saved.
    """
    # Convert ndarray to list for JSON serialization
    new_data = {
        'predictions': preds.tolist(),
        'ground_truths': trues.tolist()
    }

    # Check if the file exists
    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {'predictions': [], 'ground_truths': []}

    # Append new data to existing data
    existing_data['predictions'].extend(new_data['predictions'])
    existing_data['ground_truths'].extend(new_data['ground_truths'])

    # Write the updated data back to the file
    with open(file_path, 'w') as f:
        json.dump(existing_data, f)

def test_timesfm(
    *,
    model_name: Annotated[
        str, typer.Option(help="Specify the name of the huggingface model.")
    ] = "google/timesfm-1.0-200m",
    checkpoint_path: Annotated[
        str, typer.Option(help="The path to the local model checkpoint.")
    ] = "/home/data/times-forecasting/checkpoints/timesfm-1.0-200m-pytorch/torch_model.ckpt",
    adapter_path: Annotated[
        str, typer.Option(help="The path to the local model checkpoint.")
    ] = "",
    results_path: Annotated[
        str, typer.Option(help="The path to the saving results.")
    ] = "predictions_and_trues.json",
    context_len: Annotated[int, typer.Option(help="Length of the context window")],
    horizon_len: Annotated[int, typer.Option(help="Prediction length.")],
    batch_size: Annotated[
        int, typer.Option(help="Batch size for the randomly sampled batch")
    ] = 128,
    lora_rank: Annotated[int, typer.Option(help="LoRA Rank",),] = 8,
    lora_target_modules: Annotated[
        str,
        typer.Option(
            help="LoRA target modules of the transformer block. Allowed values: [all, attention, mlp]"
        ),
    ] = "all",
    case_id: Annotated[
        int, typer.Option(help="Batch size for the randomly sampled batch")
    ] = 0,
    is_instance_setting: Annotated[
        bool, typer.Option(help="Normalize data for eval or not")
    ] = False,
    use_dora: Annotated[
        bool, typer.Option(help="Normalize data for eval or not")
    ] = False,
    use_lora: Annotated[
        bool, typer.Option(help="Normalize data for eval or not")
    ] = False,
    data_path: Annotated[str, typer.Option(help="Path to dataset csv")]='/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/vitaldb_test_data.csv',
):
    # # Loading TimesFM in pytorch version
    # tfm = timesfm.TimesFm(
    #     hparams=timesfm.TimesFmHparams(
    #         backend="gpu",
    #         per_core_batch_size=32,      
    #         horizon_len=horizon_len,
    #     ),
    #     checkpoint=timesfm.TimesFmCheckpoint(
    #         version="torch",
    #         path=checkpoint_path),
    # )
    # print("Loading Model Finish.")

    # Loading TimesFM in pax/jax version
    tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,      
        horizon_len=horizon_len,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
         version="jax",
         path=checkpoint_path),
    )
    
    # Loading adapter for TimesFM if need
    if use_lora:
        load_adapter_checkpoint(
            model=tfm,
            adapter_checkpoint_path=adapter_path,
            lora_rank=lora_rank,
            lora_target_modules=lora_target_modules,
            use_dora=use_dora,
            )

    # Benchmark
    if is_instance_setting:
        input_data = get_batched_data_fn(
            batch_size=batch_size, data_path=data_path, context_len=context_len, 
            horizon_len=horizon_len, is_instance=True, data_flag="test", case_id=case_id)
    else:
        input_data = get_batched_data_fn(
            batch_size=batch_size, data_path=data_path, context_len=context_len, 
            horizon_len=horizon_len, is_instance=False, data_flag="test")
    
    metrics = defaultdict(list)

    ground_true_labels = []
    preds = []
    trues = []
    for i, example in enumerate(input_data()):
        if np.array(example["inputs"]).shape != (batch_size, context_len):
            continue

        start_time = time.time()
        raw_forecast, _ = tfm.forecast(
            inputs=example["inputs"], freq=[0] * len(example["inputs"])
        )
        
        print(
            f"\rFinished batch {i} linear in {time.time() - start_time} seconds",
            end="",
        )

        # print()
        raw_forecast = raw_forecast[:, :horizon_len]
        true_series = np.array(example["outputs"])[:, :horizon_len]
        ground_true_labels.extend(example["label"])
        preds.append(raw_forecast)
        trues.append(true_series)

        metrics["eval_mae_timesfm"].extend(mae(raw_forecast, true_series))
        metrics["eval_mse_timesfm"].extend(mse(raw_forecast[:, :horizon_len], true_series))
        metrics["eval_pred_lable_timesfm"].extend(user_definable_IOH(raw_forecast))

    print()

    # saving prediction results
    pds = np.array(preds)
    trs = np.array(trues)
    pds = pds.reshape(-1, pds.shape[-1], 1) # 注意，这里只对单变量预测有效
    trs = trs.reshape(-1, trs.shape[-1], 1)
    print("p_shape:{}, t_shape:{}".format(pds.shape, trs.shape))
    save_predictions(pds, trs, file_path=results_path)

    for k, v in metrics.items():
        if not is_instance_setting and k in ["eval_pred_lable_timesfm", "eval_pred_lable_xreg_timesfm", "eval_pred_lable_xreg"]:
            print(k, "--Prediction Results:")
            precision, recall, thmbps = precision_recall_curve(ground_true_labels, v)
            auprc = auc(recall, precision)

            fpr, tpr, thmbps = roc_curve(ground_true_labels, v)
            auroc = auc(fpr, tpr)
            f1 = f1_score(ground_true_labels, v)
            acc = accuracy_score(ground_true_labels, v)
            tn, fp, fn, tp = confusion_matrix(ground_true_labels, v).ravel()

            testres = 'auroc={:.3f}, auprc={:.3f} acc={:.3f}, F1={:.3f}, PPV={:.1f}, NPV={:.1f}, TN={}, fp={}, fn={}, TP={}'.format(auroc, auprc, acc, f1, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp)
            print(testres)
        else:   
            print(f"{k}: {np.mean(v)}")

if __name__ == "__main__":
    typer.run(test_timesfm)