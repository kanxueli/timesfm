# from numba import njit
import numpy as np
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve

import timesfm
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from peft import get_peft_model, LoraConfig, TaskType

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
    data_path: str=None, 
    # context_len: int = 120, 
    # horizon_len: int = 24,
):
    # 读取CSV文件
    csv_file = data_path
    data = pd.read_csv(csv_file)

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
        bts = parse_sequence(row['bts'][1:-1], skip_rate=0, sample_type='skip_sample') #采样周期是：2*skip_rate
        hrs = parse_sequence(row['hrs'][1:-1], skip_rate=0, sample_type='skip_sample')
        dbp = parse_sequence(row['dbp'][1:-1], skip_rate=0, sample_type='skip_sample')
        mbp = parse_sequence(row['mbp'][1:-1], skip_rate=0, sample_type='skip_sample')
        prediction_mbp = parse_sequence(row['prediction_mbp'][1:-1], skip_rate=0, sample_type='skip_sample')
        # print(len(bts), len(hrs), len(dbp), len(mbp), len(prediction_mbp))
        if len(bts) != 450 or len(hrs) != 450 or len(dbp) != 450 or\
            len(mbp) != 450 or len(prediction_mbp) != 150:
            continue
        
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
        for i in range(1 + (len(data) - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
    
    return data_fn

# 数据集加载与批量化
context_len = 450
horizon_len = 150
bs = 128
train_data = get_batched_data_fn(batch_size=bs, data_path='/home/likx/time_series_forecasting/IOH_Datasets_Preprocess/vitaldb/vitaldb_train_data.csv')

# Loading TimesFM in pytorch version
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,      
          horizon_len=horizon_len,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          version="torch",
          path="/home/data/times-forecasting/checkpoints/timesfm-1.0-200m-pytorch/torch_model.ckpt"),
  )
print("Loading Model Finish.")


# 使用LoRA进行微调
lora_config = LoraConfig(
    r=8,  
    lora_alpha=16,
    lora_dropout=0.1,  
    bias="none",
    target_modules=["query", "key", "value"],
)

lora_model = get_peft_model(tfm, lora_config)

# 冻结原始模型权重，只微调LoRA模块
for param in tfm.parameters():
    param.requires_grad = False
for param in lora_model.parameters():
    param.requires_grad = True

# 定义优化器
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-5)

# 微调循环
metrics = defaultdict(list)
ground_true_labels = []
for epoch in range(5):  # 微调5个epoch
    for i, example in enumerate(train_data()):
        if np.array(example["inputs"]).shape != (bs, context_len):
            continue

        optimizer.zero_grad()
        
        # 将数据送入模型进行预测
        inputs = torch.tensor(example["inputs"]).float()
        outputs = torch.tensor(example["outputs"]).float()
        
        raw_forecast, _ = lora_model.forecast(inputs=inputs, freq=[0] * len(inputs))
        
        # 计算MSE损失
        loss = mse(raw_forecast, outputs)
        loss.backward()
        
        # 更新LoRA模块的参数
        optimizer.step()
        
        # 更新评估指标
        raw_forecast = raw_forecast[:, :horizon_len]
        true_series = np.array(example["outputs"])[:, :horizon_len]
        ground_true_labels.extend(example["label"])

        metrics["eval_mae_timesfm"].extend(mae(raw_forecast, true_series))
        metrics["eval_mse_timesfm"].extend(mse(raw_forecast[:, :horizon_len], true_series))
        metrics["eval_pred_lable_timesfm"].extend(user_definable_IOH(raw_forecast))
        
        print(f"Finished batch {i} in epoch {epoch + 1}.")

# 保存微调后的模型
lora_model.save_pretrained("/path_to_save_lora_model")

# 输出最终评估结果
for k, v in metrics.items():
    if k in ["eval_pred_lable_timesfm"]:
        precision, recall, _ = precision_recall_curve(ground_true_labels, v)
        auprc = auc(recall, precision)
        fpr, tpr, _ = roc_curve(ground_true_labels, v)
        auroc = auc(fpr, tpr)
        f1 = f1_score(ground_true_labels, v)
        acc = accuracy_score(ground_true_labels, v)
        print(f"auroc={auroc:.3f}, auprc={auprc:.3f}, acc={acc:.3f}, F1={f1:.3f}")
    else:
        print(f"{k}: {np.mean(v)}")
