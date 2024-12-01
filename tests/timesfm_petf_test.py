import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, accuracy_score, roc_auc_score, precision_recall_curve, roc_curve
from collections import defaultdict
import time
import torch
import timesfm
from tqdm import tqdm

# 假设已经有微调后的LoRA模型，加载这个模型
from peft import get_peft_model

# MSE 计算
def mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

# 计算MAE
def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

# 数据预处理与加载函数（假设这部分是你提供的代码）
def get_batched_data_fn(batch_size=128):
    # 数据加载、预处理、采样的函数，假设已经提供了
    # ...
    return data_fn  # 返回一个批次生成器

# 加载经过LoRA微调后的TimesFM模型
def load_finetuned_model(model_path: str):
    # 加载 TimesFM 模型与微调后的权重
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu", 
            per_core_batch_size=32,  
            horizon_len=150,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            version="torch",
            path=model_path  # 微调后的模型路径
        )
    )
    return tfm

# 测试模型
def evaluate_model(tfm, input_data, batch_size=128, horizon_len=150):
    metrics = defaultdict(list)
    ground_true_labels = []

    for i, example in enumerate(input_data()):
        if np.array(example["inputs"]).shape != (batch_size, 450):
            continue

        start_time = time.time()
        
        # 进行推理
        raw_forecast, _ = tfm.forecast(
            inputs=example["inputs"], freq=[0] * len(example["inputs"])
        )
        
        print(f"\rFinished batch {i} in {time.time() - start_time:.2f} seconds", end="")

        raw_forecast = raw_forecast[:, :horizon_len]  # 截取前horizon_len的预测值
        true_series = np.array(example["outputs"])[:, :horizon_len]
        ground_true_labels.extend(example["label"])

        # 计算MAE与MSE
        metrics["eval_mae_timesfm"].extend(mae(raw_forecast, true_series))
        metrics["eval_mse_timesfm"].extend(mse(raw_forecast, true_series))

        # 使用自定义的IOH检测
        metrics["eval_pred_lable_timesfm"].extend(user_definable_IOH(raw_forecast))
    
    print("\nEvaluating done!")
    return metrics, ground_true_labels

# 计算最终的评估指标
def calculate_metrics(metrics, ground_true_labels):
    # 计算AUC, F1-score等
    for k, v in metrics.items():
        if k in ["eval_pred_lable_timesfm"]:
            print(k, "--Prediction Results:")
            precision, recall, _ = precision_recall_curve(ground_true_labels, v)
            auprc = auc(recall, precision)

            fpr, tpr, _ = roc_curve(ground_true_labels, v)
            auroc = auc(fpr, tpr)

            f1 = f1_score(ground_true_labels, v)
            acc = accuracy_score(ground_true_labels, v)
            tn, fp, fn, tp = confusion_matrix(ground_true_labels, v).ravel()

            testres = 'auroc={:.3f}, auprc={:.3f} acc={:.3f}, F1={:.3f}, PPV={:.1f}, NPV={:.1f}, TN={}, fp={}, fn={}, TP={}'.format(
                auroc, auprc, acc, f1, tp/(tp+fp)*100, tn/(tn+fn)*100, tn, fp, fn, tp)
            print(testres)
        else:   
            print(f"{k}: {np.mean(v)}")

# 例：加载微调后的模型并测试
if __name__ == "__main__":
    # 加载微调后的模型路径
    finetuned_model_path = "/path/to/your/finetuned_model.ckpt"

    # 加载TimesFM模型
    tfm = load_finetuned_model(finetuned_model_path)

    # 加载测试数据
    input_data = get_batched_data_fn(batch_size=128)

    # 评估模型
    metrics, ground_true_labels = evaluate_model(tfm, input_data, batch_size=128, horizon_len=150)

    # 计算并打印指标
    calculate_metrics(metrics, ground_true_labels)
