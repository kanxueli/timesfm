from utils.metrics import metric
import numpy as np
import json

def load_predictions(file_path):
    """
    Load the predictions and true values from a file.

    Parameters:
    - file_path (str): The file path from which to load the data.

    Returns:
    - preds (ndarray): The loaded predicted values.
    - trues (ndarray): The loaded ground truth values.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    preds = np.array(data['predictions'])
    trues = np.array(data['ground_truths'])

    return preds, trues



# 读取保存的预测和真实值
preds, trues = load_predictions('predictions_and_trues.json')

# 调用 metric 函数进行评测
mae, mse, rmse, mape, mspe = metric(preds, trues)
print(f'Metrics: MAE={mae}, MSE={mse}, RMSE={rmse}, MAPE={mape}, MSPE={mspe}')
