# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TF dataloaders for general timeseries datasets.

The expected input format is csv file with a datetime index.
"""

from absl import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from . import time_features
import os
from collections import defaultdict
PATCH_LEN=32

class TimeSeriesdata(object):
  """Data loader class."""

  def __init__(
      self,
      data_path,
      datetime_col,
      num_cov_cols,
      cat_cov_cols,
      ts_cols,
      train_range,
      val_range,
      test_range,
      hist_len,
      pred_len,
      batch_size,
      freq='H',
      normalize=True,
      epoch_len=None,
      holiday=False,
      permute=True,
  ):
    """Initialize objects.

    Args:
      data_path: path to csv file
      datetime_col: column name for datetime col
      num_cov_cols: list of numerical global covariates
      cat_cov_cols: list of categorical global covariates
      ts_cols: columns corresponding to ts
      train_range: tuple of train ranges
      val_range: tuple of validation ranges
      test_range: tuple of test ranges
      hist_len: historical context
      pred_len: prediction length
      batch_size: batch size (number of ts in a batch)
      freq: freq of original data
      normalize: std. normalize data or not
      epoch_len: num iters in an epoch
      holiday: use holiday features or not
      permute: permute ts in train batches or not

    Returns:
      None
    """
    self.data_df = pd.read_csv(open(data_path, 'r'))
    if not num_cov_cols:
      self.data_df['ncol'] = np.zeros(self.data_df.shape[0])
      num_cov_cols = ['ncol']
    if not cat_cov_cols:
      self.data_df['ccol'] = np.zeros(self.data_df.shape[0])
      cat_cov_cols = ['ccol']
    self.data_df.fillna(0, inplace=True)
    self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col]),
                           inplace=True)
    self.num_cov_cols = num_cov_cols
    self.cat_cov_cols = cat_cov_cols
    self.ts_cols = ts_cols
    self.train_range = train_range
    self.val_range = val_range
    self.test_range = test_range
    data_df_idx = self.data_df.index
    date_index = data_df_idx.union(
        pd.date_range(
            data_df_idx[-1] + pd.Timedelta(1, freq=freq),
            periods=pred_len + 1, 
            freq=freq,
        ))
    self.time_df = time_features.TimeCovariates(
        date_index, holiday=holiday).get_covariates()
    self.hist_len = hist_len
    self.pred_len = pred_len
    self.batch_size = batch_size
    self.freq = freq
    self.normalize = normalize
    
    self.data_mat = self.data_df[self.ts_cols].to_numpy().transpose()
    self.data_mat = self.data_mat[:, 0:self.test_range[1]] 
    self.time_mat = self.time_df.to_numpy().transpose()
    self.num_feat_mat = self.data_df[num_cov_cols].to_numpy().transpose()
    self.cat_feat_mat, self.cat_sizes = self._get_cat_cols(cat_cov_cols)
    self.normalize = normalize
    if normalize:
      self._normalize_data()
    logging.info(
        'Data Shapes: %s, %s, %s, %s',
        self.data_mat.shape,
        self.time_mat.shape,
        self.num_feat_mat.shape,
        self.cat_feat_mat.shape,
    )
    self.epoch_len = epoch_len
    self.permute = permute

  def _get_cat_cols(self, cat_cov_cols):
    """Get categorical columns."""
    cat_vars = []
    cat_sizes = []
    for col in cat_cov_cols:
      dct = {x: i for i, x in enumerate(self.data_df[col].unique())}
      cat_sizes.append(len(dct))
      mapped = self.data_df[col].map(lambda x: dct[x]).to_numpy().transpose()  # pylint: disable=cell-var-from-loop
      cat_vars.append(mapped)
    return np.vstack(cat_vars), cat_sizes

  def _normalize_data(self):
    self.scaler = StandardScaler()
    train_mat = self.data_mat[:, 0:self.train_range[1]] # only count data in train set
    self.scaler = self.scaler.fit(train_mat.transpose())
    self.data_mat = self.scaler.transform(self.data_mat.transpose()).transpose()


  def train_gen(self):
    """Generator for training data."""
    num_ts = len(self.ts_cols)
    perm = np.arange( # initialize range of train set
        self.train_range[0] + self.hist_len,
        self.train_range[1] - self.pred_len,
    )
    perm = np.random.permutation(perm)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    if not self.epoch_len:
      epoch_len = len(perm)
    else:
      epoch_len = self.epoch_len
    for idx in perm[0:epoch_len]:
      for _ in range(num_ts // self.batch_size + 1):
        if self.permute:
          tsidx = np.random.choice(num_ts, size=self.batch_size, replace=False)
        else:
          tsidx = np.arange(num_ts)
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train,
            bts_pred,
            bfeats_train,
            bfeats_pred,
            bcf_train,
            bcf_pred,
        ) = self._get_features_and_ts(dtimes, tsidx, hist_len)

        all_data = [
            bts_train,
            bfeats_train,
            bcf_train,
            bts_pred,
            bfeats_pred,
            bcf_pred,
            tsidx,
        ]
        yield tuple(all_data)

  def test_val_gen(self, mode='val', shift=1):
    """Generator for validation/test data."""
    if mode == 'val':
      start = self.val_range[0]
      end = self.val_range[1] - self.pred_len + 1
    elif mode == 'test':
      start = self.test_range[0]
      end = self.test_range[1] - self.pred_len + 1
    else:
      raise NotImplementedError('Eval mode not implemented')
    num_ts = len(self.ts_cols)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    perm = np.arange(start, end)
    if self.epoch_len:
      epoch_len = self.epoch_len
    else:
      epoch_len = len(perm)
    for i in range(0, epoch_len, shift):
      idx = perm[i]
      for batch_idx in range(0, num_ts, self.batch_size):
        tsidx = np.arange(batch_idx, min(batch_idx + self.batch_size, num_ts))
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train,
            bts_pred,
            bfeats_train,
            bfeats_pred,
            bcf_train,
            bcf_pred,
        ) = self._get_features_and_ts(dtimes, tsidx, hist_len)
        all_data = [
            bts_train,
            bfeats_train,
            bcf_train,
            bts_pred,
            bfeats_pred,
            bcf_pred,
            tsidx,
        ]
        yield tuple(all_data)
  # get only one sample
  def _get_features_and_ts(self, dtimes, tsidx, hist_len=None):
    """Get features and ts in specified windows."""
    if hist_len is None:
      hist_len = self.hist_len

    data_times = dtimes[dtimes < self.data_mat.shape[1]]
    bdata = self.data_mat[:, data_times]
    bts = bdata[tsidx, :]
    bnf = self.num_feat_mat[:, data_times]
    bcf = self.cat_feat_mat[:, data_times]
    btf = self.time_mat[:, dtimes]

    if bnf.shape[1] < btf.shape[1]:
      rem_len = btf.shape[1] - bnf.shape[1]
      rem_rep = np.repeat(bnf[:, [-1]], repeats=rem_len)
      rem_rep_cat = np.repeat(bcf[:, [-1]], repeats=rem_len)
      bnf = np.hstack([bnf, rem_rep.reshape(bnf.shape[0], -1)])
      bcf = np.hstack([bcf, rem_rep_cat.reshape(bcf.shape[0], -1)])
    bfeats = np.vstack([btf, bnf])
    bts_train = bts[:, 0:hist_len]
    bts_pred = bts[:, hist_len:]
    bfeats_train = bfeats[:, 0:hist_len]
    bfeats_pred = bfeats[:, hist_len:]
    bcf_train = bcf[:, 0:hist_len]
    bcf_pred = bcf[:, hist_len:]
    return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred

  def tf_dataset(self, mode='train', shift=1):
    """Tensorflow Dataset."""
    if mode == 'train':
      gen_fn = self.train_gen
    else:
      gen_fn = lambda: self.test_val_gen(mode, shift)
    output_types = tuple([tf.float32] * 2 + [tf.int32] + [tf.float32] * 2 +
                         [tf.int32] * 2) 
    dataset = tf.data.Dataset.from_generator(gen_fn, output_types)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class ioh_timeseriesdata(object):
  """Data loader class."""

  def __init__(
      self,
      root_path,
      data_path='ETTh1.csv',
      flag='train',
      size=None,
      num_features=1,
      batch_size=1,
      instanceLevel_flag=False,
      freq='H',
      percent=1.0,
      normalize=True,
      permute=True,
  ):
    # size [seq_len, label_len, pred_len]
    # info
    self.root_path = root_path
    self.data_path = data_path
  
    self.flag = flag
    self.seq_len = size[0]
    self.label_len = size[1]
    self.pred_len = size[2]
    self.features = num_features
    self.normalize = normalize
    self.instanceLevel_flag = instanceLevel_flag


    self.freq = freq
    self.percent = percent
    self.batch_size=batch_size
    self.permute = permute
    
    # Initialize scalers for each feature to be standardized
    # if normalize:
    #   self.scaler_bts = StandardScaler()
    #   self.scaler_hrs = StandardScaler()
    #   self.scaler_dbp = StandardScaler()
    #   self.scaler_mbp = StandardScaler()
    #   self.scaler_prediction_mbp = StandardScaler()

    self.__read_data__()

  def __read_data__(self):
      if self.instanceLevel_flag :
          if self.flag == 'train': # self.data_path是instance(caseid)的序号
              df_raw = pd.read_csv(os.path.join(self.root_path, str(self.data_path) + '_train.csv'))
          elif self.flag == 'val':
              df_raw = pd.read_csv(os.path.join(self.root_path, str(self.data_path) + '_val.csv'))
          elif self.flag == 'test':
              df_raw = pd.read_csv(os.path.join(self.root_path, str(self.data_path) + '_test.csv'))
          # df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
      else:
          if self.flag == 'train':
              df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_train_data.csv'))
          elif self.flag == 'val':
              df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_val_data.csv'))
          elif self.flag == 'test':
              df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_test_data.csv'))

          # random select train/val data
          if self.flag in ["train", "val"] and self.percent < 1.0:
            total_size = len(df_raw)
            shuffled_indices = np.random.permutation(total_size)
            selected_indices = shuffled_indices[:int(total_size * self.percent)]
            df_raw = df_raw.iloc[selected_indices].reset_index(drop=True)

      '''
      df_raw.columns: ['date', ...(other features), target feature]
      '''
      self.__preprocess_csv__(df_raw)

  def __preprocess_csv__(self, data):

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
              
              def sliding_window_average(time_series, slide_len):
                  if slide_len <= 0:
                      raise ValueError("slide_len must be greater than 0")
                  
                  # 存储滑动窗口的平均值
                  window_averages = []
                  
                  # 遍历序列，按滑动窗口大小取值
                  for i in range(0, len(time_series), slide_len):
                      # 获取当前窗口的值
                      window = time_series[i:i + slide_len]
                      # 计算窗口的平均值并存储
                      window_avg = round(np.nanmean(window), 2)
                      window_averages.append(window_avg)
                  
                  return window_averages

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
          # if index > 100:
          #     break
          bts = parse_sequence(row['bts'][1:-1], skip_rate=0, sample_type='avg_sample') #采样周期是：2*skip_rate
          hrs = parse_sequence(row['hrs'][1:-1], skip_rate=0, sample_type='avg_sample')
          dbp = parse_sequence(row['dbp'][1:-1], skip_rate=0, sample_type='avg_sample')
          mbp = parse_sequence(row['mbp'][1:-1], skip_rate=0, sample_type='avg_sample')
          prediction_mbp = parse_sequence(row['prediction_mbp'][1:-1], skip_rate=0, sample_type='avg_sample')
          # print(len(bts), len(hrs), len(dbp), len(mbp), len(prediction_mbp))
          if len(bts) != self.seq_len or len(hrs) != self.seq_len or len(dbp) != self.seq_len or\
              len(mbp) != self.seq_len or len(prediction_mbp) != self.label_len:
              continue
          
          examples['caseid'].append(row['caseid'])
          examples['stime'].append(row['stime'])
          examples['ioh_stime'].append(row['ioh_stime'])
          examples['ioh_dtime'].append(row['ioh_dtime'])
          examples['age'].append(row['age']) # np.full(len(bts), row['age'])
          examples['sex'].append(row['sex'])
          examples['bmi'].append(row['bmi'])
          examples['label'].append(row['label'])
          examples['bts'].append(bts)
          examples['hrs'].append(hrs)
          examples['dbp'].append(dbp)
          examples['mbp'].append(mbp)
          examples['prediction_mbp'].append(prediction_mbp)

      print("处理后的测试样本数量:", len(examples['caseid']))

      # 统计处理后 examples 中 label 列的分布
      label_counts = pd.Series(examples['label']).value_counts(normalize=True) * 100
      print("处理后的Label分布 (%):")
      print(label_counts)

      # # 仅在训练集上进行标准化处理
      # if self.flag == 'train' and self.scale:
      #     print("Fitting scalers on training data...")
      #     self.scaler_bts.fit(examples['bts'])
      #     self.scaler_hrs.fit(examples['hrs'])
      #     self.scaler_dbp.fit(examples['dbp'])
      #     self.scaler_mbp.fit(examples['mbp'])
      #     self.scaler_prediction_mbp.fit(examples['prediction_mbp'])

      # # 对验证集和测试集使用训练集拟合好的scaler进行标准化
      # if self.scale:
      #     print("Transforming data with fitted scalers...")
      #     examples['bts'] = self.scaler_bts.transform(examples['bts'])
      #     examples['hrs'] = self.scaler_hrs.transform(examples['hrs'])
      #     examples['dbp'] = self.scaler_dbp.transform(examples['dbp'])
      #     examples['mbp'] = self.scaler_mbp.transform(examples['mbp'])
      #     examples['prediction_mbp'] = self.scaler_prediction_mbp.transform(examples['prediction_mbp'])

      self.data = examples
      if self.normalize and self.flag == "train":
        self._normalize_data()
  
  def _normalize_data(self): # This version only considers univariates
    self.scaler = StandardScaler()
    self.scaler = self.scaler.fit(self.data['mbp']) # only count data in train set
    self.data['mbp'] = self.scaler.transform(self.data['mbp'])
    self.scaler = self.scaler.fit(self.data['prediction_mbp']) # only count data in train set
    self.data['prediction_mbp'] = self.scaler.transform(self.data['prediction_mbp'])

  def _get_features_and_ts(self, bs_data):
    bts_train=[] 
    bts_pred=[] 
    bfeats_train=[] 
    bfeats_pred=[]
    bcf_train=[]
    bcf_pred=[]
    
    for k, v in bs_data.items():
      if k == "mbp":
        for i in range(len(v)):
          bts_train.append(v[i][(self.seq_len%PATCH_LEN - self.seq_len):])
        # bfeats_train = bts_train
        # bcf_train = bts_train
        
      elif k == "prediction_mbp":
        for i in range(len(v)):
          bts_pred.append(v[i][:self.label_len - self.label_len%PATCH_LEN])
        # bfeats_pred = bts_pred
        # bcf_pred = bts_pred
      
    return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred
     
  def dataset_gen(self):
    for i in range(1 + (len(self.data['caseid']) - 1) // self.batch_size):
        bs_data = {k: v[(i * self.batch_size) : ((i + 1) * self.batch_size)] for k, v in self.data.items()}
        (
        bts_train, 
        bts_pred, 
        bfeats_train,
        bfeats_pred, 
        bcf_train, 
        bcf_pred,
        ) = self._get_features_and_ts(bs_data)

        all_data = [
          bts_train,
          bfeats_train,
          bcf_train,
          bts_pred,
          bfeats_pred,
          bcf_pred,
          ]

        yield tuple(all_data)

  # 生成一个 TensorFlow 数据集。
  def tf_dataset(self):
    """Tensorflow Dataset."""
    gen_fn = self.dataset_gen
    output_types = tuple([tf.float32] * 2 + [tf.int32] * 2 + [tf.float32] * 2) 
    # 这意味着生成器返回的数据包括多个不同类型的数据（浮点数和整数），以供 TensorFlow 训练时使用。
    
    dataset = tf.data.Dataset.from_generator(gen_fn, output_types)
    # 使用 dataset.prefetch 方法为数据集启用预取，以提高性能。AUTOTUNE 使 TensorFlow 自动选择预取的数量，以优化数据加载速度和计算效率。
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
