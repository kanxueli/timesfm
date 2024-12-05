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
      datetime_col: column name for datetime col 时间列的名称
      num_cov_cols: list of numerical global covariates
      cat_cov_cols: list of categorical global covariates
      ts_cols: columns corresponding to ts 时间序列数据列的名称列表
      train_range: tuple of train ranges 训练、验证和测试数据的时间范围
      val_range: tuple of validation ranges
      test_range: tuple of test ranges
      hist_len: historical context
      pred_len: prediction length
      batch_size: batch size (number of ts in a batch)
      freq: freq of original data
      normalize: std. normalize data or not
      epoch_len: num iters in an epoch 每个周期的迭代次数
      holiday: use holiday features or not 是否使用假期特征
      permute: permute ts in train batches or not 是否在训练批次中打乱时间序列

    Returns:
      None
    """
    # 如果没有指定数值型协变量或类别型协变量，默认会添加一列全零的列。
    self.data_df = pd.read_csv(open(data_path, 'r'))
    if not num_cov_cols:
      self.data_df['ncol'] = np.zeros(self.data_df.shape[0])
      num_cov_cols = ['ncol']
    if not cat_cov_cols:
      self.data_df['ccol'] = np.zeros(self.data_df.shape[0])
      cat_cov_cols = ['ccol']
    self.data_df.fillna(0, inplace=True) # 将所有缺失值填充为 0
    self.data_df.set_index(pd.DatetimeIndex(self.data_df[datetime_col]),
                           inplace=True) # 设置数据的索引为日期时间列，以便后续按时间顺序处理数据。
    self.num_cov_cols = num_cov_cols
    self.cat_cov_cols = cat_cov_cols
    self.ts_cols = ts_cols
    self.train_range = train_range
    self.val_range = val_range
    self.test_range = test_range
    data_df_idx = self.data_df.index
    date_index = data_df_idx.union( # union 方法用于将两个 DatetimeIndex 对象合并为一个，并去除重复的时间点。
        pd.date_range( #  用于生成一个包含指定时间范围的时间序列
            data_df_idx[-1] + pd.Timedelta(1, freq=freq),
            periods=pred_len + 1, # 指定生成的时间点数量
            freq=freq,
        ))
    self.time_df = time_features.TimeCovariates(
        date_index, holiday=holiday).get_covariates()
    self.hist_len = hist_len
    self.pred_len = pred_len
    self.batch_size = batch_size
    self.freq = freq
    self.normalize = normalize
    # 从 self.data_df 中提取出来的时间序列数据，转置后，每一行是一个时间序列，每一列是一个时间步的数据。
    self.data_mat = self.data_df[self.ts_cols].to_numpy().transpose()
    # 第二行通过 test_range 限制了时间序列数据的范围
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

  # 将原本的类别特征转换为数值形式，并记录每个分类变量的类别数量。
  def _get_cat_cols(self, cat_cov_cols): # cat_cov_cols 是一个列表，包含了数据集中的分类特征列名
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
    train_mat = self.data_mat[:, 0:self.train_range[1]]
    self.scaler = self.scaler.fit(train_mat.transpose())
    self.data_mat = self.scaler.transform(self.data_mat.transpose()).transpose()

  # 生成器函数会逐步返回训练数据，而不是一次性加载所有数据，这在处理大规模数据时非常有效。
  def train_gen(self):
    """Generator for training data."""
    num_ts = len(self.ts_cols)
    perm = np.arange( # self.train_range[0]是训练集的起始位置，self.train_range[1]是训练集的终点位置。
        self.train_range[0] + self.hist_len,
        self.train_range[1] - self.pred_len,
    )
    # 对 perm 中的索引进行随机打乱，生成一个随机顺序。这是为了增加训练过程中的随机性，避免模型对数据的顺序产生偏见。
    perm = np.random.permutation(perm)
    hist_len = self.hist_len
    logging.info('Hist len: %s', hist_len)
    if not self.epoch_len:
      epoch_len = len(perm)
    else:
      epoch_len = self.epoch_len
    for idx in perm[0:epoch_len]:
      # 遍历每个时间序列的批次。num_ts // self.batch_size 计算每个训练周期中批次数量。
      # +1 是为了确保在有余数的情况下仍然会生成额外的一个批次。即使 num_ts 不是 batch_size 的整数倍，仍然会生成额外的批次来处理剩余的数据。
      for _ in range(num_ts // self.batch_size + 1):
        if self.permute:
          # 随机选择 batch_size 个时间序列的索引（tsidx）。np.random.choice 用于从 num_ts 中随机选择指定数量的索引，
          # replace=False 表示不允许重复选择同一个时间序列。
          tsidx = np.random.choice(num_ts, size=self.batch_size, replace=False)
        else:
          tsidx = np.arange(num_ts)
        dtimes = np.arange(idx - hist_len, idx + self.pred_len)
        (
            bts_train, # 训练时的时间序列数据
            bts_pred, # 预测时的时间序列数据
            bfeats_train, # 训练时的特征数据
            bfeats_pred, # 预测时的特征数据
            bcf_train, # 训练时的类别特征数据
            bcf_pred, # 预测时的类别特征数据
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
        # yield 语句将 all_data 作为一个元组返回，这意味着这个函数是一个生成器，
        # 它会逐步返回数据，而不是一次性返回所有数据。每次调用 train_gen 时，
        # 会生成一个新的训练数据批次，直到遍历完所有数据。
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

  # 它的目的是从指定的时间窗口中提取特征和时间序列数据。
  # dtimes：时间索引，表示当前时间窗口的时间步； tsidx: 选择的时间序列的索引（通常是时间序列的行）
  def _get_features_and_ts(self, dtimes, tsidx, hist_len=None):
    """Get features and ts in specified windows."""
    if hist_len is None:
      hist_len = self.hist_len
    # 将 dtimes 中的时间步限制在数据矩阵的列范围内。self.data_mat.shape[1] 获取数据矩阵的列数（即时间步的数量）
    data_times = dtimes[dtimes < self.data_mat.shape[1]]
    # 从 self.data_mat 中提取出对应于 data_times 的时间步的数据。
    # bdata 是一个包含时间序列数据的矩阵，其中每一列是对应时间步的数据。
    bdata = self.data_mat[:, data_times]
    # 根据 tsidx（选择的时间序列的索引）从 bdata 中提取出相应的时间序列数据。
    # bts 现在是一个二维数组，包含了指定时间序列（tsidx）的所有时间步的数据。
    bts = bdata[tsidx, :]
    # 从 self.num_feat_mat和cat_feat_mat中提取出对应于 data_times 的数值/的类别型特征数据。
    bnf = self.num_feat_mat[:, data_times]
    bcf = self.cat_feat_mat[:, data_times]
    # 从 self.time_mat 中提取出与 dtimes 对应的时间特征数据。btf 是一个包含时间相关特征（如周期性等）的矩阵。
    btf = self.time_mat[:, dtimes]

    # 检查数值型特征的时间步数（bnf.shape[1]）是否少于时间特征的时间步数（btf.shape[1]）。
    # 如果是，则需要对数值型特征进行扩展，以便与时间特征矩阵的维度匹配。
    if bnf.shape[1] < btf.shape[1]:
      rem_len = btf.shape[1] - bnf.shape[1] # 计算需要添加的额外时间步数
      rem_rep = np.repeat(bnf[:, [-1]], repeats=rem_len)
      rem_rep_cat = np.repeat(bcf[:, [-1]], repeats=rem_len)
      bnf = np.hstack([bnf, rem_rep.reshape(bnf.shape[0], -1)]) # 重复的数据（rem_rep）水平拼接到 bnf 矩阵的右侧。
      bcf = np.hstack([bcf, rem_rep_cat.reshape(bcf.shape[0], -1)])
    bfeats = np.vstack([btf, bnf]) # 将时间特征（btf）和数值型特征（bnf）垂直拼接在一起，形成一个新的特征矩阵 bfeats
    bts_train = bts[:, 0:hist_len] # 历史(观察窗口)数据，用于模型的训练
    bts_pred = bts[:, hist_len:] # 预测数据，用于模型的预测
    bfeats_train = bfeats[:, 0:hist_len] # 用于训练的特征数据
    bfeats_pred = bfeats[:, hist_len:] # 用于预测的特征数据
    bcf_train = bcf[:, 0:hist_len] # 用于训练的类别特征数据
    bcf_pred = bcf[:, hist_len:] # 用于预测的类别特征数据
    return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred

  # 生成一个 TensorFlow 数据集。
  def tf_dataset(self, mode='train', shift=1):
    """Tensorflow Dataset."""
    if mode == 'train':
      gen_fn = self.train_gen
    else:
      # 在 'test' 或 'val' 模式下，使用一个匿名函数（lambda）来调用 self.test_val_gen 生成测试或验证数据。
      # test_val_gen 生成器会根据 mode 和 shift 参数生成相应的数据。
      gen_fn = lambda: self.test_val_gen(mode, shift)
    output_types = tuple([tf.float32] * 2 + [tf.int32] + [tf.float32] * 2 +
                         [tf.int32] * 2) # 这意味着生成器返回的数据包括多个不同类型的数据（浮点数和整数），以供 TensorFlow 训练时使用。
    # 使用 tf.data.Dataset.from_generator 方法从 gen_fn 生成器创建 TensorFlow 数据集。
    dataset = tf.data.Dataset.from_generator(gen_fn, output_types)
    # 使用 dataset.prefetch 方法为数据集启用预取，以提高性能。AUTOTUNE 使 TensorFlow 自动选择预取的数量，以优化数据加载速度和计算效率。
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
      normalize=False,
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
    self.instanceLevel_flag = instanceLevel_flag


    self.freq = freq
    self.batch_size=batch_size
    self.permute = permute
    
    # Initialize scalers for each feature to be standardized
    if normalize:
      self.scaler_bts = StandardScaler()
      self.scaler_hrs = StandardScaler()
      self.scaler_dbp = StandardScaler()
      self.scaler_mbp = StandardScaler()
      self.scaler_prediction_mbp = StandardScaler()

    self.__read_data__()

  def __read_data__(self):
      if self.instanceLevel_flag :
          if self.flag == 'train': # self.data_path是instance(caseid)的序号
              df_raw = pd.read_csv(os.path.join(self.root_path, str(self.data_path) + '_train.csv'))
          elif self.flag == 'val':
              df_raw = pd.read_csv(os.path.join(self.root_path, str(self.data_path) + '_val.csv'))
          elif self.flag == 'test':
              df_raw = pd.read_csv(os.path.join(self.root_path, str(self.data_path) + '_test.csv'))
      else:
          if self.flag == 'train':
              df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_train_data.csv'))
          elif self.flag == 'val':
              df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_val_data.csv'))
          elif self.flag == 'test':
              df_raw = pd.read_csv(os.path.join(self.root_path, 'vitaldb_test_data.csv'))

      '''
      df_raw.columns: ['date', ...(other features), target feature]
      '''
      # 对csv数据进行预处理
      self.__preprocess_csv__(df_raw)

  def __preprocess_csv__(self, data):

      # 数据预处理前总数据
      print("源数据长度：", len(data))
      label_counts = data['label'].value_counts(normalize=True) * 100
      print("处理前的Label分布 (%):")
      print(label_counts)

      # 定义处理序列数据的函数，直接通过空格拆分并转换为浮点数列表，且完成重采样
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
      self.scaler = StandardScaler()
      examples = defaultdict(list)

      for index, row in data.iterrows():
          # if index > 100:
          #     break
          bts = parse_sequence(row['bts'][1:-1], skip_rate=0, sample_type='skip_sample') #采样周期是：2*skip_rate
          hrs = parse_sequence(row['hrs'][1:-1], skip_rate=0, sample_type='skip_sample')
          dbp = parse_sequence(row['dbp'][1:-1], skip_rate=0, sample_type='skip_sample')
          mbp = parse_sequence(row['mbp'][1:-1], skip_rate=0, sample_type='skip_sample')
          prediction_mbp = parse_sequence(row['prediction_mbp'][1:-1], skip_rate=0, sample_type='skip_sample')
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

      # 修正统计处理后的样本数量
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

  def _get_features_and_ts(self, bs_data):
    bts_train=[] # 训练时的时间序列数据
    bts_pred=[] # 预测时的时间序列数据
    bfeats_train=[] # 训练时的特征数据
    bfeats_pred=[] # 预测时的特征数据
    bcf_train=[] # 训练时的类别特征数据
    bcf_pred=[] # 预测时的类别特征数据
    
    # 由于数据集处理方式问题，目前没办法支持多变量时序预测,为了简单，其他变量和单变量的一样。
    for k, v in bs_data.items():
      if k == "mbp":
        for i in range(len(v)):
          bts_train.append(v[i][-448:]) # 448是为了能够被Patching的长度整除，下面同理
        bfeats_train = bts_train
        bcf_train = bts_train
        
      elif k == "prediction_mbp":
        for i in range(len(v)):
          bts_pred.append(v[i][-128:])
        bfeats_pred = bts_pred
        bcf_pred = bts_pred
      
    return bts_train, bts_pred, bfeats_train, bfeats_pred, bcf_train, bcf_pred
     
  # 生成器函数会逐步返回训练数据，而不是一次性加载所有数据，这在处理大规模数据时非常有效。
  def dataset_gen(self):
    for i in range(1 + (len(self.data['caseid']) - 1) // self.batch_size):
        # 获取一个batch_size的数据[num_features, batch_size, sequence_len]
        bs_data = {k: v[(i * self.batch_size) : ((i + 1) * self.batch_size)] for k, v in self.data.items()}
        (
        bts_train, # 训练时的时间序列数据
        bts_pred, # 预测时的时间序列数据
        bfeats_train, # 训练时的特征数据
        bfeats_pred, # 预测时的特征数据
        bcf_train, # 训练时的类别特征数据
        bcf_pred, # 预测时的类别特征数据
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
