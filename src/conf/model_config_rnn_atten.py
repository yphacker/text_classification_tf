# coding=utf-8
# author=yphacker


import os
from conf import config

hidden_size = [256, 256]  # 单层LSTM结构的神经元个数

save_path = os.path.join(config.model_path, 'bilstm_attention')
model_save_path = os.path.join(save_path, 'best')
model_submission_path = os.path.join(config.data_path, 'bilstm_attention_submission.csv')

keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率

l2RegLambda = 0.0
