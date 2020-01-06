# coding=utf-8
# author=yphacker


import os
from conf import config

hidden_size = [256, 256]  # 单层LSTM结构的神经元个数

# hidden_dim = 128  # 全连接层神经元

keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
l2RegLambda = 0.0
