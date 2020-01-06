# coding=utf-8
# author=yphacker


import os
from conf import config

hidden_size = [128]  # LSTM结构的神经元个数
output_size = 128  # 从高维映射到低维的神经元个数


keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率

l2RegLambda = 0.0
