# coding=utf-8
# author=yphacker

import os
from conf import config

#
# filterSizes = [2, 3, 4, 5]
# dropoutKeepProb = 0.5
# l2RegLambda = 0.0
# checkpointEvery = 100
# learningRate = 0.001

save_path = os.path.join(config.model_path, 'cnn')
if not os.path.isdir(save_path):
    os.makedirs(save_path)
model_save_path = os.path.join(save_path, 'best')
model_submission_path = os.path.join(config.data_path, 'cnn_submission.csv')

num_filters = 256  # 卷积核数目
kernel_size = 5  # 卷积核尺寸

hidden_dim = 128  # 全连接层神经元

keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
