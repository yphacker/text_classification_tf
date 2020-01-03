# coding=utf-8
# author=yphacker


import os
from conf import config

save_path = os.path.join(config.model_path, 'transformer')
if not os.path.isdir(save_path):
    os.makedirs(save_path)
model_save_path = os.path.join(save_path, 'best')

model_submission_path = os.path.join(config.data_path, 'transformer_submission.csv')

filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
numHeads = 8  # Attention 的头数
numBlocks = 1  # 设置transformer block的数量
epsilon = 1e-8  # LayerNorm 层中的最小除数
keepProp = 0.9  # multi head attention 中的dropout

keep_prob = 0.5  # 全连接层的dropout
l2RegLambda = 0.0
learning_rate = 0.001
