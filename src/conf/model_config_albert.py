# coding=utf-8
# author=yphacker

import os
from conf import config

bert_data_path = os.path.join(config.data_path, 'albert_base')
bert_config_path = os.path.join(bert_data_path, 'albert_config.json')
bert_checkpoint_path = os.path.join(bert_data_path, 'model.ckpt-best')
bert_vocab_path = os.path.join(bert_data_path, '30k-clean.vocab')

learning_rate = 1e-5
grad_clip = 5.0
# bert max_seq_length 最大为512
