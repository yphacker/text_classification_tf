# coding=utf-8
# author=yphacker

import json
import numpy as np
from conf import config


def get_vocabulary():
    # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
    with open(config.word2id_json_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open(config.label2id_json_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    return word2idx, label2idx


def get_word_embedding():
    return np.load(config.word_vec_path)


def word2index(text_str, word_dict, max_seq_len=config.max_seq_len):
    if len(text_str) == 0 or len(word_dict) == 0:
        print('[ERROR] word2id failed! | The params: {} and {}'.format(text_str, word_dict))
        return None

    text_list = text_str.strip().split(' ')
    text_ids = list()
    for item in text_list:
        if item in word_dict:
            text_ids.append(word_dict[item])
        else:
            text_ids.append(word_dict['_UNK_'])

    if len(text_ids) < max_seq_len:
        text_ids = text_ids + [word_dict['_PAD_'] for _ in range(max_seq_len - len(text_ids))]
    else:
        text_ids = text_ids[:max_seq_len]
    return text_ids


def label2index(label, label2id):
    """
    将标签转换成索引表示
    """
    return label2id[str(label)]
