# coding=utf-8
# author=yphacker

import json
import numpy as np
from conf import config


def get_vocab():
    # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
    with open(config.word2id_json_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open(config.label2id_json_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    return word2idx, label2idx


def get_word_embedding():
    return np.load(config.word_vec_path)


def get_dataset(data):
    word2id, label2id = get_vocab()

    def solve(df):
        x_data = df['review']
        x_tensor = np.array([encode_data(text, word2id) for text in x_data])
        y_tensor = None
        if 'sentiment' in df.columns.tolist():
            y_data = df['sentiment']
            y_tensor = np.array([label2index(label, label2id) for label in y_data])
        return x_tensor, y_tensor

    def encode_data(text_str, word_dict, max_seq_len=config.max_seq_len):
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

    return solve(data)


# # def get_data_iter(x, y, batch_size):
# def bert_bacth_iter(x, y=[], batch_size=config.batch_size, shuffle=True):
#     input_ids, input_masks, segment_ids = x
#     index = np.random.permutation(len(y))
#     n_batches = len(y) // batch_size + 1
#     for batch_index in np.array_split(index, n_batches):
#         batch_input_ids, batch_input_masks, batch_segment_ids, batch_y = \
#             input_ids[batch_index], input_masks[batch_index], segment_ids[batch_index], y[batch_index]
#         yield (batch_input_ids, batch_input_masks, batch_segment_ids), batch_y


def get_data_iter(x, y=[], batch_size=config.batch_size, shuffle=True):
    data_len = len(x)

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
    else:
        indices = np.arange(len(input_ids))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def label2index(label, label2id):
    """
    将标签转换成索引表示
    """
    return label2id[str(label)]


if __name__ == '__main__':
    word2id, label2id = get_vocab()
