# coding=utf-8
# author=yphacker

import json
import numpy as np
from conf import config


def load_vocab():
    # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
    with open(config.word2id_json_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def get_dataset(data):
    word2id, label2id = load_vocab()

    def solve(df):
        x_data = df['review']
        x_tensor = np.array([encode_data(text, word2id) for text in x_data])
        y_tensor = None
        if 'sentiment' in df.columns.tolist():
            y_data = df['sentiment']
            y_tensor = np.array([label for label in y_data])
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


def get_data_iter(x, y=None, batch_size=config.batch_size, shuffle=True):
    data_len = len(x)
    if y is None:
        y = np.zeros(shape=(data_len, 1))
    if shuffle:
        indices = np.random.permutation(data_len)
    else:
        indices = np.arange(data_len)
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def build_embedding_pretrained():
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    word2idx, idx2word = load_vocab()
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(config.word_embedding_path))

    embedding_matrix = np.zeros((config.num_vocab, config.embed_dim))
    for word, i in word2idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.savez_compressed(config.pretrain_embedding_path, embeddings=embedding_matrix)


def get_pretrain_embedding():
    return np.load(config.pretrain_embedding_path)["embeddings"].astype('float32')


if __name__ == '__main__':
    word2id, id2word = load_vocab()
    print(len(word2id))
    # build_embedding_pretrained()
