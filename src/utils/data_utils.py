# coding=utf-8
# author=yphacker

import os
import re
import numpy as np
import pandas as pd
import pickle as pkl
from conf import config


def clean_text(text):
    text = text.replace('\n', ' ').lower()
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = text.strip()
    return text


def build_vocab():
    tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    vocab_dic = {}
    vocab_max_size = 100000
    vocab_min_freq = 5

    train_df = pd.read_csv(config.train_path, sep='\t')
    texts = train_df['review'].values.tolist()
    for text in texts:
        if not text:
            continue
        text = clean_text(text)
        for word in tokenizer(text):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= vocab_min_freq],
                        key=lambda x: x[1], reverse=True)[:vocab_max_size]
    vocab_dic = {word_count[0]: idx + 2 for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({'_PAD_': 0, '_UNK_': 1})
    print(len(vocab_dic))
    pkl.dump(vocab_dic, open(config.vocab_path, 'wb'))
    return vocab_dic


def load_vocab():
    if os.path.exists(config.vocab_path):
        word2idx = pkl.load(open(config.vocab_path, 'rb'))
    else:
        word2idx = build_vocab()
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def get_dataset(data):
    word2id, label2id = load_vocab()

    def solve(df):
        x_data = df['review']
        x_tensor = np.array([encode_data(clean_text(text), word2id) for text in x_data])
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
            text_ids.append(word_dict.get(item, word_dict['_UNK_']))

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
    # word2id, id2word = load_vocab()
    # print(len(word2id))
    build_embedding_pretrained()
