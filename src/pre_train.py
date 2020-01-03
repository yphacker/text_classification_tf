# coding=utf-8
# author=yphacker

import json
import gensim
import numpy as np
import pandas as pd
from gensim.models import word2vec
from collections import Counter
from conf import config


def gen_word2vec():
    sentences = word2vec.LineSentence(config.text_path)
    model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)
    model.wv.save_word2vec_format(config.word2vec_path, binary=True)


def get_word_embedding(words):
    """
    按照我们的数据集中的单词取出预训练好的word2vec中的词向量
    """

    word_vec = gensim.models.KeyedVectors.load_word2vec_format(config.word2vec_path, binary=True)
    vocab = []
    word_embedding = []

    # 添加 "pad" 和 "unk",
    vocab.append("_PAD_")
    vocab.append("_UNK_")
    word_embedding.append(np.zeros(config.embedding_size))
    word_embedding.append(np.random.randn(config.embedding_size))

    for word in words:
        try:
            vector = word_vec.wv[word]
            vocab.append(word)
            word_embedding.append(vector)
        except:
            print(word + "不存在于词向量中")
    return vocab, np.array(word_embedding)


# 生成词向量的词汇表
def gen_vocabulary():
    train = pd.read_csv(config.train_path)
    texts = train.review.tolist()
    labels = train.sentiment.tolist()

    texts = [line.strip().split() for line in texts]
    all_words = [word for text in texts for word in text]

    with open(config.stop_words_path, "r") as f:
        stop_words = f.read()
        stop_word_list = stop_words.splitlines()
        stop_word_dict = dict(zip(stop_word_list, list(range(len(stop_word_list)))))

    # 去掉停用词
    sub_words = [word for word in all_words if word not in stop_word_dict]
    count = Counter(sub_words)  # 统计词频
    sort_word_count = sorted(count.items(), key=lambda x: x[1], reverse=True)

    # 去除低频词
    words = [item[0] for item in sort_word_count if item[1] >= 5]
    vocab, word_embedding = get_word_embedding(words)
    # count_pairs = counter.most_common(config.vocab_size - 1)
    # words, _ = list(zip(*count_pairs))
    # vocab = ['_PAD_', '_UNK_'] + list(words)

    print(len(vocab))
    word2idx = dict(zip(vocab, list(range(len(vocab)))))
    uniqueLabel = list(set(labels))
    label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))

    # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
    with open(config.word2id_json_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f)

    with open(config.label2id_json_path, "w", encoding="utf-8") as f:
        json.dump(label2idx, f)

    np.save(config.word_vec_path, word_embedding)


if __name__ == '__main__':
    # gen_word2vec()
    gen_vocabulary()
