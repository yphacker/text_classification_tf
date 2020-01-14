# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
model_path = os.path.join(data_path, 'model')

train_path = os.path.join(data_path, 'labeledTrainData.tsv')
test_path = os.path.join(data_path, 'testData.tsv')
sample_submission_path = os.path.join(data_path, 'sampleSubmission.csv')

word_embedding_path = os.path.join(data_path, "glove.840B.300d.txt")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
pretrain_embedding_path = os.path.join(data_path, "pretrain_embedding.npz")
vocab_path = os.path.join(data_path, "vocab.pkl")

train_check_path = os.path.join(data_path, 'train_check.txt')

pretrain_embedding = False
# pretrain_embedding = True
embed_dim = 300

num_vocab = 28772  # 词汇表达大小
max_seq_len = 200
num_labels = 2  # 类别数量
epochs_num = 8
batch_size = 32
print_per_batch = 10
improvement_step = print_per_batch * 10
