# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
word_embedding_path = os.path.join(data_path, "glove.840B.300d.txt")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
pretrain_embedding_path = os.path.join(data_path, "pretrain_embedding.npz")

train_path = os.path.join(data_path, 'labeledTrainData.tsv')
test_path = os.path.join(data_path, 'testData.tsv')
sample_submission_path = os.path.join(data_path, 'sampleSubmission.csv')


stop_words_path = os.path.join(data_path, 'english')
word2id_json_path = os.path.join(data_path, 'word2id.json')
label2id_json_path = os.path.join(data_path, 'label2id.json')

train_check_path = os.path.join(data_path, 'train_check.txt')

model_path = os.path.join(data_path, 'model')

# pretrain_embedding = False
pretrain_embedding = True
embed_dim = 300

num_vocab = 28607  # 词汇表达大小
max_seq_len = 200
num_labels = 2  # 类别数量
epochs_num = 8
batch_size = 32
print_per_batch = 10
improvement_step = print_per_batch * 10
