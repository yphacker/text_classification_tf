# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")

train_path = os.path.join(data_path, 'labeledTrainData.tsv')
test_path = os.path.join(data_path, 'testData.tsv')
sample_submission_path = os.path.join(data_path, 'sampleSubmission.csv')

# process_data_path = os.path.join(data_path, "process_data")
# text_path = os.path.join(process_data_path, 'text.txt')
# train_path = os.path.join(process_data_path, 'train.csv')
# test_path = os.path.join(process_data_path, 'test.csv')

stop_words_path = os.path.join(data_path, 'english')
word2vec_path = os.path.join(data_path, 'word2vec.bin')
word_vec_path = os.path.join(data_path, 'word_vec.npy')
word2id_json_path = os.path.join(data_path, 'word2id.json')
label2id_json_path = os.path.join(data_path, 'label2id.json')

train_check_path = os.path.join(data_path, 'train_check.txt')

model_path = os.path.join(data_path, 'model')

vocab_size = 28607  # 词汇表达大小
embedding_size = 200
max_seq_len = 200  # 序列长度
num_labels = 2  # 类别数量
epochs_num = 8
batch_size = 32
print_per_batch = 10
improvement_step = print_per_batch * 10
