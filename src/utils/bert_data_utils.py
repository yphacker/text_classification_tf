# coding=utf-8
# author=yphacker


import numpy as np
from conf import config
from conf import model_config_bert
from utils.bert import tokenization
from utils.data_utils import get_vocab


def get_dataset(data):
    _, label2id = get_vocab()
    tokenizer = tokenization.FullTokenizer(vocab_file=model_config_bert.bert_vocab_path)

    def label2index(label, label2id):
        return label2id[str(label)]

    def solve(df):
        x_data = df['review']
        input_ids_list = []
        input_masks_list = []
        segment_ids_list = []
        for text in x_data:
            single_input_id, single_input_mask, single_segment_id = encode_data(text)
            input_ids_list.append(single_input_id)
            input_masks_list.append(single_input_mask)
            segment_ids_list.append(single_segment_id)
        input_ids = np.asarray(input_ids_list, dtype=np.int32)
        input_masks = np.asarray(input_masks_list, dtype=np.int32)
        segment_ids = np.asarray(segment_ids_list, dtype=np.int32)

        y_tensor = None
        if 'sentiment' in df.columns.tolist():
            y_data = df['sentiment']
            y_tensor = np.array([label2index(label, label2id) for label in y_data])
        return (input_ids, input_masks, segment_ids), y_tensor

    def encode_data(text, max_seq_len=config.max_seq_len):
        input_ids, input_mask, segment_ids = convert_single_example(max_seq_len, tokenizer, text)
        return input_ids, input_mask, segment_ids

    return solve(data)


def get_data_iter(x, y, batch_size=config.batch_size):
    input_ids, input_masks, segment_ids = x
    index = np.random.permutation(len(y))
    n_batches = len(y) // batch_size
    for batch_index in np.array_split(index, n_batches):
        batch_input_ids, batch_input_masks, batch_segment_ids, batch_y = \
            input_ids[batch_index], input_masks[batch_index], segment_ids[batch_index], y[batch_index]
        yield (batch_input_ids, batch_input_masks, batch_segment_ids), batch_y


# 生成位置嵌入
def fixedPositionEmbedding(batchSize, sequenceLen):
    embedd_pos = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embedd_pos.append(x)

    return np.array(embedd_pos, dtype="float32")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(max_seq_length, tokenizer, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)  # 这里主要是将中文分字
    if tokens_b:
        # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
        # 因为要为句子补上[CLS], [SEP], [SEP]
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 2
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
    # (a) 两个句子:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) 单个句子:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # 这里 "type_ids" 主要用于区分第一个第二个句子。
    # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
    # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将中文转换成ids
    # 创建mask
    input_mask = [1] * len(input_ids)
    # 对于输入进行补0
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids  # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数
