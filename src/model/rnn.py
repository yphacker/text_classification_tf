# coding=utf-8
# author=yphacker

import numpy as np
import tensorflow as tf
from conf import config
from conf import model_config_rnn as model_config
from utils.data_utils import get_pretrain_embedding


class Model(object):

    def __init__(self):
        self.learning_rate = model_config.learning_rate

        # 定义模型的输入
        self.input_x = tf.placeholder(tf.int32, [None, config.max_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        if config.pretrain_embedding:
            with tf.name_scope("embedding"):
                # 利用预训练的词向量初始化词嵌入矩阵
                embedding = get_pretrain_embedding()
                input_embedding = tf.Variable(tf.cast(embedding, dtype=tf.float32, name="word2vec"), name="W")
        else:
            with tf.variable_scope('embedding'):
                # 标准正态分布初始化
                input_embedding = tf.Variable(
                    tf.truncated_normal(shape=[config.num_vocab, config.embed_dim], stddev=0.1),
                    name='embedding')

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            x_input_embedded = tf.nn.embedding_lookup(input_embedding, self.input_x)
            for idx, hiddenSize in enumerate(model_config.hidden_size):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                  x_input_embedded,
                                                                                  dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    x_input_embedded = tf.concat(outputs, 2)

        # 去除最后时间步的输出作为全连接的输入
        finalOutput = x_input_embedded[:, -1, :]

        outputSize = model_config.hidden_size[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.num_labels]), name="outputB")
            logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            self.y_pred = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')  # 预测类别

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.input_y, depth=config.num_labels, dtype=tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')
