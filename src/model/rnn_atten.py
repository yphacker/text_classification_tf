# coding=utf-8
# author=yphacker

import tensorflow as tf
from conf import config
from conf import model_config_rnn_atten as model_config
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
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   x_input_embedded, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    x_input_embedded = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(x_input_embedded, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H)
            outputSize = model_config.hidden_size[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "outputW",
                shape=[outputSize, config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[config.num_labels]), name="outputB")
            logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.prob = tf.nn.softmax(logits, name='prob')
            self.y_pred = tf.argmax(self.prob, 1, name='y_pred')

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

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = model_config.hidden_size[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.max_seq_len])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.max_seq_len, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.keep_prob)

        return output
