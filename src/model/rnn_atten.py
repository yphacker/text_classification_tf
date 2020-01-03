# coding=utf-8
# author=yphacker

import numpy as np
import tensorflow as tf
from conf import config
from conf import bilstm_attention_model_config as model_config
from utils.train_utils import get_word_embedding, get_vocabulary, word2index, label2index
from utils.model_utils import batch_iter


class Model(object):
    def __init__(self):
        self.learning_rate = model_config.learning_rate
        # 定义模型的输入
        self.input_x = tf.placeholder(tf.int32, [None, config.max_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.word_embedding = get_word_embedding()
        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(self.word_embedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.input_x)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
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
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H)
            outputSize = model_config.hidden_size[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.num_labels]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            self.y_pred = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')
            # if config.num_labels == 1:
            #     self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            # elif config.num_labels > 1:
            #     self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            # if config.num_labels == 1:
            #     losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
            #                                                      labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
            #                                                                     dtype=tf.float32))
            # elif config.num_labels > 1:
            #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.input_y, depth=config.num_labels, dtype=tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy) + model_config.l2RegLambda * l2Loss

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

    def train(self, x_train, y_train, x_val, y_val):
        # 初始化词汇-索引映射表和词向量矩阵
        word2id, label2id = get_vocabulary()
        x_train = np.array([word2index(text, word2id) for text in x_train])
        x_val = np.array([word2index(text, word2id) for text in x_val])
        y_train = np.array([label2index(label, label2id) for label in y_train])
        y_val = np.array([label2index(label, label2id) for label in y_val])

        print('Training and evaluating...')
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved_step = 0  # 记录上一次提升批次
        data_len = len(y_train)
        adjust_num = 0
        cur_step = 0
        step_sum = (int((data_len - 1) / config.batch_size) + 1) * config.epochs_num
        flag = True
        # 配置 Saver
        saver = tf.train.Saver(max_to_keep=1)
        # session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # session_conf.gpu_options.allow_growth = True
        # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(config.epochs_num):
                for batch_x, batch_y in batch_iter(x_train, y_train):
                    feed_dict = {
                        self.input_x: batch_x,
                        self.input_y: batch_y,
                        self.keep_prob: model_config.keep_prob
                    }
                    sess.run(self.train_op, feed_dict=feed_dict)
                    cur_step += 1
                    if cur_step % config.print_per_batch == 0:
                        fetches = [self.loss, self.accuracy]
                        loss_train, acc_train = sess.run(fetches, feed_dict=feed_dict)
                        loss_val, acc_val = self.evaluate(sess, x_val, y_val)
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved_step = cur_step
                            saver.save(sess=sess, save_path=model_config.model_save_path)
                            improved_str = '*'
                        else:
                            improved_str = ''
                        cur_step_str = str(cur_step) + "/" + str(step_sum)
                        msg = 'the Current step: {0}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, {5}'
                        print(msg.format(cur_step_str, loss_train, acc_train, loss_val, acc_val, improved_str))
                    if cur_step - last_improved_step >= config.improvement_step:
                        last_improved_step = cur_step
                        print("No optimization for a long time, auto adjust learning_rate...")
                        # learning_rate = learning_rate_decay(learning_rate)
                        adjust_num += 1
                        if adjust_num > 3:
                            print("No optimization for a long time, auto-stopping...")
                            flag = False
                    if not flag:
                        break
                if not flag:
                    break

    def evaluate(self, sess, x_val, y_val):
        """评估在某一数据上的准确率和损失"""
        data_len = len(y_val)
        total_loss = 0.0
        total_acc = 0.0
        for batch_x_val, batch_y_val in batch_iter(x_val, y_val):
            feed_dict = {
                self.input_x: batch_x_val,
                self.input_y: batch_y_val,
                self.keep_prob: 1.0,
            }
            batch_len = len(batch_y_val)
            _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            total_loss += _loss * batch_len
            total_acc += _acc * batch_len
        return total_loss / data_len, total_acc / data_len

    def predict(self, x_test):
        word2id, label2id = get_vocabulary()
        x_test = [word2index(text, word2id) for text in x_test]

        data_len = len(x_test)
        num_batch = int((data_len - 1) / config.batch_size) + 1
        preds = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess=sess, save_path=model_config.model_save_path)  # 读取保存的模型
            for i in range(num_batch):  # 逐批次处理
                start_id = i * config.batch_size
                end_id = min((i + 1) * config.batch_size, data_len)
                feed_dict = {
                    self.input_x: x_test[start_id:end_id],
                    self.keep_prob: 1.0
                }
                pred = sess.run(self.y_pred, feed_dict=feed_dict)
                preds.extend(pred)
        return preds
