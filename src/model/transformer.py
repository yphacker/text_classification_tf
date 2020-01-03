# coding=utf-8
# author=yphacker

import numpy as np
import tensorflow as tf
from conf import config
from conf import transformer_model_config as model_config
from utils.train_utils import get_word_embedding, get_vocabulary, word2index, label2index
from utils.model_utils import fixedPositionEmbedding, batch_iter


class Model(object):
    """
    Transformer Encoder 用于文本分类
    """

    def __init__(self):
        self.learning_rate = model_config.learning_rate
        # 定义模型的输入
        self.input_x = tf.placeholder(tf.int32, [None, config.max_seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.embedd_pos = tf.placeholder(tf.float32, [None, config.max_seq_len, config.max_seq_len], name="embedd_pos")

        self.word_embedding = get_word_embedding()

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层, 位置向量的定义方式有两种：一是直接用固定的one-hot的形式传入，然后和词向量拼接，在当前的数据集上表现效果更好
        # 另一种就是按照论文中的方法实现，这样的效果反而更差，可能是增大了模型的复杂度，在小数据集上表现不佳。

        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(self.word_embedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embeddedWords = tf.concat([self.embedded, self.embedd_pos], -1)

        with tf.name_scope("transformer"):
            for i in range(model_config.numBlocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    # 维度[batch_size, sequence_length, embedding_size]
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.input_x, queries=self.embeddedWords,
                                                            keys=self.embeddedWords)
                    # 维度[batch_size, sequence_length, embedding_size]
                    self.embeddedWords = self._feedForward(multiHeadAtt,
                                                           [model_config.filters,
                                                            config.embedding_size + config.max_seq_len])

            outputs = tf.reshape(self.embeddedWords,
                                 [-1, config.max_seq_len * (config.embedding_size + config.max_seq_len)])

        outputSize = outputs.get_shape()[-1].value

        #         with tf.name_scope("wordEmbedding"):
        #             self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
        #             self.wordEmbedded = tf.nn.embedding_lookup(self.W, self.input_x)

        #         with tf.name_scope("positionEmbedding"):
        #             print(self.wordEmbedded)
        #             self.positionEmbedded = self._positionEmbedding()

        #         self.embeddedWords = self.wordEmbedded + self.positionEmbedded

        #         with tf.name_scope("transformer"):
        #             for i in range(config.model.numBlocks):
        #                 with tf.name_scope("transformer-{}".format(i + 1)):

        #                     # 维度[batch_size, sequence_length, embedding_size]
        #                     multiHeadAtt = self._multiheadAttention(rawKeys=self.wordEmbedded, queries=self.embeddedWords,
        #                                                             keys=self.embeddedWords)
        #                     # 维度[batch_size, sequence_length, embedding_size]
        #                     self.embeddedWords = self._feedForward(multiHeadAtt, [config.model.filters, config.model.embeddingSize])

        #             outputs = tf.reshape(self.embeddedWords, [-1, config.max_seq_len * (config.model.embeddingSize)])

        #         outputSize = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.num_labels]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name="logits")
            self.y_pred = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')  # 预测类别
            # if config.num_labels == 1:
            #     self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            # elif config.num_labels > 1:
            #     self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            one_hot_labels = tf.one_hot(self.input_y, depth=config.num_labels, dtype=tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
            # if config.num_labels == 1:
            #     losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
            #                                                      labels=tf.cast(tf.reshape(self.input_y, [-1, 1]),
            #                                                                     dtype=tf.float32))
            # elif config.num_labels > 1:
            #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)

            self.loss = tf.reduce_mean(cross_entropy) + model_config.l2RegLambda * l2Loss

            # 优化器
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.y_pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = model_config.epsilon

        inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]

        paramsShape = inputsShape[-1:]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值

        numHeads = model_config.numHeads
        keepProp = model_config.keepProp

        if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(rawKeys, [numHeads, 1])

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,
                                  scaledSimilary)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings,
                                      maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络

        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = config.batch_size
        sequenceLen = config.max_seq_len
        embeddingSize = config.embedding_size

        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded

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
                        self.keep_prob: model_config.keep_prob,
                        self.embedd_pos: fixedPositionEmbedding(len(batch_y), config.max_seq_len)
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
        data_len = len(y_val)
        total_loss = 0.0
        total_acc = 0.0
        for batch_x_val, batch_y_val in batch_iter(x_val, y_val):
            feed_dict = {
                self.input_x: batch_x_val,
                self.input_y: batch_y_val,
                self.keep_prob: 1.0,
                self.embedd_pos: fixedPositionEmbedding(len(batch_y_val), config.max_seq_len)
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
                    self.keep_prob: 1.0,
                    self.embedd_pos: fixedPositionEmbedding(end_id - start_id, config.max_seq_len)
                }
                pred = sess.run(self.y_pred, feed_dict=feed_dict)
                preds.extend(pred)
        return preds
