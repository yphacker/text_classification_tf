# coding=utf-8
# author=yphacker

import tensorflow as tf
from conf import config, model_config_cnn as model_config
from utils.data_utils import get_pretrain_embedding


# # 构建模型
# class TextCNN(object):
#
#     def __init__(self, config, wordEmbedding):
#         self.seq_length = config.seq_length
#         self.num_classes = config.num_classes
#
#         # 定义模型的输入
#         self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name="input_x")
#         self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")
#         self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
#
#         # 定义l2损失
#         l2Loss = tf.constant(0.0)
#
#         # 词嵌入层
#         with tf.name_scope("embedding"):
#
#             # 利用预训练的词向量初始化词嵌入矩阵
#             self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
#             # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
#             self.embeddedWords = tf.nn.embedding_lookup(self.W, self.input_x)
#             # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
#             self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)
#
#         # 创建卷积和池化层
#         pooledOutputs = []
#         # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
#         for i, filterSize in enumerate(config.model.filterSizes):
#             with tf.name_scope("conv-maxpool-%s" % filterSize):
#                 # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
#                 # 初始化权重矩阵和偏置
#                 filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
#                 W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
#                 b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
#                 conv = tf.nn.conv2d(
#                     self.embeddedWordsExpanded,
#                     W,
#                     strides=[1, 1, 1, 1],
#                     padding="VALID",
#                     name="conv")
#
#                 # relu函数的非线性映射
#                 h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#
#                 # 池化层，最大池化，池化是对卷积后的序列取一个最大值
#                 pooled = tf.nn.max_pool(
#                     h,
#                     ksize=[1, config.sequenceLength - filterSize + 1, 1, 1],
#                     # ksize shape: [batch, height, width, channels]
#                     strides=[1, 1, 1, 1],
#                     padding='VALID',
#                     name="pool")
#                 pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
#
#         # 得到CNN网络的输出长度
#         numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)
#
#         # 池化后的维度不变，按照最后的维度channel来concat
#         self.hPool = tf.concat(pooledOutputs, 3)
#
#         # 摊平成二维的数据输入到全连接层
#         self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])
#
#         # dropout
#         with tf.name_scope("dropout"):
#             self.hDrop = tf.nn.dropout(self.hPoolFlat, self.keep_prob)
#
#         # 全连接层的输出
#         with tf.name_scope("output"):
#             outputW = tf.get_variable(
#                 "outputW",
#                 shape=[numFiltersTotal, config.numClasses],
#                 initializer=tf.contrib.layers.xavier_initializer())
#             outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
#             l2Loss += tf.nn.l2_loss(outputW)
#             l2Loss += tf.nn.l2_loss(outputB)
#             self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
#             if config.numClasses == 1:
#                 self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
#             elif config.numClasses > 1:
#                 self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
#
#             print(self.predictions)
#
#         # 计算二元交叉熵损失
#         with tf.name_scope("loss"):
#             if config.numClasses == 1:
#                 losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
#                                                                  labels=tf.cast(tf.reshape(self.input_x, [-1, 1]),
#                                                                                 dtype=tf.float32))
#             elif config.numClasses > 1:
#                 losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
#
#             self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
#
#     def train(self):
#         pass
#
#     def eval(self):
#         pass
#
#     def predict(self):
#         pass

class Model(object):

    def __init__(self):
        self.learning_rate = model_config.learning_rate

        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

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

        with tf.name_scope("cnn"):
            # CNN layer
            x_input_embedded = tf.nn.embedding_lookup(input_embedding, self.input_x)
            conv = tf.layers.conv1d(x_input_embedded, model_config.num_filters, model_config.kernel_size, name='conv')
            # global max pooling layer
            pooling = tf.reduce_max(conv, reduction_indices=[1])

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(pooling, model_config.hidden_dim, name='fc1')
            # fc = tf.contrib.layers.dropout(fc, keep_prob)
            fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            logits = tf.layers.dense(fc, config.num_labels, name='fc2')
            self.prob = tf.nn.softmax(logits, name='prob')
            self.pred = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')  # 预测类别

        with tf.name_scope("optimize"):
            # 将label进行onehot转化
            one_hot_labels = tf.one_hot(self.input_y, depth=config.num_labels, dtype=tf.float32)
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.pred)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')
