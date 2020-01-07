# coding=utf-8
# author=yphacker

import os
import argparse
import pandas as pd
import tensorflow as tf
from importlib import import_module
from sklearn.model_selection import train_test_split
from conf import config
from utils.data_utils import get_vocab, get_dataset


def get_feed_dict(batch_x, batch_y=None, type='train'):
    if model_name in ['bert', 'albert']:
        input_ids = batch_x[0]
        input_masks = batch_x[1]
        segment_ids = batch_x[2]
        feed_dict = {
            model.input_ids: input_ids,
            model.input_masks: input_masks,
            model.segment_ids: segment_ids,
        }
        if type in ['train', 'val']:
            feed_dict[model.labels] = batch_y
        elif type in ['predict']:
            pass
        if type in ['train']:
            feed_dict[model.is_training] = True
        elif type in ['val', 'predict']:
            feed_dict[model.is_training] = False
    else:
        feed_dict = {
            model.input_x: batch_x
        }
        if type in ['train', 'val']:
            feed_dict[model.input_y] = batch_y
        elif type in ['predict']:
            pass
        if type in ['train']:
            feed_dict[model.keep_prob] = model_config.keep_prob
        elif type in ['val', 'predict']:
            feed_dict[model.keep_prob] = 1.0
    return feed_dict


def evaluate(sess, x_val, y_val):
    """评估在某一数据上的准确率和损失"""
    data_len = len(y_val)
    total_loss = 0.0
    total_acc = 0.0
    val_iter = get_data_iter(x_val, y_val, config.batch_size)
    for batch_x, batch_y in val_iter:
        batch_len = len(batch_y)
        feed_dict = get_feed_dict(batch_x, batch_y, 'val')
        _loss, _acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        total_loss += _loss * batch_len
        total_acc += _acc * batch_len
    return total_loss / data_len, total_acc / data_len


def train():
    train_df = pd.read_csv(config.train_path)
    # x_train = df['review'].values.tolist()
    # y_train = df['sentiment'].values.tolist()
    #
    # dev_sample_index = -1 * int(0.1 * float(len(y_train)))
    # # 划分训练集和验证集
    # x_train, x_val = x_train[:dev_sample_index], x_train[dev_sample_index:]
    # y_train, y_val = y_train[:dev_sample_index], y_train[dev_sample_index:]
    # print('train:{}, val:{}, all:{}'.format(len(y_train), len(y_val), df.shape[0]))
    train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
    x_train, y_train = get_dataset(train_data)
    x_val, y_val = get_dataset(val_data)
    # train_iter = get_data_iter(train_data, config.batch_size)
    # val_iter = get_data_iter(val_data, config.batch_size)

    print('training and evaluating...')
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved_step = 0  # 记录上一次提升批次
    data_len = len(y_train)
    cur_step = 0
    epoch_step = int((data_len - 1) / config.batch_size) + 1
    total_step = epoch_step * config.epochs_num
    adjust_num = 0
    flag = True
    # 配置 Saver
    saver = tf.train.Saver(max_to_keep=1)
    # session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # session_conf.gpu_options.allow_growth = True
    # session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.epochs_num):
            train_iter = get_data_iter(x_train, y_train, config.batch_size)
            for batch_x, batch_y in train_iter:
                feed_dict = get_feed_dict(batch_x, batch_y)
                sess.run(model.train_op, feed_dict=feed_dict)
                cur_step += 1
                if cur_step % config.print_per_batch == 0:
                    fetches = [model.loss, model.accuracy]
                    loss_train, acc_train = sess.run(fetches, feed_dict=feed_dict)
                    loss_val, acc_val = evaluate(sess, x_val, y_val)
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved_step = cur_step
                        saver.save(sess=sess, save_path=model_config.model_save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''
                    cur_step_str = str(cur_step) + "/" + str(total_step)
                    msg = 'the current step: {0}, train loss: {1:>6.2}, train acc: {2:>7.2%},' \
                          + ' val loss: {3:>6.2}, val acc: {4:>7.2%}, {5}'
                    print(msg.format(cur_step_str, loss_train, acc_train, loss_val, acc_val, improved_str))
                # if cur_step - last_improved_step >= config.improvement_step:
                if cur_step - last_improved_step >= epoch_step:
                    flag = False
                    # last_improved_step = cur_step
                    # print("No optimization for a long time, auto adjust learning_rate...")
                    # learning_rate = learning_rate_decay(learning_rate)
                    # adjust_num += 1
                    # if adjust_num > 3:
                    #     print("No optimization for a long time, auto-stopping...")
                    #     flag = False
                if not flag:
                    break
            if not flag:
                break


def eval():
    # df = pd.read_csv(config.train_path, sep='\t', header=None, names=['label', 'text'])
    # x_test = df['review'].values.tolist()
    #
    # preds = model.predict(x_test)
    # df['pred_label'] = preds
    # cols = ['pred_label', 'label', 'text']
    # train = df.ix[:, cols]
    # train.to_csv(config.train_check_path, index=False)
    pass


def predict():
    test_df = pd.read_csv(config.test_path)
    x_test, _ = get_dataset(test_df)
    test_iter = get_data_iter(x_test, batch_size=config.batch_size, shuffle=False)
    preds = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess=sess, save_path=model_config.model_save_path)  # 读取保存的模型
        for batch_x, _ in test_iter:  # 逐批次处理
            feed_dict = get_feed_dict(batch_x, type='predict')
            pred = sess.run(model.prob, feed_dict=feed_dict)
            preds.extend(pred[:, 1])
    submission = pd.read_csv(config.sample_submission_path)
    submission['sentiment'] = preds
    submission.to_csv(model_config.submission_path, index=False)


def main(op):
    if op == 'train':
        train()
    elif op == 'eval':
        eval()
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--model_name", default='cnn', type=str, help="model select")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num

    word2id, label2id = get_vocab()

    model_name = args.model_name
    if model_name in ['bert', 'albert']:
        from utils.bert_data_utils import get_dataset, get_data_iter
    else:
        from utils.data_utils import get_dataset, get_data_iter
    x = import_module('model.{}'.format(model_name))
    model_config = import_module('conf.model_config_{}'.format(model_name))
    model_save_path = os.path.join(config.model_path, model_name)
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    model_config.model_save_path = os.path.join(model_save_path, 'best')
    model_config.submission_path = os.path.join(config.data_path, '{}_submission.csv'.format(model_name))

    model = x.Model()

    main(args.operation)
