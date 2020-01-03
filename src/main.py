# coding=utf-8
# author=yphacker

import os
import argparse
import pandas as pd
from importlib import import_module
from conf import config


def train():
    df = pd.read_csv(config.train_path)
    df = df.sample(frac=1).reset_index(drop=True)
    x_train = df['review'].values.tolist()
    y_train = df['sentiment'].values.tolist()

    dev_sample_index = -1 * int(0.1 * float(len(y_train)))
    # 划分训练集和验证集
    x_train, x_val = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_val = y_train[:dev_sample_index], y_train[dev_sample_index:]
    print('train:{}, val:{}, all:{}'.format(len(y_train), len(y_val), df.shape[0]))

    model.train(x_train, y_train, x_val, y_val)


def eval():
    df = pd.read_csv(config.train_path, sep='\t', header=None, names=['label', 'text'])
    x_test = df['review'].values.tolist()

    preds = model.predict(x_test)
    df['pred_label'] = preds
    cols = ['pred_label', 'label', 'text']
    train = df.ix[:, cols]
    train.to_csv(config.train_check_path, index=False)


def predict():
    df = pd.read_csv(config.test_path)
    x_test = df['review'].values.tolist()
    preds = model.predict(x_test)

    submission = pd.read_csv(config.origin_submission_path)
    submission['sentiment'] = preds
    submission.to_csv(config.model_submission_path, index=False)


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
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS

    model_name = args.model_name
    x = import_module('model.{}'.format(model_name))
    model_config = import_module('conf.model_config_{}'.format(model_name))
    model_save_path = os.path.join(config.model_path, model_name)
    model_config.model_save_path = os.path.join(model_save_path, 'best')
    model_config.submission_path = os.path.join(config.data_path, '{}_submission.csv'.format(model_name))

    model = x.Model()

    main(args.operation)
