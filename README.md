# Bag of Words Meets Bags of Popcorn
[竞赛链接](https://www.kaggle.com/c/word2vec-nlp-tutorial)
## 数据下载
[data](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)
## 评估标准
roc_auc
## kaggle score:
|model|score|
|---|---|
|ml|0.95188|
|数据扩充+ml|0.96977|
|cnn|0.94647|
|cnn+预训练|0.94608|
|rnn|0.79370|
|rcnn|0.92285|
|rnn+atten|0.93044|
|transformer|0.86324|
|bert(uncased_L-12_H-768_A-12)|0.96632|
|albert(base_v2)|0.90293|

## 实验环境
Tesla P100
16G
cuda9  
python:3.6  
torch:1.2.0.dev20190722

## 参考链接
[google-research/bert](https://github.com/google-research/bert)
[google-research/ALBERT](https://github.com/google-research/ALBERT)


