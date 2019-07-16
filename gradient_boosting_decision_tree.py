# coding=utf-8

import numpy as np
import random

class GBDT(object):

    def __init__(self, max_iter, sample_rate):
        self.max_iter = max_iter
        self.sample_rate = sample_rate # 0 < sample_rate <= 1

    def fit(self, X_train, Y_train):
        # 记录每个特征的对应的预测值
        f = dict()
        # 数据集随机抽样，减少模型方差
        n, m = X_train.shape
        n_sample = int(n*self.sample_rate)
        sample_idx = np.random.permutation(n)[:n_sample]
        X_train_subset, Y_train_subset = X_train[sample_idx, :], Y_train[sample_idx, :]
        y_predict_subset = np.zeros(n_sample)
        # 用损失函数的负梯度作为回归树的残差近似值
        for i in range(n):
            y_predict_subset[i] = f[X_train_subset[i]]
        residual = Y_train - y_predict_subset


if __name__ == '__main__':
    model = GBDT(max_iter=100, sample_rate=0.5)
    X_ = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    model.fit(X_, X_)