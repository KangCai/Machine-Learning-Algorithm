# coding=utf-8

import numpy as np
import math
from collections import defaultdict

class MaxEntropy(object):

    def __init__(self, epsilon=0.001, n_iter=100):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.w = None
        # 训练样本量，特征维度，x 的经验边缘分布，(x,y) 的联合概率分布，特征函数关于 (x,y) 联合分布的期望值，特征函数 f(x,y)，记录标签集
        self.N, self.M, self.px, self.pxy, self.e_feat, self.feat_list, self.labels = \
            None, None, defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0), [], []

    def fit(self, X_train, Y_train):
        self.N, self.M = X_train.shape
        self.labels = np.arange(np.bincount(Y_train).size)
        # 统计 (x,y) 的联合概率分布，x 的经验边缘分布
        feat_set = set()
        for X,y in zip(X_train, Y_train):
            X = tuple(X)
            self.px[X] += 1.0 / self.N
            self.pxy[(X, y)] += 1.0 / self.N
            for idx, val in enumerate(X):
                key = (idx, val, y)
                feat_set.add(key)
        self.feat_list = list(feat_set)
        self.w = np.zeros(len(self.feat_list))
        print(self.px, self.pxy)
        # 计算特征的经验期望值, E_p~(f) = Sum( P~(x,y) * f(x,y) )
        for X,y in zip(X_train, Y_train):
            X = tuple(X)
            for idx, val in enumerate(X):
                key = (idx, val, y)
                self.e_feat[key] += self.pxy[(X, y)]
        # 迭代找到最优参数 self.w
        for i in range(self.n_iter):
            delta = self._GIS(X_train, Y_train)
            print('Iter %d/%d, Delta %r' % (i, self.n_iter, np.max(np.abs(delta))))
            if np.max(np.abs(delta)) < self.epsilon:
                break
            self.w += delta

    def predict(self, X):
        n, m = X.shape
        result_array = np.zeros(n)
        for i in range(n):
            output = self._cal_py_X(X[i, :])
            result_array[i] = max(output, key=output.get)
        return result_array

    def _GIS(self, X_train, Y_train):
        n_feat = len(self.feat_list)
        # 基于当前模型，获取每个特征估计期望, E_p(f) = Sum( P~(x) * P(y|x) * f(x,y) )
        delta = np.zeros(n_feat)
        estimate_feat = defaultdict(float)
        for X,y in zip(X_train, Y_train):
            X = tuple(X)
            py_x = self._cal_py_X(X)[y]
            for idx, val in enumerate(X):
                key = (idx, val, y)
                estimate_feat[key] += self.px[X] * py_x
        # 更新 delta
        for j in range(n_feat):
            feat_key = self.feat_list[j]
            e_feat_exp = self.e_feat[feat_key]
            e_feat_estimate = estimate_feat[feat_key]
            if e_feat_estimate == 0 or e_feat_exp / e_feat_estimate <= 0:
                continue
            delta[j] = 1.0 / self.M * math.log(e_feat_exp / e_feat_estimate)
        delta /= np.sum(delta)
        return delta

    def _cal_py_X(self, X):
        # 计算条件分布概率 P(y|x)
        py_X = defaultdict(float)
        for y in self.labels:
            s = 0
            for idx, val in enumerate(X):
                feat_key = (idx, val, y)
                if feat_key in self.feat_list:
                    dim_idx = self.feat_list.index(feat_key)
                    s += self.w[dim_idx]
            py_X[y] = math.exp(s)
        normalizer = sum(py_X.values())
        for label, val in py_X.items():
            py_X[label] = val / normalizer
        return py_X

datalabel = np.array(['年龄(特征1)', '有工作(特征2)', '有自己的房子(特征3)', '信贷情况(特征4)', '类别(标签)'])
train_sets = np.array([
                    ['青年', '否', '否', '一般', '否'],
                    ['青年', '否', '否', '好', '否'],
                    ['青年', '是', '否', '好', '是'],
                    ['青年', '是', '是', '一般', '是'],
                    ['青年', '否', '否', '一般', '否'],
                    ['中年', '否', '否', '一般', '否'],
                    ['中年', '否', '否', '好', '否'],
                    ['中年', '是', '是', '好', '是'],
                    ['中年', '否', '是', '非常好', '是'],
                    ['中年', '否', '是', '非常好', '是'],
                    ['老年', '否', '是', '非常好', '是'],
                    ['老年', '否', '是', '好', '是'],
                    ['老年', '是', '否', '好', '是'],
                    ['老年', '是', '否', '非常好', '是'],
                    ['老年', '否', '否', '一般', '否'],
                    ['青年', '否', '否', '一般', '是']])
validate_sets = np.array([
    ['青年', '是', '是', '好', '是'],
    ['青年', '是', '否', '一般', '是'],
    ['中年', '否', '否', '一般', '否'],
    ['老年', '是', '是', '一般', '是'],
    ['青年', '否', '否', '非常好', '是'],
])
map_table = {'青年': 0, '中年': 1, '老年': 2,
             '否': 0, '是': 1,
             '一般': 0, '好': 1, '非常好': 2}

if __name__ == '__main__':
    row_, col_ = train_sets.shape
    train_sets_encode = np.array([[map_table[train_sets[i, j]] for j in range(col_)] for i in range(row_)])
    X_t, Y_t = train_sets_encode[:, :-1], train_sets_encode[:, -1]
    model = MaxEntropy()
    model.fit(X_t, Y_t)
    res = model.predict(X_t)
    print('Ground truth   on train set: %r' % (Y_t,))
    print('Predict result on train set: %r' % (res.astype(int),))
    row_, col_ = validate_sets.shape
    validate_sets_encode = np.array([[map_table[validate_sets[i, j]] for j in range(col_)] for i in range(row_)])
    X_v, Y_v = validate_sets_encode[:, :-1], validate_sets_encode[:, -1]
    res = model.predict(X_v)
    print('Ground truth   on validate set: %r' % (Y_v,))
    print('Predict result on validate set: %r' % (res.astype(int),))