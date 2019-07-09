# coding=utf-8

"""
最大熵模型: 采用IIS最优化算法(M为常数时等同于GIS算法)
"""

import math
from collections import defaultdict

import numpy as np


class MaxEnt:
    def __init__(self, epsilon=1e-3, maxstep=10000):
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.w = None  # 特征函数的权重
        self.labels = None  # 标签
        self.fea_list = []  # 特征函数
        self.px = defaultdict(lambda: 0)  # 经验边缘分布概率
        self.pxy = defaultdict(lambda: 0)  # 经验联合分布概率,由于特征函数为取值为0，1的二值函数，所以等同于特征的经验期望值
        self.exp_fea = defaultdict(lambda: 0)  # 每个特征在数据集上的期望
        self.data_list = []  # 样本集，元素为tuple((X),y)
        self.N = None  # 样本总量
        self.M = None  # 某个训练样本包含特征的总数，这里假设每个样本的M值相同，即M为常数。其倒数类似于学习率
        self.n_fea = None  # 特征函数的总数

    def init_param(self, X_data, y_data):
        # 根据传入的数据集(数组)初始化模型参数
        self.N = X_data.shape[0]
        self.labels = np.unique(y_data)

        self.fea_func(X_data, y_data)
        self.n_fea = len(self.fea_list)
        self.w = np.zeros(self.n_fea)
        self._exp_fea(X_data, y_data)
        return

    def fea_func(self, X_data, y_data, rules=None):
        # 特征函数
        if rules is None:  # 若没有特征提取规则，则直接构造特征，此时每个样本没有缺失值的情况下的特征个数相同，等于维度
            for X, y in zip(X_data, y_data):
                X = tuple(X)
                self.px[X] += 1.0 / self.N  # X的经验边缘分布
                self.pxy[(X, y)] += 1.0 / self.N  # X,y的经验联合分布

                for dimension, val in enumerate(X):
                    key = (dimension, val, y)
                    if not key in self.fea_list:
                        self.fea_list.append(key)  # 特征函数，由 维度+维度下的值+标签 构成的元组
            self.M = X_data.shape[1]
        else:
            self.M = defaultdict(int)  # 字典存储每个样本的特征总数
            for i in range(self.N):
                self.M[i] = X_data.shape[1]
            pass  # 根据具体规则构建

    def _exp_fea(self, X_data, y_data):
        # 计算特征的经验期望值
        for X, y in zip(X_data, y_data):
            for dimension, val in enumerate(X):
                fea = (dimension, val, y)
                self.exp_fea[fea] += self.pxy[(tuple(X), y)]  # 特征存在取值为1，否则为0
        return

    def _py_X(self, X):
        # 当前w下的条件分布概率,输入向量X和y的条件概率
        py_X = defaultdict(float)

        for y in self.labels:
            s = 0
            for dimension, val in enumerate(X):
                tmp_fea = (dimension, val, y)
                if tmp_fea in self.fea_list:  # 输入X包含的特征
                    s += self.w[self.fea_list.index(tmp_fea)]
            py_X[y] = math.exp(s)

        normalizer = sum(py_X.values())
        for key, val in py_X.items():
            py_X[key] = val / normalizer
        return py_X

    def _est_fea(self, X_data, y_data):
        # 基于当前模型，获取每个特征估计期望
        est_fea = defaultdict(float)
        for X, y in zip(X_data, y_data):
            py_x = self._py_X(X)[y]
            for dimension, val in enumerate(X):
                est_fea[(dimension, val, y)] += self.px[tuple(X)] * py_x
        return est_fea

    def GIS(self, X_data, y_data):
        # GIS算法更新delta
        est_fea = self._est_fea(X_data, y_data)
        delta = np.zeros(self.n_fea)
        for j in range(self.n_fea):
            try:
                delta[j] = 1 / self.M * math.log(self.exp_fea[self.fea_list[j]] / est_fea[self.fea_list[j]])
            except:
                continue
        delta = delta / delta.sum()  # 归一化，防止某一个特征权重过大导致，后续计算超过范围
        return delta

    def IIS(self, delta, X_data, y_data):
        # IIS算法更新delta
        g = np.zeros(self.n_fea)
        g_diff = np.zeros(self.n_fea)
        for j in range(self.n_fea):
            for k in range(self.N):
                g[j] += self.px[tuple(X_data[k])] * self._py_X(X_data[k])[y_data[k]] * math.exp(delta[j] * self.M[k])
                g_diff[j] += g[j] * self.M[k]
            g[j] -= self.exp_fea[j]
            delta[j] -= g[j] / g_diff[j]
        return delta

    def fit(self, X_data, y_data):
        # 训练，迭代更新wi
        self.init_param(X_data, y_data)
        if isinstance(self.M, int):
            i = 0
            while i < self.maxstep:
                i += 1
                delta = self.GIS(X_data, y_data)
                # if max(abs(delta)) < self.epsilon:  # 所有的delta都小于阈值时，停止迭代
                #     break
                self.w += delta
                print('Iter %d/%d, Delta %r' % (i, self.maxstep, np.max(np.abs(delta))))
        else:
            i = 0
            delta = np.random.rand(self.n_fea)
            while i < self.maxstep:
                i += 1
                delta = self.IIS(delta, X_data, y_data)
                # if max(abs(delta)) < self.epsilon:
                #     break
                self.w += delta
        return

    def predict(self, X):
        # 输入x(数组)，返回条件概率最大的标签
        py_x = self._py_X(X)
        best_label = max(py_x, key=py_x.get)
        return best_label


datalabel = np.array(['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
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
map_table = {'青年': 0, '中年': 1, '老年': 2,
             '否': 0, '是': 1,
             '一般': 0, '好': 1, '非常好': 2}

if __name__ == '__main__':
    row_, col_ = train_sets.shape
    train_sets_encode = np.array([[map_table[train_sets[i, j]] for j in range(col_)] for i in range(row_)])
    X_t, Y_t = train_sets_encode[:, :-1], train_sets_encode[:, -1]
    model = MaxEnt()
    model.fit(X_t, Y_t)

