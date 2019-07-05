# coding=utf-8

import numpy as np

class DTreeID3(object):

    def __init__(self):
        self.tree = Node()

    def fit(self, X_train, Y_train):
        self._train(X_train, Y_train, node)

    def _train(self, X_train, Y_train, node):
        # 特殊情况：若 D 中所有实例属于同一类，决策树成单节点树，直接返回
        if np.any(np.bincount(Y_train) == len(Y_train)):
            return
        # 计算特征集 A 中各特征对 D 的信息增益，选择信息增益最大的特征 A_g

        # 如果 A_g 的信息增益小于阈值 epsilon，决策树成单节点树，直接返回

        # 否则，对于 A_g 的每一可能值 a_i，依据 A_g = a_i 将 D 分割
        # Calculate the information gain.


class Node(object):

    def __init__(self):
        pass

if __name__ == '__main__':
    model = DTreeID3()
    model.fit(np.zeros(5, 2), np.zeros(5))