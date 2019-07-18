# coding=utf-8

import numpy as np
import decision_tree

class GBDT(object):

    def __init__(self, max_iter=10, sample_rate=0.5, learn_rate=1.0, max_depth=3):
        self.max_iter = max_iter
        self.sample_rate = sample_rate # 0 < sample_rate <= 1
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.dtrees = dict()
        self.original_f = None

    def fit(self, X_train, Y_train):
        n, m = X_train.shape
        # 记录每个样本对应的预测值，这个偏移值需要加到GBDT的预测结果中
        f = np.ones(n) * np.mean(Y_train)
        self.original_f = np.array(f)
        # 数据集随机抽样，减少模型方差
        n_sample = int(n*self.sample_rate)
        print('<Train begins>')
        for iter_ in range(self.max_iter):
            sample_idx = np.random.permutation(n)[:n_sample]
            X_train_subset, Y_train_subset = X_train[sample_idx, :], Y_train[sample_idx]
            y_predict_subset = np.zeros(n_sample)
            # 用损失函数的负梯度作为回归树的残差近似值
            for j in range(n_sample):
                k = sample_idx[j]
                y_predict_subset[j] = f[k]
            residual = Y_train_subset - y_predict_subset
            print('Iter %r/%r: %r(residual)' % (iter_, self.max_iter, np.mean(residual)))
            # 用残差作为新标签训练一颗新树
            dtree = decision_tree.DTreeRegressionCART(max_depth=self.max_depth)
            dtree.fit(X_train_subset, residual)
            self.dtrees[iter_] = dtree
            # 更新样本预测值
            for j in range(n):
                f[j] += self.learn_rate * dtree.predict(np.array([X_train[j]]))

    def predict(self, X):
        n = X.shape[0]
        Y = np.zeros([n, self.max_iter])
        for iter_ in range(self.max_iter):
            dtree = self.dtrees[iter_]
            Y[:, iter_] = dtree.predict(X)
        # 将GBDT初始化时的偏移值需要加到预测结果中
        return np.sum(Y, axis=1) + self.original_f

if __name__ == '__main__':
    model = GBDT()
    X_ = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    Y_ = np.array([1, 2, 3, 4])
    model.fit(X_, Y_)
    print('<Label Output>')
    print(model.predict(X_))