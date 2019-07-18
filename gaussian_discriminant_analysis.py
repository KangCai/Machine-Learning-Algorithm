# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

class GDA(object):

    def __init__(self):
        self.Mu0, self.Mu1, self.Sigma = None, None, None
        self.w, self.b, self.sign = None, None, None

    def fit(self, X_train, Y_train):
        n, m = X_train.shape
        X0, X1 = X_train[Y_train==0], X_train[Y_train==1]
        self.Mu0, self.Mu1 = np.mean(X0, axis=0), np.mean(X1, axis=0)
        X_sub_Mu = np.vstack([X0 - self.Mu0, X1 - self.Mu1])
        self.Sigma = (1.0/m) * np.dot(X_sub_Mu.T, X_sub_Mu)
        # 判别平面计算
        normal_vec = self.Mu1 - self.Mu0
        normal_vec = normal_vec / np.sqrt(np.sum(normal_vec * normal_vec))
        self.w = normal_vec
        self.b = - np.dot(self.w.T, (self.Mu0 + self.Mu1) / 2.0)
        self.sign = int(np.dot(self.w.T, self.Mu1) + self.b > 0)

    def predict(self, X):
        return (np.dot(X, self.w) + self.b > 0).astype(int) * self.sign

def _GenerateData():
    import random
    m, n_train, n_val, interval = 2, 10, 2, 1
    X_train, X_val, Y_train, Y_val = [], [], [], []
    color = ['c', 'r']
    def _generateOne(X, Y, i):
        i += 1
        x, y, l = random.uniform((int(i / 2) + 0.1) * 10, (int(i / 2) + 0.9) * 10), random.uniform((int(i / 2) + 0.1) * 10, (int(i / 2) + 0.9) * 10), i
        X.append((x, y))
        Y.append(i - 1)
        return x, y
    for i_ in range(m):
        for _ in range(n_train):
            x_, y_ = _generateOne(X_train, Y_train, i_)
            plt.scatter(x_, y_, s=60, c=color[i_], alpha=0.3)
        for _ in range(n_val):
            _generateOne(X_val, X_val, i_)
    return np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)

if __name__ == '__main__':
    model = GDA()
    X_t, X_v, Y_t, Y_v = _GenerateData()
    print('<Y_t>')
    print(Y_t)
    model.fit(X_t, Y_t)
    print('<Label Output>')
    print(model.predict(X_t))
    # 画 Mu 点
    plt.scatter([model.Mu0[0], model.Mu1[0]], [model.Mu0[1], model.Mu1[1]], s=100, c=['c', 'r'])
    # 根据 Mu 画判别边界
    midPoint = [(model.Mu0[0] + model.Mu1[0]) / 2.0, (model.Mu0[1] + model.Mu1[1]) / 2.0]
    k = (model.Mu1[1] - model.Mu0[1]) / (model.Mu1[0] - model.Mu0[0])
    bx = range(-5, 25)
    by = [(-1.0 / k) * (i - midPoint[0]) + midPoint[1] for i in bx]
    plt.plot(bx, by)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.title('Gaussian discriminant analysis')
    plt.show()