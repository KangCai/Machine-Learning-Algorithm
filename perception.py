# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

class PerceptionPrimitive(object):

    def __init__(self, eta=0.1, n_iter=10000):
        self.eta = eta
        self.n_iter = n_iter
        self.w = self.b = None
        self.error_count_history = []

    def fit(self, X_train, Y_train):
        # Sum( -y(wx + b) ) -> min
        self.w, self.b = np.zeros(X_train.shape[1]), 0
        # Iteration
        for _ in range(self.n_iter):
            error_count = 0
            for xi, yi in zip(X_train, Y_train):
                if yi * self.predict(xi) <= 0:
                    self.w += self.eta * yi * xi
                    self.b += self.eta * yi
                    error_count += 1
            self.error_count_history.append(error_count)
            if error_count == 0:
                break

    def predict_raw(self, x):
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        print(self.predict_raw(x), np.sign(self.predict_raw(x)))
        return np.sign(self.predict_raw(x))


class PerceptionDual(PerceptionPrimitive):

    def __init__(self, eta=0.1, n_iter=10000):
        super(PerceptionDual, self).__init__(eta=eta, n_iter=n_iter)
        self.alpha, self.Gram_matrix = None, None

    def fit(self, X_train, Y_train):
        n_samples, dim = X_train.shape
        self.alpha, self.w, self.b = np.zeros(n_samples), np.zeros(dim), 0
        # Gram matrix
        self.Gram_matrix = np.dot(X_train, X_train.T)
        # Iteration
        i = 0
        while i < n_samples:
            # Judge end of iteration
            wx = np.sum(np.dot(self.Gram_matrix[i, :] , self.alpha * Y_train))
            # print(wx)
            if Y_train[i] * (wx + self.b) <= 0:
                # a <- a + eta, b <- b + eta * y_i
                self.alpha[i] += self.eta
                self.b += self.eta * Y_train[i]
                print('Iteration: %r/%r' % (i, n_samples))
                i = 0
            else:
                i += 1

        self.w = np.sum(X_train * np.tile((self.alpha * Y_train).reshape((n_samples, 1)), (1, dim)), axis=0)

def _GenerateData():
    import random
    m, n_train, n_val, interval = 2, 5, 2, 1
    X_train, X_val, Y_train, Y_val = [], [], [], []
    color = ['c', 'g', 'b', 'r']
    def _generateOne(X, Y, i):
        x, y, l = random.uniform((int(i / 2) + 0.1) * 10, (int(i / 2) + 0.9) * 10), random.uniform((i % 2 * interval + 0.1) * 10, (i % 2 * interval + 0.9) * 10), i
        X.append((x, y))
        Y.append((i - 0.5)*2)
        return x, y
    for i_ in range(m):
        for _ in range(n_train):
            x_, y_ = _generateOne(X_train, Y_train, i_)
            plt.scatter(x_, y_, s=100, c=color[i_])
        for _ in range(n_val):
            _generateOne(X_val, X_val, i_)
    return np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)

if __name__ == '__main__':
    model = PerceptionPrimitive()
    X_t, X_v, Y_t, Y_v = _GenerateData()
    model.fit(X_t, Y_t)
    plt.plot([10, 0], [(-model.b - model.w[0] * 10) / model.w[1], -model.b / model.w[1]])
    plt.grid()
    plt.xlim(0, 10)
    plt.ylim(0, 20)
    plt.show()

