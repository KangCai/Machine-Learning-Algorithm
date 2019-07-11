# coding=utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
import re
import time

class SVMModel(object):
    """
    SVM model
    """
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.00001):
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.kernel_func_list = {
            'linear': self._kernel_linear,
            'quadratic': self._kernel_quadratic,
        }
        self.kernel_func = self.kernel_func_list[kernel_type]
        self.C = C
        self.epsilon = epsilon
        self.alpha = None

    def fit(self, X_train, Y_train):
        """
        Training model
        :param X_train: shape = num_train, dim_feature
        :param Y_train: shape = num_train, 1
        :return: loss_history
        """
        n, d = X_train.shape[0], X_train.shape[1]
        self.alpha = np.zeros(n)
        # Iteration
        for i in range(self.max_iter):
            diff = self._iteration(X_train, Y_train)
            if i % 100 == 0:
                print('Iter %r / %r, Diff %r' % (i, self.max_iter, diff))
            if diff < self.epsilon:
                break

    def _iteration(self, X_train, Y_train):
        alpha = self.alpha
        alpha_prev = np.copy(alpha)
        n = alpha.shape[0]
        for j in range(n):
            # Find i not equal to j randomly
            i = j
            for _ in range(1000):
                if i != j:
                    break
                i = random.randint(0, n - 1)
            x_i, x_j, y_i, y_j = X_train[i, :], X_train[j, :], Y_train[i], Y_train[j]
            # Define the similarity of instances. K11 + K22 - 2K12
            k_ij = self.kernel_func(x_i, x_i) + self.kernel_func(x_j, x_j) - 2 * self.kernel_func(x_i, x_j)
            if k_ij == 0:
                continue
            a_i, a_j = alpha[i], alpha[j]
            # Calculate the boundary of alpha
            L, H = self._cal_L_H(self.C, a_j, a_i, y_j, y_i)
            # Calculate model parameters
            self.w = np.dot(X_train.T, np.multiply(alpha, Y_train))
            self.b = np.mean(Y_train - np.dot(self.w.T, X_train.T))
            # Iterate alpha_j and alpha_i according to 'Delta W(a_j)'
            E_i = self.predict(x_i) - y_i
            E_j = self.predict(x_j) - y_j
            alpha[j] = a_j + (y_j * (E_i - E_j) * 1.0) / k_ij
            alpha[j] = min(H, max(L, alpha[j]))
            alpha[i] = a_i + y_i * y_j * (a_j - alpha[j])
        diff = np.linalg.norm(alpha - alpha_prev)
        return diff

    def _kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def _kernel_quadratic(self, x1, x2):
        return np.dot(x1, x2.T) ** 2

    def _cal_L_H(self, C, a_j, a_i, y_j, y_i):
        if y_i != y_j:
            L = max(0, a_j - a_i)
            H = min(C, C - a_i + a_j)
        else:
            L = max(0, a_i + a_j - C)
            H = min(C, a_i + a_j)
        return L, H

    def predict_raw(self, X):
        return np.dot(self.w.T, X.T) + self.b

    def predict(self, X):
        return np.sign(np.dot(self.w.T, X.T) + self.b).astype(int)

def _GenerateData():
    k, m, n_train, n_val = 5, 2, 5, 2
    X_train, X_val, Y_train, Y_val = [], [], [], []
    color = ['c', 'g', 'b', 'r']
    def _generateOne(X, Y, i):
        x, y, l = random.uniform((int(i / 2) + 0.1) * 10, (int(i / 2) + 0.9) * 10), random.uniform((i % 2 * 0.5 + 0.1) * 10, (i % 2 * 0.5 + 0.9) * 10), i
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
    model = SVMModel()
    X_t, X_v, Y_t, Y_v = _GenerateData()
    model.fit(X_t, Y_t)
    plt.plot([10, 0], [(-model.b-model.w[0]*10)/model.w[1], -model.b/model.w[1]])
    alpha_idx = np.where(model.alpha > 0)[0]
    for i__ in range(len(model.alpha)):
        if model.alpha[i__] > 0.1:
            plt.scatter(X_t[i__, 0], X_t[i__, 1], color='', marker='o', s=300, edgecolors='r')
    for i__ in range(X_t.shape[0]):
        plt.text(X_t[i__, 0], X_t[i__, 1], s='%0.1f, %0.1f' % (model.alpha[i__], model.predict_raw(X_t[i__, :])))
    plt.xlim(0, 10)
    plt.ylim(0, 15)
    plt.show()