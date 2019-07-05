# coding=utf-8

import numpy as np


class PerceptionPrimitive(object):

    def __init__(self, eta=0.1, n_iter=50):
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
        return np.dot(x.T, self.w) + self.b

    def predict(self, x):
        return np.sign(self.predict_raw(x))


class PerceptionDual(PerceptionPrimitive):

    def __init__(self, eta=0.1, n_iter=50):
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
            print(wx)
            if Y_train[i] * (wx + self.b) <= 0:
                # a <- a + eta, b <- b + eta * y_i
                self.alpha += self.eta
                self.b += self.eta * Y_train[i]
            else:
                i += 1

        self.w = np.sum(X_train * self.alpha * Y_train, axis=0)

if __name__ == '__main__':
    perception = PerceptionDual()
