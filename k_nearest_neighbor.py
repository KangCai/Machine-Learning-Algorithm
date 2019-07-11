# coding=utf-8

import numpy as np
import util_kd_tree as kdtree

class KNNModel_Naive(object):

    def __init__(self, k, X_train, Y_train):
        """
        Train model implicitly. No explicit training process.
        :param:
        :param X_train:
        :param Y_train:
        """
        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train

    def validate(self, X_val, Y_val):
        """
        Validate the trained model.
        :param X_val:
        :param Y_val:
        :return:
        """
        label_list = []
        row, col = X_val.shape
        for i in range(row):
            dist = np.linalg.norm(X_val[i, :] - self.X_train, axis=1)
            res_idx = np.argsort(dist)[:self.k]
            res = [self.Y_train[i] for i in res_idx]
            label = np.argmax(np.bincount(res))
            label_list.append(label)
        label_array = np.array(label_list)
        accuracy = len(np.where(label_array == Y_val)) / row
        return accuracy, label_array

class KNNModel_Heap(KNNModel_Naive):

    def validate(self, X_val, Y_val):
        """
        Validate the trained model.
        :param X_val:
        :param Y_val:
        :return:
        """
        import heapq
        label_list = []
        row, col = X_val.shape
        for i in range(row):
            heap = []
            dist = np.linalg.norm(X_val[i, :] - self.X_train, axis=1)
            for idx, d in enumerate(dist):
                if len(heap) < self.k:
                    heapq.heappush(heap, (-d, idx))
                elif d < -heap[0][0]: # -heap[0][0] is the maximum distance in heap.
                    heapq.heappushpop(heap, (-d, idx))
            res = [self.Y_train[r[1]] for r in heap]
            label = np.argmax(np.bincount(res))
            label_list.append(label)
        label_array = np.array(label_list)
        accuracy = len(np.where(label_array == Y_val)) / row
        return accuracy, label_array

class KNNModel_KDTree(KNNModel_Naive):

    def __init__(self, k, X_train, Y_train):
        """
        Train model.
        :param X_train:
        :param Y_train:
        """
        super().__init__(k, X_train, Y_train)
        self.kd_node = kdtree.KDTree(X_train, range(X_train.shape[0]))

    def validate(self, X_val, Y_val):
        """
        Validate the trained model.
        :param X_val:
        :param Y_val:
        :return:
        """
        label_list = []
        row, col = X_val.shape
        for i in range(row):
            nn_nodes = self.kd_node.search(X_val[i], self.k)
            res = [self.Y_train[n.data] for n in nn_nodes]
            label = np.argmax(np.bincount(res))
            label_list.append(label)
        label_array = np.array(label_list)
        accuracy = len(np.where(label_array == Y_val)) / row
        return accuracy, label_array

def _TestEfficiency():
    import random
    X_train, Y_train = [], []
    # k=5, n=100000, m=200: 2.4(Naive), 6.0(Heap), 0.05(KDTree)
    k_, n_, t_num, d = 5, 1000, 200, 500
    # Train
    for _ in range(n_):
        X_train.append([random.uniform(0, 50) for _ in range(d)])
        Y_train.append(random.randint(0, 1))
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    # Validate
    X_val, Y_val = [], []
    for _ in range(t_num):
        X_val.append([random.uniform(0, 50) for _ in range(d)])
        Y_val.append(random.randint(0, 1))
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    for model_class in (KNNModel_Naive, KNNModel_Heap, KNNModel_KDTree):
        model = model_class(k_, X_train, Y_train)
        print('=' * 5 + 'Model type %r' % (model,) + '=' * 5)
        t1 = time.clock()
        print('Accuracy is %r' % (model.validate(X_val, Y_val),))
        print('Total used time is %r' % (time.clock() - t1,))

def _TestVisualization():
    import matplotlib.pyplot as plt
    k, m, n_train, n_val = 5, 4, 5, 2
    X_train, X_val, Y_train, Y_val = [], [], [], []
    color = ['c', 'g', 'b', 'r']
    for i in range(m):
        for _ in range(n_train):
            x, y, l = random.uniform(int(i/2)+0.1, int(i/2)+0.9), random.uniform(i%2+0.1, i%2+0.9), i
            X_train.append((x, y))
            Y_train.append(i)
            plt.scatter(x, y, s=100, c=color[i])
        for _ in range(n_val):
            x, y, l = random.uniform(int(i/2)+0.1, int(i/2)+0.9), random.uniform(i%2+0.1, i%2+0.9), i
            X_val.append((x, y))
            Y_val.append(i)
    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)
    for model_class in (KNNModel_KDTree,):
        model = model_class(k, X_train, Y_train)
        accuracy, label_val = model.validate(X_val, Y_val)
        for i in range(len(label_val)):
            plt.scatter(X_val[i, 0], X_val[i, 1], alpha=0.3, s=100, c=color[Y_val[i]], linewidths=2, edgecolors=color[label_val[i]])
    plt.grid()
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.show()

if __name__ == '__main__':
    import random, time
    _TestEfficiency()
    # Visualization
    # _TestVisualization()


