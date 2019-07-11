# coding=utf-8

import numpy as np
import util_kd_tree as kdtree

class KmeansModel(object):

    def cluster(self, X, k):
        """
        X: shape(n, d)
        :param X:
        :param k:
        :return:
        """
        row, col = X.shape
        # Init centers according to value range.
        rand_idx = random.sample(range(0, row), k)
        centers = X[rand_idx, :]
        # Iteration of assigning cluster ID to each point
        assignments = self._assign_points(X, centers)
        old_assignments = None
        count = 0
        while (assignments != old_assignments).any():
            count += 1
            # New centers
            centers = self._update_centers(X, k, assignments)
            # Store old assignments
            old_assignments = assignments
            # New assignments
            assignments = self._assign_points(X, centers)
        return zip(assignments, X)

    def _update_centers(self, X, k, assignments):
        # Statistic
        store_center_points = {i : [] for i in range(k)}
        for i, p in zip(assignments, X):
            store_center_points[i].append(p)
        # Calculate new centers
        row, col = X.shape
        center_points = np.zeros((k, col))
        for i, points in store_center_points.items():
            store_center_points[i] = np.array(store_center_points[i])
            center_points[i] = np.mean(store_center_points[i], axis=0)
        return center_points

    def _assign_points(self, X, centers):
        row, col = X.shape
        assignments = np.zeros(row)
        for i in range(row):
            dists = np.linalg.norm(X[i] - centers, axis=1)
            assignments[i] = np.argmin(dists)
        return assignments

class KmeansModelKDTree(KmeansModel):

    def _assign_points(self, X, centers):
        if not getattr(self, 'kd_tree', None):
            self.kd_tree = kdtree.KDTree(centers, range(0, len(centers)))
        row, col = X.shape
        assignments = np.zeros(row)
        for i in range(row):
            nn_center_nodes = self.kd_tree.search(X[i], 1)
            assignments[i] = nn_center_nodes[0].data
        return assignments


if __name__ == '__main__':
    import random, time
    a_ = []
    # k = 3
    # 100: Naive kmeans 0.004s; KD tree kmeans 0.004s
    # 1,000: Naive kmeans 0.19s; KD tree kmeans 0.05s
    # 10,000: Naive kmeans 4.9s; KD tree kmeans 0.4s
    # 100,000: Naive kmeans 66.6s; KD tree kmeans 4.2s
    for _ in range(10000):
        a_.append((random.uniform(0, 100), random.uniform(0, 100)))
    a_ = np.array(a_)
    k_ = 3
    print('=' * 5 + ' Naive kmeans ' + '=' * 5)
    model = KmeansModel()
    t1 = time.clock()
    model.cluster(a_, k_)
    print('Total used time: %r s' % (time.clock() - t1))
    # for label, point in res:
    #     print(label, point)
    print('=' * 5 + ' KD tree kmeans ' + '=' * 5)
    model = KmeansModelKDTree()
    t1 = time.clock()
    res = model.cluster(a_, k_)
    print('Total used time: %r s' % (time.clock() - t1))
    # for label, point in res:
    #     print(label, point)