# coding=utf-8

import numpy as np

class KmeansModel(object):

    def cluster(self, x, k):
        """
        x: shape(n, d)
        :param x:
        :param k:
        :return:
        """
        row, col = x.shape
        # Calculate the value range of dimension.
        max_val, min_val = np.max(x, 0), np.min(x, 0)
        # Init centers according to value range.
        centers = np.random.rand(k, col) * (max_val - min_val) + min_val
        # Iteration of assigning cluster ID to each point
        assignments = self._assign_points(x, centers)
        old_assignments = None
        while (assignments != old_assignments).any():
            # New centers
            centers = self._update_centers(x, k, assignments)
            # Store old assignments
            old_assignments = assignments
            # New assignments
            assignments = self._assign_points(x, centers)
        return zip(assignments, x)

    def _update_centers(self, x, k, assignments):
        # Statistic
        store_center_points = {i : [] for i in range(k)}
        for i, p in zip(assignments, x):
            store_center_points[i].append(p)
        # Calculate new centers
        row, col = x.shape
        center_points = np.zeros((k, col))
        for i, points in store_center_points.items():
            store_center_points[i] = np.array(store_center_points[i])
            center_points[i] = np.mean(store_center_points[i], axis=0)
        return center_points

    def _assign_points(self, x, centers):
        row, col = x.shape
        assignments = np.zeros(row)
        for i in range(row):
            dists = np.linalg.norm(x[i] - centers, axis=1)
            assignments[i] = np.argmin(dists)
        return assignments

if __name__ == '__main__':
    model_kmeans = KmeansModel()
    for label, point in model_kmeans.cluster(np.array([[0, 1], [2, 2], [4, 5]]), 2):
        print(label, point)