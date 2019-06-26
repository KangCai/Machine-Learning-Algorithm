# coding=utf-8

import numpy as np

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
            self.kd_tree = KDTree(centers, range(0, len(centers)))
        row, col = X.shape
        assignments = np.zeros(row)
        for i in range(row):
            nn_center_nodes = self.kd_tree.search(X[i], 1)
            assignments[i] = nn_center_nodes[0].data
        return assignments

class KDTree(object):

    def __init__(self, X, DATA):
        # Attach extra data to point.
        point_container_list = []
        row, col = X.shape
        for i in range(row):
            point_container_list.append(PointContainer(X[i], DATA[i]))
        # Create KD tree.
        self.kd_node = self._create_kd_tree(point_container_list, col)

    def _create_kd_tree(self, points, dim, i=0):
        """

        :param points:
        :param i:
        :return:
        """
        if len(points) > 1:
            points.sort(key=lambda p: p.x[i])
            i = (i + 1) % dim
            half = len(points) >> 1
            return self._create_kd_tree(points[:half], dim, i), self._create_kd_tree(points[half+1:], dim, i), points[half]
        elif len(points) == 1:
            return None, None, points[0]

    def search(self, x, k):
        """
        This k is not the k in 'kmeans'.
        :param x:
        :param k:
        :return:
        """
        p = PointContainer(x, None)
        dim = len(x)
        return self._search_kd_tree(self.kd_node, p, k, dim, lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(dim)))

    def _search_kd_tree(self, kd_p, p, k, dim, dist_func, i=0, heap=None):
        import heapq
        is_root = not heap
        if is_root:
            heap = []
        if kd_p and isinstance(kd_p, tuple) and len(kd_p) == 3:
            mid_kd_p = kd_p[2]
            dist = dist_func(p.x, mid_kd_p.x)
            dx = mid_kd_p.x[i] - p.x[i]
            if len(heap) < k:
                heapq.heappush(heap, (-dist, mid_kd_p))
            elif dist < -heap[0][0]: # -heap[0][0] is the maximum distance in heap.
                heapq.heappushpop(heap, (-dist, mid_kd_p))
            i = (i + 1) % dim
            self._search_kd_tree(kd_p[dx < 0], p, k, dim, dist_func, i, heap)
            if dx * dx < -heap[0][0]: # -heap[0][0] is the maximum distance in heap.
                self._search_kd_tree(kd_p[dx >= 0], p, k, dim, dist_func, i, heap)
        if is_root:
            nn_result = sorted((-h[0], h[1]) for h in heap)
            return [n[1] for n in nn_result]

class PointContainer(object):

    def __init__(self, x, data):
        self.x = x
        self.data = data

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