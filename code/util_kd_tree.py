# coding=utf-8

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