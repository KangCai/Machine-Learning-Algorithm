# coding=utf-8

import numpy as np

class DTreeID3(object):

    def __init__(self, epsilon):
        self.tree = Node()
        self.epsilon = epsilon

    def fit(self, X_train, Y_train):
        A_recorder = np.arange(X_train.shape[1])
        self._train(X_train, Y_train, self.tree, A_recorder)

    def _train(self, A, D, node, AR):
        row, col = A.shape
        # 1. 结束条件：若 D 中所有实例属于同一类，决策树成单节点树，直接返回
        if np.any(np.bincount(D) == len(D)):
            node.y = D[0]
            return
        # 2. 结束条件：若 A 为空，则返回单结点树 T，标记类别为样本默认输出最多的类别
        if A.size == 0:
            node.y = np.argmax(np.bincount(D))
            return
        # 3. 计算特征集 A 中各特征对 D 的信息增益，选择信息增益最大的特征 A_g
        entropy = self._cal_entropy(D)
        max_info_gain = None
        g = None
        for j in range(col):
            a_cls = np.bincount(A[:, j])
            condition_entropy = 0
            for k in range(len(a_cls)):
                a_row_idxs = np.argwhere(A[:, j] == k)
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                condition_entropy += a_cls[k] / np.sum(a_cls) * self._cal_entropy(D[a_row_idxs].T[0])
            if max_info_gain is None or max_info_gain < entropy - condition_entropy:
                max_info_gain = entropy - condition_entropy
                g = j
        # 4. 结束条件：如果 A_g 的信息增益小于阈值 epsilon，决策树成单节点树，直接返回
        if max_info_gain <= self.epsilon:
            node.y = np.argmax(np.bincount(D))
            return
        # 5. 对于 A_g 的每一可能值 a_i，依据 A_g = a_i 将 D 分割为若干非空子集 D_i，将当前结点的标记设为样本数最大的 D_i 对应
            # 的类别，即对第 i 个子节点，以 D_i 为训练集，以 A - {A_g} 为特征集，递归调用以上步骤，得到子树 T_i，返回 T_i
        node.label = AR[g]
        a_cls = np.bincount(A[:, g])
        new_A, AR = np.hstack((A[:, 0:g], A[:, g+1:])), np.hstack((AR[0:g], AR[g+1:]))
        for k in range(len(a_cls)):
            a_row_idxs = np.argwhere(A[:, g] == k).T[0].T
            child = Node()
            node.append(child)
            A_child, D_child= new_A[a_row_idxs, :], D[a_row_idxs]
            self._train(A_child, D_child, child, AR)

    def _cal_entropy(self, D):
        statistic = np.bincount(D)
        prob = statistic / np.sum(statistic)
        prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
        # H(D) = -SUM(p_i * log(p_i))
        entropy = -np.sum(prob * np.log2(prob))
        return entropy


class Node(object):

    def __init__(self):
        self.label = None
        self.x = None
        self.child = []
        self.y = None
        self.data = None

    def append(self, child):
        self.child.append(child)


datalabel = np.array(['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
train_sets = np.array([
                    ['青年', '否', '否', '一般', '否'],
                    ['青年', '否', '否', '好', '否'],
                    ['青年', '是', '否', '好', '是'],
                    ['青年', '是', '是', '一般', '是'],
                    ['青年', '否', '否', '一般', '否'],
                    ['中年', '否', '否', '一般', '否'],
                    ['中年', '否', '否', '好', '否'],
                    ['中年', '是', '是', '好', '是'],
                    ['中年', '否', '是', '非常好', '是'],
                    ['中年', '否', '是', '非常好', '是'],
                    ['老年', '否', '是', '非常好', '是'],
                    ['老年', '否', '是', '好', '是'],
                    ['老年', '是', '否', '好', '是'],
                    ['老年', '是', '否', '非常好', '是'],
                    ['老年', '否', '否', '一般', '否'],
                    ['青年', '否', '否', '一般', '是']])
map_table = {'青年': 0, '中年': 1, '老年': 2,
             '否': 0, '是': 1,
             '一般': 0, '好': 1, '非常好': 2}

def _visualization_DFS(node, layer=0):
    output_str = ['\n' + ' ' * 4 * layer, '%r+%r ' % (node.y, node.label)]
    if not node.child:
        return ''.join(output_str)
    for child in node.child:
        output_str.append(_visualization_DFS(child, layer=layer+1))
    return ''.join(output_str)

if __name__ == '__main__':
    model = DTreeID3(0.00001)
    row_, col_ = train_sets.shape
    train_sets_encode = np.array([[map_table[train_sets[i, j]] for j in range(col_)] for i in range(row_)])
    X_t, Y_t = train_sets_encode[:, :-1], train_sets_encode[:, -1]
    model.fit(X_t, Y_t)
    print(_visualization_DFS(model.tree))


