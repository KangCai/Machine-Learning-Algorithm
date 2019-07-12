# coding=utf-8

import numpy as np
import decision_tree

class RandomForest(object):

    def __init__(self, tree_count=10):
        self.tree_list = []
        self.tree_count = tree_count

    def fit(self, X_train, Y_train):
        # Generate decision tree
        for i in range(self.tree_count):
            dt_CART = decision_tree.DTreeCART()
            # Bagging data
            n, m = X_train.shape
            sample_idx = np.random.permutation(n)
            feature_idx = np.random.permutation(m)[:int(np.sqrt(m))]
            X_t_ = X_train[:, feature_idx]
            X_t_, Y_t_ = X_t_[sample_idx, :], Y_train[sample_idx]
            # Train
            dt_CART.fit(X_t_, Y_t_)
            self.tree_list.append((dt_CART, feature_idx))
            print('=' * 10 + ' %r/%r tree trained ' % (i + 1, self.tree_count) + '=' * 10)
            # print(dt_CART.visualization())

    def predict(self, X):
        output_matrix = np.zeros((self.tree_count, X.shape[0]))
        output_label = np.zeros(X.shape[0])
        for i, (tree, feature_idx) in enumerate(self.tree_list):
            output_matrix[i, :] = tree.predict(X[:, feature_idx])
        for col in range(output_matrix.shape[1]):
            output_label[col] = np.argmax(np.bincount(output_matrix[:, col].astype(int)))
        return output_label.astype(int)

datalabel = np.array(['年龄(特征1)', '有工作(特征2)', '有自己的房子(特征3)', '信贷情况(特征4)', '类别(标签)'])
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

if __name__ == '__main__':
    model = RandomForest()
    train_sets_encode = np.array([[map_table[train_sets[i, j]] for j in range(train_sets.shape[1])] for i in range(train_sets.shape[0])])
    X_t, Y_t = train_sets_encode[:, :-1], train_sets_encode[:, -1]
    model.fit(X_t, Y_t)
    print('Ground truth   : %r' % (Y_t,))
    print('Label predicted: %r' % (model.predict(X_t),))


