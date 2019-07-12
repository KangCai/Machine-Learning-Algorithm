# coding=utf-8

import numpy as np


def pca(data_mat, top_n_feat=1, draw=False):
    """
    pca特征维度压缩函数
    :param data_mat: 数据集矩阵
    :param top_n_feat: 需要保留的特征维度，即要压缩成的维度数
    :param draw:
    :return:
    """
    # 求数据矩阵每一列的均值
    mean_val = np.mean(data_mat, axis=0)
    # 数据矩阵每一列特征减去该列的特征均值
    mean_removed = data_mat - mean_val
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    cov_mat = np.cov(mean_removed, rowvar=False)
    # 计算协方差矩阵的特征值eig_val及对应的特征向量eig_vec
    eig_val, eig_vec = np.linalg.eig(np.mat(cov_mat))
    # argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eig_val_ind = np.argsort(eig_val)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eig_val_ind = eig_val_ind[:-(top_n_feat + 1):-1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    red_eig_vec = eig_vec[:, eig_val_ind]
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    low_dim_data_mat = mean_removed * red_eig_vec
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    recon_mat = (low_dim_data_mat * red_eig_vec.T) + mean_val
    # 画图
    if draw:
        import matplotlib.pyplot as plt
        color = np.array(['r', 'g', 'b', 'm', 'c'])
        plt.scatter(data_mat[:, 0], data_mat[:, 1], c=color)
        plt.scatter(np.array(recon_mat)[:, 0], np.array(recon_mat)[:, 1], marker='s', c=color)
        x_min, x_max, y_min, y_max = np.min(data_mat[:, 0]) - 1, np.max(data_mat[:, 0]) + 1, \
            np.min(data_mat[:, 1]) - 1, np.max(data_mat[:, 1]) + 1
        if top_n_feat == 1:
            for i in range(data_mat.shape[0]):
                plt.plot([np.array(recon_mat)[:, 0], data_mat[:, 0]], [np.array(recon_mat)[:, 1], data_mat[:, 1]], linestyle=':')
            w = float(red_eig_vec[1][0] / red_eig_vec[0][0])
            b = mean_val[1] - mean_val[0] * w
            plt.plot([x_min, x_max], [x_min * w + b, x_max * w + b], linestyle='--')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('PCA example')
        plt.show()
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return low_dim_data_mat, recon_mat


if __name__ == '__main__':
    data = np.array([[1, 0], [3, 2], [2, 2], [0, 2], [1, 3]])
    print('Raw data: %r\n' % data[:, 0])
    lowDData, recon = pca(data, draw=True)
    print('PCA data: %r\n' % lowDData)
    print('Reconstructed data: %r' % recon)
