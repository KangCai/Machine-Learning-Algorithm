# coding=utf-8

from numpy import *

def lda(c1, c2, topNfeat=1):
    """
    lda特征维度压缩函数
    :param c1: 第一类样本矩阵，每行是一个样本
    :param c2: 第二类样本矩阵，每行是一个样本
    :param topNfeat: 需要保留的特征维度，即要压缩成的维度数
    :return:
    """
    # 第一类样本均值
    m1 = mean(c1, axis=0)
    # 第二类样本均值
    m2 = mean(c2, axis=0)
    # 所有样本矩阵
    c = vstack((c1, c2))
    # 所有样本的均值
    m = mean(c, axis=0)
    # 第一类样本数
    n1 = c1.shape[0]
    # 第二类样本数
    n2 = c2.shape[0]
    # 求第一类样本的散列矩阵s1
    s1=0
    for i in range(0,n1):
        s1 += (c1[i,:]-m1).T*(c1[i,:]-m1)
    # 求第二类样本的散列矩阵 s2
    s2=0
    for i in range(0,n2):
        s2 += (c2[i,:]-m2).T*(c2[i,:]-m2)
    # 计算类内离散度矩阵Sw
    Sw = (n1*s1+n2*s2)/(n1+n2)
    # 计算类间离散度矩阵Sb
    Sb = (n1*(m-m1).T*(m-m1) + n2*(m-m2).T*(m-m2))/(n1+n2)
    # 求最大特征值对应的特征值和特征向量（重点）
    eigvalue, eigvector = linalg.eig(mat(Sw).I*Sb)
    # 对eigvalue从大到小排序，返回对应排序后的索引
    indexVec = argsort(-eigvalue)
    # 取出最大的特征值对应的索引
    nLargestIndex = indexVec[:topNfeat]
    # 取出最大的特征值对应的特征向量
    W = eigvector[:,nLargestIndex]
    # 返回降维后结果
    return W

if __name__ == '__main__':
    data1 = [[1, 0], [3, 2]]
    data2 = [[0, 1], [1, 3]]
    w = lda(array(data1), array(data2), 2)
    print(w)
