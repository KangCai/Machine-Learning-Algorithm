# coding=utf-8

from numpy import *

def pca(dataMat, topNfeat=1):
    """
    pca特征维度压缩函数
    :param dataMat: 数据集矩阵
    :param topNfeat: 需要保留的特征维度，即要压缩成的维度数
    :return:
    """
    #求数据矩阵每一列的均值
    meanVals = mean(dataMat, axis=0)
    #数据矩阵每一列特征减去该列的特征均值
    meanRemoved = dataMat - meanVals
    #计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    covMat = cov(meanRemoved, rowvar=0)
    #计算协方差矩阵的特征值eigVals及对应的特征向量eigVects
    eigVals, eigVects = linalg.eig(mat(covMat))
    #argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = argsort(eigVals)
    #从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    #将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:,eigValInd]
    #将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    #利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    #返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return lowDDataMat, reconMat

def draw(RawData, lowData):
    import matplotlib.pyplot as plt
    color = array(['r', 'g', 'b', 'm', 'c'])
    plt.scatter(RawData[:,0], RawData[:,1], c=color)
    plt.scatter(array(lowData)[:, 0], array(lowData)[:, 1], marker='s', c=color)
    plt.show()

if __name__ == '__main__':
    data = array([[1,0],[3,2],[2,2],[0,2],[1,3]])
    print(data[:,0])
    lowDData, recon = pca(data)
    print(lowDData)
    print(recon)
    draw(data, recon)