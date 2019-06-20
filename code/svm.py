# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import re
import time

class SVMModel(object):
    """
    SVM模型
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.C = None

    def train(self, x_train, y_train):
        """
        模型训练
        :param x_train: shape = num_train, dim_feature
        :param y_train: shape = num_train, 1
        :return: loss_history
        """
        num_train, dim_feature = x_train.shape
        # w * x + b
        for i in range()

