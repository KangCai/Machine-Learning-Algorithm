# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import re
import time

class RegressionModel(object):
    """
    逻辑回归模型
    """
    def __init__(self):
        self.W = None

    def train(self, x_train, y_train, learning_rate=0.1, num_iters=10000):
        """
        模型训练
        :param x_train: shape = num_train, dim_feature
        :param y_train: shape = num_train, 1
        :param learning_rate
        :param num_iters
        :return:
        """
        num_train, dim_feature = x_train.shape
        # w * x + b
        x_train_ = np.hstack((x_train, np.ones((num_train, 1))))
        self.W = 0.001 * np.random.randn(dim_feature + 1, 1)
        loss_history = []
        for i in range(num_iters+1):
            # linear transformation: w * x + b
            g = np.dot(x_train_, self.W)
            # sigmoid: 1 / (1 + e**-x)
            h = 1 / (1 + np.exp(-g))
            # cross entropy: 1/m * sum((y*np.log(h) + (1-y)*np.log((1-h))))
            loss = -np.sum(y_train * np.log(h) + (1 - y_train) * np.log(1 - h)) / num_train
            loss_history.append(loss)
            # dW = cross entropy' = 1/m * sum(h-y) * x
            dW = x_train_.T.dot(h - y_train) / num_train
            # W = W - dW
            self.W -= learning_rate * dW
            # debug
            if i % 100 == 0:
                print('Iters: %r/%r Loss: %r' % (i, num_iters, loss))
        return loss_history

    def test(self, input_feature):
        """
        预测过程
        :param input_feature: 处理过后的单词集合特征
        :return:
        """
        return

    def validate(self, x_val, y_val):
        """
        验证模型效果
        :param x_val: shape = num_val, dim_feature
        :param y_val: shape = num_val, 1
        :return: accuracy, metric
        """
        num_val, dim_feature = x_val.shape
        x_val_ = np.hstack((x_val, np.ones((num_val, 1))))
        # linear transformation: w * x + b
        g = np.dot(x_val_, self.W)
        # sigmoid: 1 / (1 + e**-x)
        h = 1 / (1 + np.exp(-g))
        # predict
        y_val_ = h
        y_val_[y_val_ >= 0.5] = 1
        y_val_[y_val_ < 0.5] = 0
        true_positive = len(np.where(((y_val_ == 1).astype(int) + (y_val == 1).astype(int) == 2) == True)[0]) * 1.0 / num_val
        true_negative = len(np.where(((y_val_ == 0).astype(int) + (y_val == 0).astype(int) == 2) == True)[0]) * 1.0 / num_val
        false_positive = len(np.where(((y_val_ == 1).astype(int) + (y_val == 0).astype(int) == 2) == True)[0]) * 1.0 / num_val
        false_negative = len(np.where(((y_val_ == 0).astype(int) + (y_val == 1).astype(int) == 2) == True)[0]) * 1.0 / num_val
        negative_instance = true_negative + false_positive
        positive_instance = false_negative + true_positive
        metric = np.array([[true_negative / negative_instance, false_positive / negative_instance],
                           [false_negative / positive_instance, true_positive / positive_instance]])
        accuracy = true_positive + true_negative
        return accuracy, metric

def feature_batch_extraction(d_list, kw_set):
    """
    特征批量提取
    :param d_list:
    :param kw_set:
    :return:
    """
    kw_2_idx_dict = dict(zip(list(kw_set), range(len(kw_set))))
    feature_data = np.zeros((len(d_list), len(kw_set)))
    label_data = np.zeros((len(d_list), 1))
    for i in range(len(d_list)):
        label, words = d_list[i]
        for word in words:
            if word in kw_2_idx_dict:
                feature_data[i, kw_2_idx_dict[word]] = 1
        label_data[i] = 1 if label == 'spam' else 0
    return feature_data, label_data


def data_pre_process(data_file_name):
    """
    句子切分成单词，由于是英文，所以这里处理方式比较暴力，按照空格和除'之外的符号来切分了；然后全部转小写
    :param data_file_name:
    :return:
    """
    fh = open(data_file_name, encoding='utf-8')
    data = list()
    for line in fh.readlines():
        label_text_pair = line.split('\t')
        word_list = re.split('[^\'a-zA-Z]', label_text_pair[1])
        word_in_doc_set = set()
        for raw_word in word_list:
            word = raw_word.lower()
            if word == '':
                continue
            word_in_doc_set.add(word)
        # 组成 [[label] [input_text_words]] 的形式
        data.append((label_text_pair[0], list(word_in_doc_set)))
    return data


def statistic_key_word(data, cut_off=None):
    """
    统计单词出现的文档次数，并试图把直观上无效（出现在的文档数目较少）的单词去掉
    :param data: data in one line: [label] [input_text]
    :param cut_off:
    :return:
    """
    # 针对各个单词，统计单词出现的文档次数
    w_dict = dict()
    total_doc_count = len(data)
    for _, word_in_doc_set in data:
        for word in word_in_doc_set:
            if word not in w_dict:
                w_dict[word] = 0
            w_dict[word] += 1
    for word in w_dict.keys():
        w_dict[word] /= total_doc_count * 1.0
    # 按出现文档次数从高到低，对单词进行排序
    w_count_list = sorted(w_dict.items(), key=lambda d: d[1], reverse=True)
    # 截断后续出现次数过低的单词
    kw_set = set()
    cut_off_length = cut_off if cut_off else len(w_count_list)
    for word, _ in w_count_list[:cut_off_length]:
        kw_set.add(word)
    return w_count_list, kw_set


def shuffle(data, k):
    """
    切分并打乱，为模型的交叉验证做准备
    :param data:
    :param k:
    :return:
    """
    # 将数据按类别归类，目的是为了切分各个fold的时候，保证数据集合中类别分布平均一些
    label_data_dict = dict()
    for label, word_in_doc_set in data:
        if label not in label_data_dict:
            label_data_dict[label] = list()
        label_data_dict[label].append((label, word_in_doc_set))
    # 切分并打乱
    k_group_data_list = [list() for _ in range(k)]
    for label, label_data_list in label_data_dict.items():
        # 打乱
        seq = np.random.permutation(range(len(label_data_list)))
        # 切分
        fold_instance_count = int(len(label_data_list) / k)
        for i in range(k):
            for idx in range(i * fold_instance_count, (i+1) * fold_instance_count):
                k_group_data_list[i].append(label_data_list[seq[idx]])
    k_fold_data_list = list()
    for i in range(k):
        train_data = []
        for j in range(k):
            if i != j:
                train_data.extend(k_group_data_list[j])
        k_fold_data_list.append((train_data, k_group_data_list[i]))
    return k_fold_data_list

def draw_loss_list(loss_list):
    """
    画出单词频次分布情况，为选择一个合适的截断提供直观的依据
    :param loss_list:
    :return:
    """
    plt.figure(figsize=(8, 4))
    plt.xlabel('Train iteration')
    plt.ylabel('Loss')
    xt_list = range(0, len(loss_list[0][1]), 1000)
    print(len(loss_list[0][1]))
    for cut_off, loss in loss_list:
        print(len(loss))
        plt.plot(range(0, len(loss)), loss, label='cut off %r' % (cut_off,))
    plt.xticks(xt_list, xt_list)
    plt.xlim(1, len(loss_list[0][1]) + 1)
    plt.ylim(0, 0.7)
    plt.legend()
    plt.show()

def performance_with_cut_off():
    """

    :return:
    """
    file_name = '../data/SMSSpamCollection.txt'
    raw_data_list = data_pre_process(file_name)
    fold_count = 4
    fold_data_list = shuffle(raw_data_list, fold_count)
    loss_list = list()
    accuracy_list = list()
    metric_list = list()
    for cut_off in (200, 500, 2000, 5000, 7956):
        data_list = fold_data_list[0]
        train_data_list, test_data_list = data_list
        word_count_list, key_word_set = statistic_key_word(train_data_list, cut_off=cut_off)
        # Feature extraction
        train_feature, train_label = feature_batch_extraction(train_data_list, key_word_set)
        validate_feature, validate_label = feature_batch_extraction(test_data_list, key_word_set)
        # Train model
        lr_model = RegressionModel()
        loss_history = lr_model.train(train_feature, train_label, num_iters=10000)
        loss_list.append((cut_off, loss_history))
        accuracy, metric = lr_model.validate(validate_feature, validate_label)
        accuracy_list.append(accuracy)
        metric_list.append(metric)
    with open('../result/lr_loss_list.txt', 'w') as f:
        f.write(str(loss_list) + '\n')
        f.write(str(accuracy_list) + '\n')
        f.write(str(metric_list))
    with open('../result/lr_loss_list.txt') as f:
        loss_list = eval(f.readline())
        draw_loss_list(loss_list)
        accuracy_list = eval(f.readline())
        metric_list = eval(f.readline())
        print(accuracy_list)
        print(metric_list)

def performance_with_fold():
    """

    :return:
    """
    file_name = '../data/SMSSpamCollection.txt'
    raw_data_list = data_pre_process(file_name)
    fold_count = 4
    fold_data_list = shuffle(raw_data_list, fold_count)
    acc_average = 0
    cut_off = 500
    t1 = time.clock()
    for fold, data_list in enumerate(fold_data_list):
        train_data_list, test_data_list = data_list
        word_count_list, key_word_set = statistic_key_word(train_data_list, cut_off=cut_off)
        # Feature extraction
        train_feature, train_label = feature_batch_extraction(train_data_list, key_word_set)
        validate_feature, validate_label = feature_batch_extraction(test_data_list, key_word_set)
        # Train model
        lr_model = RegressionModel()
        loss_history = lr_model.train(train_feature, train_label)
        # Validate
        accuracy, metric = lr_model.validate(validate_feature, validate_label)
        acc_average += accuracy
        print('Fold %r/%r - Acc:%r Metric:%r' % (fold + 1, fold_count, accuracy, metric))
    print('Average Acc:%r Average Cost Time:%r' % (acc_average / len(fold_data_list),
            (time.clock() - t1) / len(fold_data_list)))

if __name__ == '__main__':
    performance_with_cut_off()

