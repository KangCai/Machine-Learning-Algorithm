# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import re
import time

class NaiveBayesClassificationModel(object):
    """
    朴素贝叶斯模型
    """
    def __init__(self, kw_set):
        # 关键字集合，即哪些单词是我们要当做是特征属性的单词
        self.kw_set = kw_set
        # P(类别) 样本类别本身在样本中出现的先验概率
        self.label_prior_prob = dict()
        # P(关键字|类别) 这一条件概率
        self.kw_posterior_prob = dict()

    def train(self, data):
        """
        训练模型
        :param data: 以 [[label] [input_text_words]] 的形式构成的list
        :return: None
        """
        # 计算条件概率 P(关键字|类别)
        for label, input_text_words in data:
            if label not in self.kw_posterior_prob:
                self.kw_posterior_prob[label] = dict()
            if label not in self.label_prior_prob:
                self.label_prior_prob[label] = 0
            self.label_prior_prob[label] += 1
            for word in input_text_words:
                if word not in self.kw_set:
                    continue
                if word not in self.kw_posterior_prob[label]:
                    self.kw_posterior_prob[label][word] = 0
                self.kw_posterior_prob[label][word] += 1
        for label, kw_posterior_prob in self.kw_posterior_prob.items():
            for word in self.kw_set:
                if word in kw_posterior_prob:
                    self.kw_posterior_prob[label][word] /= self.label_prior_prob[label] * 1.0
                else:
                    self.kw_posterior_prob[label][word] = 0
        # 样本类别本身在样本中出现的先验概率 P(类别)
        for label in self.label_prior_prob:
            self.label_prior_prob[label] /= len(data) * 1.0

    def predict(self, input_text):
        """
        预测过程
        :param input_text: 处理过后的单词集合
        :return:
        """
        predicted_label = None
        max_prob = None
        for label in self.label_prior_prob:
            prob = 1.0
            for word in self.kw_set:
                if word in input_text:
                    prob *= self.kw_posterior_prob[label][word]
                else:
                    prob *= 1 - self.kw_posterior_prob[label][word]
            if max_prob is None or prob > max_prob:
                predicted_label = label
                max_prob = prob
        return predicted_label

    def validate(self, data):
        """
        验证模型效果
        :param data: 以 [[label] [input_text_words]] 的形式形构成的list
        :return:
        """
        # 计算 正误频次混淆矩阵
        mtc = {label_1: {label_2: 0 for label_2 in self.label_prior_prob} for label_1 in self.label_prior_prob}
        for gd_label, input_text_words in data:
            predicted_label = self.predict(input_text_words)
            mtc[gd_label][predicted_label] += 1
        # 计算 准确率混淆矩阵 和 总准确率
        acc = 0
        for gd_label in mtc:
            for predicted_label in mtc[gd_label]:
                mtc[gd_label][predicted_label] /= len(data) * self.label_prior_prob[gd_label]
                if predicted_label == gd_label:
                    acc += mtc[gd_label][predicted_label] * self.label_prior_prob[gd_label]
        return acc, mtc


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
    k_fold_data_list = [list() for _ in range(k)]
    for label, label_data_list in label_data_dict.items():
        # 打乱
        seq = np.random.permutation(range(len(label_data_list)))
        # 切分
        fold_instance_count = int(len(label_data_list) / k)
        for i in range(k):
            for idx in range(i * fold_instance_count, (i+1) * fold_instance_count):
                k_fold_data_list[i].append(label_data_list[seq[idx]])
    return k_fold_data_list


def draw(kw_count_list):
    """
    画出单词频次分布情况，为选择一个合适的截断提供直观的依据
    :param kw_count_list:
    :return:
    """
    key_word_list = list()
    count_list = list()
    for key_word, count in kw_count_list:
        key_word_list.append(key_word)
        count_list.append(count)

    plt.figure(figsize=(8, 4))
    plt.xlabel('Rank of key word')
    plt.ylabel('count of doc containing key word')
    plt.plot(key_word_list, count_list)
    xt_list = range(0, len(count_list), 1000)
    plt.xticks(xt_list, xt_list)
    plt.xlim(0, len(count_list))
    plt.ylim(0, 0.35)
    plt.grid(True)


if __name__ == '__main__':
    file_name = '../data/SMSSpamCollection.txt'
    raw_data_list = data_pre_process(file_name)
    fold_count = 4
    fold_data_list = shuffle(raw_data_list, fold_count)
    acc_average = 0
    cut_off = 5000
    t1 = time.clock()
    for fold, data_list in enumerate(fold_data_list):
        word_count_list, key_word_set = statistic_key_word(data_list, cut_off=cut_off)
        nbc_model = NaiveBayesClassificationModel(key_word_set)
        nbc_model.train(data_list)
        accuracy, metric = nbc_model.validate(data_list)
        acc_average += accuracy
        print('Fold %r/%r - Acc:%r Metric:%r' % (fold+1, fold_count, accuracy, metric))
    print('Average Acc:%r Average Cost Time:%r' % (acc_average / len(fold_data_list), (time.clock() - t1) / len(fold_data_list)))