# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import time


class OneHotFeatureExtraction(object):
    """
    One-hot编码特征提取
    """
    def __init__(self, kw_set):
        # 关键字集合，即哪些单词是我们要当做是特征属性的单词
        self.kw_set = dict(zip(list(kw_set), range(len(kw_set))))

    def extract(self, input_text_words):
        """
        独热编码特征提取
        :param input_text_words:
        :return:
        """
        for word in input_text_words:
            if word not in self.kw_set:
                continue



class RegressionModel(object):
    """
    逻辑回归模型
    """
    def __init__(self):
        return

    def train(self, data):
        for label, feature in data:
            pass

    def test(self, input_feature):
        return


def feature_batch_extraction(d_list, kw_set):
    """

    :param d_list:
    :param kw_set:
    :return:
    """
    feature_extraction = OneHotFeatureExtraction(kw_set)
    feature_data_list = []
    for label, words in d_list:
        feature = feature_extraction.extract(words)
        feature_data_list.append((label, feature))
    return feature_data_list


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
        data = feature_batch_extraction(data_list, key_word_set)
