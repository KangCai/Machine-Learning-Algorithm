# coding=utf-8

from numpy import *
import matplotlib.pyplot as plt
import re


class NaiveBayesClassificationModel(object):
    """
    NBC Model.
    """
    def __init__(self, kw_set):
        """
        Init feature and label matrix.
        """
        self.kw_set = kw_set
        self.kw_posterior_prob = dict()
        self.label_prior_prob = dict()

    def train(self, data):
        """
        Train model.
        :param data: [label] [input_text_words]
        :return:
        """
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
            for word in kw_posterior_prob.keys():
                self.kw_posterior_prob[label][word] /= self.label_prior_prob[label] * 1.0
        for label in self.label_prior_prob:
            self.label_prior_prob[label] /= len(data) * 1.0
        print(self.label_prior_prob)
        print(self.kw_posterior_prob)

    def predict(self, input_text):
        """
        Predict the label according to input_text.
        :param input_text:
        :return:
        """


def data_pre_process(data_file_name):
    """

    :param data_file_name:
    :return:
    """
    fh = open(data_file_name, encoding='utf-8')
    data_list = list()
    for line in fh.readlines():
        label_text_pair = line.split('\t')
        word_list = re.split('[^\'a-zA-Z]', label_text_pair[1])
        word_in_doc_set = set()
        for raw_word in word_list:
            word = raw_word.lower()
            if word == '':
                continue
            word_in_doc_set.add(word)
        # data_list: [label] [input_text_words]
        data_list.append((label_text_pair[0], list(word_in_doc_set)))
    return data_list


def statistic_key_word(data_list, cut_off=None):
    """
    :param data_list: data in one line: [label] [input_text]
    :param cut_off:
    :return:
    """
    w_dict = dict()
    total_doc_count = len(data_list)
    for _, word_in_doc_set in data_list:
        for word in word_in_doc_set:
            if word not in w_dict:
                w_dict[word] = 0
            w_dict[word] += 1
    for word in w_dict.keys():
        w_dict[word] /= total_doc_count * 1.0
    w_count_list = sorted(w_dict.items(), key=lambda d: d[1], reverse=True)
    kw_set = set()
    cut_off_length = cut_off if cut_off else len(w_count_list)
    for word, _ in w_count_list[:cut_off_length]:
        kw_set.add(word)
    return w_count_list, kw_set


def draw(kw_count_list):
    """
    Draw the key word count distribution for selecting a proper cut off.
    :param kw_count_list:
    :return:
    """
    key_word_list = list()
    count_list = list()
    print(kw_count_list)
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

    plt.figure(figsize=(8, 4))
    plt.xlabel('Rank of key word')
    plt.ylabel('Count of doc containing key word')
    valid_count = 500
    plt.plot(key_word_list[:valid_count], count_list[:valid_count])
    xt_list = range(0, valid_count, 50)
    plt.xticks(xt_list, xt_list)
    plt.xlim(0, valid_count)
    plt.ylim(0, 0.35)
    plt.grid(True)
    plt.show()


def validate(data_file_name, k):
    """
    k-fold cross-validation.
    :param data_file_name: str | data in one line: [label] [input_text]
    :param k: count of validation fold.
    :return:
    """
    data_list = data_pre_process(data_file_name)
    word_count_list, key_word_set = statistic_key_word(data_list, cut_off=300)
    print(len(key_word_set), key_word_set)
    nbc_model = NaiveBayesClassificationModel(key_word_set)
    nbc_model.train(data_list)


if __name__ == '__main__':
    file_name = '../data/SMSSpamCollection.txt'

    validate(file_name, 4)
