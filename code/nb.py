# coding=utf-8

from numpy import *


class NaiveBayesClassificationModel(object):
    """
    NBC Model.
    """
    def __init__(self):
        """
        Init feature and label matrix.
        """
        self.feature = []
        self.label = []

    def train(self, data):
        """
        Train model.
        :param data: [label] [input_text]
        :return:
        """


    def predict(self, input_text):
        """
        Predict the label according to input_text.
        :param input_text:
        :return:
        """

def validate(data_file_name, k):
    """
    k-fold cross-validation.
    :param data_file_name: str | data in one line: [label] [input_text]
    :param k: count of validation fold.
    :return:
    """
    fh = open(data_file_name, encoding='utf-8')
    for line in fh.readlines():
        print(line)
    model = NaiveBayesClassificationModel()
    # model.train()


if __name__ == '__main__':
    validate('../data/SMSSpamCollection.txt', 4)
