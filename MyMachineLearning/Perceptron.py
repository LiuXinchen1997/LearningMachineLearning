import numpy as np

class Perceptron:
    @staticmethod
    def __get_feats_and_labels_from_data(data):
        feats = data[:, :-1]
        labels = data[:, -1]
        nsamples = data.shape[0]
        nfeats = data.shape[1]
        return feats, labels, nsamples, nfeats

    def __init__(self, train_data, test_data):
        self.__train_data = train_data
        self.__test_data = test_data

        self.__train_feats, self.__train_labels, self.__train_nsamples, self.__train_nfeats \
            = self.__get_feats_and_labels_from_data(train_data)
        self.__test_feats, self.__test_labels, self.__test_nsamples, self.__test_nfeats \
            = self.__get_feats_and_labels_from_data(test_data)

        self.__omega = None
        self.__b = None

    def train(self, learning_rate=0.001):
        self.__omega