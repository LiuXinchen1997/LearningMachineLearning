import numpy as np
import matplotlib.pyplot as plt

import utils.CONSTANT
from MyMachineLearning.Dataset import LabeledDatasetFromFile, LabeledTrainAndTestDataset

# Just for linear separable dataset
class Perceptron:
    @staticmethod
    def __get_feats_and_labels_from_data(data):
        feats = data[:, :-1]
        labels = data[:, -1]
        nsamples = feats.shape[0]
        nfeats = feats.shape[1]
        return feats, labels, nsamples, nfeats

    def __init__(self, train_data, test_data, activate, epsilon=1e-4):
        self.__train_data = train_data
        self.__test_data = test_data

        self.__train_feats, self.__train_labels, self.__train_nsamples, self.__train_nfeats \
            = self.__get_feats_and_labels_from_data(train_data)
        self.__test_feats, self.__test_labels, self.__test_nsamples, self.__test_nfeats \
            = self.__get_feats_and_labels_from_data(test_data)

        self.__activate = activate
        self.__epsilon = epsilon

        self.__omega = None
        self.__b = None

        self.__history_omegas = []
        self.__history_bs = []

    def __calc_fx(self, feat):
        if self.__omega is None:
            return

        return self.__activate(np.dot(self.__omega.transpose(), feat) + self.__b)

    @staticmethod
    def __is_converged(before_update_omega, after_update_omega, before_update_b, after_update_b, epsilon=1e-4):
        diffs = before_update_omega - after_update_omega
        for diff in diffs:
            if np.abs(diff) > epsilon:
                return False
        if np.abs(before_update_b - after_update_b) > epsilon:
            return False

        return True

    def train(self, max_epoch=100, learning_rate=0.1):
        self.__omega = np.zeros((self.__train_nfeats, ))
        # self.__omega = np.random.ranf((self.__train_nfeats, ))
        self.__b = 0.

        for epoch in range(max_epoch):
            print('*** epoch %d ***' % epoch)
            self.__history_omegas.append(self.__omega.copy())
            self.__history_bs.append(self.__b)
            before_update_omega = self.__omega.copy()
            before_update_b = self.__b
            for feat, label in zip(self.__train_feats, self.__train_labels):
                self.__omega += learning_rate * (label - self.__calc_fx(feat)) * feat
                self.__b += learning_rate * (label - self.__calc_fx(feat))
            print('omega: ', self.__omega)
            print('b: ', self.__b)
            if self.__is_converged(before_update_omega, self.__omega, before_update_b, self.__b, epsilon=self.__epsilon):
                break

    def pred(self, feat):
        if self.__calc_fx(feat) > 0:
            return 1
        else:
            return 0

    def evaluate_train_data(self):
        if self.__omega is None:
            return -1

        accuracy = 0
        for feat, label in zip(self.__train_feats, self.__train_labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        return 1 - accuracy / self.__train_nsamples

    def visual_train_data_and_model(self, visual_process=False, step=80):
        if self.__train_nfeats != 2:
            return

        if self.__omega is None or self.__b is None:
            return

        for feat, label in zip(self.__train_feats, self.__train_labels):
            if label == 1:
                plt.plot(feat[0], feat[1], '*r')
            else:
                plt.plot(feat[0], feat[1], '+g')

        min_x = np.min(self.__train_feats[:, 0])
        max_x = np.max(self.__train_feats[:, 0])
        min_y = (-self.__b - self.__omega[0] * min_x) / self.__omega[1]
        max_y = (-self.__b - self.__omega[0] * max_x) / self.__omega[1]
        plt.plot([min_x, max_x], [min_y, max_y])

        if visual_process:  # visualize several perceptrons in the training process
            for i in range(len(self.__history_omegas)):
                if i % step != 0:
                    continue
                min_y = (-self.__history_bs[i] - self.__history_omegas[i][0] * min_x) / self.__history_omegas[i][1]
                max_y = (-self.__history_bs[i] - self.__history_omegas[i][0] * max_x) / self.__history_omegas[i][1]
                plt.plot([min_x, max_x], [min_y, max_y])
                plt.text(max_x, max_y, '%d' % i)

        plt.title('Simple Perceptron')
        plt.show()


if __name__ == '__main__':
    def f(x):
        return 1 if x > 0 else 0

    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\watermelon3.xlsx'
    datasetff = LabeledDatasetFromFile(data_address).get_data_by_sheet(0, mode=utils.CONSTANT.TRANS)
    datasetff.astype(np.float)
    train_data = datasetff[:, -3:]  # 只使用连续属性值
    linear_separable_data = train_data[[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 16], :]

    classifier = Perceptron(linear_separable_data, linear_separable_data, f, epsilon=0.000001)
    classifier.train(max_epoch=1000, learning_rate=0.0001)
    classifier.visual_train_data_and_model(visual_process=True, step=200)
    error = classifier.evaluate_train_data()
    if error != -1:
        print('error rate: %f' % error)
