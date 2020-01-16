import numpy as np
import matplotlib.pyplot as plt
import turtle

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
        # self.__b = np.random.rand()

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

    def visualize_train_data_and_model(self, visual_process=False, step=80):
        if self.__train_nfeats != 2:
            return

        if self.__omega is None or self.__b is None:
            return

        for feat, label in zip(self.__train_feats, self.__train_labels):
            if label == 1:
                plt.plot(feat[0], feat[1], '*r')
            else:
                plt.plot(feat[0], feat[1], '+g')

        start_x = np.min(self.__train_feats[:, 0])
        end_x = np.max(self.__train_feats[:, 0])
        start_y = (-self.__b - self.__omega[0] * start_x) / self.__omega[1]
        end_y = (-self.__b - self.__omega[0] * end_x) / self.__omega[1]
        plt.plot([start_x, end_x], [start_y, end_y])

        if visual_process:  # visualize several perceptrons in the training process
            for i in range(len(self.__history_omegas)):
                if i % step != 0:
                    continue
                if self.__history_omegas[i][1] == 0.:
                    continue
                start_y = (-self.__history_bs[i] - self.__history_omegas[i][0] * start_x) / self.__history_omegas[i][1]
                end_y = (-self.__history_bs[i] - self.__history_omegas[i][0] * end_x) / self.__history_omegas[i][1]
                plt.plot([start_x, end_x], [start_y, end_y])
                plt.text(end_x, end_y, '%d' % i)

        plt.title('Simple Perceptron')
        plt.show()

    def visualize_train_data_and_model_with_turtle(self):
        # hyper parameters
        margin_ratio = 0.1
        radius_ratio = 300.
        paint_speed = 5
        step = 120

        if self.__train_nfeats != 2:
            return

        min_x = np.min(self.__train_feats[:, 0])
        max_x = np.max(self.__train_feats[:, 0])
        min_y = np.min(self.__train_feats[:, 1])
        max_y = np.max(self.__train_feats[:, 1])
        origin_width = max_x - min_x
        origin_height = max_y - min_y

        radius = (max_x - min_x) / radius_ratio
        turtle.setworldcoordinates(min_x - origin_width * margin_ratio,
                                   min_y - origin_height * margin_ratio,
                                   max_x + origin_width * margin_ratio,
                                   max_y + origin_height * margin_ratio)
        turtle.speed(paint_speed)

        # paint points
        for feat, label in zip(self.__train_feats, self.__train_labels):
            if label == 1:
                turtle.color('red')
                turtle.penup()
                turtle.goto(feat[0], feat[1])
                turtle.pendown()
                turtle.begin_fill()
                turtle.fillcolor('red')
                turtle.circle(radius)
                turtle.end_fill()
            else:
                turtle.color('green')
                turtle.penup()
                turtle.goto(feat[0], feat[1])
                turtle.pendown()
                turtle.begin_fill()
                turtle.fillcolor('green')
                turtle.circle(radius)
                turtle.end_fill()

        # paint models
        turtle.color('black')
        start_x = min_x
        end_x = max_x
        for i in range(len(self.__history_omegas)):
            if i % step != 0:
                continue
            if self.__history_omegas[i][1] == 0.:
                continue
            start_y = (-self.__history_bs[i] - self.__history_omegas[i][0] * start_x) / self.__history_omegas[i][1]
            end_y = (-self.__history_bs[i] - self.__history_omegas[i][0] * end_x) / self.__history_omegas[i][1]
            turtle.penup()
            turtle.goto(start_x, start_y)
            turtle.pendown()
            turtle.goto(end_x, end_y)
            turtle.penup()
            turtle.goto((start_x + end_x) / 2., (start_y + end_y) / 2.)
            turtle.pendown()
            turtle.write('epoch ' + str(i))

        turtle.color('purple')
        start_y = (-self.__b - self.__omega[0] * start_x) / self.__omega[1]
        end_y = (-self.__b - self.__omega[0] * end_x) / self.__omega[1]
        turtle.penup()
        turtle.goto(start_x, start_y)
        turtle.pendown()
        turtle.goto(end_x, end_y)
        turtle.penup()
        turtle.goto((start_x + end_x) / 2., (start_y + end_y) / 2.)
        turtle.pendown()
        turtle.write('final model')

        turtle.mainloop()


if __name__ == '__main__':
    def f(x):
        return 1 if x > 0 else 0

    data_address = r'..\dataset\watermelon3.xlsx'
    datasetff = LabeledDatasetFromFile(data_address).get_data_by_sheet(0, mode=utils.CONSTANT.TRANS)
    datasetff.astype(np.float)
    train_data = datasetff[:, -3:]  # 只使用连续属性值
    linear_separable_data = train_data[[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 16], :]

    classifier = Perceptron(linear_separable_data, linear_separable_data, f, epsilon=0.000001)
    classifier.train(max_epoch=1000, learning_rate=0.00001)
    # classifier.visual_train_data_and_model(visual_process=True, step=200)
    classifier.visualize_train_data_and_model_with_turtle()
    error = classifier.evaluate_train_data()
    if error != -1:
        print('error rate: %f' % error)
