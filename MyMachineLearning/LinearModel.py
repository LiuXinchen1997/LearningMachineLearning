#%%
import numpy as np
from matplotlib import pyplot as plt
from MyMachineLearning.Dataset import LabeledDatasetFromFile

import utils.CONSTANT


#%%
class LinearRegression:
    def __init__(self, train_data):
        self.__train_feats = train_data[:, :-1]
        self.__train_labels = train_data[:, -1].astype(np.int32)
        self.__train_nsamples = self.__train_feats.shape[0]
        self.__train_nfeats = self.__train_feats.shape[1]

        self.__omega = None
        self.__b = None

    def train(self):
        feats = np.mat(np.concatenate((self.__train_feats, np.ones((self.__train_feats.shape[0], 1))), 1))  # 给属性增加一个偏置项
        labels = np.mat(self.__train_labels)
        omega = ((feats.T * feats).I * feats.T * labels.T).A
        self.__omega = omega[:-1].reshape((self.__train_nfeats,))
        self.__b = omega[-1][0]

    def pred(self, feat):
        if np.dot(self.__omega.transpose(), feat) + self.__b > 0:
            return 1
        else:
            return 0

    def evaluate_with_train_data(self):
        if self.__omega is None:
            return -1

        accuracy = 0
        for feat, label in zip(self.__train_feats, self.__train_labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        error = 1 - accuracy / self.__train_nsamples
        return error

    def visual_train_data_and_model(self):
        if self.__train_nfeats != 2:
            return

        if self.__omega is None or self.__b is None:
            return

        for i in range(self.__train_nsamples):
            if self.__train_labels[i] == 0:
                plt.plot(self.__train_feats[i, 0], self.__train_feats[i, 1], '+r')
            else:
                plt.plot(self.__train_feats[i, 0], self.__train_feats[i, 1], '*g')

        min_x = np.min(self.__train_feats[:, 0])
        max_x = np.max(self.__train_feats[:, 0])
        min_y = (-self.__b - self.__omega[0] * min_x) / self.__omega[1]
        max_y = (-self.__b - self.__omega[0] * max_x) / self.__omega[1]
        plt.plot([min_x, max_x], [min_y, max_y])

        plt.title('Linear Regression')
        plt.show()


class LogisticRegression:
    def __init__(self, train_data):
        self.__feats = train_data[:, :-1]
        self.__feats = np.concatenate((self.__feats, np.ones((self.__feats.shape[0], 1))), 1)  # 给属性增加一个偏置项
        self.__labels = train_data[:, -1].astype(np.int32)
        self.__nsamples = self.__feats.shape[0]
        self.__nfeats = self.__feats.shape[1]

        self.__beta = None

    def __calc_p1(self, feat):
        return np.exp(np.dot(self.__beta, feat)) / (1 + np.exp(np.dot(self.__beta, feat)))

    def __calc_p0(self, feat):
        return 1 - self.__calc_p1(self.__beta, feat)

    def __calc_l(self):
        # 极大似然法，最小化l
        l = 0
        for i in range(self.__nsamples):
            l += (-self.__labels[i] * np.dot(self.__beta.T, self.__feats[i, :].T)
                  + np.log(1 + np.exp(np.dot(self.__beta.T, self.__feats[i, :].T))))

        return l

    def __calc_dl(self):
        res = np.zeros(self.__nfeats)
        for i in range(self.__nsamples):
            p1 = self.__calc_p1(self.__feats[i, :])
            item = np.dot(self.__feats[i, :], (self.__labels[i] - p1))
            res -= item

        return res

    def __calc_d2l(self):
        res = np.zeros((self.__nfeats, self.__nfeats))
        for i in range(self.__nsamples):
            p1 = self.__calc_p1(self.__feats[i, :])
            mat = np.dot(np.expand_dims(self.__feats[i, :], 1), np.expand_dims(self.__feats[i, :], 1).T)
            res += mat * p1 * (1 - p1)

        return res

    def train(self, iteration=100, epsilon=1e-5):
        self.__beta = np.zeros(self.__nfeats)
        self.__beta[-1] = 1  # 偏置项

        pre_l = 0
        for i in range(iteration):
            cur_l = self.__calc_l()

            if np.abs(cur_l - pre_l) < epsilon:
                break

            d2l = self.__calc_d2l()
            dl = self.__calc_dl()
            self.__beta = self.__beta - np.dot(np.linalg.inv(d2l), dl)

    def pred(self, feat, threshold=0.5):
        if self.__beta is None:
            return -1

        res = self.__calc_p1(feat)
        if res > threshold:
            return 1
        else:
            return 0

    def evaluate_with_train_data(self):
        if self.__beta is None:
            return -1

        accuracy = 0
        for feat, label in zip(self.__feats, self.__labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        error = 1 - accuracy / self.__nsamples
        return error


class LinearDiscriminantAnalysis:
    def __init__(self, train_data):
        self.__feats = train_data[:, :-1]
        self.__labels = train_data[:, -1].astype(np.int32)
        self.__nsamples = self.__feats.shape[0]
        self.__nfeats = self.__feats.shape[1]

        self.__X0, self.__X1 = self.__get_x0_x1(self.__feats, self.__labels)
        self.__miu0, self.__miu1 = self.__get_miu0_miu1(self.__feats, self.__labels)

        self.__omega = None

    @staticmethod
    def __get_x0_x1(feats, labels):
        X0, X1 = [], []
        for (feat, label) in zip(feats, labels):
            if label == 0:
                X0.append(feat)
            else:
                X1.append(feat)

        X0 = np.array(X0)
        X1 = np.array(X1)
        return X0, X1

    @staticmethod
    def __get_miu0_miu1(feats, labels):
        miu0, miu1 = 0, 0
        X0, X1 = LinearDiscriminantAnalysis.__get_x0_x1(feats, labels)
        nsamples0 = X0.shape[0]
        nsamples1 = X1.shape[1]

        if nsamples0 != 0:
            miu0 = np.sum(X0, 0) / nsamples0
        if nsamples1 != 0:
            miu1 = np.sum(X1, 0) / nsamples1

        return miu0, miu1

    def __calc_sw(self):
        nsamples0 = self.__X0.shape[0]
        sigma0 = np.zeros((self.__nfeats, self.__nfeats)).astype(np.float)
        for i in range(nsamples0):
            sigma0 += np.dot(self.__X0[i, :].reshape((self.__nfeats, 1)), self.__X0[i, :].reshape((self.__nfeats, 1)).T)

        nsamples1 = self.__X1.shape[0]
        sigma1 = np.zeros((self.__nfeats, self.__nfeats)).astype(np.float)
        for i in range(nsamples1):
            sigma1 += np.dot(self.__X1[i, :].reshape((self.__nfeats, 1)), self.__X1[i, :].reshape((self.__nfeats, 1)).T)

        return sigma0 + sigma1

    def train(self):
        self.__omega = np.dot(np.linalg.inv(self.__calc_sw()), (self.__miu0 - self.__miu1))
        return self.__omega

    def get_feats(self):
        return self.__feats

    def get_labels(self):
        return self.__labels

    def pred(self, feat):
        if self.__omega is None:
            return -1
        threshold = np.dot(((self.__miu0 + self.__miu1) / 2.0).T, self.__omega)
        flag = 1
        if np.dot(self.__miu1.T, self.__omega) < threshold:
            flag = -1
        res = np.dot(self.__omega.reshape((1, self.__nfeats)), feat)
        if flag == 1:
            if res[0] > threshold:
                return 1
            else:
                return 0
        else:
            if res[0] < threshold:
                return 1
            else:
                return 0

    def evaluate_with_train_data(self):
        # 训练集和测试集是同一个集合
        if self.__omega is None:
            return -1

        accuracy = 0
        for feat, label in zip(self.__feats, self.__labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        error = 1 - accuracy / self.__nsamples
        return error

    # 此函数在功能上没有普适性，仅适用于西瓜数据集3.0
    def visual_train_data_and_lda_model(self):
        if self.__X0.shape[1] != 2:
            return -1

        if self.__omega is None:
            return -1

        for x in self.__X0:
            plt.plot(x[0], x[1], '+r')
        for x in self.__X1:
            plt.plot(x[0], x[1], '*g')
        k = -self.__omega[0] / self.__omega[1]

        line_x, line_y = [0.1, 0.9], []
        for x in line_x:
            line_y.append(k * x)
        plt.plot(line_x, line_y)

        # 绘制各个样本在LDA直线上的垂直投影
        for x in self.__X0:
            projected_x = (x[1] + x[0] / k) * (k / (k*k + 1))
            projected_y = projected_x * k
            plt.plot(projected_x, projected_y, '+r')
            plt.plot([x[0], projected_x], [x[1], projected_y], color="red", linestyle="-.")
        for x in self.__X1:
            projected_x = (x[1] + x[0] / k) * (k / (k * k + 1))
            projected_y = projected_x * k
            plt.plot(projected_x, projected_y, '*g')
            plt.plot([x[0], projected_x], [x[1], projected_y], color="green", linestyle=":")

        plt.title('Linear Discrimenant Analysis')
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.show()


#%%
if __name__ == '__main__':
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\watermelon3.xlsx'
    datasetff = LabeledDatasetFromFile(data_address)
    entire_train_data = datasetff.get_data_by_sheet(0, mode=utils.CONSTANT.TRANS)
    entire_train_data.astype(np.float)
    selected_train_data = datasetff.get_data_by_sheet(0, mode=utils.CONSTANT.TRANS, feat_indces=[6, 7], label_indces=[8])
    selected_train_data.astype(np.float)

    # 线性回归
    print('1. Linear Regression:')
    classifier = LinearRegression(selected_train_data)
    classifier.train()
    classifier.visual_train_data_and_model()
    error = classifier.evaluate_with_train_data()  # 对于线性不可分的样本，分类性能很差
    if error != -1:
        print('Error: ', error * 100, '%')

    # 对数几率回归
    print('2. Logistic Regression:')
    classifier2 = LogisticRegression(selected_train_data)
    classifier2.train()
    error = classifier2.evaluate_with_train_data()
    if error != -1:
        print('Error: ', error * 100, '%')

    # 线性判别分析
    print('3. Linear Discriminant Analysis:')
    classifier3 = LinearDiscriminantAnalysis(selected_train_data)
    classifier3.train()
    classifier3.visual_train_data_and_lda_model()
    error = classifier3.evaluate_with_train_data()  # LDA需要应用于线性可分的样本，否则性能会很差
    if error != -1:
        print('Error: ', error * 100, '%')
