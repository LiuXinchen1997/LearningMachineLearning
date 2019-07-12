"""
实现仍存在bug！

待实现需求：
1. 硬间隔
2. 软间隔
3. 核函数
"""

#%%
import numpy as np

from MyMachineLearning.Dataset import LabeledDatasetFromFile, LabeledTrainAndTestDataset
import utils.CONSTANT


#%%
# 此版本为最为简单的 硬间隔支持向量机
class SupportVectorMachine:
    def __init__(self, train_data, test_data):
        """
        :param train_data: numpy.ndarray, shape: nsamples * nfeats
        :param test_data: numpy.ndarray, shape: nsamples * nfeats
        """
        self.__train_data = train_data
        self.__test_data = test_data

        self.__train_nsamples = self.__train_data.shape[0]
        self.__train_nfeats = self.__train_data.shape[1] - 1

        # SVM核心参数
        self.__omega = np.zeros((self.__train_nfeats, ))
        self.__b = 0.0

    def get_train_feats_and_labels(self):
        return self.__train_data[:, :-1], self.__train_data[:, -1]

    def get_test_feats_and_labels(self):
        return self.__test_data[:, :-1], self.__test_data[:, -1]

    def train(self, epoch=30, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        # 求解参数，构建模型
        alpha = self.__smo(epoch=epoch, epsilon=epsilon)
        omega, b = self.__calc_omega_and_b_by_alpha(alpha, epsilon=epsilon)
        self.__omega = omega
        self.__b = b
        print('final parameter: omega: ', self.__omega)
        print('b: %f\n' % self.__b)

    # for SMO
    @staticmethod
    def __equals_zero(x, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        if np.abs(x) <= epsilon:
            return True
        return False

    def __calc_omega_and_b_by_alpha(self, alpha, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        train_feats, train_labels = self.get_train_feats_and_labels()
        omega = np.zeros((self.__train_nfeats, ))
        for i in range(self.__train_nsamples):
            if np.abs(alpha[i]) <= epsilon:
                continue
            omega += alpha[i] * train_labels[i] * train_feats[i, :]

        b = 0.0
        samples = self.__train_data[np.where(alpha > epsilon)[0], :]
        if samples.shape[0] != 0:
            for sample in samples:
                b += sample[-1] - np.dot(omega.transpose(), sample[:-1])
            b = b / samples.shape[0]

        print('omega: ', omega)
        print('b: %f\n' % b)
        return omega, b

    def __calc_fx(self, alpha, feat):
        omega, b = self.__calc_omega_and_b_by_alpha(alpha)
        return np.dot(omega.transpose(), feat) + b

    def __single_is_fit_for_kkt(self, ind, alpha, feat, label, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        if self.__equals_zero(alpha[ind], epsilon) and label * self.__calc_fx(alpha, feat) >= 1:
            return True
        elif alpha[ind] > 0 and label * self.__calc_fx(alpha, feat) == 1:
            return True
        return False

    def __calc_errors_between_fx_and_label(self, alpha, feats, labels):
        errors = np.zeros((feats.shape[0], ), dtype=np.float)
        for i in range(feats.shape[0]):
            errors[i] = self.__calc_fx(alpha, feats[i, :]) - labels[i]

        return errors

    def __choose_k1_and_k2_for_alpha(self, alpha, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        k1 = -1
        train_feats, train_labels = self.get_train_feats_and_labels()
        support_vector_indces = np.where(alpha > 0)[0]
        for ind in support_vector_indces:
            if not self.__single_is_fit_for_kkt(ind, alpha, train_feats[ind, :], train_labels[ind], epsilon=epsilon):
                k1 = ind
                break

        if k1 == -1:
            for i in range(self.__train_nsamples):
                if not self.__single_is_fit_for_kkt(i, alpha, train_feats[i, :], train_labels[i], epsilon=epsilon):
                    k1 = i
                    break

        k2 = -1
        if k1 == -1:
            return k1, k2
        errors = self.__calc_errors_between_fx_and_label(alpha, train_feats, train_labels)
        sorted_errors = np.sort(errors)
        max_ = -1
        if np.abs(sorted_errors[0] - errors[k1]) > max_:
            max_ = np.abs(sorted_errors[0] - errors[k1])
            k2 = np.where(errors == sorted_errors[0])[0][0]
        if np.abs(sorted_errors[-1] - errors[k1]) > max_:
            max_ = np.abs(sorted_errors[-1] - errors[k1])
            k2 = np.where(errors == sorted_errors[-1])[0][0]

        k1, k2 = k2, k1
        return k1, k2

    def __calc_sum_alpha_y_except_k1_and_k2(self, alpha, k1, k2):
        sum = 0
        for i in range(self.__train_nsamples):
            if i != k1 and i != k2:
                sum += alpha[i] * self.__train_data[i, -1]

        return sum

    def __solve_quadratic_program(self, alpha, k1, k2):
        train_feats, train_labels = self.get_train_feats_and_labels()

        # 计算二次函数的三个参数a, b, c，从而求解最值
        a = b = c = 0  # f(x)=ax^2+bx+c
        for i in range(self.__train_nsamples):
            if i != k1 and i != k2:
                c += alpha[i]

        L = self.__calc_sum_alpha_y_except_k1_and_k2(alpha, k1, k2)
        c += -L / train_labels[k2]
        b += (1 - train_labels[k1] / train_labels[k2])

        delta_a = delta_b = delta_c = 0
        for i in range(self.__train_nsamples):
            if i != k1 and i != k2:
                for j in range(self.__train_nsamples):
                    if j != k1 and j != k2:
                        c += -0.5 * alpha[i] * alpha[j] * train_labels[i] * train_labels[j] \
                             * np.dot(train_feats[i, :].transpose(), train_feats[j, :])

            elif i == k1:
                for j in range(self.__train_nsamples):
                    if j != k1 and j != k2:
                        tmp = -0.5 * alpha[j] * train_labels[i] * train_labels[j] \
                              * np.dot(train_feats[i, :].transpose(), train_feats[j, :])
                        delta_b += tmp
                    elif j == k1:
                        tmp = -0.5 * train_labels[k1] * train_labels[k1] \
                               * np.dot(train_feats[k1, :].transpose(), train_feats[k1, :])
                        a -= tmp
                        delta_a += tmp
                    elif j == k2:
                        tmp = -0.5 * train_labels[k1] * train_labels[k2] \
                              * np.dot(train_feats[k1, :].transpose(), train_feats[k2, :])
                        tmp1 = tmp * (-L) / train_labels[k2]
                        b -= tmp1
                        delta_b += tmp1
                        tmp2 = (-train_labels[k1] / train_labels[k2])
                        a -= tmp2
                        delta_a += tmp2
            elif i == k2:
                for j in range(self.__train_nsamples):
                    if j != k1 and j != k2:
                        tmp = -0.5 * alpha[j] * train_labels[i] * train_labels[j] \
                              * np.dot(train_feats[i, :].transpose(), train_feats[j, :])
                        tmp1 = tmp * (-L) / train_labels[k2]
                        delta_c += tmp1
                        tmp2 = tmp * (-train_labels[k1]) / train_labels[k2]
                        delta_b += tmp2
                    elif j == k1:
                        tmp = -0.5 * train_labels[k1] * train_labels[k2] \
                              * np.dot(train_feats[k1, :].transpose(), train_feats[k2, :])
                        tmp1 = tmp * (-L) / train_labels[k2]
                        b -= tmp1
                        delta_b += tmp1
                        tmp2 = (-train_labels[k1] / train_labels[k2])
                        a -= tmp2
                        delta_a += tmp2
                    elif j == k2:
                        tmp = train_labels[k2] * train_labels[k2] * \
                              np.dot(train_feats[k2, :].transpose(), train_feats[k2, :])
                        a -= tmp
                        delta_a += tmp
                        tmp1 = tmp * L * L
                        c -= tmp1
                        delta_c += tmp1
                        tmp2 = tmp1 * 2 * L * train_labels[k1]
                        b -= tmp2
                        delta_b += tmp2

        a += delta_a * 2
        b += delta_b * 2
        c += delta_c * 2

        alpha_k1 = -b / (2 * a)
        alpha_k2 = -L / train_labels[k2] - train_labels[k1] * alpha_k1 / train_labels[k2]
        return alpha_k1, alpha_k2

    def __quit_condition(self, alpha, feats, labels, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        if np.abs(np.dot(alpha.transpose(), labels)) > epsilon:
            return False
        if not np.all(alpha >= -epsilon):
            return False

        for i in range(feats.shape[0]):
            if self.__equals_zero(alpha[i], epsilon=epsilon):
                if labels[i] * self.__calc_fx(alpha, feats[i, :]) < 1:
                    return False
            elif alpha[i] > epsilon:
                if not self.__equals_zero(labels[i] * self.__calc_fx(alpha, feats[i, :]) - 1, epsilon=epsilon):
                    return False
            else:
                return False

        return True

    def __smo(self, epoch=30, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION):
        train_feats, train_labels = self.get_train_feats_and_labels()
        alpha = np.zeros((self.__train_nsamples, ))

        for i in range(epoch):
            k1, k2 = self.__choose_k1_and_k2_for_alpha(alpha, epsilon=epsilon)
            if -1 == k1:
                break
            alpha_k1, alpha_k2 = self.__solve_quadratic_program(alpha, k1, k2)
            alpha[k1] = alpha_k1
            alpha[k2] = alpha_k2
            if self.__quit_condition(alpha, train_feats, train_labels, epsilon=epsilon):
                break

        return alpha

    def pred(self, feat):
        if np.dot(self.__omega.transpose(), feat) + self.__b > 0:
            return 1
        else:
            return -1

    def evaluate_result(self):
        correct = 0
        for sample in self.__test_data:
            correct += (self.pred(sample[:-1]) == sample[-1])

        return 1 - correct / self.__test_data.shape[0]


#%%
if __name__ == '__main__':
    # 调整数据格式
    # 注意：与之前不一样，SVM的数据集中负例使用-1表示！
    # 注意：由于此版本SVM没有使用软间隔，也没有引入核技巧，因此要求数据集是线性可分的！
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\watermelon3.xlsx'
    datasetff = LabeledDatasetFromFile(data_address).get_data_by_sheet(0, mode=utils.CONSTANT.TRANS)
    datasetff.astype(np.float)
    selected_samples = [2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 16]
    datasetff = datasetff[selected_samples, -3:]  # 只使用连续属性值
    datasetff[datasetff[:, -1] == 0, -1] = -1

    dataset = LabeledTrainAndTestDataset(datasetff, test_data=datasetff, test_ratio=0.5)
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()
    # dataset.visual_data(train_data)

    # 获得模型
    svm = SupportVectorMachine(train_data, test_data)
    svm.train(epoch=100)

    print('error rate %f\n' % svm.evaluate_result())