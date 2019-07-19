"""
分类性能比较差 :-(
"""

#%%
import numpy as np
import time

from MyMachineLearning.Dataset import LabeledDatasetFromFile, LabeledTrainAndTestDataset
import utils.CONSTANT


#%%
class SupportVectorMachine:
    def __init__(self, train_data, test_data, epsilon=utils.CONSTANT.DEFAULT_ZERO_PRECISION,
                 C=1.0, kernel_option=('rbf', 1.0)):
        """
        :param train_data: numpy.ndarray, shape: nsamples * nfeats
        :param test_data: numpy.ndarray, shape: nsamples * nfeats
        """
        self.__train_data = train_data
        self.__test_data = test_data

        self.__train_nsamples = self.__train_data.shape[0]
        self.__train_nfeats = self.__train_data.shape[1] - 1

        self.__epsilon = epsilon
        self.__C = C
        self.__K = self.__calc_kernel_matrix(self.__train_data, kernel_option)
        # 记录每个样本的error，第一个0/1表示error是否被优化，第二个记录error值
        self.__errors_cache = np.zeros((self.__train_nsamples, 2))

        # SVM核心参数
        self.__omega = np.zeros((self.__train_nfeats, ))
        self.__b = 0.0

    @staticmethod
    def __calc_kernel_value(feats, ind, kernel_option):
        nsamples = feats.shape[0]
        feat = feats[ind, :]

        value = np.zeros((nsamples, ))
        kernel_type = kernel_option[0]
        if 'linear' == kernel_type:
            value = np.dot(feats, feat.transpose())
        elif 'rbf' == kernel_type:
            sigma = kernel_option[1]
            if 0. == sigma:
                sigma = 1.0

            for i in range(nsamples):
                diff = feat - feats[i, :]
                value[i] = np.exp(np.dot(diff, diff.transpose()) / (-2. * sigma ** 2))

        return value

    @staticmethod
    def __calc_kernel_matrix(train_data, kernel_option=('rbf', 1.0)):
        nsamples = train_data.shape[0]
        feats = train_data[:, :-1]
        kernel_matrix = np.zeros((nsamples, nsamples))
        for i in range(nsamples):
            kernel_matrix[:, i] = SupportVectorMachine.__calc_kernel_value(feats, i, kernel_option)

        return kernel_matrix

    def get_train_feats_and_labels(self):
        return self.__train_data[:, :-1], self.__train_data[:, -1]

    def get_test_feats_and_labels(self):
        return self.__test_data[:, :-1], self.__test_data[:, -1]

    def train(self, epoch=100):
        # 求解参数，构建模型
        starttime = time.time()
        alphas = self.__smo(epoch=epoch)
        omega = self.__calc_omega_by_alphas(alphas)
        self.__omega = omega  # b已在训练过程中更新完毕
        endtime = time.time()
        print('final parameter: omega: ', self.__omega)
        print('b: %f\n' % self.__b)
        print('cost time: %d\n' % (endtime - starttime))

    # for SMO
    def __x_equals_y(self, x, y):
        if y - self.__epsilon <= x <= y + self.__epsilon:
            return True
        return False

    def __calc_omega_by_alphas(self, alphas):
        train_feats, train_labels = self.get_train_feats_and_labels()
        omega = np.zeros((self.__train_nfeats, ))
        for i in range(self.__train_nsamples):
            if np.abs(alphas[i]) <= self.__epsilon:
                continue
            omega += alphas[i] * train_labels[i] * train_feats[i, :]

        return omega

    def __calc_fx(self, alphas, feat, update_omega=True):
        omega = self.__calc_omega_by_alphas(alphas)
        if update_omega:
            self.__omega = omega

        return np.dot(omega.transpose(), feat) + self.__b

    def __calc_errors_between_fx_and_label(self, alphas, feats, labels):
        errors = np.zeros((feats.shape[0], ), dtype=np.float)
        for i in range(feats.shape[0]):
            errors[i] = self.__calc_fx(alphas, feats[i, :]) - labels[i]

        return errors

    def __calc_error_for_sample_i(self, alphas, feat, label):
        return self.__calc_fx(alphas, feat) - label

    def __update_errors_cache(self, alphas, ind, feat, label):
        error = self.__calc_error_for_sample_i(alphas, feat, label)
        self.__errors_cache[ind, 0] = 1
        self.__errors_cache[ind, 1] = error

    def __choose_k2_from_k1_for_alphas(self, alphas, k1):
        feats, labels = self.get_train_feats_and_labels()
        error_1 = self.__calc_error_for_sample_i(alphas, feats[k1, :], labels[k1])
        self.__update_errors_cache(alphas, k1, feats[k1, :], labels[k1])
        candidate_alphas = np.nonzero(self.__errors_cache[:, 0])[0]

        max_ = -1
        k2 = 0
        if len(candidate_alphas) > 1:
            for i in range(alphas.shape[0]):
                if i == k1:
                    continue
                error_2 = self.__calc_error_for_sample_i(alphas, feats[i, :], labels[i])
                if max_ < np.abs(error_1 - error_2):
                    max_ = np.abs(error_1 - error_2)
                    k2 = i
        else:
            k2 = k1
            while k2 == k1:
                k2 = int(np.random.uniform(0, self.__train_nsamples))

        return k2

    def __single_is_fit_for_kkt(self, alphas, alpha_ind, feat, label):
        target = label * self.__calc_fx(alphas, feat)
        if target >= 1. and self.__x_equals_y(alphas[alpha_ind], 0):
            return True
        elif self.__x_equals_y(target, 1.) and 0. < alphas[alpha_ind] < self.__C:
            return True
        elif target <= 1. and self.__x_equals_y(alphas[alpha_ind], self.__C):
            return True
        return False

    def __inner_loop(self, alphas, alpha_ind):
        train_feats, train_labels = self.get_train_feats_and_labels()
        feat = train_feats[alpha_ind, :]
        label = train_labels[alpha_ind]
        error_1 = self.__calc_error_for_sample_i(alphas, feat, label)

        # KKT 条件
        # 1) y_i * f(x_i) >= 1 and alpha_i == 0 (样本点位于边界外，对SVM模型不构成任何影响)
        # 2) y_i * f(x_i) == 1 and 0 < alpha_i < C (支持向量，在边界上)
        # 3) y_i * f(x_i) <= 1 and alpha_i == C (支持向量，在两边界之间，正确划分但置信度低或者被错误划分)
        # y_i * E_i = y_i * f(x_i) - y_i^2 = y_i * f(x_i) - 1
        if not self.__single_is_fit_for_kkt(alphas, alpha_ind, feat, label):
            # 此时可以将alpha[alpha_ind]作为alpha_i，即被选择的第一个变量
            k1 = alpha_ind

            # 1. 根据alpha k1选择另一个alpha_k2
            k2 = self.__choose_k2_from_k1_for_alphas(alphas, k1)
            feat2 = train_feats[k2, :]
            label2 = train_labels[k2]
            error_2 = self.__calc_error_for_sample_i(alphas, feat2, label2)

            alpha_1_old = alphas[k1]
            alpha_2_old = alphas[k2]

            # 2. 计算alpha_k2的L和H
            if label == label2:
                L = np.max((0., alphas[k2] - alphas[k1]))
                H = np.min((self.__C, self.__C + alphas[k2] - alphas[k1]))
            else:
                L = np.max((0., alphas[k2] + alphas[k1] - self.__C))
                H = np.min((self.__C, alphas[k1] + alphas[k2]))
            if L == H:
                return 0

            # 3. 计算eta
            eta = 2 * self.__K[k1, k2] - self.__K[k1, k1] - self.__K[k2, k2]
            if eta >= 0:  # 此处代码待完善
                return 0

            # 4. 更新alpha_k2
            alphas[k2] = alpha_2_old + label2 * (error_1 - error_2) / eta

            # 5. 裁剪alpha
            if alphas[k2] > H:
                alphas[k2] = H
            elif alphas[k2] < L:
                alphas[k2] = L

            # 6. 如果alpha_k2基本没有变化，则认为收敛了
            if np.abs(alphas[k2] - alpha_2_old) <= self.__epsilon:
                self.__update_errors_cache(alphas, k2, feat2, label2)
                return 0

            # 7. 更新alpha_k1
            alphas[k1] = alpha_1_old + label * label2 * (alpha_2_old - alphas[k2])

            # 8. 更新b
            b1 = self.__b - error_1 - label * (alphas[k1] - alpha_1_old) * self.__K[k1, k1] - \
                 label2 * (alphas[k2] - alpha_2_old) * self.__K[k1, k2]
            b2 = self.__b - error_2 - label * (alphas[k1] - alpha_1_old) * self.__K[k1, k2] - \
                 label2 * (alphas[k2] - alpha_2_old) * self.__K[k2, k2]
            if 0 < alphas[k1] < self.__C:
                self.__b = b1
            elif 0 < alphas[k2] < self.__C:
                self.__b = b2
            else:
                self.__b = (b1 + b2) / 2.

            # 9. 更新error
            self.__update_errors_cache(alphas, k1, feat, label)
            self.__update_errors_cache(alphas, k2, feat2, label2)

            return 1
        else:
            return 0

    def __smo(self, epoch):
        # alphas = np.zeros((self.__train_nsamples, ))
        alphas = np.random.ranf((self.__train_nsamples, ))

        epoch_count = 0
        entire_set = True  # 是否要遍历全部样本集
        changed_alpha_pairs = 0  # 发生了变化的alpha对数
        while (epoch_count < epoch) and (changed_alpha_pairs > 0 or entire_set):
            if entire_set:
                for i in range(self.__train_nsamples):
                    changed_alpha_pairs += self.__inner_loop(alphas, i)
            else:
                non_bound_alphas = np.nonzero((alphas > 0) * (alphas < 0))[0]
                for i in non_bound_alphas:
                    changed_alpha_pairs += self.__inner_loop(alphas, i)

            if entire_set:
                entire_set = False
            elif changed_alpha_pairs == 0:
                entire_set = True

            epoch_count += 1

        return alphas

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
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    datasetff = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    datasetff.astype(np.float)
    datasetff = datasetff[:, -3:]  # 只使用连续属性值
    datasetff[datasetff[:, -1] == 0, -1] = -1

    train_data = datasetff[:100, :]
    test_data = datasetff[100:, :]
    dataset = LabeledTrainAndTestDataset(train_data, test_data=test_data)
    dataset.visual_data(train_data)

    # 获得模型
    svm = SupportVectorMachine(train_data, test_data, epsilon=0.0001, C=200, kernel_option=('rbf', 1.3))
    svm.train(epoch=200000)

    print('error rate %f\n' % svm.evaluate_result())