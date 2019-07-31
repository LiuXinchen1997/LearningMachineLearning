import numpy as np
import types
import time

import utils.CALC_FUNCTIONS
import utils.CONSTANT
from MyMachineLearning.Dataset import LabeledDatasetFromFile


class FullyConnectedNeuralNetwork:
    def __init__(self, train_data, test_data, activate='sigmoid', activate_derivative='sigmoid',
                 num_hidden_layers_nodes=[5]):
        self.__train_data = train_data  # attention: not np.matrix!
        self.__test_data = test_data

        self.__train_feats = self.__train_data[:, :-1]
        self.__train_labels = self.__train_data[:, -1]
        self.__train_nsamples = self.__train_feats.shape[0]
        self.__train_nfeats = self.__train_feats.shape[1]

        self.__test_feats = test_data[:, :-1]
        self.__test_labels = test_data[:, -1]
        self.__test_nsamples = self.__test_feats.shape[0]
        self.__test_nfeats = self.__test_feats.shape[1]

        if isinstance(activate, str):
            if 'sigmoid' == activate:
                self.__activate = utils.CALC_FUNCTIONS.sigmoid
                self.__activate_derivative = utils.CALC_FUNCTIONS.sigmoid_derivative
            elif 'tanh' == activate:
                self.__activate = utils.CALC_FUNCTIONS.tanh
                self.__activate_derivative = utils.CALC_FUNCTIONS.tanh_derivative
            elif 'relu' == activate:
                self.__activate = utils.CALC_FUNCTIONS.relu
                self.__activate_derivative = utils.CALC_FUNCTIONS.relu_derivative
            elif 'none' == activate:  # no activate functions
                self.__activate = utils.CALC_FUNCTIONS._none
                self.__activate_derivative = utils.CALC_FUNCTIONS._none_derivative
            else:
                raise Exception("wrong activate functions.")
        elif isinstance(self.__activate, types.FunctionType):
            self.__activate = activate
            self.__activate_derivative = activate_derivative

        self.__num_input_layer_nodes = self.__train_nfeats
        self.__num_output_layer_nodes = len(set(self.__train_labels))
        self.__num_hidden_layers_nodes = num_hidden_layers_nodes

        # parameters of FCNN
        self.__omegas = []
        self.__bs = []

    def __forward_propagation(self, feat):
        if (not self.__omegas) or (not self.__bs):
            return -1

        # record input and output of every layer
        _as = []  # input of a layer
        _ys = []  # output of a layer

        feat = np.mat(feat)
        _as.append(feat)
        _ys.append(feat)
        for (omega, b) in zip(self.__omegas, self.__bs):
            feat = feat * omega + b
            _as.append(feat)
            feat = self.__activate(feat)
            _ys.append(feat)

        return _as, _ys

    @staticmethod
    def __label2vec(label, dim):
        vec = np.zeros((dim, ))
        vec[int(label)] = 1.
        return np.mat(vec)

    def __get_num_all_layers_nodes(self):
        num_all_layers_nodes = self.__num_hidden_layers_nodes.copy()
        num_all_layers_nodes.insert(0, self.__num_input_layer_nodes)
        num_all_layers_nodes.append(self.__num_output_layer_nodes)

        return num_all_layers_nodes

    def __backward_propagation(self, feat, label, learning_rate):
        label_vec = self.__label2vec(label, self.__num_output_layer_nodes)
        _as, _ys = self.__forward_propagation(feat)
        _deltas = []
        for ind in reversed(range(len(self.__omegas))):  # layer ind --> layer ind+1
            derivative = self.__activate_derivative(_as[ind + 1])

            if not _deltas:  # if _deltas is not a []
                delta = np.multiply(derivative, (label_vec - _ys[ind + 1]))
            else:
                delta = np.multiply(derivative, _deltas[-1] * self.__omegas[ind + 1].T)

            _delta_omega = _ys[ind].T * delta
            self.__omegas[ind] += learning_rate * _delta_omega
            _delta_b = delta
            self.__bs[ind] += learning_rate * _delta_b
            _deltas.append(delta)

    def __init_parameters(self):
        num_all_layers_nodes = self.__get_num_all_layers_nodes()
        for i in range(len(num_all_layers_nodes) - 1):
            self.__omegas.append(np.mat(np.random.ranf((num_all_layers_nodes[i], num_all_layers_nodes[i+1]))))
            self.__bs.append(np.mat(np.random.ranf((num_all_layers_nodes[i+1], ))))

    def __print_parameters(self):
        print('****************')
        print('omegas: ')
        for omega in self.__omegas:
            print(omega.shape)
            print(omega)
        print('bs: ')
        for b in self.__bs:
            print(b.shape)
            print(b)

    @staticmethod
    def __calc_loss(output, label):
        return np.squeeze(0.5 * ((output - label) ** 2))

    def __sampling_for_training(self, sampling_mode, mini_batch_size=0.4):
        np.random.shuffle(self.__train_data)
        self.__train_feats = self.__train_data[:, :-1]
        self.__train_labels = self.__train_data[:, -1]
        if 'batch' == sampling_mode or sampling_mode.startswith('b'):
            train_feats_one_epoch = self.__train_feats.copy()
            train_labels_one_epoch = self.__train_labels.copy()
        elif 'stochastic' == sampling_mode or sampling_mode.startswith('s'):
            train_feats_one_epoch = self.__train_feats[:1, :].copy()
            train_labels_one_epoch = self.__train_labels[:1].copy()
        elif 'mini-batch' == sampling_mode or sampling_mode.startswith('m'):
            if isinstance(mini_batch_size, float):
                mini_batch_size = int(self.__train_nsamples * mini_batch_size)
            elif mini_batch_size > self.__train_nsamples:
                mini_batch_size = self.__train_nsamples
            train_feats_one_epoch = self.__train_feats[:mini_batch_size, :].copy()
            train_labels_one_epoch = self.__train_labels[:mini_batch_size].copy()
        else:
            raise Exception('wrong sampling mode.')

        return train_feats_one_epoch, train_labels_one_epoch

    def train(self, max_epoch=20000, learning_rate=0.001, sampling_mode='batch', mini_batch_size=0.4):
        """
        :param sampling_mode: mode of sampling, including 'batch', 'stochastic' and 'mini-batch'.
        """
        start_time = time.time()

        # init parameters
        self.__init_parameters()
        self.__print_parameters()

        cur_epoch = 0
        while cur_epoch < max_epoch:
            train_feats_one_epoch, train_labels_one_epoch = \
                self.__sampling_for_training(sampling_mode=sampling_mode, mini_batch_size=mini_batch_size)

            loss = 0.
            for (feat, label) in zip(train_feats_one_epoch, train_labels_one_epoch):
                # backward propagation
                self.__backward_propagation(feat, label, learning_rate)

                # calculate loss
                loss += self.__calc_loss(self.pred(feat), label)

            print("epoch {} / {} loss: {}".format(cur_epoch + 1, max_epoch, loss / train_feats_one_epoch.shape[0]))
            if loss <= utils.CONSTANT.DEFAULT_ZERO_PRECISION:
                break
            cur_epoch += 1

        self.__print_parameters()

        end_time = time.time()
        print("cost time for training: {} seconds.".format(end_time - start_time))

    def pred(self, feat):
        if (not self.__omegas) or (not self.__bs):
            return -1

        # forward propagation
        feat = np.mat(feat)
        for (omega, b) in zip(self.__omegas, self.__bs):
            feat = feat * omega + b
            feat = self.__activate(feat)

        return np.argmax(feat)

    def evaluate_test_dataset(self):
        correct = 0
        for (feat, label) in zip(self.__test_feats, self.__test_labels):
            judge = self.pred(feat)
            correct += (judge == label)

        return 1 - correct / self.__test_nsamples


if __name__ == '__main__':
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    train_data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    train_data[train_data[:, 2] == -1, 2] = 0.  # preprocess
    # train_data = train_data[:, -3:] # step for watermelon dataset
    train_data.astype(np.float)
    np.random.shuffle(train_data)

    classifier = FullyConnectedNeuralNetwork(train_data[:100, :], test_data=train_data[100:, :],
                                             num_hidden_layers_nodes=[6], activate='tanh', activate_derivative='tanh')
    classifier.train(max_epoch=20000, learning_rate=0.0001)
    print('error rate: {}'.format(classifier.evaluate_test_dataset()))
