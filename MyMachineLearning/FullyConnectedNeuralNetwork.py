import numpy as np
import types

import utils.CALC_FUNCTIONS
from MyMachineLearning.Dataset import LabeledDatasetFromFile


class FullyConnectedNeuralNetwork:
    def __init__(self, train_data, test_data, activate='sigmoid', activate_derivative='sigmoid',
                 num_hidden_layers_nodes=[5]):
        self.__train_data = np.mat(train_data)
        self.__test_data = np.mat(test_data)

        self.__train_feats = train_data[:, :-1]
        self.__train_labels = train_data[:, -1]
        self.__train_nsamples = self.__train_feats.shape[0]
        self.__train_nfeats = self.__train_feats.shape[1]

        self.__test_feats = test_data[:, :-1]
        self.__test_labels = test_data[:, -1]
        self.__test_nsamples = self.__test_feats.shape[0]
        self.__test_nfeats = self.__test_feats.shape[1]

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
            if isinstance(self.__activate, str):
                if 'sigmoid' == self.__activate:
                    feat = utils.CALC_FUNCTIONS.sigmoid(feat)
            elif isinstance(self.__activate, types.FunctionType):
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
            derivative = 0.
            if isinstance(self.__activate_derivative, str):
                if 'sigmoid' == self.__activate_derivative:
                    derivative = utils.CALC_FUNCTIONS.sigmoid_derivative(_as[ind + 1])
            elif isinstance(self.__activate_derivative, types.FunctionType):
                derivative = self.__activate_derivative(_as[ind + 1])

            if not _deltas:  # if _deltas is not a []
                delta = np.multiply((_ys[ind + 1] - label_vec), derivative)
            else:
                delta = np.multiply(_deltas[-1] * self.__omegas[ind + 1].T, derivative)

            _delta_omega = _ys[ind].T * delta
            self.__omegas[ind] -= learning_rate * _delta_omega
            _delta_b = delta
            self.__bs[ind] -= learning_rate * _delta_b
            _deltas.append(delta)

    def train(self, max_epoch=500, learning_rate=0.001):
        # init parameters
        for i in range(len(self.__num_hidden_layers_nodes)):
            if 0 == i:
                self.__omegas.append(np.mat(np.random.ranf((self.__num_input_layer_nodes, self.__num_hidden_layers_nodes[i]))))
                self.__bs.append(np.random.random())
            else:
                self.__omegas.append(np.mat(np.random.ranf((self.__num_hidden_layers_nodes[i - 1], self.__num_hidden_layers_nodes[i]))))
                self.__bs.append(np.random.random())
        self.__omegas.append(np.mat(np.random.ranf((self.__num_hidden_layers_nodes[-1], self.__num_output_layer_nodes))))
        self.__bs.append(np.random.random())

        num_all_layers_nodes = self.__get_num_all_layers_nodes()
        cur_epoch = 0
        while cur_epoch <= max_epoch:
            for (feat, label) in zip(self.__train_feats, self.__train_labels):
                # backward propagation
                self.__backward_propagation(feat, label, learning_rate)

            cur_epoch += 1

    def pred(self, feat):
        if (not self.__omegas) or (not self.__bs):
            return -1

        # forward propagation
        feat = np.mat(feat)
        for (omega, b) in zip(self.__omegas, self.__bs):
            feat = feat * omega + b
            if isinstance(self.__activate, str):
                if 'sigmoid' == self.__activate:
                    feat = utils.CALC_FUNCTIONS.sigmoid(feat)
            elif isinstance(self.__activate, types.FunctionType):
                feat = self.__activate(feat)

        return feat

    def evaluate_test_dataset(self):
        correct = 0
        for (feat, label) in zip(self.__test_feats, self.__test_labels):
            output_vec = self.pred(feat)
            judge = np.where(output_vec == np.max(output_vec))[0][0]
            correct += (judge == label)

        return 1 - correct / self.__test_nsamples


if __name__ == '__main__':
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    train_data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    train_data[train_data[:, 2] == -1, 2] = 0.  # preprocess
    train_data.astype(np.float)

    classifier = FullyConnectedNeuralNetwork(train_data, test_data=train_data, num_hidden_layers_nodes=[5, 4])
    classifier.train()
    print(classifier.evaluate_test_dataset())
