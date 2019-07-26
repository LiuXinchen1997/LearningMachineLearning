import numpy as np
import types

import utils.CALC_FUNCTIONS
from MyMachineLearning.Dataset import LabeledDatasetFromFile


class FullyConnectedNeuralNetwork:
    def __init__(self, train_data, test_data, activate='sigmoid', hidden_layers_nodes=[5]):
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

        self.__num_input_layer_nodes = self.__train_nfeats
        self.__num_output_layer_nodes = len(set(self.__train_labels))
        self.__hidden_layers_nodes = hidden_layers_nodes

        # parameters of FCNN
        self.__omegas = []
        self.__bs = []

    def train(self, max_epoch=500):
        # init parameters
        for i in range(len(self.__hidden_layers_nodes)):
            if 0 == i:
                self.__omegas.append(np.mat(np.random.ranf((self.__num_input_layer_nodes, self.__hidden_layers_nodes[i]))))
                self.__bs.append(np.random.random())
            else:
                self.__omegas.append(np.mat(np.random.ranf((self.__hidden_layers_nodes[i - 1], self.__hidden_layers_nodes[i]))))
                self.__bs.append(np.random.random())
        self.__omegas.append(np.mat(np.random.ranf((self.__hidden_layers_nodes[-1], self.__num_output_layer_nodes))))
        self.__bs.append(np.random.random())

        cur_epoch = 0
        while cur_epoch <= max_epoch:

            cur_epoch += 1

    def pred(self, feat):
        if self.__omegas is [] or self.__bs is []:
            return -1

        feat = np.mat(feat)
        for (omega, b) in zip(self.__omegas, self.__bs):
            feat = feat * omega + b
            if isinstance(self.__activate, str):
                if 'sigmoid' == self.__activate:
                    feat = utils.CALC_FUNCTIONS.sigmoid(feat)
            elif isinstance(self.__activate, types.FunctionType):
                feat = self.__activate(feat)

        return feat


if __name__ == '__main__':
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    train_data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    train_data.astype(np.float)

    classifier = FullyConnectedNeuralNetwork(train_data, test_data=train_data, hidden_layers_nodes=[5, 4, 3])
    classifier.train()
    print(classifier.pred(train_data[0, :-1]))
