import numpy as np
import types
import time
import os
import turtle
from matplotlib import pyplot as plt

import utils.CALC_FUNCTIONS
import utils.CONSTANT
from MyMachineLearning.Dataset import LabeledDatasetFromFile
from utils import Visualization


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

        self.__MODEL_NAME = './generated/model.para'
        self.__history_loss = []

    def get_train_data(self):
        return self.__train_data

    def get_test_data(self):
        return self.__test_data

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
        self.__history_loss = []

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
            self.__history_loss.append(loss)
            if loss <= utils.CONSTANT.DEFAULT_ZERO_PRECISION:
                break

            # record model parameters during training for visualization
            if cur_epoch % 50 == 0:
                self.save_model(filename='./generated/model_' + str(cur_epoch) + '.para')

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

    def evaluate_train_dataset(self):
        correct = 0
        for (feat, label) in zip(self.__train_feats, self.__train_labels):
            judge = self.pred(feat)
            correct += (judge == label)

        return 1 - correct / self.__train_nsamples

    def save_model(self, filename=None):
        if (not self.__omegas) or (not self.__bs):
            return

        if filename is not None:
            self.__MODEL_NAME = filename

        with open(self.__MODEL_NAME, 'w') as file:
            file.write(str(len(self.__omegas)) + '\n')
            for omega in self.__omegas:
                nrows = omega.shape[0]
                ncols = omega.shape[1]
                file.write(str(nrows) + ' ' + str(ncols) + '\n')
                for i in range(nrows):
                    for j in range(ncols):
                        file.write(str(omega[i, j]) + ' ')
                    file.write('\n')

            file.write(str(len(self.__bs)) + '\n')
            for b in self.__bs:
                nlens = b.shape[1]
                file.write(str(nlens) + '\n')
                for i in range(nlens):
                    file.write(str(b[0, i]) + ' ')
                file.write('\n')

    def load_model(self, filename=None):
        if filename is None:
            filename = self.__MODEL_NAME

        if not os.path.exists(filename):
            raise Exception('model parameter file does not exist.')

        with open(filename, 'r') as file:
            self.__omegas = []
            self.__bs = []
            nomegas = int(file.readline().strip())
            for i in range(nomegas):
                eles = file.readline().strip().split(' ')
                nrows = int(eles[0])
                ncols = int(eles[1])
                omega = np.mat(np.zeros((nrows, ncols), dtype=np.float))
                for j in range(nrows):
                    eles = file.readline().strip().split(' ')
                    for k in range(ncols):
                        omega[j, k] = float(eles[k])
                self.__omegas.append(omega)

            nbs = int(file.readline().strip())
            for i in range(nbs):
                eles = file.readline().strip()
                nlens = int(eles)
                b = np.mat(np.zeros((nlens, ), dtype=np.float))
                eles = file.readline().strip().split(' ')
                for j in range(nlens):
                    b[0, j] = float(eles[j])
                self.__bs.append(b)

    # visualization
    def visualize_train_data_and_model(self):
        cost_time = Visualization.visualize_data_and_model(self.__train_data, self, title='FCNN for train dataset')
        return cost_time

    def visualize_train_data_and_several_models(self):
        cost_time = 0.
        if os.path.exists('./generated/model_50.para'):
            self.load_model(filename='./generated/model_50.para')
            cost_time += Visualization.visualize_data_and_model(self.__train_data, self,
                                                                title='FCNN for train dataset: epoch 50')

        if os.path.exists('./generated/model_100.para'):
            self.load_model(filename='./generated/model_100.para')
            cost_time += Visualization.visualize_data_and_model(self.__train_data, self,
                                                                title='FCNN for train dataset: epoch 100')

        if os.path.exists('./generated/model_150.para'):
            self.load_model(filename='./generated/model_150.para')
            cost_time += Visualization.visualize_data_and_model(self.__train_data, self,
                                                                title='FCNN for train dataset: epoch 150')

        self.load_model()
        cost_time += Visualization.visualize_data_and_model(self.__train_data, self,
                                                            title='FCNN for train dataset')

        return cost_time

    def visualize_test_data_and_model(self):
        cost_time = Visualization.visualize_data_and_model(self.__test_data, self, title='FCNN for test dataset')
        return cost_time

    def visualize_all_scene_samples_with_labels(self):
        cost_time = Visualization.visualize_all_scene_samples_with_labels(self.__train_data, self, step_ratio=5,
                                                                          title='FCNN for all scene samples')
        return cost_time

    def visualize_random_samples_with_labels(self):
        cost_time = Visualization.visualize_random_samples_with_labels(self.__train_data, self,
                                                                       title='FCNN for random samples')
        return cost_time

    def visualize_train_loss(self, step=0.05):
        if not self.__history_loss:
            return

        plt.xlabel('epoch')
        plt.ylabel('loss')

        points = []
        for i in range(len(self.__history_loss)):
            if isinstance(step, int):
                if i % step != 0:
                    continue
            else:
                if i % int(len(self.__history_loss) * step) != 0:
                    continue

            plt.plot(i, self.__history_loss[i], '*r')
            points.append([i, self.__history_loss[i]])

        for i in range(len(points) - 1):
            plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], 'b')

        plt.title('train loss')
        plt.show()

    def visualize_network_with_turtle(self):
        # hyper parameters
        width = 400
        height = 400
        node_radius = 10

        num_all_layers = self.__get_num_all_layers_nodes()

        turtle.setworldcoordinates(0, 0, width, height)
        layers_nodes = []
        width_step = width / float(len(num_all_layers) + 1)

        # print nodes
        for i in range(len(num_all_layers)):
            num_cur_layer = num_all_layers[i]
            height_step = height / float(num_cur_layer + 1)
            layer_nodes = []
            for j in range(num_cur_layer):
                pos_x = width_step * (i+1)
                pos_y = height_step * (j+1)
                layer_nodes.append([pos_x, pos_y])

                turtle.penup()
                turtle.goto(pos_x, pos_y)
                turtle.pendown()
                turtle.begin_fill()
                if 0 == i:
                    turtle.color('red')
                    turtle.fillcolor('red')
                elif len(num_all_layers) - 1 == i:
                    turtle.color('blue')
                    turtle.fillcolor('blue')
                else:
                    turtle.color('green')
                    turtle.fillcolor('green')
                turtle.circle(node_radius)
                turtle.end_fill()
            layers_nodes.append(layer_nodes)

        # print links
        turtle.color('black')
        for i in range(len(layers_nodes)-1):
            for j in range(num_all_layers[i]):
                for k in range(num_all_layers[i+1]):
                    turtle.penup()
                    turtle.goto(layers_nodes[i][j][0] + node_radius, layers_nodes[i][j][1] + node_radius)
                    turtle.pendown()
                    turtle.goto(layers_nodes[i+1][k][0] - node_radius, layers_nodes[i+1][k][1] + node_radius)

        turtle.mainloop()


if __name__ == '__main__':
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    train_data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    train_data[train_data[:, 2] == -1, 2] = 0.  # preprocess
    # train_data = train_data[:, -3:] # step for watermelon dataset
    train_data.astype(np.float)
    np.random.shuffle(train_data)

    classifier = FullyConnectedNeuralNetwork(train_data[:100, :], test_data=train_data[100:, :],
                                             num_hidden_layers_nodes=[6], activate='tanh', activate_derivative='tanh')
    classifier.train(max_epoch=2000, learning_rate=0.01)
    classifier.visualize_train_loss()
    classifier.save_model()
    classifier.load_model()
    print('error rate of test dataset: {}'.format(classifier.evaluate_test_dataset()))
    print('error rate of train dataset: {}'.format(classifier.evaluate_train_dataset()))

    # cost_time = classifier.visualize_train_data_and_model()
    cost_time = classifier.visualize_train_data_and_several_models()
    print('cost time for visualization (for train dataset) is %d sec.' % cost_time)
    cost_time = classifier.visualize_test_data_and_model()
    print('cost time for visualization (for test dataset) is %d sec.' % cost_time)
    cost_time = classifier.visualize_all_scene_samples_with_labels()
    print('cost time for visualization (FCNN for all scene samples) is %d sec.' % cost_time)
    cost_time = classifier.visualize_random_samples_with_labels()
    print('cost time for visualization (FCNN for random samples) is %d sec.' % cost_time)

    # classifier.visualize_network_with_turtle()
