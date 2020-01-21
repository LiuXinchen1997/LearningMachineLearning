import numpy as np
from tqdm import tqdm

from MyMachineLearning.utils.CALC_FUNCTIONS import sigmoid, sigmoid_derivative
from MyMachineLearning.Dataset import LabeledDatasetFromFile


class Node:
    def __init__(self, activate, activate_derivative, in_val=None, is_input_node=False):
        self.activate = activate
        self.activate_derivative = activate_derivative
        self.__in_val = in_val
        self.__is_input_node = is_input_node
        if self.__in_val is None:
            self.__out_val = None
        else:
            if is_input_node:
                self.__out_val = self.__in_val
            else:
                self.__out_val = activate(self.__in_val)

    def set_in_val(self, in_val):
        self.__in_val = in_val
        if self.__is_input_node:
            self.__out_val = self.__in_val
        else:
            self.__out_val = self.activate(self.__in_val)

    def set_out_val(self, out_val):
        self.__out_val = out_val

    def update_out_val(self):
        if self.__is_input_node:
            self.__out_val = self.__in_val
        else:
            self.__out_val = self.activate(self.__in_val)

    def get_in_val(self):
        return self.__in_val

    def get_out_val(self):
        return self.__out_val

    def get_activate(self):
        return self.activate

    def get_activate_derivative(self):
        return self.activate_derivative


class Layer:
    def __init__(self, nodes, next_layer=None):
        self.__nodes = nodes
        self.__next_layer = next_layer
        if next_layer is None:
            self.__omega = None
            self.__b = None
        else:
            self.__omega = np.random.random((len(self.__nodes), len(next_layer.get_nodes())))
            self.__b = np.random.random((len(next_layer.get_nodes())))

    def update_out_vals(self):
        for node in self.__nodes:
            node.update_out_val()

    def get_nodes(self):
        return self.__nodes

    def get_next_layer(self):
        return self.__next_layer

    def set_next_layer(self, next_layer):
        self.__next_layer = next_layer
        self.__omega = np.random.random((len(self.__nodes), len(next_layer.get_nodes())))
        self.__b = np.random.random(len(next_layer.get_nodes()))

    def get_omega(self):
        return self.__omega

    def set_omega(self, omega):
        self.__omega = omega

    def get_b(self):
        return self.__b

    def set_b(self, b):
        self.__b = b


class FullyConnectedNeuralNetwork2:
    def __init__(self, layers, train_data, test_data):
        self.__layers = layers
        self.__is_trained = False

        self.__train_feats = train_data[:, :-1]
        self.__train_labels = train_data[:, -1]
        self.__test_feats = test_data[:, :-1]
        self.__test_labels = test_data[:, -1]

    def __feed_input_layer(self, feat):
        input_layer = self.__layers[0]
        input_nodes = input_layer.get_nodes()
        for i in range(len(input_nodes)):
            input_nodes[i].set_in_val(feat[i])
            input_nodes[i].set_out_val(feat[i])

    def __feed_next_layer(self, cur_layer, next_layer):
        cur_layer_nodes = cur_layer.get_nodes()
        next_layer_nodes = next_layer.get_nodes()

        for i in range(len(next_layer_nodes)):
            val = 0.
            for j in range(len(cur_layer_nodes)):
                val += cur_layer.get_omega()[j, i] * cur_layer_nodes[j].get_out_val()
            val += cur_layer.get_b()[i]
            next_layer_nodes[i].set_in_val(val)

        next_layer.update_out_vals()

    def __forward_propagate(self, feat):
        self.__feed_input_layer(feat)

        cur_layer = self.__layers[0]
        next_layer = cur_layer.get_next_layer()
        while next_layer is not None:
            self.__feed_next_layer(cur_layer, next_layer)
            cur_layer = next_layer
            next_layer = cur_layer.get_next_layer()

    def __label2vec(self, label):
        vec = np.zeros((len(self.__layers[-1].get_nodes()), ))
        vec[int(label)] = 1
        return vec

    def __backward_propagate(self, feat, label, learning_rate):
        self.__forward_propagate(feat)
        label_vec = self.__label2vec(label)

        ind = len(self.__layers) - 1
        delta = np.zeros((len(self.__layers[ind].get_nodes()),))
        while ind > 0:
            cur_layer = self.__layers[ind]
            cur_layer_noeds = cur_layer.get_nodes()
            num_cur_layer_nodes = len(cur_layer_noeds)
            prev_layer = self.__layers[ind - 1]
            prev_layer_nodes = prev_layer.get_nodes()
            if ind == len(self.__layers) - 1:
                for j in range(num_cur_layer_nodes):
                    cur_node = cur_layer_noeds[j]
                    delta[j] = (cur_node.get_out_val() - label_vec[j]) * cur_node.get_activate_derivative()(cur_node.get_out_val())
                    for i in range(len(prev_layer_nodes)):
                        prev_layer.get_omega()[i, j] -= learning_rate * delta[j] * prev_layer_nodes[i].get_out_val()
                    prev_layer.get_b()[j] -= learning_rate * delta[j]
            else:
                old_delta = delta.copy()
                delta = np.zeros((num_cur_layer_nodes, ))
                for j in range(num_cur_layer_nodes):
                    res = 0.
                    for k in range(old_delta.shape[0]):
                        res += old_delta[k] * cur_layer.get_omega()[j, k]
                    cur_node = cur_layer_noeds[j]
                    res *= cur_node.get_activate_derivative()(cur_node.get_out_val())
                    delta[j] = res
                    for i in range(len(prev_layer_nodes)):
                        prev_layer.get_omega()[i, j] -= learning_rate * delta[j] * prev_layer_nodes[i].get_out_val()
                    prev_layer.get_b()[j] -= learning_rate * delta[j]

            ind -= 1

    def train(self, max_epoch=20000, learning_rate=0.01):
        for i in tqdm(range(max_epoch)):
            #if i % 100 == 0:
             #   self.print_parameters()
            for (feat, label) in zip(self.__train_feats, self.__train_labels):
                self.__backward_propagate(feat, label, learning_rate=learning_rate)

        self.__is_trained = True

    def test(self):
        if not self.__is_trained:
            raise Exception("Network has not been trained.")

        correct = 0
        for (feat, label) in zip(self.__test_feats, self.__test_labels):
            correct += (self.pred(feat) == label)

        return correct / self.__test_labels.shape[0]

    def pred(self, feat):
        if not self.__is_trained:
            raise Exception("Network has not been trained.")

        self.__forward_propagate(feat)
        nodes = self.__layers[-1].get_nodes()

        y_ = 0
        max_ = nodes[0].get_out_val()
        for i in range(len(nodes)):
            if max_ < nodes[i].get_out_val():
                max_ = nodes[i].get_out_val()
                y_ = i

        return y_

    def print_layers_vals(self):
        for layer in self.__layers:
            nodes = layer.get_nodes()
            for node in nodes:
                print(node.get_in_val(), end=' ')
            print(end='\n')
            for node in nodes:
                print(node.get_out_val(), end=' ')
            print(end='\n')
            print('-------------------')

    def print_parameters(self):
        for layer in self.__layers:
            if layer.get_omega() is None:
                continue

            omega = layer.get_omega()
            print('omega: ')
            for i in range(omega.shape[0]):
                for j in range(omega.shape[1]):
                    print(omega[i, j], end=' ')
                print(end='\n')
            b = layer.get_b()
            print('b: ')
            for i in range(b.shape[0]):
                print(b[i], end=' ')
            print(end='\n')

    # modify network structure
    def append_layer_node(self, layer_id, node):
        if layer_id < 0:
            return

        cur_layer = self.__layers[layer_id]
        cur_layer.get_nodes().append(node)

        if layer_id > 0:
            prev_layer = self.__layers[layer_id-1]
            prev_layer.set_omega(np.random.random((len(prev_layer.get_nodes()), len(cur_layer.get_nodes()))))
            prev_layer.set_b(np.random.random((len(cur_layer.get_nodes()))))
        if layer_id < len(self.__layers) - 1:
            next_layer = cur_layer.get_next_layer()
            cur_layer.set_omega(np.random.random((len(cur_layer.get_nodes()), len(next_layer.get_nodes()))))
            cur_layer.set_b(np.random.random((len(next_layer.get_nodes()))))

        self.__is_trained = False


def construct_network(num_layers_nodes, train_data, test_data, activate, activate_derivative):
    """
    :param num_layers_nodes: a list of the number of several layers
    :return: object of FullyConnectedNeuralNetwork2
    """
    if len(num_layers_nodes) < 2:
        raise Exception("The number of layers is at least 2.")

    layers = []
    for i in range(len(num_layers_nodes)):
        num_layer_nodes = num_layers_nodes[i]
        nodes = []
        for _ in range(num_layer_nodes):
            if i == 0:
                nodes.append(Node(activate, activate_derivative, is_input_node=True))
            else:
                nodes.append(Node(activate, activate_derivative))
        layer = Layer(nodes)
        layers.append(layer)

    for i in range(len(layers) - 1):
        layers[i].set_next_layer(layers[i+1])

    return FullyConnectedNeuralNetwork2(layers, train_data, test_data)


if __name__ == '__main__':
    data_address = r'..\dataset\demodata.xls'
    data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    data[data[:, 2] == -1, 2] = 0.  # preprocess
    data.astype(np.float)
    np.random.shuffle(data)
    train_data = data[:100, :]
    test_data = data[100:, :]

    nn = construct_network([2, 4, 2], train_data, test_data, sigmoid, sigmoid_derivative)
    nn.train(max_epoch=3000, learning_rate=0.01)
    print(nn.test())

    # modify network structure
    nn.append_layer_node(1, Node(sigmoid, sigmoid_derivative))
    nn.train(max_epoch=3000, learning_rate=0.01)
    print(nn.test())
