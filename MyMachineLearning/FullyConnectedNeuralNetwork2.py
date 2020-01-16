import numpy as np


class Node:
    def __init__(self, activate, in_val=None):
        self.activate = activate
        self.__in_val = in_val
        if self.__in_val is None:
            self.__out_val = None
        else:
            self.__out_val = activate(self.__in_val)

    def set_in_val(self, in_val):
        self.__in_val = in_val
        self.__out_val = self.activate(self.__in_val)

    def set_out_val(self, out_val):
        self.__out_val = out_val

    def get_in_val(self):
        return self.__in_val

    def get_out_val(self):
        return self.__out_val


class Layer:
    def __init__(self, nodes):
        self.__nodes = nodes

    def get_nodes(self):
        return self.__nodes


class FullyConnectedNeuralNetwork2:
    def __init__(self, layers):
        self.__layers = layers


if __name__ == '__main__':
    print('hello')