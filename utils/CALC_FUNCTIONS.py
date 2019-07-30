import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    y = 1. / (1. + np.exp(-x))
    if isinstance(y, np.matrix):
        return np.multiply(y, (1 - y))
    return y * (1 - y)
