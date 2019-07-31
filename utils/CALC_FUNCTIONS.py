import numpy as np


# activate functions
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_derivative(x):
    y = sigmoid(x)
    if isinstance(y, np.matrix):
        return np.multiply(y, (1 - y))
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    if isinstance(x, np.matrix):
        return 1 - np.multiply(np.tanh(x), np.tanh(x))
    else:
        return 1 - np.tanh(x) * np.tanh(x)


def relu(x):
    return (np.abs(x) + x) / 2.


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def _none(x):
    return x


def _none_derivative(x):
    return 1
