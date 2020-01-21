import sys, os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

import MyMachineLearning.FullyConnectedNeuralNetwork2 as fcnn


class UIFCNN_Window(object):
    def __init__(self, window):
        self.window = window

        self._setup_ui()

    def _setup_ui(self):
        self.window.resize(600, 400)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = UIFCNN_Window(main_window)
    ui.window.show()

    sys.exit(app.exec_())
