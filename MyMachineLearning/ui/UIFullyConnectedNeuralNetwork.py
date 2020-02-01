import sys, os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt

from MyMachineLearning.Dataset import LabeledDatasetFromFile
import MyMachineLearning.FullyConnectedNeuralNetwork2 as FCNN


class UIFCNNWindow(QtWidgets.QWidget):
    def __init__(self):
        super(UIFCNNWindow, self).__init__()
        self._setup_ui()

        self.has_data = False
        self.is_constructed = False
        self.is_trained = False

        self.structure = []

    def _setup_ui(self):
        # hyper-parameters
        self.window_width = 800
        self.window_height = 1000
        self.operate_height = 70
        self.margin = 20
        self.node_margin = 150
        self.node_radius = 30
        self.line_width = 3

        self.setObjectName("Main Window")
        self.resize(self.window_width, self.window_height)

        vertical_layout = QtWidgets.QVBoxLayout()

        layout1 = QtWidgets.QHBoxLayout()
        self.input_btn = QtWidgets.QPushButton("Input Data")
        self.label = QtWidgets.QLabel("Number of output nodes: ")
        self.output_nodes_spin_box = QtWidgets.QSpinBox()
        layout1.addWidget(self.input_btn)
        layout1.addWidget(self.label)
        layout1.addWidget(self.output_nodes_spin_box)
        vertical_layout.addLayout(layout1)

        layout2 = QtWidgets.QHBoxLayout()
        self.label2 = QtWidgets.QLabel("Layer Operate Pos: ")
        self.layer_pos_spin_box = QtWidgets.QSpinBox()
        self.layer_pos_spin_box.setValue(1)
        self.insert_layer_btn = QtWidgets.QPushButton("Insert")
        self.remove_layer_btn = QtWidgets.QPushButton("Remove")
        layout2.addWidget(self.label2)
        layout2.addWidget(self.layer_pos_spin_box)
        layout2.addWidget(self.insert_layer_btn)
        layout2.addWidget(self.remove_layer_btn)
        vertical_layout.addLayout(layout2)

        layout3 = QtWidgets.QHBoxLayout()
        self.label3 = QtWidgets.QLabel("Node Operate Pos: ")
        self.node_pos_spin_box = QtWidgets.QSpinBox()
        self.node_pos_spin_box.setValue(1)
        self.insert_node_btn = QtWidgets.QPushButton("Insert")
        self.remove_node_btn = QtWidgets.QPushButton("Remove")
        layout3.addWidget(self.label3)
        layout3.addWidget(self.node_pos_spin_box)
        layout3.addWidget(self.insert_node_btn)
        layout3.addWidget(self.remove_node_btn)
        vertical_layout.addLayout(layout3)

        self.vis_label = QtWidgets.QLabel()
        self.vis_label.setFixedWidth(self.window_width)
        self.vis_label.setFixedHeight(self.window_height-self.operate_height)
        vertical_layout.addWidget(self.vis_label)

        self.setLayout(vertical_layout)

        self._widgets_state_init_status()
        self._set_connect()

    def _widgets_state_init_status(self):
        self.output_nodes_spin_box.setEnabled(False)
        self.layer_pos_spin_box.setEnabled(False)
        self.insert_layer_btn.setEnabled(False)
        self.remove_layer_btn.setEnabled(False)
        self.node_pos_spin_box.setEnabled(False)
        self.insert_node_btn.setEnabled(False)
        self.remove_node_btn.setEnabled(False)

    def _set_connect(self):
        self.input_btn.clicked.connect(self._input_btn_clicked)
        self.output_nodes_spin_box.valueChanged.connect(self._output_nodes_spin_box_value_changed)

        self.insert_layer_btn.clicked.connect(self._insert_layer_btn_clicked)
        self.remove_layer_btn.clicked.connect(self._remove_layer_btn_clicked)

        self.insert_node_btn.clicked.connect(self._insert_node_btn_clicked)
        self.remove_node_btn.clicked.connect(self._remove_node_btn_clicked)

    def _insert_node_btn_clicked(self):
        pos = self.node_pos_spin_box.value()
        if pos <= 0 or pos >= len(self.structure) - 1:
            return
        self.structure[pos] += 1
        self.update()

    def _remove_node_btn_clicked(self):
        pos = self.layer_pos_spin_box.value()
        if pos <= 0 or pos >= len(self.structure) - 1:
            return
        if self.structure[pos] == 1:
            return
        self.structure[pos] -= 1
        self.update()

    def _insert_layer_btn_clicked(self):
        pos = self.layer_pos_spin_box.value()
        if pos <= 0 or pos >= len(self.structure):
            return
        self.structure.insert(pos, 1)
        self.update()

    def _remove_layer_btn_clicked(self):
        if len(self.structure) == 2:
            return
        pos = self.layer_pos_spin_box.value()
        if pos <= 0 or pos >= len(self.structure) - 1:
            return
        self.structure.pop(pos)
        self.update()

    def _input_btn_clicked(self):
        self.data_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Input the Data File', './',
                                                     'Data Files(*.xls)')
        if self.data_file_path is None or self.data_file_path == '':
            return
        else:
            self._has_data_status()
            self.structure = [self.raw_data.shape[1]-1, 2]
            self.update()
            #QtWidgets.QMessageBox.information(None, "Message", "Input the data file done.", QtWidgets.QMessageBox.Ok)

    def _output_nodes_spin_box_value_changed(self, val):
        if len(self.structure) == 0:
            return
        if val < 2:
            val = 2
            self.output_nodes_spin_box.setValue(val)
        self.structure[-1] = val
        self.update()

    def _has_data_status(self):
        self.has_data = True
        data = LabeledDatasetFromFile(self.data_file_path).get_data_by_sheet(0)
        self.raw_data = data

        self.output_nodes_spin_box.setEnabled(True)
        self.layer_pos_spin_box.setEnabled(True)
        self.insert_layer_btn.setEnabled(True)
        self.remove_layer_btn.setEnabled(True)
        self.node_pos_spin_box.setEnabled(True)
        self.insert_node_btn.setEnabled(True)
        self.remove_node_btn.setEnabled(True)

    def mousePressEvent(self, event):
        s = event.windowPos()
        self.setMouseTracking(True)
        print(str(int(s.x())) + ':' + str(int(s.y())))

    def paintEvent(self, event):
        if len(self.structure) == 0:
            return

        painter = QPainter()

        width_step = (self.window_width - 2 * self.node_margin) / (len(self.structure) - 1)
        width_cur = self.node_margin

        last_layer_nodes = []
        for i in range(len(self.structure)):
            if 0 == i:
                brush = QBrush(Qt.red)
                pen = QPen(Qt.red)
            elif len(self.structure) - 1 == i:
                brush = QBrush(Qt.green)
                pen = QPen(Qt.green)
            else:
                brush = QBrush(Qt.blue)
                pen = QPen(Qt.blue)

            if self.structure[i] == 1:
                height_cur = (self.operate_height + self.node_margin + self.window_height) / 2
                height_step = 0
            else:
                height_cur = self.operate_height + self.node_margin
                height_step = (self.window_height - self.operate_height - 2 * self.node_margin) / (self.structure[i] - 1)
            last_layer_nodes_tmp = []
            for _ in range(self.structure[i]):
                painter.begin(self)
                painter.setBrush(brush)
                painter.setPen(pen)
                painter.drawEllipse(width_cur-self.node_radius, height_cur-self.node_radius, self.node_radius*2, self.node_radius*2)
                last_layer_nodes_tmp.append((width_cur, height_cur))

                for node in last_layer_nodes:
                    pen2 = QPen(Qt.black)
                    pen2.setWidth(self.line_width)
                    painter.setPen(pen2)
                    painter.drawLine(node[0] + self.node_radius, node[1], width_cur - self.node_radius, height_cur)
                painter.end()
                height_cur += height_step
            last_layer_nodes = last_layer_nodes_tmp
            width_cur += width_step


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UIFCNNWindow()
    ui.show()

    sys.exit(app.exec_())
