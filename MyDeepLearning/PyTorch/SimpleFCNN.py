import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from MyMachineLearning.Dataset import LabeledDatasetFromFile


class Net(torch.nn.Module):
    """
    # Why not work???
    def __init__(self, structure):
        super(Net, self).__init__()
        self.structure = structure

        nlayers = len(structure)
        if nlayers < 2:
            return

        self.hiddens = []
        for i in range(nlayers):
            if 0 == i:
                continue
            elif nlayers - 1 == i:
                self.out = torch.nn.Linear(structure[i-1], structure[i])
            else:
                self.hiddens.append(torch.nn.Linear(structure[i-1], structure[i]))
        print(self.hiddens)

    def forward(self, x):
        x = x.float()
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        x = self.out(x)
        return x
    """

    def __init__(self, structure):
        super(Net, self).__init__()
        self.structure = structure

        nlayers = len(structure)
        if nlayers != 3:  # Now only support 3 layers(input layer --> 1 hidden layer --> output layer)
            return

        self.hidden = torch.nn.Linear(structure[0], structure[1])
        self.out = torch.nn.Linear(structure[1], structure[2])

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


class SimpleFCNN:
    def __init__(self, structure, train_data, test_data):
        self.net = Net(structure)
        print(self.net)

        self.train_data = train_data
        self.test_data = test_data

        self._is_trained = False

    def train(self, max_epoch=10000, learning_rate=0.01):
        x = self.train_data[:, :2]
        y = self.train_data[:, 2]

        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

        for i in tqdm(range(max_epoch)):
            out = self.net(x)                 # input x and predict based on x
            y = y.long()
            loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        out = self.net(x)
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(accuracy)

        self._is_trained = True

    def test(self):
        if not self._is_trained:
            return

        x = self.test_data[:, :2]
        y = self.test_data[:, 2]

        out = self.net(x)
        pred = torch.max(out, 1)[1]
        pred_y = pred.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

        return accuracy


if __name__ == '__main__':
    data_address = r'..\..\dataset\demodata.xls'
    train_data = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    train_data[train_data[:, 2] == -1, 2] = 0.  # preprocess
    train_data.astype(np.float)
    np.random.shuffle(train_data)

    test_data = torch.from_numpy(train_data[150:, :])
    train_data = torch.from_numpy(train_data[:150, :])

    fcnn = SimpleFCNN([2, 5, 2], train_data, test_data)
    fcnn.train(max_epoch=20000)
    print(fcnn.test())
