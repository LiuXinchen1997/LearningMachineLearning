#%%
import xlrd
import numpy as np
import utils.CONSTANT

#%%
class LabeledDatasetFromFile:
    def __init__(self, data_address, datafile_type='xlsx'):
        self.datafile_type = datafile_type
        self.readbook = None

        if datafile_type in ['xlsx', 'xls']:
            self.readbook = xlrd.open_workbook(data_address)

    def get_data_by_sheet(self, index, mode=''):
        """
        :param index: index of sheet
        :param normal:
        :return: nsamples * nfeats
        """
        sheet = self.readbook.sheet_by_index(index)
        nrows = sheet.nrows
        ncols = sheet.ncols

        data = np.zeros((nrows, ncols)).astype(np.float)
        for i in range(nrows):
            for j in range(ncols):
                data[i, j] = sheet.cell(i, j).value

        if mode == utils.CONSTANT.TRANS:
            data = data.transpose()

        return data

    def get_feats_and_labels_by_sheet(self, index, mode=''):
        data = self.get_data_by_sheet(index, mode)
        nfeats = data.shape[1]
        feats = data[:, :nfeats - 1]
        labels = data[:, nfeats - 1]

        return feats, labels


# 仅使用numpy
class LabeledDataset:
    def __init__(self, feats, labels, columns=None, feats_values=None, seq_attrs=set()):
        self.__feats = feats
        self.__labels = labels
        self.__columns = columns  # 用于存储各列属性的名字（字符串）
        self.__seq_attrs = seq_attrs
        if not seq_attrs:  # 用于自动检测出连续属性
            self.__seq_attrs = LabeledDataset.__generate_seq_attrs(self.__feats)

        self.__feats_values = feats_values
        if self.__feats_values is None:
            self.__feats_values = self.__get_feats_values(self.__feats)

    @staticmethod
    def __generate_seq_attrs(feats):
        seq_attrs = set()
        nfeats = feats.shape[1]
        for i in range(nfeats):
            if len(set(feats[:, i])) > nfeats * 0.5:
                seq_attrs.add(i)

        return seq_attrs

    @staticmethod
    def __get_feats_values(feats):
        """
        :return: 获得各个属性的取值集合
        """
        feats_values = []
        for i in range(feats.shape[1]):
            feat_values = list(set([sample[i] for sample in feats]))
            feats_values.append(feat_values)

        return feats_values

    def get_feats_and_labels(self):
        return self.__feats, self.__labels

    def get_nsamples_and_nfeats(self):
        return self.__feats.shape[0], self.__feats.shape[1]

    def get_columns(self):
        return self.__columns

    def get_feats_values(self):
        return self.__feats_values

    def get_seq_attrs(self):
        return self.__seq_attrs


class LabeledTrainAndTestDataset:
    def __init__(self, train_data, test_data=None, test_ratio=0.4):
        nsamples = train_data.shape[0]
        if test_data is None:
            indces = [i for i in range(nsamples)]
            np.random.shuffle(indces)
            test_data = train_data[indces[:int(nsamples * test_ratio)], :]
            train_data = train_data[indces[int(nsamples * test_ratio):], :]
        self.__train_data = train_data
        self.__test_data = test_data

    def get_train_data(self):
        return self.__train_data

    def get_train_feats_and_labels(self):
        return self.__train_data[:, :-1], self.__train_data[:, -1]

    def get_test_data(self):
        return self.__test_data

    def get_test_feats_and_labels(self):
        return self.__test_data[:, :-1], self.__test_data[:, -1]

# 待开发，使用pandas改写！
