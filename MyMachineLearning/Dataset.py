#%%
import xlrd
import numpy as np


#%%
# 仅使用numpy
class LabeledDataset:
    def __init__(self, feats, labels, columns=None, feats_values=None, attr_is_seq=[]):
        """
        :param prob_is_seq: 表示连续属性的编号
        """
        self.__feats = feats
        self.__labels = labels
        self.__columns = columns  # 用于存储各列属性的名字（字符串）
        self.__attr_is_seq = attr_is_seq

        self.__feats_values = feats_values
        if self.__feats_values is None:
            self.__feats_values = self.__get_feats_values(self.__feats)

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

    def get_attr_is_seq(self):
        return self.__attr_is_seq


class LabeledDatasetFromFile:
    def __init__(self, data_address, datafile_type='xlsx'):
        self.datafile_type = datafile_type
        self.readbook = None

        if datafile_type in ['xlsx']:
            self.readbook = xlrd.open_workbook(data_address)

    def get_data_by_sheet(self, index, mode='trans'):
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

        if mode == 'trans':
            data = data.transpose()

        return data

    def get_feats_and_labels_by_sheet(self, index, mode='trans'):
        data = self.get_data_by_sheet(index, mode)
        nfeats = data.shape[1]
        feats = data[:, :nfeats - 1]
        labels = data[:, nfeats - 1]

        return feats, labels


# 待开发，使用pandas改写！
