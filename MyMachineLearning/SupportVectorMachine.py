#%%
import numpy as np
import xlrd
from MyMachineLearning.Dataset import LabeledDatasetFromFile, LabeledTrainAndTestDataset


#%%
class SupportVectorMachine:
    def __init__(self, train_data, test_data):
        self.__train_data = train_data
        self.__test_data = test_data


#%%
if __name__ == '__main__':
    # 调整数据格式
    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\demodata.xls'
    datasetff = LabeledDatasetFromFile(data_address).get_data_by_sheet(0)
    datasetff = datasetff[:, -3:]  # 只使用连续属性值
    datasetff[datasetff[:, -1] == 0, -1] = -1

    dataset = LabeledTrainAndTestDataset(datasetff)
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    # 获得模型
    svm = SupportVectorMachine(train_data, test_data)
