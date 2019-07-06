#%%
import numpy as np
from MyMachineLearning.Dataset import *


#%%
class DecisionTree:
    def __init__(self, dataset):
        """
        feats: nsamples * nfeats
        断言：columns列的名字与feats列的顺序是一一对应的
        """
        self.__feats, self.__labels = dataset.get_feats_and_labels()
        self.__nsamples, self.__nfeats = dataset.get_nsamples_and_nfeats()
        self.__columns = dataset.get_columns()
        self.__feats_values = dataset.get_feats_values()
        self.__attr_is_seq = dataset.get_attr_is_seq()
        self.__tree = None

        self.__IV = {}  # for C4.5

    @staticmethod
    def __samples_same_on_attrs(dataset, attrs):
        for attr_ind in attrs:
            if dataset[:, attr_ind].tolist().count(dataset[0][attr_ind]) != dataset.shape[0]:
                return False
        return True

    @staticmethod
    def __samples_on_most_labels(dataset):
        labels_count = {}
        labels = dataset[:, -1]
        for i in range(labels.shape[0]):
            if labels[i] not in labels_count.keys():
                labels_count[labels[i]] = 0
            labels_count[labels[i]] += 1

        max_key = -1
        max_value = -1
        for key in labels_count.keys():
            if max_value < labels_count[key]:
                max_value = labels_count[key]
                max_key = key
        return max_key

    @staticmethod
    def __calc_dataset_pk(dataset):
        labels = dataset[:, -1]
        labels_count = {}
        for label in labels:
            if label not in labels_count.keys():
                labels_count[label] = 0
            labels_count[label] = labels_count[label] + 1

        return labels_count

    @staticmethod
    def __calc_dataset_info_entropy(dataset):
        labels_count = DecisionTree.__calc_dataset_pk(dataset)

        ent = 0.0
        for key in labels_count.keys():
            p = labels_count[key] / labels.shape[0]
            ent = ent + p * np.log2(p)

        return ent

    @staticmethod
    def __split_dataset_by_feat_values(dataset, attr, feat_values):
        splits = []
        for feat_value in feat_values:
            splits.append(np.where(dataset[:, attr] == feat_value)[0].tolist())

        return splits

    def __calc_partition_attr_info_gain(self, dataset, attr):
        ent = self.__calc_dataset_info_entropy(dataset)
        ent2 = 0.0
        splits = self.__split_dataset_by_feat_values(dataset, attr, self.__feats_values[attr])
        for split in splits:
            ent2 += float(len(split)) / dataset.shape[0] * self.__calc_dataset_info_entropy(dataset[split, :])
        gain = ent - ent2

        return gain

    def __calc_attr_IV(self, dataset, attr):
        iv = 0.0
        splits = self.__split_dataset_by_feat_values(dataset, attr, self.__feats_values[attr])
        for split in splits:
            iv += -float(len(split)) / dataset.shape[0] * np.log2(float(len(split)) / dataset.shape[0])
        return iv

    def __calc_partition_attr_gain_ratio(self, dataset, attr):
        gain = self.__calc_partition_attr_info_gain(dataset, attr)
        if attr not in self.__IV.keys():
            self.__IV[attr] = self.__calc_attr_IV(dataset, attr)

        return gain / self.__IV[attr]

    def __choose_best_partition_attr_ID3(self, dataset, attrs):
        max_ = -1
        best_attr = attrs[0]
        best_attr_ind = 0
        for i in range(len(attrs)):
            attr = attrs[i]
            cur_ = self.__calc_partition_attr_info_gain(dataset, attr)
            if max_ < cur_:
                max_ = cur_
                best_attr = attr
                best_attr_ind = i

        return best_attr, best_attr_ind

    def __choose_best_partition_attr_C45(self, dataset, attrs):
        max_ = -1
        best_attr = attrs[0]
        best_attr_ind = 0
        gain_ratios = np.zeros((len(attrs), 1), dtype=np.float)
        for i in range(len(attrs)):
            gain_ratios[i] = self.__calc_partition_attr_gain_ratio(dataset, attrs[i])
        mean_ = np.mean(gain_ratios)
        candidate_attrs = []
        for i in range(len(attrs)):
            attr = attrs[i]
            if gain_ratios[i] > mean_:
                candidate_attrs.append(attr)

        for i in range(len(candidate_attrs)):
            attr = candidate_attrs[i]
            cur_ = self.__calc_partition_attr_gain_ratio(dataset, attr)
            if max_ > cur_:
                max_ = cur_
                best_attr = attr
                best_attr_ind = i

        return best_attr, best_attr_ind

    @staticmethod
    def __calc_Gini(dataset):
        labels_count = DecisionTree.__calc_dataset_pk(dataset)
        gini = 1.0
        for key in labels_count.keys():
            gini = gini - (labels_count[key] / dataset.shape[0]) ** 2

        return gini

    def __calc_Gini_index(self, dataset, attr):
        gini_index = 0.0
        splits = self.__split_dataset_by_feat_values(dataset, attr, self.__feats_values[attr])
        for split in splits:
            gini_index = gini_index + len(split) / dataset.shape[0] * self.__calc_Gini(dataset[split, :])

        return gini_index

    def __choose_best_partition_attr_CART(self, dataset, attrs):
        # 使用基尼指数选择划分属性
        min_ = np.Inf
        best_attr = 0
        best_attr_ind = 0
        for i in range(len(attrs)):
            attr = attrs[i]
            cur_ = self.__calc_Gini_index(dataset, attr)
            if min_ > cur_:
                min_ = cur_
                best_attr = attr
                best_attr_ind = i

        return best_attr, best_attr_ind

    def __generate_nodes(self, dataset, attrs, mode=1):
        labels = dataset[:, -1]
        if labels.tolist().count(labels[0]) == labels.shape[0]:
            return labels[0]
        if not attrs or self.__samples_same_on_attrs(dataset, attrs):
            return self.__samples_on_most_labels(dataset)

        if mode == 1:
            best_attr, best_attr_ind = self.__choose_best_partition_attr_ID3(dataset, attrs)
        elif mode == 2:
            best_attr, best_attr_ind = self.__choose_best_partition_attr_C45(dataset, attrs)
        elif mode == 3:
            best_attr, best_attr_ind = self.__choose_best_partition_attr_CART(dataset, attrs)
        else:
            return None

        nodes = {best_attr: {}}
        splits = self.__split_dataset_by_feat_values(dataset, best_attr, self.__feats_values[best_attr])
        for i in range(len(splits)):
            split = splits[i]
            if not split:
                return self.__samples_on_most_labels(dataset)
            else:
                new_attrs = attrs.copy()
                new_attrs.pop(best_attr_ind)
                nodes[best_attr][self.__feats_values[best_attr][i]] = self.__generate_nodes(dataset[split, :], new_attrs)

        return nodes

    def generate_tree(self, mode=1):
        """
        :param mode: 选择最优划分属性的方式。1 for ID3, 2 for C4.5, 3 for CART
        :return:
        """
        # 先暂不考虑连续属性，假定所有属性都是离散的！！！
        # 暂定代码，后面会需要删除
        attrs = [i for i in range(self.__nfeats) if i not in self.__attr_is_seq]
        dataset = np.concatenate((self.__feats[:, attrs], self.__labels.reshape((self.__nsamples, 1))), 1)

        self.__tree = self.__generate_nodes(dataset, attrs, mode=mode)

    def get_tree(self):
        return self.__tree

    @staticmethod
    def __dfs_tree(tree, feat):
        if type(tree).__name__ == 'dict':
            key = list(tree.keys())[0]
            _tree = tree[key][feat[key]]
            return DecisionTree.__dfs_tree(_tree, feat)
        else:
            return tree

    def pred(self, feat):
        if self.__tree is None:
            return -1
        else:
            return self.__dfs_tree(self.__tree, feat)

    def evaluate_result(self):
        if self.__tree is None:
            return -1
        accuracy = 0
        for feat, label in zip(self.__feats, self.__labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        error = 1 - accuracy / self.__nsamples
        return error


#%%
if __name__ == '__main__':
    data_address = r'F:/工作/GitHub/LearningMachineLearning/dataset/watermelon3.xlsx'
    datasetff = LabeledDatasetFromFile(data_address)
    feats, labels = datasetff.get_feats_and_labels_by_sheet(0)
    dataset = LabeledDataset(feats, labels,
                             columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'], attr_is_seq=[6, 7])
    # print(feats)

    tree = DecisionTree(dataset)

    # ID3
    print('ID3 algorithm')
    tree.generate_tree(mode=1)
    nodes = tree.get_tree()
    print(nodes)
    print(tree.evaluate_result())

    # C4.5
    print('C4.5 algorithm')
    tree.generate_tree(mode=2)
    nodes = tree.get_tree()
    print(nodes)
    print(tree.evaluate_result())

    # CART
    print('CART algorithm')
    tree.generate_tree(mode=3)
    nodes = tree.get_tree()
    print(nodes)
    print(tree.evaluate_result())
