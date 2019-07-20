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
        self.__seq_attrs = dataset.get_seq_attrs()
        self.__attrs_are_seq = DecisionTree.__get_attrs_are_seq(self.__nfeats, self.__seq_attrs)
        self.__tree = None

    @staticmethod
    def __get_attrs_are_seq(nfeats, seq_attrs):
        attrs_are_seq = []
        for i in range(nfeats):
            if i in seq_attrs:
                attrs_are_seq.append(True)
            else:
                attrs_are_seq.append(False)

        return np.array(attrs_are_seq)

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

    @staticmethod
    def __split_dataset_by_candidate_t(dataset, attr, candidate_t):
        splits = []
        splits.append(np.where(dataset[:, attr] >= candidate_t)[0].tolist())
        splits.append(np.where(dataset[:, attr] < candidate_t)[0].tolist())

        return splits

    def __calc_partition_attr_info_gain(self, dataset, attr):
        # 只处理离散属性
        ent = self.__calc_dataset_info_entropy(dataset)
        ent2 = 0.0
        splits = self.__split_dataset_by_feat_values(dataset, attr, self.__feats_values[attr])
        for split in splits:
            ent2 += float(len(split)) / dataset.shape[0] * self.__calc_dataset_info_entropy(dataset[split, :])
        gain = ent - ent2

        return gain

    def __calc_partition_seq_attr_info_gain(self, dataset, attr):
        candidate_ts = self.__get_candidate_ts(dataset, attr)
        max_gain = -np.inf
        best_candidate_t = -1
        for candidate_t in candidate_ts:
            cur_gain = self.__calc_dataset_info_entropy(dataset)
            splits = self.__split_dataset_by_candidate_t(dataset, attr, candidate_t)
            for split in splits:
                cur_gain = cur_gain - self.__calc_dataset_info_entropy(dataset[split, :])
            if max_gain < cur_gain:
                max_gain = cur_gain
                best_candidate_t = candidate_t

        return best_candidate_t, max_gain

    def __calc_attr_iv(self, dataset, attr, candidate_t=None):
        epsilon = 1e-6
        iv = 0.0
        if self.__attrs_are_seq[attr] and candidate_t is not None:
            splits = self.__split_dataset_by_candidate_t(dataset, attr, candidate_t)
        else:
            splits = self.__split_dataset_by_feat_values(dataset, attr, self.__feats_values[attr])
        for split in splits:
            iv += -float(len(split)) / dataset.shape[0] * np.log2(float(len(split)+epsilon) / dataset.shape[0])

        return iv

    def __calc_partition_attr_gain_ratio(self, dataset, attr):
        gain = self.__calc_partition_attr_info_gain(dataset, attr)
        iv = self.__calc_attr_iv(dataset, attr)

        return gain / iv

    def __calc_partition_seq_attr_gain_ratio(self, dataset, attr):
        candidate_ts = self.__get_candidate_ts(dataset, attr)
        max_gain_ratio = -np.inf
        best_candidate_t = -1
        for candidate_t in candidate_ts:
            cur_gain = self.__calc_dataset_info_entropy(dataset)
            splits = self.__split_dataset_by_candidate_t(dataset, attr, candidate_t)
            for split in splits:
                cur_gain = cur_gain - self.__calc_dataset_info_entropy(dataset[split, :])
            cur_iv = self.__calc_attr_iv(dataset, attr, candidate_t=candidate_t)
            cur_gain_ratio = cur_gain / cur_iv
            if max_gain_ratio < cur_gain_ratio:
                max_gain_ratio = cur_gain_ratio
                best_candidate_t = candidate_t

        return best_candidate_t, max_gain_ratio

    def __choose_best_partition_attr_id3(self, dataset, attrs):
        max_ = -1
        best_attr = attrs[0]
        best_attr_ind = 0
        best_candidate_t = -1
        for i in range(len(attrs)):
            attr = attrs[i]
            t = -1
            if self.__attrs_are_seq[attr]:
                t, cur_ = self.__calc_partition_seq_attr_info_gain(dataset, attr)
            else:
                cur_ = self.__calc_partition_attr_info_gain(dataset, attr)

            if max_ < cur_:
                max_ = cur_
                best_attr = attr
                best_attr_ind = i
                if self.__attrs_are_seq[attr]:
                    best_candidate_t = t

        return best_attr, best_attr_ind, best_candidate_t

    def __choose_best_partition_attr_c45(self, dataset, attrs):
        max_ = -1
        best_attr = attrs[0]
        best_attr_ind = 0
        gain_ratios = np.zeros((len(attrs), 1), dtype=np.float)
        best_candidate_t = -1
        seq_attr2candidate_t = dict()
        for i in range(len(attrs)):
            if self.__attrs_are_seq[attrs[i]]:
                t, gain_ratios[i] = self.__calc_partition_seq_attr_gain_ratio(dataset, attrs[i])
                seq_attr2candidate_t[attrs[i]] = t
            else:
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
                if self.__attrs_are_seq[best_attr]:
                    best_candidate_t = seq_attr2candidate_t[best_attr]

        return best_attr, best_attr_ind, best_candidate_t

    @staticmethod
    def __calc_gini(dataset):
        labels_count = DecisionTree.__calc_dataset_pk(dataset)
        gini = 1.0
        for key in labels_count.keys():
            gini = gini - (labels_count[key] / dataset.shape[0]) ** 2

        return gini

    def __calc_partition_attr_gini_index(self, dataset, attr):
        gini_index = 0.0
        splits = self.__split_dataset_by_feat_values(dataset, attr, self.__feats_values[attr])
        for split in splits:
            gini_index = gini_index + len(split) / dataset.shape[0] * self.__calc_gini(dataset[split, :])

        return gini_index
    
    def __calc_partition_seq_attr_gini_index(self, dataset, attr):
        candidate_ts = self.__get_candidate_ts(dataset, attr)
        min_gini_index = np.inf
        best_candidate_t = -1
        for candidate_t in candidate_ts:
            cur_gini_index = 0.0
            splits = self.__split_dataset_by_candidate_t(dataset, attr, candidate_t)
            for split in splits:
                cur_gini_index = cur_gini_index + len(split) / dataset.shape[0] * self.__calc_gini(dataset[split, :])
            if min_gini_index > cur_gini_index:
                min_gini_index = cur_gini_index
                best_candidate_t = candidate_t

        return best_candidate_t, min_gini_index

    def __choose_best_partition_attr_cart(self, dataset, attrs):
        # 使用基尼指数选择划分属性
        min_ = np.Inf
        best_attr = 0
        best_attr_ind = 0
        best_candidate_t = -1
        for i in range(len(attrs)):
            attr = attrs[i]
            t = -1
            if self.__attrs_are_seq[attr]:
                t, cur_ = self.__calc_partition_seq_attr_gini_index(dataset, attr)
            else:
                cur_ = self.__calc_partition_attr_gini_index(dataset, attr)
            if min_ > cur_:
                min_ = cur_
                best_attr = attr
                best_attr_ind = i
                if self.__attrs_are_seq[attr]:
                    best_candidate_t = t

        return best_attr, best_attr_ind, best_candidate_t

    @staticmethod
    def __get_candidate_ts(dataset, attr):
        candidate_ts = []
        feat_values = list(set(dataset[:, attr].tolist()))
        sorted_feat_values = np.sort(feat_values)
        for i in range(len(sorted_feat_values) - 1):
            candidate_ts.append((sorted_feat_values[i] + sorted_feat_values[i+1]) / 2.0)

        return candidate_ts

    def __get_feats_and_labels(self):
        return self.__feats, self.__labels

    def __get_entire_dataset(self):
        feats, labels = self.__get_feats_and_labels()
        dataset = np.concatenate((feats, labels.reshape((self.__nsamples, 1))), 1)

        return dataset

    def __get_entire_attrs(self):
        attrs = [i for i in range(self.__nfeats)]
        return attrs

    def __generate_nodes(self, dataset, attrs, mode=1):
        labels = dataset[:, -1]
        if labels.tolist().count(labels[0]) == labels.shape[0]:
            return labels[0]
        if not attrs or self.__samples_same_on_attrs(dataset, attrs):
            return self.__samples_on_most_labels(dataset)

        if mode == 1:
            best_attr, best_attr_ind, best_candidate_t = self.__choose_best_partition_attr_id3(dataset, attrs)
        elif mode == 2:
            best_attr, best_attr_ind, best_candidate_t = self.__choose_best_partition_attr_c45(dataset, attrs)
        elif mode == 3:
            best_attr, best_attr_ind, best_candidate_t = self.__choose_best_partition_attr_cart(dataset, attrs)
        else:
            raise Exception

        nodes = {best_attr: {}}
        if self.__attrs_are_seq[best_attr]:
            splits = self.__split_dataset_by_candidate_t(dataset, best_attr, best_candidate_t)
        else:
            splits = self.__split_dataset_by_feat_values(dataset, best_attr, self.__feats_values[best_attr])
        for i in range(len(splits)):
            split = splits[i]
            if not split:
                if self.__attrs_are_seq[best_attr]:
                    s = '>=' if 0 == i else '<'
                    s += str(best_candidate_t)
                    nodes[best_attr][s] = self.__samples_on_most_labels(dataset)
                else:
                    nodes[best_attr][self.__feats_values[best_attr][i]] = self.__samples_on_most_labels(dataset)
            else:
                new_attrs = attrs.copy()
                new_attrs.pop(best_attr_ind)
                if self.__attrs_are_seq[best_attr]:
                    s = '>=' if 0 == i else '<'
                    s += str(best_candidate_t)
                    nodes[best_attr][s] = self.__generate_nodes(dataset[split, :], new_attrs)
                else:
                    nodes[best_attr][self.__feats_values[best_attr][i]] = \
                        self.__generate_nodes(dataset[split, :], new_attrs)

        return nodes

    def generate_tree(self, mode=1):
        """
        :param mode: 选择最优划分属性的方式。1 for ID3, 2 for C4.5, 3 for CART
        :return:
        """
        dataset = self.__get_entire_dataset()
        attrs = self.__get_entire_attrs()

        self.__tree = self.__generate_nodes(dataset, attrs, mode=mode)

    def get_tree(self):
        return self.__tree

    @staticmethod
    def __get_candidate_t_from_node(tree):
        flag_s = list(tree.keys())[0]
        if flag_s.startswith('>='):
            flag_s = flag_s[2:]
        else:
            flag_s = flag_s[1:]

        return float(flag_s)

    def __dfs_tree(self, tree, feat):
        if type(tree).__name__ == 'dict':
            key = list(tree.keys())[0]

            if self.__attrs_are_seq[int(key)]:
                candidate_t = self.__get_candidate_t_from_node(tree[key])
                if feat[key] >= candidate_t:
                    _tree = tree[key]['>='+str(candidate_t)]
                else:
                    _tree = tree[key]['<'+str(candidate_t)]
            else:
                _tree = tree[key][feat[key]]
            return self.__dfs_tree(_tree, feat)
        else:
            return tree

    def pred(self, feat):
        if self.__tree is None:
            return -1
        else:
            return self.__dfs_tree(self.__tree, feat)

    def evaluate_result_with_train_data(self):
        if self.__tree is None:
            return -1
        accuracy = 0
        for feat, label in zip(self.__feats, self.__labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        error = 1 - accuracy / self.__nsamples
        return error

    def evaluate_result_with_test_data(self, feats, labels):
        if self.__tree is None:
            return -1
        accuracy = 0
        for feat, label in zip(feats, labels):
            judge = self.pred(feat)
            accuracy += (judge == label)

        nsamples = feats.shape[0]
        error = 1 - accuracy / nsamples
        return error

    def evaluate_result_with_test_data2(self, dataset):
        feats = dataset[:, :-1]
        labels = dataset[:, -1]
        return self.evaluate_result_with_test_data(feats, labels)


#%%
if __name__ == '__main__':
    import utils.CONSTANT

    data_address = r'D:\Project\Github\LearningMachineLearning\dataset\watermelon3.xlsx'
    datasetff = LabeledDatasetFromFile(data_address)
    feats, labels = datasetff.get_feats_and_labels_by_sheet(0, mode=utils.CONSTANT.TRANS)
    dataset = LabeledDataset(feats, labels,
                             columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'], seq_attrs={6, 7})
    # print(feats)
    tree = DecisionTree(dataset)

    # ID3
    print('ID3 algorithm')
    tree.generate_tree(mode=1)
    nodes = tree.get_tree()
    print(nodes)
    print(tree.evaluate_result_with_train_data())

    # C4.5
    print('C4.5 algorithm')
    tree.generate_tree(mode=2)
    nodes = tree.get_tree()
    print(nodes)
    print(tree.evaluate_result_with_train_data())

    # CART
    print('CART algorithm')
    tree.generate_tree(mode=3)
    nodes = tree.get_tree()
    print(nodes)
    print(tree.evaluate_result_with_train_data())
