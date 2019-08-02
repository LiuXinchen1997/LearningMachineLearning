import numpy as np
import time
from matplotlib import pyplot as plt


def calc_visual_step(data, ratio=0.5):
    min_ = np.inf
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if j == i:
                continue
            diff = data[i, -3:-1] - data[j, -3:-1]
            tmp = np.sqrt(np.dot(diff.transpose(), diff))
            if 0 < tmp < min_:
                min_ = tmp

    step = min_ * ratio
    return step


def visualize_data_and_model(data, model, title=''):
    start_time = time.time()
    if data.shape[1] - 1 != 2:  # 特征必须是二维才能可视化！
        return

    # print sample points
    for sample in data:
        if sample[-1] == 1:
            plt.plot(sample[0], sample[1], '+r')
        else:
            plt.plot(sample[0], sample[1], '*g')

    # print model
    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])
    step = calc_visual_step(data, ratio=0.5)

    path_points = []
    cur_x = min_x
    cur_y = min_y
    pre_label = None
    pre_x = pre_y = 0
    while cur_x <= max_x:
        while cur_y <= max_y:
            cur_label = model.pred(np.array([cur_x, cur_y]))
            if (pre_label is not None) and (pre_label != cur_label):
                path_points.append([(cur_x + pre_x) / 2., (cur_y + pre_y) / 2.])
            pre_label = cur_label
            pre_x = cur_x
            pre_y = cur_y
            cur_y += step

        pre_label = 0
        pre_x = pre_y = 0
        cur_y = min_y
        cur_x += step

    sorted_path_points = []
    indces = [i for i in range(len(path_points))]
    cur_ind = 0
    indces.remove(cur_ind)
    sorted_path_points.append(path_points[cur_ind])
    min_dist = np.inf
    min_ind = -1
    while len(indces) > 0:
        for ind in indces:
            diff = np.array(path_points[cur_ind]) - np.array(path_points[ind])
            if min_dist > np.dot(diff.transpose(), diff):
                min_dist = np.dot(diff.transpose(), diff)
                min_ind = ind

        cur_ind = min_ind
        min_dist = np.inf
        min_ind = -1
        indces.remove(cur_ind)
        sorted_path_points.append(path_points[cur_ind])

    for i in range(len(sorted_path_points)):
        if i == len(sorted_path_points) - 1:
            plt.plot([sorted_path_points[i][0], sorted_path_points[0][0]],
                     [sorted_path_points[i][1], sorted_path_points[0][1]], 'b')
            continue
        plt.plot([sorted_path_points[i][0], sorted_path_points[i + 1][0]],
                 [sorted_path_points[i][1], sorted_path_points[i + 1][1]], 'b')

    plt.title(title)
    plt.show()
    end_time = time.time()
    return end_time - start_time


def visualize_all_scene_samples_with_labels(data, model, step_ratio=5, title=''):
    start_time = time.time()
    if data.shape[1] - 1 != 2:
        return

    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])
    step = calc_visual_step(data, ratio=step_ratio)

    cur_x = min_x
    cur_y = min_y
    while cur_x <= max_x:
        while cur_y <= max_y:
            label = model.pred(np.array([cur_x, cur_y]))
            if label == 1:
                plt.plot(cur_x, cur_y, '*r')
            else:
                plt.plot(cur_x, cur_y, '*g')
            cur_y += step
        cur_x += step
        cur_y = min_y

    plt.title(title)
    plt.show()
    end_time = time.time()
    return end_time - start_time


def visualize_random_samples_with_labels(data, model, vis_nsamples=6000, title=''):
    start_time = time.time()
    if data.shape[1] - 1 != 2:
        return

    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])

    for i in range(vis_nsamples):
        x = min_x + (max_x - min_x) * np.random.random()
        y = min_y + (max_y - min_y) * np.random.random()
        label = model.pred(np.array([x, y]))
        if label == 1:
            plt.plot(x, y, '*r')
        else:
            plt.plot(x, y, '*g')

    plt.title(title)
    plt.show()
    end_time = time.time()
    return end_time - start_time
