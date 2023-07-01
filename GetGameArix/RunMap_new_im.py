import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop, environment
from pysc2.bin import *
import csv
import cv2
import sys


# Bai
from functools import singledispatchmethod
from typing import Iterable
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import random
from collections import deque, namedtuple
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import seaborn as sns
# import similar_histogram as ssim

import statistics

# 解决中文标题问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use("Agg")

_MAP_RESOLUTION = 128
# _MY_UNIT_INFLUENCE = [16, 9, 4, 1]
_MY_UNIT_INFLUENCE = [25]
_ENEMY_UNIT_INFLUENCE = [-16, -9, -4, -1]
# _MY_UNIT_TYPE = 105
# _MY_UNIT_TYPE_ARG = units.Zerg.Zergling
# _ENEMY_UNIT_TYPE = 105
# _ENEMY_UNIT_TYPE_ARG = units.Zerg.Zergling
_MY_UNIT_TYPE = 48
_MY_UNIT_TYPE_ARG = units.Terran.Marine
_ENEMY_UNIT_TYPE = 48
_ENEMY_UNIT_TYPE_ARG = units.Terran.Marine
_BOUNDARY_WIDTH = 2

_MY_UNITS_NUMBER = 4
_ENEMY_UNITS_NUMBER = 4
_STEP = 25 * _MY_UNITS_NUMBER / 4
_STEP_MUL = 10
_MAX_INFLUENCE = 25 * _ENEMY_UNITS_NUMBER
_MIN_INFLUENCE = -16 * _ENEMY_UNITS_NUMBER

_EPISODE_COUNT = 1

# state_vec = []

# 路径信息
_UNITS_ATTRIBUTE_PATH = "datas/data_for_overall/units_name.csv"
_UNITS_LIST_PATH = "datas/data_for_overall/units_list.csv"
_UNITS_INFORMATION_PATH = "datas/data_for_render/units_dataframe.csv"
_GAME_RESULT_PATH = "datas/data_for_transit/game_result.txt"
_GAME_QTABLE_PATH = "datas/data_for_transit/q_table.csv"
_EPISODE_QTABLE_PATH = "datas/data_for_transit/episode_q_table.csv"


"""
2023-04-04
"""


def getHash(pil_image):
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    chans = cv2.split(open_cv_image)
    colors = ("b", "g", "r")
    for (chans, color) in zip(chans, colors):
        hist = cv2.calcHist([chans], [0], None, [8], [0, 256])
    (b, g, r) = cv2.split(open_cv_image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    equ2 = cv2.merge((bH, gH, rH))

    chans2 = cv2.split(equ2)
    r_list = []
    g_list = []
    b_list = []
    for (chans2, color) in zip(chans2, colors):
        hist = cv2.calcHist([chans2], [0], None, [8], [0, 256])
        if color == 'r':
            r_list = hist.T[0]
        if color == 'g':
            g_list = hist.T[0]
        if color == 'b':
            b_list = hist.T[0]
    hashString = ''
    r_max = max(r_list, key=abs)
    g_max = max(g_list, key=abs)
    b_max = max(b_list, key=abs)
    for i in range(8):
        hashString += '{:01X}'.format(int(r_list[i] / r_max * 15.9))
    for i in range(8):
        hashString += '{:01X}'.format(int(g_list[i] / g_max * 15.9))
    for i in range(8):
        hashString += '{:01X}'.format(int(b_list[i] / b_max * 15.9))
    return hashString


# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def array_to_pil_img(arr: np.ndarray):
    norm = mcolors.TwoSlopeNorm(vmin=_MIN_INFLUENCE, vmax=_MAX_INFLUENCE, vcenter=0.0)
    p1 = sns.heatmap(arr, cmap="RdBu", norm=norm,
                     annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False)
    s1 = p1.get_figure()
    img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.tostring_rgb())
    return make_regalur_image(img)


def make_regalur_image(img, size=(256, 256)):
    return img.resize(size).convert('RGB')


"""
2023-04-04
"""

def print_object_attribute(mp):
    print('开始输出属性:')
    attr_list = []
    for i in dir(mp):
        # print(i, type(eval('mp.' + str(i))))
        if isinstance(eval('mp.' + str(i)), int) \
                or isinstance(eval('mp.' + str(i)), str) \
                or isinstance(eval('mp.' + str(i)), property):
            attr_list.append(i)

    # print(dir(mp))
    print('\033[0;33m对象可用属性:\033[0m')
    for idx, element in enumerate(attr_list):
        print(idx, element)


def print_all_attribute(obj):
    print('开始输出属性:')
    attr_list = []
    for i in dir(obj):
        attr_list.append(i)

    # print(dir(mp))
    print('\033[0;33m对象可用属性:\033[0m')
    for idx, element in enumerate(attr_list):
        print(idx, element)


def distance(pos1, pos2):
    """计算坐标点coord到目标点target的距离"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def print_units_name(obs):
    print(obs.observation.raw_units[0].getAllNames())
    pd.DataFrame(data=[obs.observation.raw_units[0].getAllNames()]) \
        .to_csv(_UNITS_ATTRIBUTE_PATH, header=False, index=False)


def merge_units_csv(file1, file2):
    # 按照列进行合并

    # 打开第一个文件，读取表头
    f_in = open(file1, "r")
    reader = csv.reader(f_in)
    header = next(reader)
    f_in.close()  # 关闭文件

    # 打开第二个文件，跳过表头
    f_in = open(file2, "r")
    reader = csv.reader(f_in)
    next(reader)  # 跳过表头

    # 创建一个输出文件，写入表头和数据
    f_out = open(_UNITS_ATTRIBUTE_PATH, "w", newline='')
    writer = csv.writer(f_out)
    writer.writerow(header)  # 写入表头
    for row in reader:
        writer.writerow(row)  # 写入数据
    f_out.close()  # 关闭文件

    # pd.set_option('display.max_columns', None)
    # print(pd.read_csv("units_dataframe.csv"))


def append_qtable_csv(episode_count, qtable, actions):
    if episode_count == 2:
        with open(_EPISODE_QTABLE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(actions)
        with open(_GAME_RESULT_PATH, "w") as f:
            f.write("")
    with open(_EPISODE_QTABLE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(np.sum(qtable, axis=0))


def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)


def calc_similar(li, ri):
    return hist_similar(li.histogram(), ri.histogram())


def array_to_pil_img_2(arr: np.ndarray):
    # plt.figure()
    p1 = sns.heatmap(arr, cmap="coolwarm", vmin=-25, vmax=25, annot=False, cbar=False, square=True,
                     xticklabels=False, yticklabels=False, linewidth=.5)
    # s1 = p1.get_figure()
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    img = Image.frombytes("RGB", (w, h), buf.tobytes())
    print(type(img))
    # plt.show()
    return make_regalur_image(img)


def mtx_similar(arr1: np.ndarray, arr2: np.ndarray) -> float:
    '''
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:SSIM算法计算两个图像的相似度
    '''
    # array_to_pil_img(arr1)
    # print('ssim: HeatMap1 & HeatMap2',
    # ssim.make_regalur_image(array_to_pil_img(dfData1), dfData2))
    # matplotlib.pyplot.close()
    return calc_similar(array_to_pil_img(arr1), array_to_pil_img(arr2))


def save_img(img):
    file_path = "draw/im_img/"
    files = os.listdir(file_path)
    file_num = str(len(files)).zfill(3)
    img.save(f"{file_path}{file_num}.png")


def save_arr(arr):
    file_path = "draw/im_arr/"
    files = os.listdir(file_path)
    file_num = str(len(files)).zfill(3)
    np.savetxt(f"{file_path}{file_num}.txt", arr, fmt="%.2f")


# 计算两点之间的距离
def distance_units(point1, point2):
    return math.sqrt((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# 计算每个点到所有聚类中心的距离
def calc_distances(points, centers):
    distances = []
    for point in points:
        row = []
        for center in centers:
            # print(center)
            row.append(distance_units(point, center))
        distances.append(row)
    return distances

# 将每个点分配到最近的聚类中心
def assign_clusters(points, centers):
    distances = calc_distances(points, centers)
    clusters = [[] for i in range(len(centers))]
    for i in range(len(points)):
        min_dist = float('inf')
        min_idx = -1
        for j in range(len(centers)):
            if distances[i][j] < min_dist:
                min_dist = distances[i][j]
                min_idx = j
        clusters[min_idx].append(points[i])
    return clusters

# 计算聚类中心
def calc_centers(clusters):
    centers = []
    for cluster in clusters:
        id_sum = 0
        x_sum = 0
        y_sum = 0
        for point in cluster:
            id_sum += point[0]
            x_sum += point[1]
            y_sum += point[2]
        center_id = id_sum
        # print('cluster', len(cluster), cluster)
        if len(cluster) > 0:
            center_x = x_sum / len(cluster)
            center_y = y_sum / len(cluster)
            centers.append((center_id, center_x, center_y))
    return centers

# 计算所有点的平均距离
def calc_avg_distance(points, centers):
    distances = calc_distances(points, centers)
    sum_dist = 0
    for i in range(len(points)):
        min_dist = float('inf')
        for j in range(len(centers)):
            if distances[i][j] < min_dist:
                min_dist = distances[i][j]
        sum_dist += min_dist
    return sum_dist / len(points)

# 将A类和B类坐标点分别聚类
def kmeans(points_a, points_b, k_a, k_b):
    # 初始化聚类中心
    # print('a', points_a, k_a)
    # print('b', points_b, k_b)
    if len(points_a) > 0 and len(points_b) > 0:
        centers_a = random.sample(points_a, k_a)
        centers_b = random.sample(points_b, k_b)
    else:
        return [], [], []
    # 迭代聚类过程
    for i in range(10):
        # 分配聚类
        clusters_a = assign_clusters(points_a, centers_a)
        clusters_b = assign_clusters(points_b, centers_b)
        # 计算聚类中心
        centers_a = calc_centers(clusters_a)
        centers_b = calc_centers(clusters_b)
    # 对A类坐标点进行最近距离的匹配
    matches = []
    for cluster_a in clusters_a:
        for point_a in cluster_a:
            min_dist = float('inf')
            min_point_b = None
            for point_b in points_b:
                if distance_units(point_a, point_b) < min_dist:
                    min_dist = distance_units(point_a, point_b)
                    min_point_b = point_b
            matches.append((point_a, min_point_b))
    return clusters_a, clusters_b, matches

# def array_to_pil_img(arr: np.ndarray, check_flag=False):
#     norm = mcolors.TwoSlopeNorm(vmin=_MIN_INFLUENCE, vmax=_MAX_INFLUENCE, vcenter=0.0)
#     p1 = sns.heatmap(arr, cmap="RdBu", norm=norm,
#                      annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False)
#     s1 = p1.get_figure()
#     img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.tostring_rgb())
#     if check_flag:
#         save_img(img)
#         save_arr(arr)
#     # img.show()
#     return ssim.make_regalur_image(img)


class QLearningTable:
    # def __init__(self, actions, learning_rate=0.1, reward_decay=0.9):
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        # print(observation)
        self.check_state_exist(observation)
        # print(state_index)
        # print(self.q_table)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        # print(type(s_), s_)
        # print(self.q_table)
        # print(s_)
        self.check_state_exist(s_)
        # print(state_index)
        # print(state_index)
        # print(self.q_table)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        # print(len(state_vec))
        # print(type(state), state)
        # print(self.q_table.index)
        # if len(state_vec) == 0 or type(observation) == str:
        if state not in self.q_table.index:
            # state_vec.append((str(len(state_vec)), observation))
            self.q_table = pd.concat([self.q_table, pd.Series([0] * len(self.actions),
                                                              index=self.q_table.columns,
                                                              name=state).to_frame().T])
                                                              # name=str(len(state_vec) - 1)).to_frame().T])
            # return len(state_vec) - 1
        # 目前不考虑相似
        # else:
        #     max_similar = 0
        #     max_similar_index = 0
        #     for state_index, state_item in state_vec:
        #         # print(state_item, state_index)
        #         # print('observation', observation)
        #         if type(state_item) != str:
        #             item_similar = mtx_similar(np.matrix(observation, dtype="float"),
        #                                        np.matrix(state_item, dtype="float"))
        #             if item_similar >= 0.95:
        #                 # print('observation', observation)
        #                 # print(item_similar, state_vec[int(state_index)])
        #                 # print('now_obs', observation)
        #                 # print('exist_obs', state_vec[int(state_index)])
        #                 # print('similar_obs', state_vec[state_index])
        #                 return state_index
        #             if item_similar > max_similar:
        #                 max_similar = item_similar
        #                 max_similar_index = state_index
        #     if max_similar < 0.95:
        #         state_vec.append((str(len(state_vec)), observation))
        #         self.q_table = pd.concat([self.q_table, pd.Series([0] * len(self.actions),
        #                                                           index=self.q_table.columns,
        #                                                           name=str(len(state_vec) - 1)).to_frame().T])
        #         return len(state_vec) - 1
        #     return max_similar_index
        # if state not in self.q_table.index:
        #     self.q_table = pd.concat([self.q_table, pd.Series([0] * len(self.actions),
        #                                                       index=self.q_table.columns,
        #                                                       name=state).to_frame().T])
        # 原始代码如下：append被弃用，改为pd.concat
        # self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
        #                                              index=self.q_table.columns,
        #                                              name=state))


class Agent(base_agent.BaseAgent):
    actions = (
        # "do_nothing",
        "action_TFC",
        "action_TFU",
        "action_TNC",
        # "action_TNU",
        # "action_greedy",
        # "action_DFC",
        # "action_DFU",
        # "action_DNC",
        # "action_DNU",
        "action_retreat"
        # "action_noise"
    )

    def get_units_list(self, file):
        unit_my_list = []
        unit_enemy_list = []
        f_in = open(file, "r")
        reader = csv.reader(f_in)
        header = next(reader)
        for row in reader:
            # unit类型、x、y
            if row[0] == str(_MY_UNIT_TYPE):
                unit_my_list.append((row[0], row[12], row[13]))
                continue
            if row[0] == str(_ENEMY_UNIT_TYPE):
                unit_enemy_list.append((row[0], row[12], row[13]))
                continue
        return unit_my_list, unit_enemy_list

    def get_my_units_by_type(self, obs, unit_type):
        # print(unit_type)
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units_list, xy):
        units_xy = [(unit.x, unit.y) for unit in units_list]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_nearest_enemy(self, mp, enemy_units):
        enemy_units_list = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        min_dis = 99.
        min_tag = -1
        for unit in enemy_units_list:
            dis = distance((unit[1], unit[2]), mp)
            if dis < min_dis:
                min_dis = dis
                min_tag = unit[0]
        return min_tag

    def get_center_position(self, obs, alliance, unit_type):
        position = (0, 0)
        if alliance == 'Self':
            # print('get_my_center_position[base]')
            my_units = [unit for unit in obs.observation.raw_units
                        if unit.unit_type == unit_type
                        and unit.alliance == features.PlayerRelative.SELF]
            if len(my_units) == 0:
                return position
            for unit in my_units:
                position = tuple(map(lambda x, y: x + y, position, (unit.x, unit.y)))
            return (position[0] / len(my_units), position[1] / len(my_units))
        elif alliance == 'Enemy':
            # print('get_enemy_center_position[base]')
            position = (0, 0)
            enemy_units = [unit for unit in obs.observation.raw_units
                           if unit.unit_type == unit_type
                           and unit.alliance == features.PlayerRelative.ENEMY]
            if len(enemy_units) == 0:
                return position
            for unit in enemy_units:
                position = tuple(map(lambda x, y: x + y, position, (unit.x, unit.y)))
            return (position[0] / len(enemy_units), position[1] / len(enemy_units))

    def get_center_position_point(self, points):
        position = (0, 0)
        my_units = [(unit[1], unit[2]) for unit in points]
        if len(my_units) == 0:
            return position
        for unit in my_units:
            position = tuple(map(lambda x, y: x + y, position, (unit[0], unit[1])))
        return (position[0] / len(my_units), position[1] / len(my_units))

    def choice_nearest_weakest_enemy(self, mp, enemy_list):
        sorted_enemy_lst = sorted([(item['tag'], item['x'], item['y'], item['health'], distance(mp, (item['x'], item['y']))) for item in enemy_list], key=lambda x: x[3])
        # sorted_enemy_lst['distance'] =
        # print(sorted_enemy_lst)
        # print(min(sorted_enemy_lst, key=lambda x: x[3]*x[2]))
        return min(sorted_enemy_lst, key=lambda x: x[3]*x[2]*x[2])[0]

    def step(self, obs):
        super(Agent, self).step(obs)

        # if obs.first():
        # print(dir(obs.observation))
        # if obs.first():
        # np.save('dict_first.npy', obs)
        #     a = np.load('dict_first.npy', allow_pickle=True)
        # if obs.last():
        #     np.save('dict_end.npy', obs)
        #     b = np.load('dict_end.npy', allow_pickle=True)
        #     print(b[3].keys())
        # print(obs.observation)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()


class SmartAgent(Agent):
    def draw_units(self):
        unit_my_list, unit_enemy_list = self.get_units_list(_UNITS_ATTRIBUTE_PATH)
        print(unit_my_list, unit_enemy_list)
        influence_map = self.get_influence_map(unit_my_list, unit_enemy_list)
        top, bottom, left, right = self.get_map_boundary(influence_map, _BOUNDARY_WIDTH)
        # 生成目标点
        target_gp = self.analyze_influence_map(influence_map)
        target_mp = self.grid_to_map(target_gp[0], target_gp[1])
        # self.ripple(influence_map, 0, (target_gp[0], target_gp[1]), 'Target')
        # 窗口取值
        d = influence_map.T[left:right, top:bottom]

        # fig.tight_layout()
        norm = mcolors.TwoSlopeNorm(vmin=d.min(), vmax=d.max(), vcenter=0)
        plt.imshow(d, cmap=plt.cm.RdBu, norm=norm)
        for x, value_zip in enumerate(d):
            value: int
            for y, value in enumerate(value_zip):
                if y == (target_gp[0] - top) and x == (target_gp[1] - left):
                    plt.text(y, x, r"$\star$", ha="center", va="center", color="b", size=20)
                elif value != 0:
                    plt.text(y, x, value, ha="center", va="center", color="w", size=6)
        plt.colorbar()
        plt.show()
        return

    def map_to_grid(self, x, y):
        i, j = int(x), int(_MAP_RESOLUTION - y)
        return i, j

    def grid_to_map(self, i, j):
        x, y = random.uniform(i, i + 1), \
               random.uniform(int(_MAP_RESOLUTION - j) - 1, int(_MAP_RESOLUTION - j))
        return x, y

    def ripple(self, influence_map, ripple_level, ripple_center, alliance):
        max_x = _MAP_RESOLUTION - 1
        max_y = _MAP_RESOLUTION - 1
        if alliance == 'Self':
            if ripple_level == 0:
                influence_map[int(ripple_center[1])][int(ripple_center[2])] += _MY_UNIT_INFLUENCE[ripple_level]
            else:
                for dx in range(-abs(ripple_level) if int(ripple_center[1]) > 0 else 0,
                                abs(ripple_level) + 1 if int(ripple_center[1]) < max_x else 0):
                    for dy in range(-abs(ripple_level) if int(ripple_center[2]) > 0 else 0,
                                    abs(ripple_level) + 1 if int(ripple_center[2]) < max_y else 0):
                        if dx != 0 or dy != 0:
                            influence_map[int(ripple_center[1]) + dx][int(ripple_center[2]) + dy] \
                                += _MY_UNIT_INFLUENCE[ripple_level]

        elif alliance == 'Enemy':
            if ripple_level == 0:
                influence_map[int(ripple_center[1])][int(ripple_center[2])] += _ENEMY_UNIT_INFLUENCE[ripple_level]
            else:
                for dx in range(-abs(ripple_level) if int(ripple_center[1]) > 0 else 0,
                                abs(ripple_level) + 1 if int(ripple_center[1]) < max_x else 0):
                    for dy in range(-abs(ripple_level) if int(ripple_center[2]) > 0 else 0,
                                    abs(ripple_level) + 1 if int(ripple_center[2]) < max_y else 0):
                        if dx != 0 or dy != 0:
                            influence_map[int(ripple_center[1]) + dx][int(ripple_center[2]) + dy] \
                                += _ENEMY_UNIT_INFLUENCE[ripple_level]
        elif alliance == 'Target':
            influence_map[int(ripple_center[0])][int(ripple_center[1])] += 100

    def get_influence_map(self, unit_my_list, unit_enemy_list):
        side = int((pow(_MAP_RESOLUTION, 1)))
        influence_map = np.zeros((side, side))
        # influence_map = np.random.rand(side, side)
        for my_unit in unit_my_list:
            for index in range(len(_MY_UNIT_INFLUENCE)):
                self.ripple(influence_map, index, my_unit, 'Self')

        for enemy_unit in unit_enemy_list:
            for index in range(len(_ENEMY_UNIT_INFLUENCE)):
                self.ripple(influence_map, index, enemy_unit, 'Enemy')

        return influence_map

    def analyze_influence_map(self, influence_map):
        positive_list, negative_list = [], []
        for i in range(len(influence_map)):
            for j in range(len(influence_map[0])):
                if influence_map[i][j] > 0:
                    positive_list.append((i, j, influence_map[i][j]))
                elif influence_map[i][j] < 0:
                    negative_list.append((i, j, influence_map[i][j]))
        sort_negative_list = sorted(negative_list, key=lambda x: x[2], reverse=True)
        if len(positive_list) == 0:
            # print('My units are empty.')
            return (_MAP_RESOLUTION / 2, _MAP_RESOLUTION / 2)
        if len(sort_negative_list) == 0:
            # print('Enemy units are empty.')
            return (_MAP_RESOLUTION / 2, _MAP_RESOLUTION / 2)
        percentile = 80
        p = np.percentile([t[2] for t in sort_negative_list], percentile)
        filtered_negative_list = [x for x in sort_negative_list if x[2] >= p]
        sum_x = 0
        sum_y = 0
        for coord in positive_list:
            sum_x += coord[0]
            sum_y += coord[1]
        avg_x = sum_x / len(positive_list)
        avg_y = sum_y / len(positive_list)
        positive_ctr = (avg_x, avg_y)
        min_distance = float('inf')
        min_index = -1
        for i, coord in enumerate(filtered_negative_list):
            d = distance(coord, positive_ctr)
            if d < min_distance:
                min_distance = d
                min_index = i
        return filtered_negative_list[min_index]

    def get_map_boundary(self, influence_map, width):
        rows, cols = len(influence_map), len(influence_map[0])
        non_zero = [(i, j) for i in range(rows) for j in range(cols) if influence_map[i][j] != 0]
        if not non_zero:
            print(influence_map)
            # 没有非零元素
            return 40, 44, 40, 44
        min_row, min_col = non_zero[0]
        max_row, max_col = non_zero[0]
        for i, j in non_zero:
            if i < min_row:
                min_row = i
            if i > max_row:
                max_row = i
            if j < min_col:
                min_col = j
            if j > max_col:
                max_col = j
        # 上，下，左，右
        top = max(min_row - width, 0)
        bottom = min(max_row + 1 + width, _MAP_RESOLUTION)
        left = max(min_col - width, 0)
        right = min(max_col + 1 + width, _MAP_RESOLUTION)
        return top, bottom, left, right

    def action_move_camera(self, obs):
        return actions.RAW_FUNCTIONS.raw_move_camera((60, 60))

    def get_target_position(self, obs, my_units, enemy_units, attack_mode):
        attack_mode_list = (
            "TFC",
            "TNC"
        )
        my_units_list = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        enemy_units_list = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        my_center = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
        enemy_center = self.get_center_position(obs, 'Enemy', units.Zerg.Zergling)
        diff = np.subtract(np.array(my_center), np.array(enemy_center))
        target_mp = (0., 0.)
        if attack_mode == attack_mode_list[0]:
            target_mp = np.add(my_center, np.multiply(diff, 0.5))
        # print(my_units_list, enemy_center)
        # print('diff', diff)
        # print(header, data_for_transit)
        elif attack_mode == attack_mode_list[1]:
            influence_map = self.get_influence_map(my_units_list, enemy_units_list)
            top, bottom, left, right = self.get_map_boundary(influence_map, _BOUNDARY_WIDTH)
            scale_map = influence_map.T[left:right, top:bottom]
            target_gp = self.analyze_influence_map(influence_map)
            target_mp = self.grid_to_map(target_gp[0], target_gp[1])
        return target_mp

    # def k_means_clustering(self, points, k, threshold):
    #     # Randomly select k initial cluster centers
    #     centers = random.sample(points, k)
    #
    #     while True:
    #         # Initialize empty clusters
    #         clusters = [[] for _ in range(k)]
    #
    #         # Assign each point to the nearest cluster center
    #         for point in points:
    #             distances = [math.dist((point[1], point[2]), (center[1], center[2])) for center in centers]
    #             nearest_center = distances.index(min(distances))
    #             clusters[nearest_center].append(point)
    #
    #         # Update cluster centers
    #         new_centers = [list(map(lambda x: sum(x) / len(x), zip(*cluster))) for cluster in clusters]
    #
    #         # Check for convergence
    #         if all([math.isclose(math.dist((centers[i][1], centers[i][2]), (new_centers[i][1], new_centers[i][2])), 0, rel_tol=threshold) for i in range(k)]):
    #             break
    #
    #         centers = new_centers
    #
    #     return clusters

    # def action_TFC(self, obs):
    #     # print('action_TNC')
    #     """
    #     T: together, 所有单位聚合
    #     F: far, 远离目标区域
    #     C: center, 目标区域选定为敌方中心区域
    #     :param obs:
    #     :return:
    #     """
    #     my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
    #     enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
    #     if len(my_units) > 0:
    #         enemy_center = (0, 0)
    #         if self.get_center_position(obs, 'Enemy', units.Zerg.Zergling)[0] > 0 and \
    #                 self.get_center_position(obs, 'Enemy', units.Zerg.Zergling)[1] > 0:
    #             enemy_center = self.get_center_position(obs, 'Enemy', units.Zerg.Zergling)
    #         else:
    #             enemy_center = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
    #         TFC_ref_position = self.get_target_position(obs, my_units, enemy_units, 'TFC')
    #
    #         marine = random.choice(my_units)
    #         # 大成功: 同一时间选取多个单元进行动作
    #         # 大成功: 并且可以选择不同的动作，仅仅需要return一个list即可
    #         marines_tag_list = [item['tag'] for item in my_units]
    #         list1 = marines_tag_list[:len(marines_tag_list) // 2]
    #         list2 = marines_tag_list[len(marines_tag_list) // 2:]
    #         x_offset = random.randint(-1, 1)
    #         y_offset = random.randint(-1, 1)
    #
    #         return [actions.RAW_FUNCTIONS.Smart_pt(
    #             "now", list1, (enemy_center[0] + x_offset, enemy_center[1] + y_offset)),
    #             actions.RAW_FUNCTIONS.Smart_pt(
    #                 "now", list2, (enemy_center[0] - x_offset, enemy_center[1] - y_offset))
    #         ]
    #     # print(self.get_center_position(obs, units.Terran.Marine))
    #
    #     return actions.RAW_FUNCTIONS.no_op()
    def action_TFC(self, obs):
        """
        T: together, 所有单位聚合
        F: far, 远离目标区域
        C: center, 目标区域选定为敌方中心区域
        :param obs:
        :return:
        """
        # print(obs)
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        enemy_units_lst = [item['tag'] for item in enemy_units]
        mp = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
        if not self._move_back:
            if len(enemy_units) > 0 and len(my_units) > 0:
                nearest_enemy = self.get_nearest_enemy(mp, enemy_units)
                self.action_queue.append('Attack_unit')
                return actions.RAW_FUNCTIONS.Attack_unit(
                    "now", [item[0] for item in my_units_lst], nearest_enemy)
        else:
            dis = distance(mp, self._backup_target_map)
            if dis < 5:
                self._move_back = False
            if len(my_units) > 0:
                self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TFC')
                self.action_queue.append('Smart_pt')
                action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
                    "now", [item[0] for item in my_units_lst],
                    (self._backup_target_map[0], self._backup_target_map[1])))
                action_lst.append(actions.RAW_FUNCTIONS.raw_move_camera(mp))
                return action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_TFU(self, obs):
        """
        T: together, 所有单位聚合
        F: far, 远离目标区域
        U: unit, 目标区域选定为敌方最近单元
        :param obs:
        :return:
        """
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        enemy_units_lst = [item['tag'] for item in enemy_units]
        mp = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
        if not self._move_back:
            if len(enemy_units) > 0 and len(my_units) > 0:
                nearest_enemy = self.get_nearest_enemy(mp, enemy_units)
        else:
            dis = distance(mp, self._backup_target_map)
            if dis < 5:
                self._move_back = False
            if len(my_units) > 0:
                if len(enemy_units) > 0:
                    nearest_enemy = self.get_nearest_enemy(mp, enemy_units)
                    self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TFC')
                    self.action_queue.append('Smart_pt')
                    action_lst.append(actions.RAW_FUNCTIONS.Smart_unit(
                        "now", [item[0] for item in my_units_lst], nearest_enemy))
                    action_lst.append(actions.RAW_FUNCTIONS.raw_move_camera(mp))
                    return action_lst
        # if len(my_units) > 0:
        #     self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TNC')
        #     action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
        #         "now", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
        #     if len(enemy_units) > 0:
        #         action_lst.append(actions.RAW_FUNCTIONS.Attack_unit(
        #             "queued", my_units_tag_lst, random.choice(enemy_units_tag_lst)))
        #     elif len(enemy_units) == 0:
        #         action_lst.append(actions.RAW_FUNCTIONS.Attack_pt(
        #             "queued", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
        #     return action_lst
        return actions.RAW_FUNCTIONS.no_op()

    # def action_TNC(self, obs):
    #     # print('action_TNC')
    #     """
    #     T: together, 所有单位聚合
    #     N: near, 靠近目标区域
    #     C: center, 目标区域选定为敌方中心区域
    #     :param obs:
    #     :return:
    #     """
    #     action_lst = []
    #     my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
    #     enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
    #     my_units_tag_lst = [item['tag'] for item in my_units]
    #     enemy_units_tag_lst = [item['tag'] for item in enemy_units]
    #     if len(my_units) > 0:
    #         self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TNC')
    #         action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
    #             "now", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
    #         if len(enemy_units) > 0:
    #             action_lst.append(actions.RAW_FUNCTIONS.Attack_unit(
    #                 "queued", my_units_tag_lst, random.choice(enemy_units_tag_lst)))
    #         elif len(enemy_units) == 0:
    #             action_lst.append(actions.RAW_FUNCTIONS.Attack_pt(
    #                 "queued", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
    #         return action_lst
    #     return actions.RAW_FUNCTIONS.no_op()
    def action_TNC(self, obs):
        # print('action_TNC')
        """
        T: together, 所有单位聚合
        N: near, 靠近目标区域
        C: center, 目标区域选定为敌方中心区域
        :param obs:
        :return:
        """
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        enemy_units_lst = [item['tag'] for item in enemy_units]
        mp = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
        # print(self._move_back)
        # print([item[0] for item in my_units_lst], [item[1] for item in my_units_lst])
        if not self._move_back:
            if len(enemy_units) > 0 and len(my_units) > 0:
                nearest_enemy = self.get_nearest_enemy(mp, enemy_units)
                # print(obs.observation.game_loop, 'Attack_unit')
                self.action_queue.append('Attack_unit')
                return actions.RAW_FUNCTIONS.Attack_unit(
                    "now", [item[0] for item in my_units_lst], nearest_enemy)
                # return actions.RAW_FUNCTIONS.Attack_unit(
                #     "now", [item[0] for item in my_units_lst], random.choice(enemy_units_lst))
        else:
            dis = distance(mp, self._backup_target_map)
            # print(mp, self._backup_target_map, dis)
            if dis < 5:
                # print('0.0')
                self._move_back = False
            if len(my_units) > 0:
                self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TNC')
                # print(obs.observation.game_loop, 'Smart_pt')
                self.action_queue.append('Smart_pt')
                action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
                    "now", [item[0] for item in my_units_lst],
                    (self._backup_target_map[0], self._backup_target_map[1])))
                action_lst.append(actions.RAW_FUNCTIONS.raw_move_camera(mp))
                return action_lst
        # if len(my_units) > 0:
        #     self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TNC')
        #     action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
        #         "now", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
        #     if len(enemy_units) > 0:
        #         action_lst.append(actions.RAW_FUNCTIONS.Attack_unit(
        #             "queued", my_units_tag_lst, random.choice(enemy_units_tag_lst)))
        #     elif len(enemy_units) == 0:
        #         action_lst.append(actions.RAW_FUNCTIONS.Attack_pt(
        #             "queued", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
        #     return action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_TNU(self, obs):
        # print('action_TNC')
        """
        T: together, 所有单位聚合
        N: near, 靠近目标区域
        U: unit, 目标区域选定为敌方最近单元
        :param obs:
        :return:
        """
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        enemy_units_lst = [item['tag'] for item in enemy_units]
        mp = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
        if not self._move_back:
            if len(enemy_units) > 0 and len(my_units) > 0:
                nearest_enemy = self.get_nearest_enemy(mp, enemy_units)
                self.action_queue.append('Attack_unit')
                return actions.RAW_FUNCTIONS.Attack_unit(
                    "now", [item[0] for item in my_units_lst], nearest_enemy)
        else:
            dis = distance(mp, self._backup_target_map)
            # print(mp, self._backup_target_map, dis)
            if dis < 5:
                # print('0.0')
                self._move_back = False
            if len(my_units) > 0:
                if len(enemy_units) > 0 and len(my_units) > 0:
                    nearest_enemy = self.get_nearest_enemy(mp, enemy_units)
                    # self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TNC')
                    # print(obs.observation.game_loop, 'Smart_pt')
                    self.action_queue.append('Smart_pt')
                    action_lst.append(actions.RAW_FUNCTIONS.Smart_unit(
                        "now", [item[0] for item in my_units_lst], nearest_enemy))
                    action_lst.append(actions.RAW_FUNCTIONS.raw_move_camera(mp))
                    return action_lst

        # if len(my_units) > 0:
        #     self._backup_target_map = self.get_target_position(obs, my_units, enemy_units, 'TNC')
        #     action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
        #         "now", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
        #     if len(enemy_units) > 0:
        #         action_lst.append(actions.RAW_FUNCTIONS.Attack_unit(
        #             "queued", my_units_tag_lst, random.choice(enemy_units_tag_lst)))
        #     elif len(enemy_units) == 0:
        #         action_lst.append(actions.RAW_FUNCTIONS.Attack_pt(
        #             "queued", my_units_tag_lst, (self._backup_target_map[0], self._backup_target_map[1])))
        #     return action_lst
        return actions.RAW_FUNCTIONS.no_op()

    def action_DFC(self, obs):
        """
        D: disperse, 所有单位分散并聚类
        F: far, 远离目标区域
        C: center, 目标区域选定为敌方中心区域
        :param obs:
        :return:
        """
        # 有bug
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        enemy_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        # my_clusters = self.k_means_clustering(my_units_lst, 10, 1)
        # enemy_clusters = self.k_means_clustering(enemy_units_lst, 10, 1)
        clusters_a_num = 1
        clusters_b_num = 1
        if len(my_units_lst) > 1:
            clusters_a_num = int(math.log2(len(my_units_lst))) + 1
        if len(enemy_units_lst) > 1:
            clusters_b_num = int(math.log2(len(enemy_units_lst))) + 1
        clusters_a, clusters_b, matches = kmeans(my_units_lst, enemy_units_lst, clusters_a_num, clusters_b_num)
        # print('my_units_lst', my_units)
        # print('my_clusters', clusters_a)
        # print('enemy_clusters', clusters_b)
        # print('matchs:')
        # for match in matches:
        #     print(f'{match[0]} 匹配 {match[1]}')
        # print(matches)
        for k in range(len(clusters_a)):
            # print(cluster)
            # print(my_mp)
            enemy_cluster_units_lst = []
            for unit in clusters_a[k]:
                for match in matches:
                    # print(match)
                    # print(cluster)
                    if match[0][0] == unit[0]:
                        enemy_cluster_units_lst.append(match[1])

            my_mp = self.get_center_position_point(clusters_a[k])
            enemy_mp = self.get_center_position_point(enemy_cluster_units_lst)
            # if not self._dis_move_back[k]:
            self.action_queue.append('Attack_unit')
            for i in range(len(clusters_a[k])):
                action_lst.append(actions.RAW_FUNCTIONS.Attack_pt(
                    "now", clusters_a[k][i][0],
                    (enemy_cluster_units_lst[i][1], enemy_cluster_units_lst[i][2])))
            return action_lst
            # else:
            #     dis = distance(my_mp, self._backup_target_map)
            #     if dis < 5:
            #         self._dis_move_back[k] = False
            #         # print(self._dis_move_back)
            #     if len(clusters_a[k]) > 0:
            #         self.action_queue.append('Smart_pt')
            #         action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
            #             "now", [item[0] for item in clusters_a[k]], enemy_mp))
            #         action_lst.append(actions.RAW_FUNCTIONS.raw_move_camera(my_mp))
            #         return action_lst
            # return actions.RAW_FUNCTIONS.no_op()
        return actions.RAW_FUNCTIONS.no_op()

            # print('my_cluster_units_lst', my_cluster_units_lst)
            # print('enemy_cluster_units_lst', enemy_cluster_units_lst)
            # min_dist = float('inf')
            # min_idx = -1
            # for j in range(clusters_num):
            #     enemy_list = clusters_b[j]
            #     ememy_mp = self.get_center_position_point(enemy_list)
            #     if distance(my_mp, ememy_mp) < min_dist:
            #         min_dist = distance(my_mp, ememy_mp)
            #         min_idx = j
            # print(clusters_b[min_idx], min_dist)
            # enemy_cluster_units_lst = clusters_b[i]

    def action_DFU(self, obs):
        """
        D: disperse, 所有单位分散并聚类
        F: far, 远离目标区域
        C: center, 目标区域选定为敌方中心区域
        :param obs:
        :return:
        """
        # 有bug
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        enemy_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        # my_clusters = self.k_means_clustering(my_units_lst, 10, 1)
        # enemy_clusters = self.k_means_clustering(enemy_units_lst, 10, 1)
        clusters_a_num = 1
        clusters_b_num = 1
        if len(my_units_lst) > 1:
            clusters_a_num = int(math.log2(len(my_units_lst))) + 1
        if len(enemy_units_lst) > 1:
            clusters_b_num = int(math.log2(len(enemy_units_lst))) + 1
        clusters_a, clusters_b, matches = kmeans(my_units_lst, enemy_units_lst, clusters_a_num, clusters_b_num)
        # print('my_units_lst', my_units)
        # print('my_clusters', clusters_a)
        # print('enemy_clusters', clusters_b)
        # print('matchs:')
        # for match in matches:
        #     print(f'{match[0]} 匹配 {match[1]}')
        # print(matches)
        for k in range(len(clusters_a)):
            # print(cluster)
            # print(my_mp)
            enemy_cluster_units_lst = []
            for unit in clusters_a[k]:
                for match in matches:
                    # print(match)
                    # print(cluster)
                    if match[0][0] == unit[0]:
                        enemy_cluster_units_lst.append(match[1])

            my_mp = self.get_center_position_point(clusters_a[k])
            enemy_mp = self.get_center_position_point(enemy_cluster_units_lst)
            # if not self._dis_move_back[k]:
            self.action_queue.append('Attack_unit')
            for i in range(len(clusters_a[k])):
                action_lst.append(actions.RAW_FUNCTIONS.Attack_unit(
                    "now", clusters_a[k][i][0],
                    enemy_cluster_units_lst[i][0]))
            return action_lst
            # else:
            #     dis = distance(my_mp, self._backup_target_map)
            #     if dis < 5:
            #         self._dis_move_back[k] = False
            #         # print(self._dis_move_back)
            #     for i in range(len(clusters_a[k])):
            #         action_lst.append(actions.RAW_FUNCTIONS.Smart_unit(
            #             "now", clusters_a[k][i][0],
            #             enemy_cluster_units_lst[i][0]))
            #     return action_lst
        return actions.RAW_FUNCTIONS.no_op()

            # print('my_cluster_units_lst', my_cluster_units_lst)
            # print('enemy_cluster_units_lst', enemy_cluster_units_lst)
            # min_dist = float('inf')
            # min_idx = -1
            # for j in range(clusters_num):
            #     enemy_list = clusters_b[j]
            #     ememy_mp = self.get_center_position_point(enemy_list)
            #     if distance(my_mp, ememy_mp) < min_dist:
            #         min_dist = distance(my_mp, ememy_mp)
            #         min_idx = j
            # print(clusters_b[min_idx], min_dist)
            # enemy_cluster_units_lst = clusters_b[i]

    def action_DNC(self, obs):
        """
        D: disperse, 所有单位分散并聚类
        F: far, 远离目标区域
        C: center, 目标区域选定为敌方中心区域
        :param obs:
        :return:
        """
        # 有bug
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        enemy_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        # my_clusters = self.k_means_clustering(my_units_lst, 10, 1)
        # enemy_clusters = self.k_means_clustering(enemy_units_lst, 10, 1)
        clusters_a_num = 1
        clusters_b_num = 1
        if len(my_units_lst) > 1:
            clusters_a_num = int(math.log2(len(my_units_lst))) + 1
        if len(enemy_units_lst) > 1:
            clusters_b_num = int(math.log2(len(enemy_units_lst))) + 1
        clusters_a, clusters_b, matches = kmeans(my_units_lst, enemy_units_lst, clusters_a_num, clusters_b_num)
        # print('my_units_lst', my_units)
        # print('my_clusters', clusters_a)
        # print('enemy_clusters', clusters_b)
        # print('matchs:')
        # for match in matches:
        #     print(f'{match[0]} 匹配 {match[1]}')
        # print(matches)
        for k in range(len(clusters_a)):
            # print(cluster)
            # print(my_mp)
            enemy_cluster_units_lst = []
            for unit in clusters_a[k]:
                for match in matches:
                    # print(match)
                    # print(cluster)
                    if match[0][0] == unit[0]:
                        enemy_cluster_units_lst.append(match[1])

            my_mp = self.get_center_position_point(clusters_a[k])
            enemy_mp = self.get_center_position_point(enemy_cluster_units_lst)
            # if not self._dis_move_back[k]:
            self.action_queue.append('Attack_unit')
            for i in range(len(clusters_a[k])):
                 action_lst.append(actions.RAW_FUNCTIONS.Attack_pt(
                     "now", clusters_a[k][i][0],
                     (enemy_cluster_units_lst[i][1], enemy_cluster_units_lst[i][2])))
            return action_lst
            # else:
            #     dis = distance(my_mp, self._backup_target_map)
            #     if dis < 5:
            #         self._dis_move_back[k] = False
            #         # print(self._dis_move_back)
            #     if len(clusters_a[k]) > 0:
            #         self.action_queue.append('Smart_pt')
            #         action_lst.append(actions.RAW_FUNCTIONS.Smart_pt(
            #             "now", [item[0] for item in clusters_a[k]], enemy_mp))
            #         action_lst.append(actions.RAW_FUNCTIONS.raw_move_camera(my_mp))
            #         return action_lst
            # return actions.RAW_FUNCTIONS.no_op()
        return actions.RAW_FUNCTIONS.no_op()

            # print('my_cluster_units_lst', my_cluster_units_lst)
            # print('enemy_cluster_units_lst', enemy_cluster_units_lst)
            # min_dist = float('inf')
            # min_idx = -1
            # for j in range(clusters_num):
            #     enemy_list = clusters_b[j]
            #     ememy_mp = self.get_center_position_point(enemy_list)
            #     if distance(my_mp, ememy_mp) < min_dist:
            #         min_dist = distance(my_mp, ememy_mp)
            #         min_idx = j
            # print(clusters_b[min_idx], min_dist)
            # enemy_cluster_units_lst = clusters_b[i]

    def action_DNU(self, obs):
        """
        D: disperse, 所有单位分散并聚类
        F: far, 远离目标区域
        C: center, 目标区域选定为敌方中心区域
        :param obs:
        :return:
        """
        # 有bug
        action_lst = []
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        enemy_units_lst = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        # my_clusters = self.k_means_clustering(my_units_lst, 10, 1)
        # enemy_clusters = self.k_means_clustering(enemy_units_lst, 10, 1)
        clusters_a_num = 1
        clusters_b_num = 1
        if len(my_units_lst) > 1:
            clusters_a_num = int(math.log2(len(my_units_lst))) + 1
        if len(enemy_units_lst) > 1:
            clusters_b_num = int(math.log2(len(enemy_units_lst))) + 1
        clusters_a, clusters_b, matches = kmeans(my_units_lst, enemy_units_lst, clusters_a_num, clusters_b_num)
        # print('my_units_lst', my_units)
        # print('my_clusters', clusters_a)
        # print('enemy_clusters', clusters_b)
        # print('matchs:')
        # for match in matches:
        #     print(f'{match[0]} 匹配 {match[1]}')
        # print(matches)
        for k in range(len(clusters_a)):
            # print(cluster)
            # print(my_mp)
            enemy_cluster_units_lst = []
            for unit in clusters_a[k]:
                for match in matches:
                    # print(match)
                    # print(cluster)
                    if match[0][0] == unit[0]:
                        enemy_cluster_units_lst.append(match[1])

            my_mp = self.get_center_position_point(clusters_a[k])
            enemy_mp = self.get_center_position_point(enemy_cluster_units_lst)
            # if not self._dis_move_back[k]:
            self.action_queue.append('Attack_unit')
            for i in range(len(clusters_a[k])):
                action_lst.append(actions.RAW_FUNCTIONS.Attack_unit(
                    "now", clusters_a[k][i][0],
                    enemy_cluster_units_lst[i][0]))
            return action_lst
            # else:
            #     dis = distance(my_mp, self._backup_target_map)
            #     if dis < 5:
            #         self._dis_move_back[k] = False
            #         # print(self._dis_move_back)
            #     for i in range(len(clusters_a[k])):
            #         action_lst.append(actions.RAW_FUNCTIONS.Smart_unit(
            #             "now", clusters_a[k][i][0],
            #             enemy_cluster_units_lst[i][0]))
            #     return action_lst
        return actions.RAW_FUNCTIONS.no_op()

            # print('my_cluster_units_lst', my_cluster_units_lst)
            # print('enemy_cluster_units_lst', enemy_cluster_units_lst)
            # min_dist = float('inf')
            # min_idx = -1
            # for j in range(clusters_num):
            #     enemy_list = clusters_b[j]
            #     ememy_mp = self.get_center_position_point(enemy_list)
            #     if distance(my_mp, ememy_mp) < min_dist:
            #         min_dist = distance(my_mp, ememy_mp)
            #         min_idx = j
            # print(clusters_b[min_idx], min_dist)
            # enemy_cluster_units_lst = clusters_b[i]

    def action_greedy(self, obs):
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        mp = self.get_center_position(obs, 'Self', _MY_UNIT_TYPE_ARG)
        if len(my_units) > 0 and len(enemy_units) > 0:
            return actions.RAW_FUNCTIONS.Attack_unit(
                "now", [item[0] for item in my_units_lst], self.choice_nearest_weakest_enemy(mp, enemy_units))

        return actions.RAW_FUNCTIONS.no_op()

    def action_noise(self, obs):
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        if len(my_units) > 0:
            return actions.RAW_FUNCTIONS.Attack_pt(
                    "now", [item[0] for item in my_units_lst],
                    (50, 50))
        # if len(my_units) > 0:
        #     return actions.RAW_FUNCTIONS.Stop_quick(
        #             "now", [item[0] for item in my_units_lst])
        return actions.RAW_FUNCTIONS.no_op()

    def action_retreat(self, obs):
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        if len(my_units) > 0:
            return actions.RAW_FUNCTIONS.Smart_pt(
                    "now", [item[0] for item in my_units_lst],
                    (90, 90))
        # if len(my_units) > 0:
        #     return actions.RAW_FUNCTIONS.Stop_quick(
        #             "now", [item[0] for item in my_units_lst])
        return actions.RAW_FUNCTIONS.no_op()


    def __init__(self):
        super(SmartAgent, self).__init__()
        self.previous_state = None
        self.previous_action = None
        self._move_back = True
        self._dis_move_back = [True, True, True, True, True, True, True, True]
        self.score_cumulative_attack_last = 0
        self.score_cumulative_defense_last = 0
        self.score_cumulative_attack_now = 0
        self.score_cumulative_defense_now = 0
        self.score_attack_max = 0
        self.score_defense_max = 0
        self._backup_target_grid = (0, 0)
        self._backup_target_map = (0., 0.)
        self.action_queue = deque()
        self.end_game_frames = _STEP * _STEP_MUL
        self.end_game_state = 'Dogfall'
        self.qtable = QLearningTable(self.actions)
        self.new_game()

    def new_game(self):
        self.end_game_frames = _STEP * _STEP_MUL
        self.end_game_state = 'Dogfall'

    def get_window_im(self, obs):
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        unit_my_list = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        unit_enemy_list = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        influence_map = self.get_influence_map(unit_my_list, unit_enemy_list)
        # return (len(unit_my_list), len(unit_enemy_list))
        # print(influence_map)
        top, bottom, left, right = self.get_map_boundary(influence_map, _BOUNDARY_WIDTH)
        # 生成目标点
        target_gp = self.analyze_influence_map(influence_map)
        target_mp = self.grid_to_map(target_gp[0], target_gp[1])
        # self.ripple(influence_map, 0, (target_gp[0], target_gp[1]), 'Target')
        # 窗口取值
        window_map = influence_map.T[left:right, top:bottom]
        return window_map

    def get_state(self, obs):
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        unit_my_list = sorted([(item['tag'], item['x'], item['y']) for item in my_units], key=lambda x: x[0])
        unit_enemy_list = sorted([(item['tag'], item['x'], item['y']) for item in enemy_units], key=lambda x: x[0])
        influence_map = self.get_influence_map(unit_my_list, unit_enemy_list)
        # return (len(unit_my_list), len(unit_enemy_list))
        # print(influence_map)
        top, bottom, left, right = self.get_map_boundary(influence_map, _BOUNDARY_WIDTH)
        # 生成目标点
        target_gp = self.analyze_influence_map(influence_map)
        target_mp = self.grid_to_map(target_gp[0], target_gp[1])
        # self.ripple(influence_map, 0, (target_gp[0], target_gp[1]), 'Target')
        # 窗口取值
        window_map = influence_map.T[left:right, top:bottom]
        # array_to_pil_img(window_map)
        hash = getHash(array_to_pil_img(window_map))
        # print('hash', hash)
        return hash

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        # action = random.choice(self.actions)
        # print_all_attribute(obs.observation)
        # print(obs.observation.game_loop)
        unit_list_my = []
        if obs.first():
            global _EPISODE_COUNT
            _EPISODE_COUNT += 1
            # self.print_units_name(obs)
            unit_list_my = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
            unit_list_enemy = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
            self.score_attack_max = sum([item['health'] for item in unit_list_enemy])
            self.score_defense_max = sum([item['health'] for item in unit_list_my])
            # print(self.score_attack_max, self.score_defense_max)
            self.score_cumulative_attack_last = sum([item['health'] for item in unit_list_enemy])
            self.score_cumulative_defense_last = sum([item['health'] for item in unit_list_my])
            unit_list_both = unit_list_my + unit_list_enemy
            dataframe = pd.DataFrame(data=unit_list_both)
            dataframe.to_csv(_UNITS_LIST_PATH, header=True, index=False, sep=',')
            merge_units_csv(_UNITS_ATTRIBUTE_PATH, _UNITS_LIST_PATH)
            # return getattr(self, 'action_move_camera')(obs)
            # self.draw_units()

        # print(obs.observation.game_loop, 'step')
        my_units = self.get_my_units_by_type(obs, _MY_UNIT_TYPE_ARG)
        enemy_units = self.get_enemy_units_by_type(obs, _ENEMY_UNIT_TYPE_ARG)
        my_units_lst = sorted([(item['tag'], item['weapon_cooldown']) for item in my_units], key=lambda x: x[0])
        if (self.action_queue.count('Attack_unit') + self.action_queue.count('Smart_pt')) != 0:
            for tup in [elem for elem in my_units_lst]:
                if tup[1] != 0:
                    self._move_back = True

        # Zergling & Marine
        if len(enemy_units) == 0 and obs.observation['score_cumulative'][5] == obs.observation['score_cumulative'][3]:
            self.end_game_state = 'Win'
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop
            # print('You Win.')
        if len(my_units) == 0:
            self.end_game_state = 'Loss'
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop
            # print('You loss.')
            # print(self.qtable.q_table)
            # print(self.qtable.q_table)

        # array_to_pil_img(self.get_window_im(obs), True)
        # array_to_pil_img(self.get_window_im(obs), False)

        state = str(self.get_state(obs))
        # state = len(state_vec)
        # !!!!!!!
        # action = self.actions[2]
        action = self.qtable.choose_action(state)
        self.score_cumulative_attack_now = sum([item['health'] for item in enemy_units])
        self.score_cumulative_defense_now = sum([item['health'] for item in my_units])
        reward_attack = - (self.score_cumulative_attack_now - self.score_cumulative_attack_last)
        reward_defense = self.score_cumulative_defense_now - self.score_cumulative_defense_last
        self.score_cumulative_attack_last = self.score_cumulative_attack_now
        self.score_cumulative_defense_last = self.score_cumulative_defense_now
        # print(reward_attack, reward_defense)

        reward_cumulative = reward_attack + reward_defense
        # reward_cumulative = obs.observation['score_cumulative'][0] + obs.observation['score_cumulative'][5] - obs.observation['score_cumulative'][3]
        # path = r'data_for_transit\reward_cumulative_log.txt'
        # f = open(path, 'a', encoding='UTF-8')
        reward_d = obs.observation['score_cumulative'][0]
        reward_a = obs.observation['score_cumulative'][5]
        # f.write(
        #     f'{reward_d}\t{reward_a}\t{reward_cumulative}\n')
        # f.close()
        # print(obs.observation['score_cumulative'][0],
        #       obs.observation['score_cumulative'][5],
        #       obs.observation['score_cumulative'][3],
        #       reward_cumulative)
        # action = self.actions[self.actions.index("action_TFC_finish")]
        # action = random.choice(self.actions)
        # print(obs.reward)
        # print(self.previous_action, reward_cumulative)
        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              # obs.reward,
                              reward_cumulative,
                              'terminal' if obs.last() else state)

        # print(obs.observation['score_cumulative'])
        # print(self.qtable.q_table)
        # print('observation', dir(obs.observation))
        # print(obs.score_by_category, obs.score_by_vital, obs.score_cumulative)
        self.previous_state = state
        self.previous_action = action
        # print(obs.observation['score_cumulative'])
        # print()

        if obs.last():
            plt.close()
            matplotlib.pyplot.figure().clear()
            matplotlib.pyplot.close()
            # print(sys.getsizeof(plt) / 1024 / 1024, 'MB')
            # print('last.')
            # print(obs.reward)
            # 累积奖励
            # print(obs.observation['score_cumulative'])
            # print(obs.observation['score_cumulative'][5])
            f = open(_GAME_RESULT_PATH, 'a', encoding='UTF-8')
            reward_d = - (self.score_cumulative_attack_now - self.score_attack_max)
            reward_a = self.score_cumulative_defense_now - self.score_defense_max
            # print(self.score_cumulative_attack_now, self.score_cumulative_defense_now)
            f.write(
                f'{self.end_game_state}\t{self.end_game_frames}\t{reward_d}\t{reward_a}\n')
            f.close()
            # print(self.qtable.q_table)
            # print(self.end_game_state, self.end_game_frames)
            self.qtable.q_table.to_csv(_GAME_QTABLE_PATH, header=True, index=True, sep=',')
            self.end_game_frames = _STEP * _STEP_MUL
            self.end_game_state = 'Dogfall'
            if _EPISODE_COUNT == 2 or _EPISODE_COUNT % 10 == 0:
                append_qtable_csv(_EPISODE_COUNT, self.qtable.q_table, self.actions)
        return getattr(self, action)(obs)


def main(unused_argv):
    steps = _STEP
    step_mul = _STEP_MUL
    try:
        with sc2_env.SC2Env(
                # map_name="MarineMicro",
                # map_name="MarineMicro_TNC_1",
                # map_name="MarineMicro_MvsM_4",
                # map_name="MarineMicro_MvsM_8",
                # map_name="MarineMicro_MvsM_4_dist",
                # map_name="MarineMicro_MvsM_8_dist",
                # map_name="MarineMicro_MvsM_4_far",
                map_name="MarineMicro_MvsM_8_far",
                # map_name="MarineMicro_ZvsM_4",
                # map_name="MarineMicro_MvsM_8_dilemma",
                # map_name="MarineMicro_MvsM_8_dilemma_2",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         # sc2_env.Agent(sc2_env.Race.terran)],
                         # sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)],
                         sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=_MAP_RESOLUTION
                ),
                score_index=-1,
                # discount_zero_after_timeout=True,
                # realtime=True,
                # visualize=True,
                disable_fog=False,
                step_mul=step_mul,
                game_steps_per_episode=steps * step_mul
                # save_replay_episodes=1,
                # replay_dir="D:/白春辉/实验平台/pysc2-tutorial/replay"
        ) as env:
            # run_loop.run_loop([agent1, agent2], env, max_frames=10, max_episodes=1000)
            agent1 = SmartAgent()
            agent2 = Agent()
            run_loop.run_loop([agent1, agent2], env, max_episodes=500)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
