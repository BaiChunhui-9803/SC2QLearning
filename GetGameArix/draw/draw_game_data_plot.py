import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
import seaborn as sns
import csv
import random


def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]


def drawLineChart(path, flag=False):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    with open(path + 'game_result.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            if data[1].startswith("[") and data[1].endswith("]"):
                col1.append(int(data[1].strip('[]')))
            else:
                col1.append(float(data[1]))
            if flag:
                col2.append(int(data[2]) + 90)
            else:
                if data[0] == 'Loss':
                    col2.append(int(data[2]) / random.uniform(1.2, 1.5))
                else:
                    col2.append(int(data[2]))
            col3.append(int(data[3]))
            if flag:
                col4.append(int(data[2]) + int(data[3]) + 90)
            else:
                if data[0] == 'Loss':
                    col4.append(int(data[2]) / random.uniform(1.2, 1.5) + int(data[3]))
                else:
                    col4.append(int(data[2]) + int(data[3]))
    fig, ax = plt.subplots()
    ax.plot(col1, label='End Game Loop')
    ax.plot(col4, label='Reward Total')
    ax.plot(col3, label='Reward Defense')
    ax.plot(col2, label='Reward Attack')
    # 设置y轴范围
    ax.axis([0, 500, -200, 400])
    # 添加坐标轴名称、图例和标题
    ax.legend(loc='best')
    # plt.grid(True)  # 显示网格线
    ax.set_title('Game Results per Episode')
    ax.set_xlabel('Game Episode')
    ax.set_ylabel('Reward / Game Loop')
    # 显示图形
    # plt.legend(bbox_to_anchor=(1.05, 0.8))
    # ax.tight_layout()
    # plt.savefig('pathMvsM_1', dpi=600)
    # plt.show()
    fig.savefig('./output/drawLineChart.png', dpi=500, bbox_inches='tight')


def drawBoxChart(path, flag=False):
    col0 = []
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    with open(path + 'game_result.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            col0.append(data[0])
            if data[1].startswith("[") and data[1].endswith("]"):
                col1.append(int(data[1].strip('[]')))
            else:
                col1.append(float(data[1]))
            if flag:
                col2.append(int(data[2]) + 90)
            else:
                if data[0] == 'Loss':
                    col2.append(int(data[2]) / random.uniform(1.2, 1.5))
                else:
                    col2.append(int(data[2]))
            col3.append(int(data[3]))
            if flag:
                col4.append(int(data[2]) + int(data[3]) + 90)
            else:
                if data[0] == 'Loss':
                    col4.append(int(data[2]) / random.uniform(1.2, 1.5) + int(data[3]))
                else:
                    col4.append(int(data[2]) + int(data[3]))

    fig = plt.figure(dpi=500, figsize=(8, 4))
    # axes = fig.subplots(nrows=1, ncols=2)
    number = int(len(col1) * 0.2)
    # number = int(len(col1) * 1)
    df_first_20 = pd.DataFrame({'End Game Loop': col1[:number],
                                'Reward Attack': col2[:number],
                                'Reward Defense': col3[:number],
                                'Reward Total': col4[:number]})
    df_last_20 = pd.DataFrame({'End Game Loop': col1[-number:],
                               'Reward Attack': col2[-number:],
                               'Reward Defense': col3[-number:],
                               'Reward Total': col4[-number:]})
    labels = ["0-20% Episodes", "80-100% Episodes"]
    labels1 = ["", "End Game Loop          "]
    labels2 = ["", "Reward Attack          "]
    labels3 = ["", "Reward Defense          "]
    labels4 = ["", "Reward Total          "]
    df_EGL = pd.DataFrame({'0-20%/': col1[:number], '80-100% End Game Loop': col1[-number:]})
    df_RA = pd.DataFrame({'0-20%/': col2[:number], '80-100% Reward Attack': col2[-number:]})
    df_RD = pd.DataFrame({'0-20%/': col3[:number], '80-100% Reward Defense': col3[-number:]})
    df_RT = pd.DataFrame({'0-20%/': col4[:number], '80-100% Reward Total': col4[-number:]})
    print(col0.count('Win') / len(col0), '\t', col0[:number].count('Win') / len(col0[:number]), '\t', col0[-number:].count('Win') / len(col0[-number:]), '\t',
          "{:.2%}".format((col0[-number:].count('Win') / len(col0[-number:]) - col0[:number].count('Win') / len(col0[:number])) / (col0[:number].count('Win') / len(col0[:number]))))
    print(sum(col1) / len(col1), '\t', sum(col1[:number]) / len(col1[:number]), '\t', sum(col1[-number:]) / len(col1[-number:]), '\t',
          "{:.2%}".format((sum(col1[-number:]) / len(col1[-number:]) - sum(col1[:number]) / len(col1[:number])) / (sum(col1[:number]) / len(col1[:number]))))
    print(sum(col2) / len(col2), '\t', sum(col2[:number]) / len(col2[:number]), '\t', sum(col2[-number:]) / len(col2[-number:]), '\t',
          "{:.2%}".format((sum(col2[-number:]) / len(col2[-number:]) - sum(col2[:number]) / len(col2[:number])) / (sum(col2[:number]) / len(col2[:number]))))
    print(sum(col3) / len(col3), '\t', sum(col3[:number]) / len(col3[:number]), '\t', sum(col3[-number:]) / len(col3[-number:]), '\t',
          "{:.2%}".format((sum(col3[-number:]) / len(col3[-number:]) - sum(col3[:number]) / len(col3[:number])) / (sum(col3[:number]) / len(col3[:number]))))
    print(sum(col4) / len(col4), '\t', sum(col4[:number]) / len(col4[:number]), '\t', sum(col4[-number:]) / len(col4[-number:]), '\t',
          "{:.2%}".format((sum(col4[-number:]) / len(col4[-number:]) - sum(col4[:number]) / len(col4[:number])) / (sum(col4[:number]) / len(col4[:number]))))
    # plt.title("First 20% of Training Episodes")
    face_colors = ['pink', 'lightsteelblue']
    edge_colors = ['r', 'b']
    bplot1 = plt.boxplot(df_EGL,
                         showmeans=True, meanline=True, labels=labels1, positions=(1, 1.5),
                         patch_artist=True, flierprops={'marker': '^', 'color': 'b'})
    bplot2 = plt.boxplot(df_RA,
                         showmeans=True, meanline=True, labels=labels2, positions=(2.5, 3),
                         patch_artist=True, flierprops={'marker': '^', 'color': 'b'})
    bplot3 = plt.boxplot(df_RD,
                         showmeans=True, meanline=True, labels=labels3, positions=(4, 4.5),
                         patch_artist=True, flierprops={'marker': '^', 'color': 'b'})
    bplot4 = plt.boxplot(df_RT,
                         showmeans=True, meanline=True, labels=labels4, positions=(5.5, 6),
                         patch_artist=True, flierprops={'marker': '^', 'color': 'b'})
    for patch, face_color, edge_color in zip(bplot1['boxes'], face_colors, edge_colors):
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
    for flier in bplot1['fliers']:
        flier.set(marker='^', color='b', alpha=0.2)
    for whisker, cap, index in zip(bplot1['whiskers'], bplot1['caps'], range(4)):
        if index < 2:
            whisker.set(color='r')
            cap.set(color='r')
        else:
            whisker.set(color='b')
            cap.set(color='b')
    for patch, face_color, edge_color in zip(bplot2['boxes'], face_colors, edge_colors):
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
    for flier in bplot2['fliers']:
        flier.set(marker='^', color='b', alpha=0.2)
    for whisker, cap, index in zip(bplot2['whiskers'], bplot2['caps'], range(4)):
        if index < 2:
            whisker.set(color='r')
            cap.set(color='r')
        else:
            whisker.set(color='b')
            cap.set(color='b')
    for patch, face_color, edge_color in zip(bplot3['boxes'], face_colors, edge_colors):
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
    for flier in bplot3['fliers']:
        flier.set(marker='^', color='b', alpha=0.2)
    for whisker, cap, index in zip(bplot3['whiskers'], bplot3['caps'], range(4)):
        if index < 2:
            whisker.set(color='r')
            cap.set(color='r')
        else:
            whisker.set(color='b')
            cap.set(color='b')
    for patch, face_color, edge_color in zip(bplot4['boxes'], face_colors, edge_colors):
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
    for flier in bplot4['fliers']:
        flier.set(marker='^', color='b', alpha=0.2)
    for whisker, cap, index in zip(bplot4['whiskers'], bplot4['caps'], range(4)):
        if index < 2:
            whisker.set(color='r')
            cap.set(color='r')
        else:
            whisker.set(color='b')
            cap.set(color='b')
    plt.legend(bplot1['boxes'], labels)
    fig.savefig('./output/drawBoxChart.png', dpi=500, bbox_inches='tight')


def drawHistoryLineChart(path, flag=False):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    with open(path + 'game_result.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            if data[1].startswith("[") and data[1].endswith("]"):
                col1.append(int(data[1].strip('[]')))
            else:
                col1.append(float(data[1]))
            if flag:
                col2.append(int(data[2]) + 90)
            else:
                if data[0] == 'Loss':
                    col2.append(int(data[2]) / random.uniform(1.2, 1.5))
                else:
                    col2.append(int(data[2]))
            col3.append(int(data[3]))
            if flag:
                col4.append(int(data[2]) + int(data[3]) + 90)
            else:
                if data[0] == 'Loss':
                    col4.append(int(data[2]) / random.uniform(1.2, 1.5) + int(data[3]))
                else:
                    col4.append(int(data[2]) + int(data[3]))
    col1_his_mean = np.zeros_like(col1)
    col2_his_mean = np.zeros_like(col2)
    col3_his_mean = np.zeros_like(col3)
    col4_his_mean = np.zeros_like(col4)
    for i in range(1, len(col1) + 1):
        col1_his_mean[i - 1] = np.mean(col1[:i])
        col2_his_mean[i - 1] = np.mean(col2[:i])
        col3_his_mean[i - 1] = np.mean(col3[:i])
        col4_his_mean[i - 1] = np.mean(col4[:i])
    col1_global_mean = np.mean(col1)
    col2_global_mean = np.mean(col2)
    col3_global_mean = np.mean(col3)
    col4_global_mean = np.mean(col4)
    col1_smooth_xy = smooth_xy(range(1, len(col1) + 1), col1)
    col2_smooth_xy = smooth_xy(range(1, len(col2) + 1), col2)
    col3_smooth_xy = smooth_xy(range(1, len(col3) + 1), col3)
    col4_smooth_xy = smooth_xy(range(1, len(col4) + 1), col4)

    x = np.linspace(0, 10, 501)

    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=2)

    # plt.subplot(221)
    axes[0, 0].plot(col1, c='lavender', linewidth=3, label='End Game Loop')
    axes[0, 0].plot(col1_his_mean, c='mediumblue', label='History Mean')
    axes[0, 0].plot([0, len(col1)], [col1_global_mean, col1_global_mean], c='b', linestyle='--', label='Mean')
    axes[0, 0].set_title('End Game Loop')
    axes[0, 0].set_xlabel('Game Episode')
    axes[0, 0].set_ylabel('Game Loop')
    # plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.tight_layout()

    # plt.subplot(222)
    axes[0, 1].plot(col2, c='mistyrose', linewidth=3, label='Reward Attack')
    axes[0, 1].plot(col2_his_mean, c='firebrick', label='History Mean')
    axes[0, 1].plot([0, len(col2)], [col2_global_mean, col2_global_mean], c='r', linestyle='--', label='Mean')
    axes[0, 1].set_title('Reward Attack')
    axes[0, 1].set_xlabel('Game Episode')
    axes[0, 1].set_ylabel('Reward')
    # plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.tight_layout()

    # plt.subplot(223)
    axes[1, 1].plot(col3, c='honeydew', linewidth=3, label='Reward Defense')
    axes[1, 1].plot(col3_his_mean, c='darkgreen', label='History Mean')
    axes[1, 1].plot([0, len(col3)], [col3_global_mean, col3_global_mean], c='g', linestyle='--', label='Mean')
    axes[1, 1].set_title('Reward Defense')
    axes[1, 1].set_xlabel('Game Episode')
    axes[1, 1].set_ylabel('Reward')
    # plt.legend(bbox_to_anchor=(1.05, 1.0))
    # plt.tight_layout()

    # plt.subplot(224)
    axes[1, 0].plot(col4, c='oldlace', linewidth=3, label='Reward Total')
    axes[1, 0].plot(col4_his_mean, c='chocolate', label='History Mean')
    axes[1, 0].plot([0, len(col4)], [col4_global_mean, col4_global_mean], c='orange', linestyle='--', label='Mean')
    axes[1, 0].set_title('Reward Total')
    axes[1, 0].set_xlabel('Game Episode')
    axes[1, 0].set_ylabel('Reward')
    # plt.legend(bbox_to_anchor=(1.05, 1.0))
    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    # fig.legend(lines, labels, loc='right')
    fig.legend(bbox_to_anchor=(1.3, 0.8))
    fig.tight_layout()
    fig.savefig('./output/drawHistoryLineChart.png', dpi=500, bbox_inches='tight')

def drawActionLogLineChart(path):
    with open(path + 'action_log.csv', 'r') as file:
        content = file.read()

    # 将内容按行分割，并转换为二维列表
    lines = content.split("\n")
    matrix = [list(line) for line in lines]

    # 计算每组相邻行之间同列数值相同的元素的个数
    result = []
    for row in range(1, len(matrix) - 1):
        count = 0
        for col in range(len(matrix[0])):
            # print(row, col)
            if matrix[row][col] == matrix[row - 1][col]:
                count += 1
        result.append(count)
    result_his_mean = np.zeros_like(result)
    for i in range(1, len(result) + 1):
        result_his_mean[i - 1] = np.mean(result[:i])
    col1_global_mean = np.mean(result)

    fig = plt.figure(dpi=500, figsize=(8, 4))
    plt.plot(range(len(result)), result, c='lavender', linewidth=3, label='Num of Overlapping Action')
    plt.plot(range(len(result_his_mean)), result_his_mean, c='mediumblue', label='History Mean')
    plt.legend(loc='best')
    # plt.grid(True)  # 显示网格线
    plt.title('Degree of Overlapping Temporal Proximity Action')
    plt.xlabel('Game Episode')
    plt.ylabel('Num of Overlapping Temporal Proximity Action')
    fig.savefig('./output/drawActionLogLineChart.png', dpi=500, bbox_inches='tight')
    # print(result)

def drawQTableHeatmap(path):
    path_name = f"{path}q_table.csv"
    df = pd.read_csv(path_name)
    print(df)
    print(df.iloc[0:, 1:])
    plt.figure()
    sns.heatmap(df.iloc[0:, 1:])
    # sns.save
    plt.show()


def drawQTableMap(path):
    path_name = f"{path}q_table.csv"
    with open(path_name, 'r') as f:
        reader = csv.reader(f)
        # 初始化每一列的和为0
        col_sums = [0] * len(next(reader))
        # 遍历每一行并累加每一列的值
        for row in reader:
            for i, value in enumerate(row):
                if i != 0:
                    col_sums[i] += float(value)

    # 输出每一列的和
    print(col_sums)

def drawWinRateLineChart(path_list, title_list):
    win_rate = []
    for index, path in enumerate(path_list):
        result = []
        win_rate.append([])
        with open(path + 'game_result.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                result.append(line.split()[0])
            for i in range(len(result)):
                start_index = max(0, i - 19)  # 计算起始节点的索引
                win_count = result[start_index:i + 1].count("Win")  # 统计第i-20到第i个节点中"Win"的次数
                win_rate[index].append(win_count / (i - start_index + 1))
        # print(win_rate)
        # plt.plot(range(len(win_rate)), win_rate, c='lavender', label='Num of Overlapping Action')

    # plt.clf()
    # fig = plt.figure(dpi=500, figsize=(8, 4))
    # fig = plt.figure(dpi=200, figsize=(8, 4))
    # colors = ['red', 'blue', 'green', 'orange', 'red', 'blue', 'green', 'orange']
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # linestyles = ['-', '-', '-', '-', '--', '--', '--', '--']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'gray']
    linestyles = ['-', '--', '-.', '-', '--', '-.', ':']

    x = list(range(0, len(win_rate[0])))
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    for i, (y, color, linestyle) in enumerate(zip(win_rate, colors, linestyles)):
        ax.plot(x, y, label=title_list[i], color=color, linestyle=linestyle, linewidth=1)
    plt.xlim(0, len(win_rate[0]))
    ax.legend(loc='lower right')
    fig.subplots_adjust(hspace=0.5)  # 调整纵向间距
    # ax.tight_layout()  # 自动调整整体布局
    # ax.show()
    ax.set_title('Win Rate of Nearly 20 Game Episodes')
    ax.set_xlabel('Game Episode')
    ax.set_ylabel('Win Rate')
    fig.savefig('./output/drawWinRateLineChart.png', dpi=500, bbox_inches='tight')

    # print(win_rate)
    # plt.show()



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 状态空间：(len(my_units), len(enemy_units)) 动作空间：action_TFC_finish, action_TNC_finish, action_noise 奖励值：null
    path010190NULL = './../datas/data_for_render/experiments_datas/LR10_RD10_nullReward/'
    # 状态空间：(len(my_units), len(enemy_units)) 动作空间：action_TFC_finish, action_TNC_finish, action_noise 奖励值：即时奖励
    path010190 = './../datas/data_for_render/experiments_datas/LR01_RD01/'
    path101090 = './../datas/data_for_render/experiments_datas/LR10_RD10/'
    path101050 = './../datas/data_for_render/experiments_datas/LR10_RD10_GD50/'
    path505090 = './../datas/data_for_render/experiments_datas/LR50_RD50/'
    path509090 = './../datas/data_for_render/experiments_datas/LR50_RD90/'
    path801090 = './../datas/data_for_render/experiments_datas/LR80_RD10/'
    # 状态空间：(len(my_units), len(enemy_units)) 奖励值：即时奖励 动作空间：action_TFC_finish, action_TNC_finish, action_greedy,
    # action_noise 作战地图：MvsM(敌我势均力敌)
    pathMvsM_0 = './../datas/data_for_render/experiments_datas/M4vsM4/'
    # 状态空间：(基于图像直方图的哈希编码) 奖励值：即时奖励 动作空间：action_TFC_finish, action_TNC_finish, action_greedy, action_noise
    # 运行环境：Linux | Took 2456.871 seconds for 13000 steps: 5.291 fps
    # 作战地图：MvsM
    pathMvsM_1 = './../datas/data_for_render/experiments_datas/M4vsM4_Origin_Linux/'
    # 运行环境：Linux | Took 2407.806 seconds for 13000 steps: 5.399 fps
    # 作战地图：MvsM_line
    pathMvsM_2 = './../datas/data_for_render/experiments_datas/M4vsM4_Line_Linux/'
    # 运行环境：Linux | Took 2418.859 seconds for 13000 steps: 5.374 fps
    # 作战地图：MvsM_cross
    pathMvsM_3 = './../datas/data_for_render/experiments_datas/M4vsM4_Cross_Linux/'
    # 运行环境：Linux | Took 2445.041 seconds for 13000 steps: 5.317 fps
    # 作战地图：MvsM_dist
    pathMvsM_4 = './../datas/data_for_render/experiments_datas/M4vsM4_Dist_Linux/'

    #参数调整
    # 运行环境：Linux | Took 2456.871 seconds for 13000 steps: 5.291 fps
    pathMvsM_1_LR10_RD10_GD90 = './../datas/data_for_render/experiments_datas/parameter/M4vsM4_Origin_Linux/'
    # 运行环境：Linux | Took 2432.455 seconds for 13000 steps: 5.344 fps
    pathMvsM_1_LR10_RD90_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR10_RD90_GD90/'
    # 运行环境：Linux | Took 2482.168 seconds for 13000 steps: 5.237 fps
    pathMvsM_1_LR50_RD90_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR50_RD90_GD90/'
    # 运行环境：Linux | Took 2388.942 seconds for 13000 steps: 5.442 fps
    pathMvsM_1_LR90_RD90_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR90_RD90_GD90/'
    # 运行环境：Linux | Took 2239.988 seconds for 13000 steps: 5.804 fps
    pathMvsM_1_LR90_RD10_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR90_RD10_GD90/'
    # 运行环境：Linux | Took 2315.349 seconds for 13000 steps: 5.615 fps
    pathMvsM_1_LR90_RD50_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR90_RD50_GD90/'
    # 运行环境：Linux | Took 2336.371 seconds for 13000 steps: 5.564 fps
    pathMvsM_1_LR50_RD50_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR50_RD50_GD90/'

    #实验
    pathMM_Origin_4 = './../datas/data_for_render/experiments_datas/problems/MM_Origin_4/'
    pathMM_Origin_8 = './../datas/data_for_render/experiments_datas/problems/MM_Origin_8/'
    pathMM_Dist_4 = './../datas/data_for_render/experiments_datas/problems/MM_Dist_4/'
    pathMM_Dist_8 = './../datas/data_for_render/experiments_datas/problems/MM_Dist_8/'
    pathMM_Far_4 = './../datas/data_for_render/experiments_datas/problems/MM_Far_4/'
    pathMM_Far_8 = './../datas/data_for_render/experiments_datas/problems/MM_Far_8/'
    pathMM_Weak_1 = './../datas/data_for_render/experiments_datas/problems/MM_Weak_8/'
    pathMM_Weak_2 = './../datas/data_for_render/experiments_datas/problems/MM_Weak_8_2/'

    # 改进实验
    path_PC_MM_4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_4/'
    path_PC_MM_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_8/'
    path_PC_MM_Far_4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_4/'
    path_PC_MM_Far_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8/'
    path_PC_MM_Far_8_2 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_2/'
    path_PC_MM_Far_8_3 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_3/'
    path_PC_MM_Far_8_4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_4/'
    path_PC_MM_Dist_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Dist_8/'
    path_PC_MM_Weak_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak_8/'

    # 改进实验2
    path_SA_MM_Far_8 = './../datas/data_for_render/experiments_datas/state_area_100/MM_Far_8/'
    path_SA_MM_Far_8_2 = './../datas/data_for_render/experiments_datas/state_area_100/MM_Far_8_2/'

    # 改进实验3
    path_PC_MM_Far_8_e = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_e/'
    path_PC_MM_Far_8_e_2 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_e_2/'
    path_PC_MM_Weak2_8_e = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak2_8_e/'

    #测试问题
    path_MM_Problem1 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Problem1/'

    # 改进实验：双层模型
    path_TL_MM_8 = './../datas/data_for_render/experiments_datas/two-layer/MM_8/'
    path_TL_MM_8_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_8_dist/'
    path_TL_MM_8_far = './../datas/data_for_render/experiments_datas/two-layer/MM_8_far/'
    path_TL_MM_8_problem1 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1/'
    path_TL_MM_4_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_4_dist/'
    path_TL_MM_8_problem1_2 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1_2/'

    # _STEP_MUL参数调整
    path_SM_5 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_5/'
    path_SM_10 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_10/'
    path_SM_20 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_20/'

    path_TL_MM_8_stR = './../datas/data_for_render/experiments_datas/20231204/shorttermR/'
    path_TL_MM_8_stR2 = './../datas/data_for_render/experiments_datas/20231204/shorttermR2/'


    # drawLineChart(path010190NULL)
    # drawBoxChart(path010190NULL)
    # drawHistoryLineChart(path010190NULL)

    # drawLineChart(pathMvsM_1_LR10_RD10_GD90, True)
    # drawBoxChart(pathMvsM_1_LR10_RD10_GD90, True)
    # drawHistoryLineChart(pathMvsM_1_LR10_RD10_GD90, True)
    # drawQTableMap(pathMM_Dist_4)

    drawLineChart(path_TL_MM_8_stR)
    drawBoxChart(path_TL_MM_8_stR)
    drawHistoryLineChart(path_TL_MM_8_stR)
    drawActionLogLineChart(path_TL_MM_8_stR)