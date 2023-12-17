import matplotlib.pyplot as plt
import pandas as pd



def fun1(path):
    path_name = f"{path}episode_q_table.csv"
    data = pd.read_csv(path_name, header=None, skiprows=[0])
    # 对正负值进行归一化
    for index, row in data.iterrows():
        min_val = min(row)
        max_val = max(row)
        for j in range(len(row)):
            row[j] = (float(row[j]) - float(min_val)) / (float(max_val) - float(min_val))
        total = sum(row)
        for j in range(len(row)):
            row[j] = float(row[j]) / float(total)
    # 绘制堆叠柱状图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data.index, data[0], label='action_TFC')
    ax.bar(data.index, data[1], bottom=data[0], label='action_TFU')
    ax.bar(data.index, data[2], bottom=data[[0, 1]].sum(axis=1), label='action_TNC')
    ax.bar(data.index, data[3], bottom=data[[0, 1, 2]].sum(axis=1), label='action_TNU')
    ax.bar(data.index, data[4], bottom=data[[0, 1, 2, 3]].sum(axis=1), label='action_DFC')
    ax.bar(data.index, data[5], bottom=data[[0, 1, 2, 3, 4]].sum(axis=1), label='action_DFU')
    ax.bar(data.index, data[6], bottom=data[[0, 1, 2, 3, 4, 5]].sum(axis=1), label='action_DNC')
    ax.bar(data.index, data[7], bottom=data[[0, 1, 2, 3, 4, 5, 6]].sum(axis=1), label='action_DNU')
    ax.bar(data.index, data[8], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7]].sum(axis=1), label='action_retreat')

    # 添加图例和标签
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.8))
    plt.tight_layout()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Action Ratios')
    ax.set_title('Action Selection Trends in MM_Origin_8')
    fig.savefig('drawAllTimeStackedChart.png', dpi=500, bbox_inches='tight')


def fun2(path):
    path_name = f"{path}q_table.csv"
    data = pd.read_csv(path_name, header=None, skiprows=[0], usecols=range(1, 10))
    # 对正负值进行归一化
    for index, row in data.iterrows():
        min_val = min(row)
        max_val = max(row)
        for j in range(len(row)):
            if min_val != max_val:
                row[j + 1] = (float(row[j + 1]) - float(min_val)) / (float(max_val) - float(min_val))
            else:
                row[j + 1] = 1. / 9.
        total = sum(row)
        for j in range(len(row)):
            row[j + 1] = float(row[j + 1]) / float(total)
    # 绘制堆叠柱状图
    data = data
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data.index, data[1], label='action_TFC')
    ax.bar(data.index, data[2], bottom=data[1], label='action_TFU')
    ax.bar(data.index, data[3], bottom=data[[1, 2]].sum(axis=1), label='action_TNC')
    ax.bar(data.index, data[4], bottom=data[[1, 2, 3]].sum(axis=1), label='action_TNU')
    ax.bar(data.index, data[5], bottom=data[[1, 2, 3, 4]].sum(axis=1), label='action_DFC')
    ax.bar(data.index, data[6], bottom=data[[1, 2, 3, 4, 5]].sum(axis=1), label='action_DFU')
    ax.bar(data.index, data[7], bottom=data[[1, 2, 3, 4, 5, 6]].sum(axis=1), label='action_DNC')
    ax.bar(data.index, data[8], bottom=data[[1, 2, 3, 4, 5, 6, 7]].sum(axis=1), label='action_DNU')
    ax.bar(data.index, data[9], bottom=data[[1, 2, 3, 4, 5, 6, 7, 8]].sum(axis=1), label='action_retreat')

    # 添加图例和标签
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.8))
    plt.tight_layout()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Action Ratios')
    ax.set_title('Action Selection Trends in MM_Origin_8')
    fig.savefig('drawSomeStackedChart.png', dpi=500, bbox_inches='tight')


def fun3(path):
    path_name = f"{path}q_table.csv"
    data = pd.read_csv(path_name, header=None, skiprows=[0], nrows=1, usecols=range(1, 10))
    print(data)
    # 对正负值进行归一化
    for index, row in data.iterrows():
        min_val = min(row)
        max_val = max(row)
        for j in range(len(row)):
            if min_val != max_val:
                row[j + 1] = (float(row[j + 1]) - float(min_val)) / (float(max_val) - float(min_val))
            else:
                row[j + 1] = 1. / 9.
        total = sum(row)
        for j in range(len(row)):
            row[j + 1] = float(row[j + 1]) / float(total)
    # 绘制堆叠柱状图
    data = data
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data.index, data[1], label='action_TFC')
    ax.bar(data.index, data[2], bottom=data[1], label='action_TFU')
    ax.bar(data.index, data[3], bottom=data[[1, 2]].sum(axis=1), label='action_TNC')
    ax.bar(data.index, data[4], bottom=data[[1, 2, 3]].sum(axis=1), label='action_TNU')
    ax.bar(data.index, data[5], bottom=data[[1, 2, 3, 4]].sum(axis=1), label='action_DFC')
    ax.bar(data.index, data[6], bottom=data[[1, 2, 3, 4, 5]].sum(axis=1), label='action_DFU')
    ax.bar(data.index, data[7], bottom=data[[1, 2, 3, 4, 5, 6]].sum(axis=1), label='action_DNC')
    ax.bar(data.index, data[8], bottom=data[[1, 2, 3, 4, 5, 6, 7]].sum(axis=1), label='action_DNU')
    ax.bar(data.index, data[9], bottom=data[[1, 2, 3, 4, 5, 6, 7, 8]].sum(axis=1), label='action_retreat')

    # 添加图例和标签
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.8))
    plt.tight_layout()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Action Ratios')
    ax.set_title('Action Selection Trends in MM_Origin_8')
    fig.savefig('drawSomeStackedChart.png', dpi=500, bbox_inches='tight')


def fun4(path):
    path_name = f"{path}episode_q_table.csv"
    data = pd.read_csv(path_name, header=None, skiprows=[0])
    print(data)
    # 对正负值进行归一化
    for index, row in data.iterrows():
        min_val = min(row)
        max_val = max(row)
        for j in range(len(row)):
            row[j] = (float(row[j]) - float(min_val)) / (float(max_val) - float(min_val))
        total = sum(row)
        for j in range(len(row)):
            row[j] = float(row[j]) / float(total)
    # 绘制堆叠柱状图
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(data.index, data[0], label='action_TFC_000')
    ax.bar(data.index, data[1], bottom=data[0], label='action_TFU_000')
    ax.bar(data.index, data[2], bottom=data[[0, 1]].sum(axis=1), label='action_TNC_000')
    ax.bar(data.index, data[3], bottom=data[[0, 1, 2]].sum(axis=1), label='action_TNU_000')
    ax.bar(data.index, data[4], bottom=data[[0, 1, 2, 3]].sum(axis=1), label='action_TFC_025')
    ax.bar(data.index, data[5], bottom=data[[0, 1, 2, 3, 4]].sum(axis=1), label='action_TFU_025')
    ax.bar(data.index, data[6], bottom=data[[0, 1, 2, 3, 4, 5]].sum(axis=1), label='action_TNC_025')
    ax.bar(data.index, data[7], bottom=data[[0, 1, 2, 3, 4, 5, 6]].sum(axis=1), label='action_TNU_025')
    ax.bar(data.index, data[8], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7]].sum(axis=1), label='action_TFC_050')
    ax.bar(data.index, data[9], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8]].sum(axis=1), label='action_TFU_050')
    ax.bar(data.index, data[10], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].sum(axis=1), label='action_TNC_050')
    ax.bar(data.index, data[11], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].sum(axis=1), label='action_TNU_050')
    ax.bar(data.index, data[12], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].sum(axis=1), label='action_TFC_075')
    ax.bar(data.index, data[13], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].sum(axis=1), label='action_TFU_075')
    ax.bar(data.index, data[14], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].sum(axis=1), label='action_TNC_075')
    ax.bar(data.index, data[15], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].sum(axis=1), label='action_TNU_075')
    ax.bar(data.index, data[16], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]].sum(axis=1), label='action_TFC_100')
    ax.bar(data.index, data[17], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].sum(axis=1), label='action_TFU_100')
    ax.bar(data.index, data[18], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].sum(axis=1), label='action_TNC_100')
    ax.bar(data.index, data[19], bottom=data[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]].sum(axis=1), label='action_TNU_100')

    # 添加图例和标签
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Action Ratios')
    ax.set_title('Action Selection Trends')
    fig.savefig('drawAllStackedChart.png', dpi=500, bbox_inches='tight')

def fun5(path):
    path_name = f"{path}episode_q_table.csv"
    data = pd.read_csv(path_name, header=None, skiprows=[0])
    # print(data)
    # 对正负值进行归一化
    for index, row in data.iterrows():
        min_val = min(row)
        max_val = max(row)
        for j in range(len(row)):
            row[j] = (float(row[j]) - float(min_val)) / (float(max_val) - float(min_val))
        total = sum(row)
        for j in range(len(row)):
            row[j] = float(row[j]) / float(total)
    # 绘制堆叠柱状图
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(data.index, data[0], label='action_kmeans_000')
    ax.bar(data.index, data[1], bottom=data[0], label='action_kmeans_025')
    ax.bar(data.index, data[2], bottom=data[[0, 1]].sum(axis=1), label='action_kmeans_050')
    ax.bar(data.index, data[3], bottom=data[[0, 1, 2]].sum(axis=1), label='action_kmeans_075')
    ax.bar(data.index, data[4], bottom=data[[0, 1, 2, 3]].sum(axis=1), label='action_kmeans_100')

    # 添加图例和标签
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Action Ratios')
    ax.set_title('Action Selection Trends')
    fig.savefig('./output/drawAllStackedChart.png', dpi=500, bbox_inches='tight')


MM_Origin_4 = "datas/data_for_render/experiments_datas/problems/MM_Origin_4/"
MM_Origin_8 = "datas/data_for_render/experiments_datas/problems/MM_Origin_8/"
MM_Dist_4 = "datas/data_for_render/experiments_datas/problems/MM_Dist_4/"
MM_Dist_8 = "datas/data_for_render/experiments_datas/problems/MM_Dist_8/"
pathMM_Far_4 = "datas/data_for_render/experiments_datas/problems/MM_Far_4/"
pathMM_Far_8 = "datas/data_for_render/experiments_datas/problems/MM_Far_8/"
pathMM_Weak_1 = "datas/data_for_render/experiments_datas/problems/MM_Weak_8/"
pathMM_Weak_2 = "datas/data_for_render/experiments_datas/problems/MM_Weak_8_2/"

# 改进实验
path_PC_MM_4 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_4/'
path_PC_MM_8 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_8/'
path_PC_MM_Far_4 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_4/'
path_PC_MM_Far_8 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8/'
path_PC_MM_Far_8_2 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_2/'
path_PC_MM_Far_8_3 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_3/'
path_PC_MM_Far_8_4 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_4/'
path_PC_MM_Dist_8 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Dist_8/'
path_PC_MM_Weak_8 = '../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak_8/'

# 双层模型
path_TL_MM_8 = '../datas/data_for_render/experiments_datas/two-layer/MM_8/'
path_TL_MM_8_dist = '../datas/data_for_render/experiments_datas/two-layer/MM_8_dist/'
path_TL_MM_8_far = '../datas/data_for_render/experiments_datas/two-layer/MM_8_far/'
path_TL_MM_8_problem1 = '../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1/'
path_TL_MM_4_dist = '../datas/data_for_render/experiments_datas/two-layer/MM_4_dist/'
path_TL_MM_8_problem1_2 = '../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1_2/'

path_SM_5 = '../datas/data_for_render/experiments_datas/two-layer-2/MM_8_5/'
path_SM_10 = '../datas/data_for_render/experiments_datas/two-layer-2/MM_8_10/'
path_SM_20 = '../datas/data_for_render/experiments_datas/two-layer-2/MM_8_20/'


path_TL_MM_8_stR = '../datas/data_for_render/experiments_datas/shorttermR/shorttermR/'

# fun1(MM_Origin_4)
# fun3(MM_Origin_4)
# fun4(path_TL_MM_8)
fun5(path_TL_MM_8_stR)
# plt.show()