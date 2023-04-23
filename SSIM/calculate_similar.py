import os

import similar_histogram as similar_histogram
from PIL import Image

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


_MAX_INFLUENCE = 25
_MIN_INFLUENCE = -16 * 4


def array_to_pil_img(arr: np.ndarray, check_flag=False):
    # plt.figure()
    # print(arr.min(), arr.max())
    norm = mcolors.TwoSlopeNorm(vmin=_MIN_INFLUENCE, vmax=_MAX_INFLUENCE, vcenter=0.0)
    p1 = sns.heatmap(arr, cmap="RdBu", norm=norm,
                     annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False)
    # p1 = sns.heatmap(arr, cmap="coolwarm", vmin=-25, vmax=25, annot=False, cbar=False, square=True,
                     # xticklabels=False, yticklabels=False, linewidth=.5)
    s1 = p1.get_figure()
    img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.tostring_rgb())
    # img.show()
    return similar_histogram.make_regalur_image(img)


def calculate_similar_from_arr(arr_list, img_list):
    f_out = open("similar_pos.txt", "w", newline='')
    for i, element in enumerate(img_list):
        for j in range(i + 1):
            arr_left = img_list[i]
            arr_right = img_list[j]
            mat[i][j] = similar_histogram.calc_similar(arr_left, arr_right)
            if mat[i][j] > 0.85:
                f_out.write(f"{i}\t{j}\n")  # 写入数据
        # print(i)
    f_out.close()  # 关闭文件
    m = 0.0
    n = 0.0
    for i in range(len(points_2d)):
        for j in range(len(points_2d[i])):
            if points_2d[i][j]:
                # print(mat[i][j])
                m += 1.0
                if mat[i][j] > 0.5:
                    n += 1.0
    print(n/m)
    img_path = "GetGameArix/draw/im_img/"
    img_list = os.listdir(img_path)
    df = pd.DataFrame(mat, index=img_list, columns=img_list)
    colors = ['white', 'white', 'lightskyblue', 'darkblue', 'white', 'pink', 'indianred', 'crimson']
    nodes = [0, 0.1, 0.5, 0.8, 0.8, 0.8, 0.95, 1]
    cmap1 = mcolors.LinearSegmentedColormap.from_list(
        'cmap1', list(zip(nodes, colors))
    )
    # sns.set(font_scale=1.25)
    plt.close()
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(data=df,
                     square=True,
                     cmap=cmap1,
                     # linewidths=0.,
                     # annot=True,
                     vmin=0, center=0.5, vmax=1,
                     xticklabels=5,
                     yticklabels=5,
                     mask=df == 0)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.xlabel('right_image')  # 设置坐标名称
    plt.ylabel('left_image')
    plt.title('Heatmap of Left-Right Image Similarity Matrix')  # 标题
    # plt.show()
    plt.savefig("heatmap.png", dpi=500)
    # s1 = p1.get_figure()
    # img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.tostring_rgb())
    # img.save("heatmap.png")
    # print(mat)


def create_arr_list(file_list):
    file_arr_list = [0 for index in file_list]
    for idx, element in enumerate(file_list):
        # print(element)
        file_arr_list[idx] = np.loadtxt(f"{file_path}{element}", dtype=np.float32)
    return file_arr_list


def create_img_list(file_list):
    file_img_list = [0 for index in file_list]
    for idx, element in enumerate(file_list):
        # print(element)
        file_img_list[idx] = array_to_pil_img(element)
        print(f"完成图像转录{idx}")
    return file_img_list


file_path = "GetGameArix/draw/im_arr/"
files_list = os.listdir(file_path)
print(files_list)
files_arr_list = create_arr_list(files_list)
files_img_list = create_img_list(files_arr_list)
mat = np.zeros((len(files_list), len(files_list)))
# print(files_list)
with open('similar_hash_pos.txt') as f:
    lines = f.readlines()
points = []
points_2d = [[0] * len(files_list) for _ in range(len(files_list))]
for line in lines:
    x, y, *_ = line.split()
    points.append((int(x), int(y)))
    points_2d[int(x)][int(y)] = 1
print(len(points_2d))

calculate_similar_from_arr(files_list, files_img_list)