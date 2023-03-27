import os

import ssim2 as ssim
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
    return ssim.make_regalur_image(img)


def calculate_similar_from_arr(arr_list, img_list):
    for i, element in enumerate(img_list):
        for j in range(i + 1):
            arr_left = img_list[i]
            arr_right = img_list[j]
            mat[i][j] = ssim.calc_similar(arr_left, arr_right)
        print(i)
    data = pd.DataFrame(mat)
    colors = ['white', 'white', 'lightskyblue', 'darkblue', 'white', 'pink', 'indianred', 'crimson']
    nodes = [0, 0.1, 0.5, 0.8, 0.8, 0.8, 0.95, 1]
    cmap1 = mcolors.LinearSegmentedColormap.from_list(
        'cmap1', list(zip(nodes, colors))
    )
    sns.set(font_scale=1.25)
    p1 = sns.heatmap(data,
                     square=True,
                     # linewidths=0.003,
                     annot=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     vmin=0, center=0.5, vmax=1,
                     cmap=cmap1,
                     xticklabels=5,
                     yticklabels=5)
    s1 = p1.get_figure()
    img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.tostring_rgb())
    img.save("heatmap.png")
    print(mat)


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
    return file_img_list


file_path = "arr/"
files_list = os.listdir(file_path)
print(files_list)
files_arr_list = create_arr_list(files_list)
files_img_list = create_img_list(files_arr_list)
mat = np.zeros((len(files_list), len(files_list)))
# print(files_list)
calculate_similar_from_arr(files_list, files_img_list)