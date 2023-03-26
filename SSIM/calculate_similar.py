import os

import ssim2 as ssim
from PIL import Image

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    # canvas = FigureCanvasAgg(plt.gcf())
    # canvas.draw()
    # w, h = canvas.get_width_height()
    # buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    # buf.shape = (w, h, 4)
    # buf = np.roll(buf, 3, axis=2)
    # img = Image.frombytes("RGB", (w, h), buf.tobytes())
    # print('img', type(img))
    # plt.show()
    return ssim.make_regalur_image(img)


def calculate_similar():
    file_path = "img/"
    files = os.listdir(file_path)
    print(files)
    file_num = len(files)
    mat = np.zeros((file_num, file_num))
    for i in range(file_num):
        for j in range(file_num):
            img_left, img_right = Image.open(f"{file_path}{files[i]}"), Image.open(f"{file_path}{files[j]}")
            img_left.save("100.png")
            mat[i][j] = ssim.calc_similar(img_left, img_right)
            # print(ssim.calc_similar(img_left, img_right))
    print(mat)


def calculate_similar_from_arr():
    file_path = "arr/"
    files = os.listdir(file_path)
    print(files)
    file_num = len(files)
    mat = np.zeros((file_num, file_num))
    for i in range(file_num):
        for j in range(file_num):
            arr_left = np.loadtxt(f"{file_path}{files[i]}", dtype=np.float32)
            arr_right = np.loadtxt(f"{file_path}{files[j]}", dtype=np.float32)
            # img_left, img_right = Image.open(f"{file_path}{files[i]}"), Image.open(f"{file_path}{files[j]}")
            # img_left.save("100.png")
            # print(arr_left)
            mat[i][j] = ssim.calc_similar(array_to_pil_img(arr_left), array_to_pil_img(arr_right))
            # print(ssim.calc_similar(img_left, img_right))
        print(i, j)
    dataDf = pd.DataFrame(mat)
    p1 = sns.heatmap(dataDf, square=True, linewidths=0.3,
                     cmap='RdBu_r', annot=True, xticklabels=1, yticklabels=1)
    s1 = p1.get_figure()
    img = Image.frombytes('RGB', s1.canvas.get_width_height(), s1.canvas.tostring_rgb())
    img.save("heatmap_22.png")
    print(mat)


# calculate_similar()

calculate_similar_from_arr()