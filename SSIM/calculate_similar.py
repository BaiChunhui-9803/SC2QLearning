import os

import ssim2 as ssim
from PIL import Image
import numpy as np


def calculate_similar():
    file_path = "img/"
    files = os.listdir(file_path)
    print(files)
    file_num = len(files)
    mat = np.zeros((file_num, file_num))
    for i in range(file_num):
        for j in range(file_num):
            img_left, img_right = Image.open(f"{file_path}{files[i]}"), Image.open(f"{file_path}{files[j]}")
            mat[i][j] = ssim.calc_similar(img_left, img_right)
            # print(ssim.calc_similar(img_left, img_right))
    print(mat)


calculate_similar()