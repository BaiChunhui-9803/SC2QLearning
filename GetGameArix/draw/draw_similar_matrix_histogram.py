import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

_MAX_INFLUENCE = 25
_MIN_INFLUENCE = -16 * 4


def make_regular_image(img, size=(256, 256)):
    return img.resize(size).convert('RGB')


def getHash(image):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    for (chans, color) in zip(chans, colors):
        hist = cv2.calcHist([chans], [0], None, [8], [0, 256])
        plt.plot(hist,color = color)
    # plt.savefig("1.png", dpi=500)
    plt.close()

    (b, g, r) = cv2.split(image)
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
        plt.plot(hist, color=color)
        if color == 'r':
            r_list = hist.T[0]
        if color == 'g':
            g_list = hist.T[0]
        if color == 'b':
            b_list = hist.T[0]
    # plt.savefig("2.png", dpi=500)
    plt.close()
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
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def create_img_list(file_list):
    file_img_list = [0 for index in file_list]
    for idx, element in enumerate(file_list):
        file_img_list[idx] = getHash(cv2.imread(file_path + element))
        print(f"完成图像转录{idx}")
    return file_img_list


def calculate_hash_similar(arr_list, img_list):
    f_out = open("../../白春辉/实验平台/QLearningTest/SSIM/similar_hash_pos.txt", "w", newline='')
    for i, element in enumerate(img_list):
        for j in range(i + 1):
            arr_left = img_list[i]
            arr_right = img_list[j]
            mat[i][j] = campHash(arr_left, arr_right)
            if mat[i][j] <= 1:
                f_out.write(f"{i}\t{j}\n")  # 写入数据
    f_out.close()  # 关闭文件
    m = 0.0
    n = 0.0
    for i in range(len(points_2d)):
        for j in range(len(points_2d[i])):
            if points_2d[i][j]:
                m += 1.0
                if mat[i][j] == 0:
                    n += 1.0
    print(n/m)
    img_path = "im_img/"
    img_list = os.listdir(img_path)
    df = pd.DataFrame(mat, index=img_list, columns=img_list)
    plt.close()
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(data=df,
                     square=True,
                     cmap="Blues_r",
                     linewidths=0.03,
                     # annot=True,
                     # vmin=0, center=0.5, vmax=1,
                     xticklabels=5,
                     yticklabels=5)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.xlabel('right_image')  # 设置坐标名称
    plt.ylabel('left_image')
    plt.title('Heatmap of Left-Right Image Hash Matrix')  # 标题
    # plt.show()
    plt.savefig("heatmap_hash.png", dpi=500)


file_path = "im_img/"
files_list = os.listdir(file_path)
files_img_list = create_img_list(files_list)
mat = [[15] * len(files_list) for _ in range(len(files_list))]
with open('../../白春辉/实验平台/QLearningTest/SSIM/similar_pos.txt') as f:
    lines = f.readlines()
points = []
points_2d = [[0] * len(files_list) for _ in range(len(files_list))]
for line in lines:
    x, y, *_ = line.split()
    points.append((int(x), int(y)))
    points_2d[int(x)][int(y)] = 1
print(len(points_2d))

calculate_hash_similar(files_list, files_img_list)