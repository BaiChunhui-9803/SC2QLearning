# 矩阵生成

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import similar_histogram as ssim
from PIL import Image

data1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
         [0, -1, -5, -1, 0, 0, 0, 0, 0, 0],
         [0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 5, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
dfData1 = np.matrix(data1, dtype="float")
data2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 5, 1, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
         [0, -1, -5, -1, 0, 0, 0, 0, 0, 0],
         [0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
dfData2 = np.matrix(data2, dtype="float")

data3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 1, 1, 2, 5, 1, 0],
         [0, 0, 0, 0, 1, 5, 2, 1, 1, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -5, -5, -5, 0, 0, 0, 0, 0, 0],
         [0, -5, -25, -5, 0, 0, 0, 0, 0, 0],
         [0, -5, -5, -5, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
dfData3 = np.matrix(data3, dtype="float")

# cmap: https://blog.csdn.net/weixin_39580795/article/details/102622004
fig = plt.figure()
p1 = sns.heatmap(dfData1, cmap="coolwarm", vmin=-25, vmax=25, annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False, linewidth=.5)
s1 = p1.get_figure()
s1.savefig('HeatMap1.jpg',bbox_inches='tight')

fig = plt.figure()
p2 = sns.heatmap(dfData2, cmap="coolwarm", vmin=-25, vmax=25, annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False, linewidth=.5)
s2 = p2.get_figure()
s2.savefig('HeatMap2.jpg',bbox_inches='tight')

fig = plt.figure()
p3 = sns.heatmap(dfData3, cmap="coolwarm", vmin=-25, vmax=25, annot=False, cbar=False, square=True, xticklabels=False, yticklabels=False, linewidth=.5)
s3 = p3.get_figure()
s3.savefig('HeatMap3.jpg',bbox_inches='tight')

img1 = Image.open('HeatMap1.jpg')
img1 = ssim.make_regalur_image(img1)
img2 = Image.open('HeatMap2.jpg')
img2 = ssim.make_regalur_image(img2)
img3 = Image.open('HeatMap3.jpg')
img3 = ssim.make_regalur_image(img3)
print('ssim: HeatMap1 & HeatMap2', ssim.calc_similar(img1, img2))
print('ssim: HeatMap1 & HeatMap3', ssim.calc_similar(img1, img3))
print('ssim: HeatMap2 & HeatMap3', ssim.calc_similar(img2, img3))
# ssim.calc_similar(Image.open(s1), Image.open(s2))