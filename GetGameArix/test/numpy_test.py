import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('1.png')
image2 = cv2.imread('2.png')

# 将图像转换为RGB格式
color = ('blue', 'green', 'red')
colors = ("b", "g", "r")
# for i, Color in enumerate(color):
    # hist = cv2.calcHist([image], [i], None, [8], [0.0, 256.0])
    # plt.plot(hist, color=Color)
    # plt.xlim([0, 7])
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
    # 256 灰度级的个数 bins / histSize
    plt.plot(hist, color=color)
    plt.fill_between(np.arange(len(hist)), hist.flatten(), color=color, alpha=0.2)
    # ax.xlim([0, 256])

# plt.show()
plt.savefig('hashimg.png', dpi=500, bbox_inches='tight')
#
# for i, Color in enumerate(color):
#     hist = cv2.calcHist([image2], [i], None, [8], [0.0, 256.0])
#     plt.plot(hist, color=Color)
#     plt.xlim([0, 7])
#
# plt.show()