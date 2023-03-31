import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import seaborn as sns
import similar_histogram as ssim

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
ndArray1 = np.array(data1, dtype="float")
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
ndArray2 = np.array(data2, dtype="float")
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
ndArray3 = np.array(data3, dtype="float")
dfData3 = np.matrix(data3, dtype="float")

data4 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, -1, -1, 0, 0, 0, 0, 0],
         [0, 0, -1, -5, -1, 0, 0, 0, 0, 0],
         [0, 0, -1, -1, -1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 5, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
ndArray4 = np.array(data4, dtype="float")
dfData4 = np.matrix(data4, dtype="float")


def array_to_pil_img(arr: np.ndarray):
    plt.figure()
    p1 = sns.heatmap(arr, cmap="coolwarm", vmin=-25, vmax=25, annot=False, cbar=False, square=True,
                     xticklabels=False, yticklabels=False, linewidth=.5)
    s1 = p1.get_figure()
    canvas = FigureCanvasAgg(plt.gcf())
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    img = Image.frombytes("RGB", (w, h), buf.tobytes())
    return ssim.make_regalur_image(img)


def mtx_similar0(arr1: np.ndarray, arr2: np.ndarray) ->float:
    '''
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:SSIM算法计算两个图像的相似度
    '''
    # array_to_pil_img(arr1)
    # print('ssim: HeatMap1 & HeatMap2',
    # ssim.make_regalur_image(array_to_pil_img(dfData1), dfData2))
    return ssim.calc_similar(array_to_pil_img(arr1), array_to_pil_img(arr2))


def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) ->float:
    '''
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    '''
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom
    return  (similar+1) / 2     # 姑且把余弦函数当线性


def mtx_similar2(arr1: np.ndarray, arr2: np.ndarray) ->float:
    '''
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ**2)
    denom = np.sum(arr1**2)
    similar = 1 - (numera / denom)
    return similar


def mtx_similar3(arr1: np.ndarray, arr2: np.ndarray) ->float:
    '''
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)     # 普通模长
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    return similar


print(mtx_similar0(dfData1, dfData2))
print(mtx_similar0(dfData1.T, dfData4))
print(mtx_similar0(dfData1, dfData3))
print(mtx_similar0(dfData2, dfData3))

for i in range(3):
    func = locals().get("mtx_similar{}".format(i+1))
    func(ndArray1, ndArray2)
    print("\033[0;33m比较法{}：\033[0m".format(i+1))
    print("ndArray1 && ndArray2", func(ndArray1, ndArray2))
    print("ndArray1旋转 && ndArray2", func(np.rot90(ndArray1, 1), ndArray2))
    print("ndArray1翻转 && ndArray2", func(ndArray1.T, ndArray2))
    print("ndArray1 && ndArray3", func(ndArray1, ndArray3))
    print("ndArray2 && ndArray3", func(ndArray2, ndArray3))