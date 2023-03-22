from PIL import Image
from numpy import average, dot, linalg
import cv2 as cv
import numpy as np


def get_thum(image, size=(64, 64), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res


def random_noise(image, noise_num):
    img = cv.imread(image)
    img_noise = img
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, :] = 255
    return img_noise


origin_image = Image.open('HeatMap1.jpg')
test_image = Image.open('HeatMap3.jpg')
print("_____________________________原始图片相似度_____________________________________")
cosin = image_similarity_vectors_via_numpy(origin_image, test_image)
print('图片余弦相似度', cosin)

print("_____________________________图片旋转90度相似度_____________________________________")
rotate90_img = origin_image.rotate(90)
# rotate90_img.show()
cosin2 = image_similarity_vectors_via_numpy(rotate90_img, test_image)
print('图片旋转90度余弦相似度', cosin2)
print("_____________________________图片旋转180度相似度_____________________________________")
rotate180_img = origin_image.rotate(180)
# rotate180_img.show()
cosin3 = image_similarity_vectors_via_numpy(rotate180_img, test_image)
print('图片旋转90度余弦相似度', cosin3)

print("_____________________________图片旋转270度相似度_____________________________________")
rotate270_img = origin_image.rotate(270)
# rotate270_img.show()
cosin4 = image_similarity_vectors_via_numpy(rotate270_img, test_image)
print('图片旋转120度余弦相似度', cosin4)

cv.waitKey(0)