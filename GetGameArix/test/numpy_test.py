"""
numpy test
"""
import numpy as np

# 基本操作
# 初始化 - 创建一个全0的numpy数组
n0 = np.zeros((3, 2))
print(n0)

# 初始化 - 创建一个全1的numpy数组
n1 = np.ones((3, 2))
print(n1)

# 向量化 - 由数组创建numpy数组
na = np.array([1, 2, 3, 4, 5])
print(na)

# 获取数组尺寸 - shape属性
print(n0.shape)

# 递增/递减序列 - arange() numpy.arange(start, stop, step)
nk1 = np.arange(3, 7)
nk2 = np.arange(7, 3, -1)
print(nk1, nk2)

# 区间等距分布序列 - linspace() numpy.linspace(start, stop, num)
nlin = np.linspace(0, 1, 5)
print(nlin)

# 产生随机值为0-1的序列 - nd.random.rand(row, line)
nr = np.random.rand(2, 4)
print(nr)

# 基本运算
# 四则运算 - 相同尺寸的数组可以直接进行四则运算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b, a / b)

# 点乘: 向量进行点乘运算 - np.dot()
print(np.dot(a, b))

# 矩阵乘法运算 - a @ b 等价于 np.matmul()
print(a @ b)

# 求平方根 - np.sqrt()
print(np.sqrt(a))

# 指数运算 - np.power()
print(np.power(a, 2))

# 特殊运算
# 不同大小的矩阵进行运算时会自动扩充为相同大小, 所扩充的元素等同于原始元素
a = np.array([[1], [10], [20]])
b = np.array([0, 1, 2])
print(a + b)

# 获取矩阵中的最大、最小值
print(a.min(), a.max())