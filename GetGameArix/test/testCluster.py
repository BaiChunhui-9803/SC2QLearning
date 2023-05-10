import random
import math
import matplotlib.pyplot as plt

# 生成随机坐标
def generate_points(n, x_range, y_range):
    points = []
    for i in range(n):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        points.append((i, x, y))
    return points

# 计算两点之间的距离
def distance(point1, point2):
    return math.sqrt((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

# 计算每个点到所有聚类中心的距离
def calc_distances(points, centers):
    distances = []
    for point in points:
        row = []
        for center in centers:
            # print(center)
            row.append(distance(point, center))
        distances.append(row)
    return distances

# 将每个点分配到最近的聚类中心
def assign_clusters(points, centers):
    distances = calc_distances(points, centers)
    clusters = [[] for i in range(len(centers))]
    for i in range(len(points)):
        min_dist = float('inf')
        min_idx = -1
        for j in range(len(centers)):
            if distances[i][j] < min_dist:
                min_dist = distances[i][j]
                min_idx = j
        clusters[min_idx].append(points[i])
    return clusters

# 计算聚类中心
def calc_centers(clusters):
    centers = []
    for cluster in clusters:
        id_sum = 0
        x_sum = 0
        y_sum = 0
        for point in cluster:
            id_sum += point[0]
            x_sum += point[1]
            y_sum += point[2]
        center_id = id_sum
        center_x = x_sum / len(cluster)
        center_y = y_sum / len(cluster)
        centers.append((center_id, center_x, center_y))
    return centers

# 计算所有点的平均距离
def calc_avg_distance(points, centers):
    distances = calc_distances(points, centers)
    sum_dist = 0
    for i in range(len(points)):
        min_dist = float('inf')
        for j in range(len(centers)):
            if distances[i][j] < min_dist:
                min_dist = distances[i][j]
        sum_dist += min_dist
    return sum_dist / len(points)

# 将A类和B类坐标点分别聚类
def kmeans(points_a, points_b, k):
    # 初始化聚类中心
    centers_a = random.sample(points_a, k)
    centers_b = random.sample(points_b, k)
    # 迭代聚类过程
    for i in range(10):
        # 分配聚类
        clusters_a = assign_clusters(points_a, centers_a)
        clusters_b = assign_clusters(points_b, centers_b)
        # 计算聚类中心
        centers_a = calc_centers(clusters_a)
        centers_b = calc_centers(clusters_b)
    # 对A类坐标点进行最近距离的匹配
    matches = []
    for cluster_a in clusters_a:
        for point_a in cluster_a:
            min_dist = float('inf')
            min_point_b = None
            for point_b in points_b:
                if distance(point_a, point_b) < min_dist:
                    min_dist = distance(point_a, point_b)
                    min_point_b = point_b
            matches.append((point_a, min_point_b))
    return clusters_a, clusters_b, matches

# 测试
points_a = generate_points(8, (0, 10), (0, 10))
points_b = generate_points(8, (0, 10), (0, 10))
clusters_a, clusters_b, matches = kmeans(points_a, points_b, 3)
print('A类坐标点聚类结果：')
for i in range(len(clusters_a)):
    print(f'聚类{i}: {len(clusters_a[i])}个点')
    for j in range(len(clusters_a[i])):
        print(clusters_a[i][j][0], end=' ')
    print("")
print('B类坐标点聚类结果：')
for i in range(len(clusters_b)):
    print(f'聚类{i}: {len(clusters_b[i])}个点')
    for j in range(len(clusters_b[i])):
        print(clusters_b[i][j][0], end=' ')
    print("")
print('A类坐标点与B类坐标点的匹配结果：')
for match in matches:
    print(f'{match[0]} 匹配 {match[1]}')


# 绘制A的点
x_A = [point[1] for point in points_a]
y_A = [point[2] for point in points_a]
id_A = [point[0] for point in points_a]
plt.scatter(x_A, y_A, c='red')
for i in range(len(id_A)):
    plt.annotate(id_A[i], xy=(x_A[i], y_A[i]), xytext=(x_A[i]+0.1, y_A[i]+0.1))

# 绘制B的点
x_B = [point[1] for point in points_b]
y_B = [point[2] for point in points_b]
id_B = [point[0] for point in points_b]
plt.scatter(x_B, y_B, c='blue')
for i in range(len(id_B)):
    plt.annotate(id_B[i], xy=(x_B[i], y_B[i]), xytext=(x_B[i]+0.1, y_B[i]+0.1))

# 添加图例
plt.legend(['A', 'B'])

plt.show()