import networkx as nx
import matplotlib.pyplot as plt
import random

# 创建有向图
G = nx.DiGraph()

def find_Pos(node):
    # print(nodeMap.items())
    for key, value in nodeMap.items():
        # print(key, value)
        if node in value:
            # print(f"节点{node}的位置是: {key}")
            return key
    else:
        # print(f"节点{node}在nodeMap中未找到。")
        return -1

nodeMap = {
    0: [0],
    1: [1, 2],
    2: [3, 4],
    3: [5, 6, 7],
    4: [8, 9, 10, 11, 12],
    5: [13, 14, 15, 16, 17],
    # 6: [18, 19, 20, 21, 22, 23],
    # 7: [24, 25, 26, 27, 29, 30, 31, 32],
    # 8: [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    # 9: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
}

for i in range(10):
    nodes = []
    for level, node in nodeMap.items():
        nodeX = random.choice(node)
        nodes.append(nodeX)
        G.add_node(nodeX, level=level)

    for j in range(len(nodes) - 1):
        G.add_edge(nodes[j], nodes[j + 1])

# 计算每个节点的度数，并将其作为节点属性
degree_values = dict(G.degree())

# 为每个节点分配一个颜色，颜色的深浅与节点度数成正比
node_colors = [find_Pos(node) for node in G.nodes]

# 创建一个虚拟的图像对象以便创建颜色条
img = plt.imshow([[0, 1]], cmap=plt.cm.Reds)
img.set_visible(False)  # 隐藏虚拟图像

# 定义节点位置
# pos = nx.spring_layout(G)
# print(pos)

# 绘制图
pos = nx.spring_layout(G, pos=None, k=0.3, fixed=None, iterations=50, threshold=1e-4,
                       weight='weight', scale=0.5, center=None, dim=2)

# 计算每个level的最大值
max_level = max([node[1]['level'] for node in G.nodes(data=True)])

# 设置节点的x坐标
for node, attr in G.nodes(data=True):
    level = attr['level']
    pos[node][0] = level / (max_level + 1)

nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.Reds, with_labels=True, node_size=250)
plt.draw()

# 添加颜色条以显示节点属性值与颜色之间的映射关系
plt.colorbar(img, ax=plt.gca(), orientation='vertical', label='Node Degree')

plt.show()