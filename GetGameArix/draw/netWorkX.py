import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities

n = 16
m = 40

# 创建一个图
G = nx.gnm_random_graph(n, m)

# 绘制聚类前的图
plt.figure(figsize=(10, 5))
plt.subplot(121)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.title('Before Clustering')

# 使用Girvan-Newman算法进行聚类
comp = louvain_communities(G, 123)
print(comp)
clusters = comp
clustered_G = nx.Graph()
clustered_G2 = nx.Graph()
color_map = []
for i, cluster in enumerate(clusters):
    # print(cluster)
    for node in cluster:
        # print(node)
        clustered_G.add_node(node)
        color_map.append(i) # 为每个节点分配一个颜色
        for neighbor in G.neighbors(node):
            if neighbor in cluster:
                clustered_G.add_edge(node, neighbor)
            else:
                clustered_G2.add_edge(node, neighbor)
        print(color_map)
        # print(G.edges)
        # for edge in G.edges:
        #     # print(edge(0))
        #     clustered_G.add_edge(edge[0], edge[1])


# 绘制聚类后的图
plt.subplot(122)
nx.draw(clustered_G2, pos, with_labels=True)
nx.draw(clustered_G, pos, with_labels=True, node_color=color_map, cmap=plt.cm.tab20)
plt.title('After Clustering')

plt.show()