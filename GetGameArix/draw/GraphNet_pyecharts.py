from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import SymbolType, ThemeType
import random


nodeMap1 = {
    0: [0],
    1: [1, 2],
    2: [3, 4],
    3: [5, 6, 7],
    4: [8, 9, 10, 11, 12],
    5: [13, 14, 15, 16, 17],
    6: [18, 19, 20, 21, 22, 23],
    7: [24, 25, 26, 27, 29, 30, 31, 32],
    8: [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    9: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
}

nodeMap2 = {
    0: [0],
    1: [56, 1],
    2: [57, 4],
    3: [58],
    4: [59, 65],
    5: [60, 66],
    6: [61, 67, 68],
    7: [62, 69, 70, 71, 72],
    8: [63, 73, 74, 75, 64],
    9: [64]
}

nodes = []
links = []
for i in range(15):
    nodesX = []
    for level, node in nodeMap1.items():
        nodeX = random.choice(node)
        nodesX.append({"name": str(nodeX)})
        if str(nodeX) not in [item['name'] for item in nodes]:
            valueX = random.randint(5, 12)
            nodes.append({"name": str(nodeX), "symbolSize": valueX, "category": level, "value": valueX})

    for index in range(len(nodesX) - 1):
        source = nodesX[index]['name']
        target = nodesX[index + 1]['name']
        links.append({"source": source, "target": target, "is_ignore_force_layout": True})
        links = [{'source': edge['source'], 'target': edge['target'], 'value': edge.get('value', 0) + 1}
                 if edge['source'] == source and edge['target'] == target
                 else edge
                 for edge in links]

for i in range(5):
    nodesX = []
    for level, node in nodeMap2.items():
        nodeX = random.choice(node)
        nodesX.append({"name": str(nodeX)})
        if str(nodeX) not in [item['name'] for item in nodes]:
            valueX = random.randint(5, 12)
            nodes.append({"name": str(nodeX), "symbolSize": valueX, "category": level, "value": valueX})

    for index in range(len(nodesX) - 1):
        source = nodesX[index]['name']
        target = nodesX[index + 1]['name']
        links.append({"source": source, "target": target, "is_ignore_force_layout": True})
        links = [{'source': edge['source'], 'target': edge['target'], 'value': edge.get('value', 0) + 1}
                 if edge['source'] == source and edge['target'] == target
                 else edge
                 for edge in links]

categories = [
    {"name": "level0"},
    {"name": "level1"},
    {"name": "level2"},
    {"name": "level3"},
    {"name": "level4"},
    {"name": "level5"},
    {"name": "level6"},
    {"name": "level7"},
    {"name": "level8"},
    {"name": "level9"},
]

print(nodes)
print(links)


# graph = Graph(init_opts=opts.InitOpts(width='1550px', height='650px', theme=ThemeType.LIGHT))
graph = Graph(init_opts=opts.InitOpts(width='1600px', height='800px'))
graph.add(
    "有向图示例",
    nodes=nodes,
    links=links,
    categories=categories,
    layout="force",
    is_rotate_label=True,
    is_roam=True,
    is_draggable=True,
    is_focusnode=False,
    gravity=0,
    # symbol_size=50,
    edge_length=[10, 50],
    edge_symbol=['', 'arrow'],
    edge_symbol_size=6,
    linestyle_opts=opts.LineStyleOpts(width=2, opacity=0.8),
    # edge_label=opts.LabelOpts(
    #     is_show=True, position="middle", formatter="{c}", color='#0000FF',  # 使用函数来设置边的颜色
    # ),
    label_opts=opts.LabelOpts(is_show=True),
)

graph.set_global_opts(
    # width="800px",  # 设置宽度
    # height="600px",  # 设置高度
    title_opts=opts.TitleOpts(title="有向图"))
graph.render("有向图示例.html")
