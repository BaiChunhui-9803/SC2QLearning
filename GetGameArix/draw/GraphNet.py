from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.charts import Scatter
from pyecharts.globals import CurrentConfig, SymbolType, ThemeType
import random
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities

path1 = "./../datas/data_for_render/experiments_datas/tests/MM8far_1/"
# path1 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_1/"
# path1 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_test_1/"

_GAME_STATE_NODE_PATH = path1 + "graph/state_node.txt"
_GAME_NODE_LOG_PATH = path1 + "graph/node_log.txt"

nodes = []
links = []
link_set = set()
nodes_size = []
links_size = []

category_list = ['good_1', 'good_2', 'good_3', 'neutral', 'bad_3', 'bad_2', 'bad_1']

def get_category(param):
    result = 'neutral'
    if param >= 120:
        result = 'good_1'
    elif 60 <= param < 120:
        result = 'good_2'
    elif 15 <= param < 60:
        result = 'good_3'
    elif -15 <= param < 15:
        result = 'neutral'
    elif -60 <= param < -15:
        result = 'bad_3'
    elif -120 <= param < -60:
        result = 'bad_2'
    elif param < -120:
        result = 'bad_1'
    return result

def draw_node_graph(node_path, log_path, param1=0, param2=1500):
    with open(node_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 去除行尾的换行符并分割行内容
            parts = line.strip().split()
            # print(parts[1])
            node_data = {
                'name': parts[1],
                'value': (float(parts[2]) + 180.0) / 360,
                'symbolSize': (float(parts[2]) + 180.0) / 360 * 30,
                # 'category': 'good' if float(parts[2]) > 0 else 'bad'
                'category': get_category(float(parts[2])),
                'level': 0
            }
            # 将字典添加到nodes列表中
            nodes.append(node_data)
    print(f'图共有{len(nodes)}条个结点')
    # print(nodes)

    global links
    with open(log_path, 'r') as file:
        lines = file.readlines()[param1:param2]
        for line_number, line in enumerate(lines):
            # 去除行尾的换行符并分割行内容
            numbers = [int(num) for num in line.strip().split()]
            for index in range(len(numbers) - 1):
                source = numbers[index]
                target = numbers[index + 1]
                link = {"source": source, "target": target, "is_ignore_force_layout": True}

                if (source, target) in link_set:
                    for edge in links:
                        if edge['source'] == source and edge['target'] == target:
                            edge['value'] += 1
                            break
                else:
                    link['value'] = 1
                    links.append(link)
                    link_set.add((source, target))
            # for index in range(len(numbers) - 1):
            #     source = numbers[index]
            #     target = numbers[index + 1]
            #     links.append({"source": source, "target": target, "is_ignore_force_layout": True})
            #     links = [{'source': edge['source'], 'target': edge['target'], 'value': edge.get('value', 0) + 1}
            #              if edge['source'] == source and edge['target'] == target
            #              else edge
            #              for edge in links]
            print(f'正在处理log第{line_number}行, links集合大小{len(links)}')
            links_size.append(len(links))
    print(f'图共有{len(links)}条边')

    position_dict = {}
    with open(_GAME_NODE_LOG_PATH, 'r') as file:
        lines = file.readlines()[param1:param2]

    # 处理每一行
    for line_number, line in enumerate(lines):
        # 去除每行末尾的换行符，并分割成列表
        numbers = line.strip().split()
        # 初始化一个字典来存储当前行的数字位置信息
        current_line_positions = {}

        # 遍历每个数字
        for number_index, number in enumerate(numbers):
            # 将数字转换为整数
            number = int(number)
            # 如果该数字还没有在当前行出现过，则添加其位置信息
            if number not in current_line_positions:
                current_line_positions[number] = [number_index]
            # 如果已经出现过，则添加其当前位置
            else:
                current_line_positions[number].append(number_index)

        # 将当前行的数字位置信息添加到全局字典中
        position_dict[line_number] = current_line_positions

    # 计算每个数字在每一行出现的位置的均值
    mean_positions = {}

    # 遍历每一行和数字
    for line_number, line_positions in position_dict.items():
        for number, positions in line_positions.items():
            # 计算均值
            mean_position = sum(positions) / len(positions)
            # 将结果添加到均值字典中
            if number not in mean_positions:
                mean_positions[number] = []
            mean_positions[number].append(mean_position)
    node_level_list = []
    for number, mean_positions_list in mean_positions.items():
        node_level_list.append((number, sum(mean_positions_list) / len(mean_positions_list) + 1))
    # print(node_level_list)
    for node_with_level in node_level_list:
        found_nodes = [item for item in nodes if item.get('name') == str(node_with_level[0])]
        for node in found_nodes:
            if node['name'] == '0':
                node['level'] = 0
            else:
                node['level'] = node_with_level[1]
                # node['level'] = 'level' + str(int(node_with_level[1]))
                # node['category'] = 'level' + str(int(node_with_level[1]))

    # print()

def distance(sol1, sol2):
    print(sol1)

loop_categories = [
    {"name": "start"},
]
for i in range(49):
    loop_categories.append({"name": "level" + str(i + 1)})
loop_categories.append({"name": "end"})

win_categories = [
    {"name": "good_1"},
    {"name": "good_2"},
    {"name": "good_3"},
    {"name": "neutral"},
    {"name": "bad_3"},
    {"name": "bad_2"},
    {"name": "bad_1"},
]

draw_node_graph(_GAME_STATE_NODE_PATH, _GAME_NODE_LOG_PATH, 0, 1500)

node_degree = {str(i): {"in_degree": 0, "out_degree": 0} for i in range(len(nodes))}
for edge in links:
    # print(edge)
    src, dst = edge['source'], edge['target']
    node_degree[str(src)]["out_degree"] += 1
    node_degree[str(dst)]["in_degree"] += 1

for node, degree in node_degree.items():
    [elem for elem in nodes if elem['name'] == node][0]['symbolSize'] = (float(degree['in_degree'] + degree['out_degree']-1)/36+1)*10

# CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/npm/echarts/dist/"

colors = ['#4B1A18', '#AB352B', '#D6904E', '#EAE4E0', '#B2DFE3', '#459CD7', '#315CB5']

# graph = Graph(init_opts=opts.InitOpts(width='1600px', height='800px', theme=ThemeType.WHITE)))
graph = Graph(init_opts=opts.InitOpts(bg_color='black', width='1600px', height='800px'))
graph.add(
    "状态转移图",
    nodes=nodes,
    links=links,
    categories=win_categories,
    layout="force",
    is_rotate_label=True,
    is_roam=True,
    is_draggable=True,
    is_focusnode=True,
    gravity=0,
    # symbol_size=50,
    edge_length=[10, 50],
    edge_symbol=['', 'arrow'],
    edge_symbol_size=4,
    linestyle_opts=opts.LineStyleOpts(width=0.5, opacity=0.8),
    # edge_label=opts.LabelOpts(
    #     is_show=True, position="middle", formatter="{c}", color='#0000FF',  # 使用函数来设置边的颜色
    # ),
    label_opts=opts.LabelOpts(is_show=False),
)

graph.set_colors(colors)

graph.set_global_opts(
    # width="800px",  # 设置宽度
    # height="600px",  # 设置高度
    title_opts=opts.TitleOpts(title="解状态转移图"),
)
graph.render("解状态转移图.html")

# print(nodes)
node_x = [value['level'] for value in nodes]
node_y = [value['value'] for value in nodes]

scatter = (
    Scatter()
    .add_xaxis(xaxis_data=node_x)
    .add_yaxis(
        series_name="",
        y_axis=node_y,
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
    )
)

scatter.render('状态散点图.html')