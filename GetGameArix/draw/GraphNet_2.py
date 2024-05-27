from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.charts import Scatter
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig, SymbolType, ThemeType
import random
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities


path1 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_1/"
# path1 = "./../datas/data_for_render/experiments_datas/tests/MM8far_1/"

# path2 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_2/"
path2 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_1/"
# path2 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_test_1/"
# path2 = "./../datas/data_for_render/experiments_datas/offline_data_to_model/8faraction7_to_8faraction7_1"
# path2 = "./../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_test_2/"

_GAME_STATE_NODE_PATH_1 = path1 + "graph/state_node.txt"
_GAME_NODE_LOG_PATH_1 = path1 + "graph/node_log.txt"
_GAME_STATE_NODE_PATH_2 = path2 + "graph/state_node.txt"
_GAME_NODE_LOG_PATH_2 = path2 + "graph/node_log.txt"


def add_nodes(node_path, log_path, param1=0, param2=1500):
    nodes = []
    links = []
    link_set = set()
    nodes_size = []
    links_size = []
    with open(node_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # 去除行尾的换行符并分割行内容
            parts = line.strip().split()
            # print(parts[1])
            node_data = {
                'name': parts[1],
                'id': parts[0],
                'value': (float(parts[2]) + 180.0) / 360,
                'level': 0
            }
            # 将字典添加到nodes列表中
            nodes.append(node_data)
    print(f'图共有{len(nodes)}条个结点')
    # print(nodes)

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
            print(f'正在处理log第{line_number}行, links集合大小{len(links)}')
            links_size.append(len(links))
    print(f'图共有{len(links)}条边')

    position_dict = {}
    with open(log_path, 'r') as file:
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
    return nodes



def distance(sol1, sol2):
    print(sol1)


nodes_1 = add_nodes(_GAME_STATE_NODE_PATH_1, _GAME_NODE_LOG_PATH_1, 0, 1500)
nodes_2 = add_nodes(_GAME_STATE_NODE_PATH_2, _GAME_NODE_LOG_PATH_2, 0, 1500)
print(nodes_1)

node_x_1 = [value['level'] for value in nodes_1]
node_y_1 = [value['value'] for value in nodes_1]
node_id_1 = [value['id'] for value in nodes_1]
node_x_2 = [value['level'] for value in nodes_2]
node_y_2 = [value['value'] for value in nodes_2]
node_id_2 = [value['id'] for value in nodes_2]
node_x_3 = [value['level'] for value in nodes_2 if value['id'] not in node_id_1]
node_y_3 = [value['value'] for value in nodes_2 if value['id'] not in node_id_1]
node_id_3 = [value['id'] for value in nodes_2 if value['id'] not in node_id_1]


scatter = (
    Scatter()
    .add_xaxis(xaxis_data=node_x_1)
    .add_yaxis(
        series_name="original",
        y_axis=node_y_1,
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_xaxis(xaxis_data=node_x_2)
    .add_yaxis(
        series_name="model_transfer",
        y_axis=node_y_2,
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .add_xaxis(xaxis_data=node_x_3)
    .add_yaxis(
        series_name="model_transfer_new_state",
        y_axis=node_y_3,
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts(
        # label_opts=opts.LabelOpts(is_show=False)
    )
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name="Rank", type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            name="Mean of Score",
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False)
        # tooltip_opts=opts.TooltipOpts(
        #     formatter=JsCode(  # formatter为标签内容格式器{a}：系列名;{b}：数据名;{c}：数值数组也可以是回调函数
        #         """function(params) {
        #             var nodes = %s;
        #             var value = nodes[params.dataIndex];
        #             return 'Node ID: ' + value;
        #     }""" % node_id
        #     ),
        # ),
    )
)

scatter.render('状态散点图.html')