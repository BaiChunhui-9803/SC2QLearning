import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
import seaborn as sns
import csv
import random
from pyecharts import options as opts
from pyecharts.charts import Scatter, Timeline, Tab, Grid, Line, Page
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode
import json


def get_point_boundry(point_list):
    x_min = point_list[0][0]
    x_max = point_list[0][0]
    y_min = point_list[0][1]
    y_max = point_list[0][1]

    # 遍历列表，更新最大最小值
    for point in point_list:
        x = point[0]
        y = point[1]
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    return x_min, x_max, y_min, y_max


def drawClustersResult_unit4(path):
    with open(path + 'clusters.csv', 'r') as f:
        lines = f.readlines()
        print(lines)
        lines = [line.strip() for line in lines if line.strip()]
        # 提取list3和list4
        list2 = []
        list3 = []
        list4 = []

        for line in lines:
            index, data, val1, val2 = eval(line)
            if index == 2:
                list2.append((data, val1, val2))
            elif index == 3:
                list3.append((data, val1, val2))
            elif index == 4:
                list4.append((data, val1, val2))

        figsise = opts.InitOpts(width='600px', height='600px')
        tl = Timeline(init_opts=figsise)
        for i in range(len(list4)):
            x_data = [d[0] for d in list4[i][0]]
            y_data = [d[1] for d in list4[i][0]]
            x_min, x_max, y_min, y_max = get_point_boundry(list4[i][0])
            scatter = (
                Scatter()
                    .add_xaxis(xaxis_data=x_data)
                    .add_yaxis(
                    series_name="",
                    y_axis=y_data,
                    symbol_size=20,
                    label_opts=opts.LabelOpts(is_show=False),
                )
                    .set_series_opts()
                    .set_global_opts(
                    xaxis_opts=opts.AxisOpts(
                        type_="value",
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        axislabel_opts=opts.LabelOpts(interval=0),
                        min_=x_min - 1,
                        max_=x_max + 1
                    ),
                    yaxis_opts=opts.AxisOpts(
                        type_="value",
                        axislabel_opts=opts.LabelOpts(interval=0),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                        min_=y_min - 1,
                        max_=y_max + 1
                    ),
                    tooltip_opts=opts.TooltipOpts(is_show=False),
                    title_opts=opts.TitleOpts(
                        title="Cluster Result for 4 Units: Sample{}".format(i),
                        subtitle="clu_uniformity: {}, clu_crowding: {}".format(list4[i][1], list4[i][2]),
                        pos_top='2%',  # 标题的垂直位置
                        pos_left='center',  # 标题的水平位置
                        title_textstyle_opts=opts.TextStyleOpts(
                            font_size=16,  # 标题的字体大小
                            font_weight='bold'  # 标题的字体粗细
                        ),
                        subtitle_textstyle_opts=opts.TextStyleOpts(
                            font_size=12,  # 副标题的字体大小
                            color='red'  # 副标题的颜色
                        )
                    )
                )
            )
            # .set_global_opts(title_opts=opts.TitleOpts("某商店{}年营业额".format(i)))
            # )
            tl.add(scatter, "sample{}".format(i))
        tl.add_schema(
            is_auto_play=True,  # 自动播放
            play_interval=100,  # 播放间隔，单位为毫秒
            # pos_left='5%',  # Timeline的水平位置
            # pos_bottom='5%',  # Timeline的垂直位置
            # width='90%',  # Timeline的宽度
            # label_opts=opts.LabelOpts(
            #     is_show=True,  # 是否显示标签
            #     color='black',  # 标签的颜色
            #     font_size=12  # 标签的字体大小
            # )
        )
        tl.render("drawClustersResult.html")


def drawClustersResult_unit8(path):
    with open(path + 'clusters.csv', 'r') as f:
        lines = f.readlines()
        # print(lines)
        lines = [line.strip() for line in lines if line.strip()]
        list_all = [[], [], [], [], [], [], [], []]

        for line in lines:
            index, data, val1, val2 = eval(line)
            list_all[index - 1].append((data, val1, val2))
            # if index == 8:
            #     list8.append((data, val1, val2))
            # elif index == 6:
            #     list6.append((data, val1, val2))
            # elif index == 4:
            #     list4.append((data, val1, val2))

        figsise = opts.InitOpts(width='600px', height='600px')
        tab = Tab()
        for list_idx in range(len(list_all)):
            tl = Timeline(init_opts=figsise)
            for i in range(len(list_all[list_idx])):
                x_data = [d[0] for d in list_all[list_idx][i][0]]
                y_data = [d[1] for d in list_all[list_idx][i][0]]
                x_min, x_max, y_min, y_max = get_point_boundry(list_all[list_idx][i][0])
                scatter = (
                    Scatter()
                        .add_xaxis(xaxis_data=x_data)
                        .add_yaxis(
                        series_name="",
                        y_axis=y_data,
                        symbol_size=20,
                        label_opts=opts.LabelOpts(is_show=False),
                    )
                        .set_series_opts()
                        .set_global_opts(
                        xaxis_opts=opts.AxisOpts(
                            type_="value",
                            splitline_opts=opts.SplitLineOpts(is_show=True),
                            axislabel_opts=opts.LabelOpts(interval=0),
                            min_=x_min - 1,
                            max_=x_max + 1
                        ),
                        yaxis_opts=opts.AxisOpts(
                            type_="value",
                            axislabel_opts=opts.LabelOpts(interval=0),
                            splitline_opts=opts.SplitLineOpts(is_show=True),
                            min_=y_min - 1,
                            max_=y_max + 1
                        ),
                        tooltip_opts=opts.TooltipOpts(is_show=False),
                        title_opts=opts.TitleOpts(
                            title="Cluster Result for {} Units: Sample{}".format(list_idx + 1, i),
                            subtitle="clu_uniformity: {}, clu_crowding: {}".format(list_all[list_idx][i][1],
                                                                                   list_all[list_idx][i][2]),
                            pos_top='0%',  # 标题的垂直位置
                            pos_left='center',  # 标题的水平位置
                            title_textstyle_opts=opts.TextStyleOpts(
                                font_size=16,  # 标题的字体大小
                                font_weight='bold'  # 标题的字体粗细
                            ),
                            subtitle_textstyle_opts=opts.TextStyleOpts(
                                font_size=24,  # 副标题的字体大小
                                color='red'  # 副标题的颜色
                            )
                        )
                    )
                )
                # .set_global_opts(title_opts=opts.TitleOpts("某商店{}年营业额".format(i)))
                # )
                tl.add(scatter, "sample{}".format(i))
            tl.add_schema(
                is_auto_play=True,  # 自动播放
                play_interval=1000,  # 播放间隔，单位为毫秒
                pos_left='0%',  # Timeline的水平位置
                # pos_bottom='5%',  # Timeline的垂直位置
                width='90%',  # Timeline的宽度
                # label_opts=opts.LabelOpts(
                #     is_show=True,  # 是否显示标签
                #     color='black',  # 标签的颜色
                #     font_size=12  # 标签的字体大小
                # )
            )
            tab.add(tl, "units_size_{}".format(list_idx + 1))
        tab.render("drawClustersResult.html")


tag_list = []
my_tag_list = []
enemy_tag_list = []


def update_tag_list(unit_list):
    for unit in unit_list:
        if unit[0] in tag_list:
            pass
        else:
            tag_list.append(unit[0])


def update_my_tag_list(unit_list):
    for unit in unit_list:
        if unit[0] in my_tag_list:
            pass
        else:
            my_tag_list.append(unit[0])


def update_enemy_tag_list(unit_list):
    for unit in unit_list:
        if unit[0] in enemy_tag_list:
            pass
        else:
            enemy_tag_list.append(unit[0])


def drawClustersHealthResult(path):
    tab = Tab()
    for file_id in range(1, 500):
        if file_id == 1 or file_id % 10 == 0:
            file_name = str(file_id) + '.csv'
            file_path = path + 'sub_episode/'
            file_path_result = path + 'short_term_result/'

            with open(file_path + file_name, 'r') as f:
                # todo
                result_step_list = []
                result_dict = {}
                with open(file_path_result + file_name, 'r') as f1:
                    result_lines = f1.readlines()
                    for line_index, line in enumerate(result_lines):
                        if line.startswith("step"):
                            result_step_list.append(line_index)
                    #     for step_index, item in enumerate(step_list):
                    #         print(result_lines[item])
                    for result_step_index, result_item in enumerate(result_step_list):
                        result_dict[result_lines[result_item].strip("step[]\n").zfill(3)] = result_lines[
                            result_item + 1].strip("\t\n")
                    # print(result_dict)
                    # for key, value in result_dict.items():
                    # print(key, value)
                lines = f.readlines()
                step_list = []
                cluster_list = []
                for line_index, line in enumerate(lines):
                    if line.startswith("step") and not lines[line_index + 1].strip('\t').startswith("cluster_-1"):
                        step_list.append(line_index)
                    if line.strip('\t').startswith("cluster_"):
                        if line.strip('\t').startswith("cluster_-1") and lines[line_index - 1].startswith("step"):
                            pass
                        else:
                            cluster_list.append(line_index)
                game_dict = {}
                for step_index, item in enumerate(step_list):
                    if step_index == len(step_list) - 1:
                        game_dict[lines[item].strip("step[]\n").zfill(3)] = {'line': item, 'nline': 9999}
                    else:
                        game_dict[lines[item].strip("step[]\n").zfill(3)] = {'line': item,
                                                                             'nline': step_list[step_index + 1]}
                # print(game_dict)
                for i, line in enumerate(cluster_list):
                    for key, value in game_dict.items():
                        if line >= value['line'] and line <= value['nline']:
                            count = len(value) - 2
                            value['c{}_line'.format(count)] = line
                    game_dict = dict(sorted(game_dict.items(), key=lambda x: x[0]))
                for key, value in game_dict.items():
                    # print(value)
                    value['c-1_line'] = value.pop('c{}_line'.format(len(value) - 3))
                    value.pop('nline')
                for step_key, step_value in game_dict.items():
                    for cluster_key, cluster_value in step_value.items():
                        if cluster_key.startswith('c'):
                            if cluster_key.startswith("c-1"):
                                # print([tuple(map(int, substr.split(','))) for substr in lines[cluster_value + 1].strip('\t\n').split(';') if substr])
                                step_value[cluster_key] = [tuple(map(int, substr.split(','))) for substr in
                                                           lines[cluster_value + 1].strip('\t\n').split(';') if substr]
                                update_tag_list(step_value[cluster_key])
                                update_enemy_tag_list(step_value[cluster_key])
                            else:
                                step_value[cluster_key] = [tuple(map(int, substr.split(','))) for substr in
                                                           lines[cluster_value + 1].strip('\t\n').split(';') if substr]
                                update_tag_list(step_value[cluster_key])
                                update_my_tag_list(step_value[cluster_key])
                # print(my_tag_list)
                # print(enemy_tag_list)
                figsise = opts.InitOpts(width='1550px', height='650px')
                tl = Timeline(init_opts=figsise)
                grid_x_list = []
                grid_y1_list = []
                grid_y2_list = []
                grid_y3_list = []
                grid_y4_list = []

                grid_r_kill_list = []
                grid_r_fall_list = []
                grid_r_inferior_list = []
                grid_r_dominant_list = []
                grid_r_self_health_loss_ratio_list = []
                grid_r_ememy_health_loss_ratio_list = []
                grid_r_fire_coverage_list = []
                grid_r_covered_in_fire_list = []
                for step_key, step_value in game_dict.items():
                    grid_x_list.append(step_key)
                    x_data = [t[1] for sublist in step_value.values() if isinstance(sublist, list) for t in sublist]
                    y_data = [t[2] for sublist in step_value.values() if isinstance(sublist, list) for t in sublist]
                    point_data = sorted(
                        [(t[0], t[1], t[2]) for sublist in step_value.values() if isinstance(sublist, list) for t in
                         sublist], key=lambda x: x[0])
                    y_list = []
                    clusters_health_dict = {}
                    for cluster_key, cluster_value in step_value.items():
                        if cluster_key.startswith('c'):
                            clusters_health_dict[cluster_key.strip('c_line')] = round(
                                sum(item[4] for item in cluster_value) / len(cluster_value) / 255.0 if len(
                                    cluster_value) > 0 else 0, 2)
                            for point in cluster_value:
                                y_list.append([point[2], point[0], point[3], int(cluster_key.strip('c_line'))])
                    x_list = [x[1] for x in point_data]
                    complete_list = []
                    none_tag_list = []
                    for tag in tag_list:
                        found = False
                        for item in y_list:
                            if item[1] == tag:
                                complete_list.append(item)
                                found = True
                                break
                        if not found:
                            # if
                            # print(y_list)
                            if item[1] in my_tag_list:
                                complete_list.append([0, tag, 0, -3])
                                none_tag_list.append(tag)
                            if item[1] in enemy_tag_list:
                                complete_list.append([0, tag, 0, -4])
                                none_tag_list.append(tag)
                    new_y_list = sorted(complete_list, key=lambda x: x[1])
                    add_x_list = []
                    for tag in none_tag_list:
                        add_x_list.append([unit[1] for unit in new_y_list].index(tag))
                    for index in sorted(add_x_list):
                        x_list.insert(index, 0)
                    positive_list = [point[2] for point in new_y_list if point[3] >= 0]
                    positive_sum = sum(positive_list)
                    positive_count = len(positive_list)
                    negative_list = [point[2] for point in new_y_list if point[3] == -1]
                    negative_sum = sum(negative_list)
                    negative_count = len(negative_list)
                    positive_live_list = [point[2] for point in new_y_list if point[3] >= 0 and point[2] > 0]
                    # print(positive_live_list)
                    # print(positive_list)
                    positive_live_sum = sum(positive_live_list)
                    positive_live_count = len(my_tag_list)
                    negative_live_list = [point[2] for point in new_y_list if point[3] == -1 and point[2] > 0]
                    negative_live_sum = sum(negative_live_list)
                    negative_live_count = len(enemy_tag_list)
                    # print(new_y_list)
                    # for key, value in clusters_health_dict.items():
                    #     print(key, value)
                    #     if float(key) >= 0:
                    #         positive_sum += float(value)
                    #         positive_count += 1
                    #     else:
                    #         negative_sum += float(value)
                    #         negative_count += 1
                    positive_mean = round(positive_sum / positive_count if positive_count > 0 else 0, 2)
                    grid_y1_list.append(positive_mean)
                    negative_mean = round(negative_sum / negative_count if negative_count > 0 else 0, 2)
                    grid_y2_list.append(negative_mean)
                    positive_live_mean = round(
                        positive_live_sum / positive_live_count if positive_live_count > 0 else 0, 2)
                    grid_y3_list.append(positive_live_mean)
                    negative_live_mean = round(
                        negative_live_sum / negative_live_count if negative_live_count > 0 else 0, 2)
                    grid_y4_list.append(negative_live_mean)
                    game_health_dict = {'self_units': positive_mean, 'enemy_units': negative_mean}
                    # print(result_dict[step_key].strip(' ')[1])
                    # print(int(result_dict[step_key].split()[1]))
                    grid_r_kill_list.append(float(result_dict[step_key].split()[0]))
                    grid_r_fall_list.append(float(result_dict[step_key].split()[1]))
                    grid_r_inferior_list.append(float(result_dict[step_key].split()[2]))
                    grid_r_dominant_list.append(float(result_dict[step_key].split()[3]))
                    grid_r_self_health_loss_ratio_list.append(float(result_dict[step_key].split()[4]))
                    grid_r_ememy_health_loss_ratio_list.append(float(result_dict[step_key].split()[5]))
                    grid_r_fire_coverage_list.append(float(result_dict[step_key].split()[6]))
                    grid_r_covered_in_fire_list.append(float(result_dict[step_key].split()[7]))
                    x_min = min(x_data)
                    x_max = max(x_data)
                    y_min = min(y_data)
                    y_max = max(y_data)
                    grid = Grid()
                    scatter = (
                        Scatter()
                            .add_xaxis(xaxis_data=x_list)
                            .add_yaxis(
                            series_name="",
                            y_axis=new_y_list,
                            symbol_size=20,
                            color='green',
                            label_opts=opts.LabelOpts(
                                is_show=True,
                                color="blue",
                                formatter=JsCode(
                                    "function(params){return '('+params.value[0]+','+params.value[1]+'):'+params.value[3];}"
                                )),
                        )
                            .set_series_opts(itemstyle_opts=opts.ItemStyleOpts(opacity=1))
                            .set_global_opts(
                            xaxis_opts=opts.AxisOpts(
                                type_="value",
                                name='X Axis',
                                splitline_opts=opts.SplitLineOpts(is_show=True),
                                axislabel_opts=opts.LabelOpts(interval=0),
                                min_=x_min - 1,
                                max_=x_max + 1
                            ),
                            yaxis_opts=opts.AxisOpts(
                                type_="value",
                                name='Y Axis',
                                axislabel_opts=opts.LabelOpts(interval=0),
                                splitline_opts=opts.SplitLineOpts(is_show=True),
                                min_=y_min - 1,
                                max_=y_max + 1
                            ),
                            visualmap_opts=opts.VisualMapOpts(
                                type_="color", max_=len(step_value.keys()) - 2, min_=-1, dimension=4
                            ),
                            legend_opts=opts.LegendOpts(pos_top="5%", pos_left="5%"),
                            title_opts=opts.TitleOpts(
                                title=json.dumps(game_health_dict).strip('\{\}').replace('"', ""),
                                subtitle=json.dumps(clusters_health_dict).strip('\{\}').replace('"', ""),
                                pos_top='0%',  # 标题的垂直位置
                                pos_left='center',  # 标题的水平位置
                                title_textstyle_opts=opts.TextStyleOpts(
                                    font_size=16,  # 标题的字体大小
                                    font_weight='bold'  # 标题的字体粗细
                                ),
                                subtitle_textstyle_opts=opts.TextStyleOpts(
                                    font_size=16,  # 副标题的字体大小
                                    color='red',  # 副标题的颜色
                                    font_weight='bold'  # 标题的字体粗细
                                )
                            )
                        )
                    )
                    line = (
                        Line()
                            .add_xaxis(grid_x_list)
                            .add_yaxis("selfliveHm", grid_y1_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="red"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="red", color0="red",
                                                                         border_color="red"),
                                       label_opts=opts.LabelOpts(is_show=False, color="red")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("enemyliveHm", grid_y2_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="blue"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="blue", color0="blue",
                                                                         border_color="blue"),
                                       label_opts=opts.LabelOpts(is_show=False, color="blue")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='blue')
                                       )
                            .add_yaxis("selfHm", grid_y3_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="pink"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="pink", color0="pink",
                                                                         border_color="pink"),
                                       label_opts=opts.LabelOpts(is_show=False, color="pink")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("enemyHm", grid_y4_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="skyblue"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="skyblue", color0="skyblue",
                                                                         border_color="skyblue"),
                                       label_opts=opts.LabelOpts(is_show=False, color="skyblue")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='blue')
                                       )
                            .add_yaxis("globalReward", [x - y for x, y in zip(grid_y3_list, grid_y4_list)],
                                       is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#228B3B"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#228B3B", color0="#228B3B",
                                                                         border_color="#228B3B"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#228B3B"),
                                       areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='#228B3B')
                                       )
                            # .set_series_opts(
                            # areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                            # label_opts=opts.LabelOpts(is_show=True),
                            # )
                            .set_global_opts(
                            xaxis_opts=opts.AxisOpts(
                                axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                                is_scale=False,
                                boundary_gap=False,
                            ),
                            legend_opts=opts.LegendOpts(pos_top="5%", pos_right="5%"),
                            # visualmap_opts=opts.VisualMapOpts(
                            #     type_="color", max_=1, min_=1
                            # ),
                        )
                    )
                    # y_data_2 = [list(map(float, data.split())) for data in result_dict.values()]
                    # y_data_2 = list(zip(*y_data_2))
                    line2 = (
                        Line()
                            .add_xaxis(grid_x_list)
                            .add_yaxis("rKill", grid_r_kill_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#FF1F5B"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#FF1F5B", color0="#FF1F5B",
                                                                         border_color="#FF1F5B"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#FF1F5B")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rFall", grid_r_fall_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#009ADE"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#009ADE", color0="#009ADE",
                                                                         border_color="#009ADE"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#009ADE")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rOutperform", grid_r_inferior_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#F28522"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#F28522", color0="#F28522",
                                                                         border_color="#F28522"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#F28522")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rUnderperform", grid_r_dominant_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#00CD6C"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#00CD6C", color0="#00CD6C",
                                                                         border_color="#00CD6C"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#00CD6C")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rSelfH", grid_r_self_health_loss_ratio_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#AF58BA"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#AF58BA", color0="#AF58BA",
                                                                         border_color="#AF58BA"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#AF58BA")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rEmemyH", grid_r_ememy_health_loss_ratio_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#FFC61E"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#FFC61E", color0="#FFC61E",
                                                                         border_color="#FFC61E"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#FFC61E")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rFCoverage", grid_r_fire_coverage_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#A6761D"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#A6761D", color0="#A6761D",
                                                                         border_color="#A6761D"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#A6761D")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("rFCovered", grid_r_covered_in_fire_list, is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#A0B1BA"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#A0B1BA", color0="#A0B1BA",
                                                                         border_color="#A0B1BA"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#A0B1BA")
                                       # areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='red')
                                       )
                            .add_yaxis("reward", [a + b + c + d + e + f + g + h for a, b, c, d, e, f, g, h in zip(
                                                grid_r_kill_list, grid_r_fall_list,
                                                grid_r_inferior_list, grid_r_dominant_list,
                                                grid_r_self_health_loss_ratio_list, grid_r_ememy_health_loss_ratio_list,
                                                grid_r_fire_coverage_list, grid_r_covered_in_fire_list)],
                                       is_smooth=True,
                                       symbol='',
                                       linestyle_opts=opts.LineStyleOpts(color="#9CCEA7"),
                                       itemstyle_opts=opts.ItemStyleOpts(color="#9CCEA7", color0="#9CCEA7",
                                                                         border_color="#9CCEA7"),
                                       label_opts=opts.LabelOpts(is_show=False, color="#9CCEA7"),
                                       areastyle_opts=opts.AreaStyleOpts(opacity=0.9, color='#9CCEA7')
                                       )
                            .set_global_opts(
                            xaxis_opts=opts.AxisOpts(
                                axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                                is_scale=False,
                                boundary_gap=False,
                            ),
                            legend_opts=opts.LegendOpts(pos_top="50%", pos_right="15%", pos_left="45%"),
                            # visualmap_opts=opts.VisualMapOpts(
                            #     type_="color", max_=1, min_=1
                            # ),
                        )
                    )
                    # x_data_2 = [int(i) for i in result_dict.keys()]
                    # y_data_2 = [list(map(float, data.split())) for data in result_dict.values()]
                    # y_data_2 = list(zip(*y_data_2))
                    # for i in range(len(y_data_2)):
                    #     line2.add_xaxis(x_data_2)
                    #     line2.add_yaxis("short_term_result", y_data_2[i], is_smooth=True)
                    # # for key, value in result_dict.items():
                    # #     step = int(key)
                    # #     values = [float(x) for x in value.split()]
                    # #     line2.add_xaxis([f'step[{step}]'])
                    # #     line2.add_yaxis('', values, linestyle_opts=opts.LineStyleOpts(width=1))
                    # line2.set_global_opts(title_opts=opts.TitleOpts(title='Line Chart'),
                    #                       xaxis_opts=opts.AxisOpts(name='Step'),
                    #                       yaxis_opts=opts.AxisOpts(name='Value'))
                    grid.add(scatter, grid_opts=opts.GridOpts(pos_left="5%", pos_right="60%", pos_top="15%"))
                    grid.add(line,
                             grid_opts=opts.GridOpts(
                                 pos_left="45%", pos_right="5%", pos_top="10%", pos_bottom="55%"
                             )
                             )
                    grid.add(line2,
                             grid_opts=opts.GridOpts(
                                 pos_left="45%", pos_right="5%", pos_top="55%", pos_bottom="10%"
                             )
                             )
                    tl.add(grid, "step{}".format(step_key))
                tl.add_schema(
                    # is_auto_play=True,  # 自动播放
                    play_interval=3000,  # 播放间隔，单位为毫秒
                    pos_left='5%',  # Timeline的水平位置
                    # pos_bottom='-5%',  # Timeline的垂直位置
                    width='90%',  # Timeline的宽度
                    # label_opts=opts.LabelOpts(
                    #     is_show=True,  # 是否显示标签
                    #     color='black',  # 标签的颜色
                    #     font_size=12  # 标签的字体大小
                    # )
                )
                # page.render("drawClustersHealthResult.html")
                tab.add(tl, "{}".format(file_id))
    tab.render("./output/drawClustersHealthResult.html")


if __name__ == '__main__':
    path_TL_MM_8 = './../datas/data_for_render/experiments_datas/two-layer/MM_8/'
    path_TL_MM_8_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_8_dist/'
    path_TL_MM_8_far = './../datas/data_for_render/experiments_datas/two-layer/MM_8_far/'
    path_TL_MM_8_problem1 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1/'
    path_TL_MM_4_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_4_dist/'
    path_TL_MM_8_problem1_2 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1_2/'
    # drawClustersResult_unit4(path_MM_8)
    path_TL_MM_8_stR = './../datas/data_for_render/experiments_datas/shorttermR/shorttermR/'
    path_TL_MM_8_stR2 = './../datas/data_for_render/experiments_datas/shorttermR/shorttermR2/'

    path_SM_5 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_5/'
    path_SM_10 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_10/'
    path_SM_20 = './../datas/data_for_render/experiments_datas/20231204/MM_8_20/'

    path_strtest = './../datas/data_for_render/experiments_datas/tests/strtest/'

    # drawClustersResult_unit8(path_TL_MM_8_problem1_2)
    drawClustersHealthResult(path_TL_MM_8_stR)
