import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
import seaborn as sns
import csv
import random
from pyecharts import options as opts
from pyecharts.charts import Scatter, Timeline, Tab

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
                            subtitle="clu_uniformity: {}, clu_crowding: {}".format(list_all[list_idx][i][1], list_all[list_idx][i][2]),
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



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    path_TL_MM_8 = './../datas/data_for_render/experiments_datas/two-layer/MM_8/'
    path_TL_MM_8_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_8_dist/'
    path_TL_MM_8_far = './../datas/data_for_render/experiments_datas/two-layer/MM_8_far/'
    path_TL_MM_8_problem1 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1/'
    path_TL_MM_4_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_4_dist/'
    path_TL_MM_8_problem1_2 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1_2/'
    # drawClustersResult_unit4(path_MM_8)
    drawClustersResult_unit8(path_TL_MM_8_problem1_2)