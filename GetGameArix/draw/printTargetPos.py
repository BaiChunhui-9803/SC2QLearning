import csv
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Scatter, Timeline
from pyecharts.faker import Faker


cdata = open(r'../data_for_render/target_pos.csv', mode='r', encoding='utf8')
clist = csv.reader(cdata)  # 注意：文件打开后每次只能进行一次操作
my_list = []
enemy_list = []
target_list = []
for i in clist:
    my_string_list = i[0].split(',')
    m_x = my_string_list[0][1:-1].strip('\'').strip().split(' ')[0]
    m_y = my_string_list[0][1:-1].strip('\'').strip().split(' ')[-1]
    my_list.append([float(m_x), float(m_y)])
    enemy_string_list = i[1].split(',')
    e_x = enemy_string_list[0][1:-1].strip('\'').strip().split(' ')[0]
    e_y = enemy_string_list[0][1:-1].strip('\'').strip().split(' ')[-1]
    enemy_list.append([float(e_x), float(e_y)])
    target_string_list = i[2].split(',')
    t_x = target_string_list[0][1:-1].strip('\'').strip().split(' ')[0]
    t_y = target_string_list[0][1:-1].strip('\'').strip().split(' ')[-1]
    target_list.append([float(t_x), float(t_y)])
    # print(string_list)
    # print('x', x, 'y', y)
    # print(x, y)
    # print(i, type(i[0]))
print(my_list)
print(enemy_list)
print(target_list)
data = [[]] * len(my_list)
for i in range(len(my_list)):
    data[i].append([my_list[i], enemy_list[i], target_list[i]])

mx_data = [data[0][i][0][0] for i in range(len(my_list))]
my_data = [data[0][i][0][1] for i in range(len(my_list))]
ex_data = [data[0][i][1][0] for i in range(len(my_list))]
ey_data = [data[0][i][1][1] for i in range(len(my_list))]
tx_data = [data[0][i][2][0] for i in range(len(my_list))]
ty_data = [data[0][i][2][1] for i in range(len(my_list))]
min_x = min(mx_data + ex_data + tx_data)
min_y = min(my_data + ey_data + ty_data)
max_x = max(mx_data + ex_data + tx_data)
max_y = max(my_data + ey_data + ty_data)
min_x_diff = round(min_x - (max_x - min_x) * 0.1, 2)
min_y_diff = round(min_y - (max_y - min_y) * 0.1, 2)
max_x_diff = round(max_x + (max_x - min_x) * 0.1, 2)
max_y_diff = round(max_y + (max_y - min_y) * 0.1, 2)

print('mx_data', mx_data)
print('my_data', my_data)
print('ex_data', ex_data)
print('ey_data', ey_data)
print('tx_data', tx_data)
print('ty_data', ty_data)

# data_for_transit[0].sort(key=lambda x: x[0])

tl = Timeline()
for i in range(0, len(mx_data)):
    c = (
        Scatter()
        .add_xaxis(xaxis_data=[mx_data[i]])
        .add_yaxis(
            series_name="",
            y_axis=[my_data[i]],
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_xaxis(xaxis_data=[ex_data[i]])
        .add_yaxis(
            series_name="",
            y_axis=[ey_data[i]],
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_xaxis(xaxis_data=[tx_data[i]])
        .add_yaxis(
            series_name="",
            y_axis=[ty_data[i]],
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_series_opts()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                min_=min_x_diff,
                max_=max_x_diff,
                type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                min_=min_y_diff,
                max_=max_y_diff,
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            tooltip_opts=opts.TooltipOpts(is_show=False),
        )
    )
    tl.add(c, format(i))
    tl.add_schema(
        play_interval=1000,
        is_auto_play=True
    )
tl.render("./render/timeline_bar_reversal.html")

