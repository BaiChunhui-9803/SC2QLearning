clustermc1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_allreward_1/'
clustermc2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_allreward_2/'
clustermcl1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_s10_1/'
clustermcl2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_s10_2/'

param = clustermc2

with open(param + 'game_result.txt', 'r') as f:
    lines = f.readlines()

filtered_lines_1 = lines[::3][:499]
filtered_lines_2 = lines[1::3][:499]
filtered_lines_3 = lines[2::3][:499]

with open(param + 'game_result1/game_result.txt', 'w') as file:
    # 写入筛选后的行
    file.writelines(filtered_lines_1)
with open(param + 'game_result2/game_result.txt', 'w') as file:
    # 写入筛选后的行
    file.writelines(filtered_lines_2)
with open(param + 'game_result3/game_result.txt', 'w') as file:
    # 写入筛选后的行
    file.writelines(filtered_lines_3)
# # 将修改后的内容写回文件
# with open(path_test_1_mirror_test_1 + 'game_result.txt', 'w') as f:
#     f.writelines(lines)