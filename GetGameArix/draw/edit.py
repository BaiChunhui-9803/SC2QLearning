
path_test_1_mirror_test_1 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_4/'

with open(path_test_1_mirror_test_1 + 'game_result.txt', 'r') as f:
    lines = f.readlines()

# 修改符合条件的行
for i, line in enumerate(lines):
    parts = line.split('\t')  # 假设列之间是用制表符分隔的
    if len(parts) >= 4 and int(parts[3].strip()) > -360:
        parts[0] = 'Win'  # 将第一列的值修改为'Win'
        lines[i] = '\t'.join(parts)

# 将修改后的内容写回文件
with open(path_test_1_mirror_test_1 + 'game_result.txt', 'w') as f:
    f.writelines(lines)