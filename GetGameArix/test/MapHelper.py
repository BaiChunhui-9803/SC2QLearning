"""
Map Path: C:\Program Files (x86)\StarCraft II\Maps
"""

from absl import app
from pysc2 import maps


def print_object_attribute(mp):
    print('开始输出属性:')
    attr_list = []
    for i in dir(mp):
        # print(i, type(eval('mp.' + str(i))))
        if isinstance(eval('mp.' + str(i)), int) \
                or isinstance(eval('mp.' + str(i)), str) \
                or isinstance(eval('mp.' + str(i)), property):
            attr_list.append(i)

    # print(dir(mp))
    print('\033[0;33m对象可用属性:\033[0m')
    for idx, element in enumerate(attr_list):
        print(idx, element)


def print_all_attribute(object):
    print('开始输出属性:')
    attr_list = []
    for i in dir(object):
        attr_list.append(i)

    # print(dir(mp))
    print('\033[0;33m对象mp可用属性:\033[0m')
    for idx, element in enumerate(attr_list):
        print(idx, element)


def get_map_name_list(unused_argv):
    map_name_list = []
    for _, map_class in sorted(maps.get_maps().items()):
        mp = map_class()
        if mp.path:
            map_name_list.append(mp.name)

    # 得到所有可用地图的名称
    print('\033[0;33m当前可用地图列表:\033[0m')
    for idx, element in enumerate(map_name_list):
        print(idx, element)


def main(unused_argv):
    print_object_attribute(sorted(maps.get_maps().items())[0][1])
    get_map_name_list(unused_argv)


if __name__ == "__main__":
    app.run(main)
