import draw_clusters_result
import draw_game_data_plot
import stackplottest

def draw_manage(path):
    draw_game_data_plot.drawLineChart(path)
    print('全局奖励折线图输出完成')
    draw_game_data_plot.drawBoxChart(path)
    print('全局奖励盒箱图输出完成')
    draw_game_data_plot.drawHistoryLineChart(path)
    print('全局奖励历史折线图输出完成')
    draw_game_data_plot.drawActionLogLineChart(path)
    print('动作选择趋势折线图输出完成')
    stackplottest.fun5(path)
    print('动作选择趋势堆叠图输出完成')
    draw_clusters_result.drawClustersHealthResult(path)
    print('局部奖励可视化输出完成')

if __name__ == '__main__':
    # 状态空间：(len(my_units), len(enemy_units)) 动作空间：action_TFC_finish, action_TNC_finish, action_noise 奖励值：null
    path010190NULL = './../datas/data_for_render/experiments_datas/LR10_RD10_nullReward/'
    # 状态空间：(len(my_units), len(enemy_units)) 动作空间：action_TFC_finish, action_TNC_finish, action_noise 奖励值：即时奖励
    path010190 = './../datas/data_for_render/experiments_datas/LR01_RD01/'
    path101090 = './../datas/data_for_render/experiments_datas/LR10_RD10/'
    path101050 = './../datas/data_for_render/experiments_datas/LR10_RD10_GD50/'
    path505090 = './../datas/data_for_render/experiments_datas/LR50_RD50/'
    path509090 = './../datas/data_for_render/experiments_datas/LR50_RD90/'
    path801090 = './../datas/data_for_render/experiments_datas/LR80_RD10/'
    # 状态空间：(len(my_units), len(enemy_units)) 奖励值：即时奖励 动作空间：action_TFC_finish, action_TNC_finish, action_greedy,
    # action_noise 作战地图：MvsM(敌我势均力敌)
    pathMvsM_0 = './../datas/data_for_render/experiments_datas/M4vsM4/'
    # 状态空间：(基于图像直方图的哈希编码) 奖励值：即时奖励 动作空间：action_TFC_finish, action_TNC_finish, action_greedy, action_noise
    # 运行环境：Linux | Took 2456.871 seconds for 13000 steps: 5.291 fps
    # 作战地图：MvsM
    pathMvsM_1 = './../datas/data_for_render/experiments_datas/M4vsM4_Origin_Linux/'
    # 运行环境：Linux | Took 2407.806 seconds for 13000 steps: 5.399 fps
    # 作战地图：MvsM_line
    pathMvsM_2 = './../datas/data_for_render/experiments_datas/M4vsM4_Line_Linux/'
    # 运行环境：Linux | Took 2418.859 seconds for 13000 steps: 5.374 fps
    # 作战地图：MvsM_cross
    pathMvsM_3 = './../datas/data_for_render/experiments_datas/M4vsM4_Cross_Linux/'
    # 运行环境：Linux | Took 2445.041 seconds for 13000 steps: 5.317 fps
    # 作战地图：MvsM_dist
    pathMvsM_4 = './../datas/data_for_render/experiments_datas/M4vsM4_Dist_Linux/'

    #参数调整
    # 运行环境：Linux | Took 2456.871 seconds for 13000 steps: 5.291 fps
    pathMvsM_1_LR10_RD10_GD90 = './../datas/data_for_render/experiments_datas/parameter/M4vsM4_Origin_Linux/'
    # 运行环境：Linux | Took 2432.455 seconds for 13000 steps: 5.344 fps
    pathMvsM_1_LR10_RD90_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR10_RD90_GD90/'
    # 运行环境：Linux | Took 2482.168 seconds for 13000 steps: 5.237 fps
    pathMvsM_1_LR50_RD90_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR50_RD90_GD90/'
    # 运行环境：Linux | Took 2388.942 seconds for 13000 steps: 5.442 fps
    pathMvsM_1_LR90_RD90_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR90_RD90_GD90/'
    # 运行环境：Linux | Took 2239.988 seconds for 13000 steps: 5.804 fps
    pathMvsM_1_LR90_RD10_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR90_RD10_GD90/'
    # 运行环境：Linux | Took 2315.349 seconds for 13000 steps: 5.615 fps
    pathMvsM_1_LR90_RD50_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR90_RD50_GD90/'
    # 运行环境：Linux | Took 2336.371 seconds for 13000 steps: 5.564 fps
    pathMvsM_1_LR50_RD50_GD90 = './../datas/data_for_render/experiments_datas/parameter/MvsM_1_LR50_RD50_GD90/'

    #实验
    pathMM_Origin_4 = './../datas/data_for_render/experiments_datas/problems/MM_Origin_4/'
    pathMM_Origin_8 = './../datas/data_for_render/experiments_datas/problems/MM_Origin_8/'
    pathMM_Dist_4 = './../datas/data_for_render/experiments_datas/problems/MM_Dist_4/'
    pathMM_Dist_8 = './../datas/data_for_render/experiments_datas/problems/MM_Dist_8/'
    pathMM_Far_4 = './../datas/data_for_render/experiments_datas/problems/MM_Far_4/'
    pathMM_Far_8 = './../datas/data_for_render/experiments_datas/problems/MM_Far_8/'
    pathMM_Weak_1 = './../datas/data_for_render/experiments_datas/problems/MM_Weak_8/'
    pathMM_Weak_2 = './../datas/data_for_render/experiments_datas/problems/MM_Weak_8_2/'

    # 改进实验
    path_PC_MM_4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_4/'
    path_PC_MM_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_8/'
    path_PC_MM_Far_4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_4/'
    path_PC_MM_Far_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8/'
    path_PC_MM_Far_8_2 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_2/'
    path_PC_MM_Far_8_3 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_3/'
    path_PC_MM_Far_8_4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_4/'
    path_PC_MM_Dist_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Dist_8/'
    path_PC_MM_Weak_8 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak_8/'

    # 改进实验2
    path_SA_MM_Far_8 = './../datas/data_for_render/experiments_datas/state_area_100/MM_Far_8/'
    path_SA_MM_Far_8_2 = './../datas/data_for_render/experiments_datas/state_area_100/MM_Far_8_2/'

    # 改进实验3
    path_PC_MM_Far_8_e = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_e/'
    path_PC_MM_Far_8_e_2 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_e_2/'
    path_PC_MM_Weak2_8_e = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak2_8_e/'

    #测试问题
    path_MM_Problem1 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Problem1/'

    # 改进实验：双层模型
    path_TL_MM_8 = './../datas/data_for_render/experiments_datas/two-layer/MM_8/'
    path_TL_MM_8_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_8_dist/'
    path_TL_MM_8_far = './../datas/data_for_render/experiments_datas/two-layer/MM_8_far/'
    path_TL_MM_8_problem1 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1/'
    path_TL_MM_4_dist = './../datas/data_for_render/experiments_datas/two-layer/MM_4_dist/'
    path_TL_MM_8_problem1_2 = './../datas/data_for_render/experiments_datas/two-layer/MM_8_problem1_2/'

    # _STEP_MUL参数调整
    path_SM_5 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_5/'
    path_SM_10 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_10/'
    path_SM_20 = './../datas/data_for_render/experiments_datas/two-layer-2/MM_8_20/'

    path_TL_MM_8_stR = './../datas/data_for_render/experiments_datas/shorttermR/shorttermR/'
    # path_TL_MM_8_stR2 = './../datas/data_for_render/experiments_datas/shorttermR/shorttermR2/'

    draw_manage(path_TL_MM_8_stR)