import draw_clusters_result
import draw_game_data_plot
import stackplottest

maxStep = 1500


def draw_manage(path):
    # draw_game_data_plot.drawLineChart(path, maxStep)
    # print('全局奖励折线图输出完成')
    # draw_game_data_plot.drawBoxChart(path)
    # print('全局奖励盒箱图输出完成')
    # draw_game_data_plot.drawHistoryLineChart(path)
    # print('全局奖励历史折线图输出完成')
    # draw_game_data_plot.drawActionLogLineChart(path)
    # print('动作选择趋势折线图输出完成')
    # stackplottest.fun5(path)
    # print('动作选择趋势堆叠图输出完成')
    # draw_clusters_result.drawClustersHealthResult(path, maxStep)
    print('局部奖励可视化输出完成')


def multi_draw_manage(path_list, title_list, separate, separate_title_list, colors, linestyles):
    # draw_game_data_plot.drawWinRateLineChart(path_list, title_list, separate, separate_title_list, colors, linestyles, maxStep)
    # draw_game_data_plot.drawFitnessLineChart(path_list, title_list, separate, separate_title_list, colors, linestyles, maxStep)
    draw_game_data_plot.drawBoth(path_list, title_list, separate, separate_title_list, colors, linestyles, maxStep)
    # draw_game_data_plot.drawParetoChart(path_list, title_list, separate, separate_title_list)


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

    # 参数调整
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

    # 实验
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

    # 测试问题
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

    path_action1 = './../datas/data_for_render/experiments_datas/shorttermR/action1/'
    path_action1_2 = './../datas/data_for_render/experiments_datas/shorttermR/action1_2/'
    path_action1_clu = './../datas/data_for_render/experiments_datas/shorttermR/action1_clu/'
    path_action1_clu_2 = './../datas/data_for_render/experiments_datas/shorttermR/action1_clu_2/'
    path_action1_random = './../datas/data_for_render/experiments_datas/shorttermR/action1_random/'
    path_action1_random_2 = './../datas/data_for_render/experiments_datas/shorttermR/action1_random_2/'
    path_action2 = './../datas/data_for_render/experiments_datas/shorttermR/action2/'
    path_action2_2 = './../datas/data_for_render/experiments_datas/shorttermR/action2_2/'
    path_action2_13 = './../datas/data_for_render/experiments_datas/shorttermR/action2_13/'
    path_action2_13_2 = './../datas/data_for_render/experiments_datas/shorttermR/action2_13_2/'
    path_action2_13_3 = './../datas/data_for_render/experiments_datas/shorttermR/action2_13_3/'
    path_action2_24 = './../datas/data_for_render/experiments_datas/shorttermR/action2_24/'
    path_action2_24_2 = './../datas/data_for_render/experiments_datas/shorttermR/action2_24_2/'
    path_action3 = './../datas/data_for_render/experiments_datas/shorttermR/action3/'
    path_action4 = './../datas/data_for_render/experiments_datas/shorttermR/action4/'
    path_action4_2 = './../datas/data_for_render/experiments_datas/shorttermR/action4_2/'
    path_action5 = './../datas/data_for_render/experiments_datas/shorttermR/action5/'
    path_action6 = './../datas/data_for_render/experiments_datas/shorttermR/action6/'
    path_8far_action7_s5 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s5/'
    path_8far_action7_s5_2 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s5_2/'
    path_8far_action7_s5_3 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s5_3/'
    path_8far_action7_s5_4 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s5_4/'
    path_8far_action7_s10 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s10/'
    path_8far_action7_s10_2 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s10_2/'
    path_8far_action7_s10_3 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s10_3/'
    path_8far_action7_s10_4 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s10_4/'
    path_8far_action7_s15 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s15/'
    path_8far_action7_s15_2 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_s15_2/'

    path_8far_action2_24_1 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action2_24_1/'
    path_8far_action2_24_2 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action2_24_2/'

    path_8far_action7_2 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_2/'
    path_8far_action7_3 = './../datas/data_for_render/experiments_datas/shorttermR/8far_action7_3/'

    # 1500
    path_8far_1500_action1_s10_1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action1_s10_1/'
    path_8far_1500_action1_s10_2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action1_s10_2/'
    path_8far_1500_action7_s10_1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action7_s10_1/'
    path_8far_1500_action7_s10_2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action7_s10_2/'
    path_8far_action7_allreward_1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action7_allreward_1/'
    path_8far_action7_allreward_2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action7_allreward_2/'
    path_8far_1500_action2_24_s10_1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action2_24_s10_1/'
    path_8far_1500_action2_24_s10_2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action2_24_s10_2/'

    # data_to_model
    path_8faraction7_to_8faraction7_1 = './../datas/data_for_render/experiments_datas/offline_data_to_model/8faraction7_to_8faraction7_1/'
    path_8faraction7_to_8faraction7_2 = './../datas/data_for_render/experiments_datas/offline_data_to_model/8faraction7_to_8faraction7_2/'
    path_8faraction7_to_8faraction7_similar_1 = './../datas/data_for_render/experiments_datas/offline_data_to_model/8faraction7_to_8faraction7_similar_1/'
    path_8faraction7_to_8faraction7_similar_2 = './../datas/data_for_render/experiments_datas/offline_data_to_model/8faraction7_to_8faraction7_similar_2/'
    path_8faraction7_to_8faraction7_similar2_1 = './../datas/data_for_render/experiments_datas/offline_data_to_model/8faraction7_to_8faraction7_similar2_1/'


    # defense
    path_action1_DEF = './../datas/data_for_render/experiments_datas/defense/action1_DEF/'
    path_action5_noDEF_1 = './../datas/data_for_render/experiments_datas/defense/action5_noDEF_1/'
    path_action5_noDEF_2 = './../datas/data_for_render/experiments_datas/defense/action5_noDEF_2/'
    path_action6_1 = './../datas/data_for_render/experiments_datas/defense/action6_1/'
    path_action6_2 = './../datas/data_for_render/experiments_datas/defense/action6_2/'
    path_action8_8far_1 = './../datas/data_for_render/experiments_datas/defense/action8_8far_1/'
    path_action8_8far_2 = './../datas/data_for_render/experiments_datas/defense/action8_8far_2/'

    # test 20240506
    path_test_1_1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_1/'
    path_test_1_2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_2/'
    path_test_1_mirror_1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_1/'
    path_test_1_mirror_2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_2/'
    path_test_1_mirror_3 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_3/'
    path_test_1_mirror_4 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_4/'
    path_test_1_test_1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_test_1/'
    path_test_1_test_2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_test_2/'
    path_test_1_mirror_test_1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_test_1/'
    path_test_1_mirror_test_2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_test_2/'


    path_8far_1 = './../datas/data_for_render/experiments_datas/tests/MM8far_1/'
    path_8far_2 = './../datas/data_for_render/experiments_datas/tests/MM8far_2/'
    path_8far_mirror_1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_1/'
    path_8far_mirror_act12345_1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act12345_1/'
    path_8far_mirror_act12345_2 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act12345_2/'
    path_8far_mirror_act123458_1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act123458_1/'
    path_8far_mirror_act123458_2 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act123458_2/'

    path_unfair_1 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_1/'
    path_unfair_2 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_2/'
    path_unfair_3 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_3/'
    path_unfair_4 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_4/'
    path_unfair_test_1 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_test_1/'
    path_unfair_test_2 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_test_2/'
    # path_unfair_test_3 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_test_3/'

    # draw_manage(path_8far_1500_action7_s10_1)
    # path_list = [path_action2, path_action2_13, path_action2_24, path_action3, path_action4, path_action5, path_8far_action7, path_action1]
    # title_list = ['2action-nN_____', '2action-n_w____', '2action-_N_W___', '3action-nNw____',
    #               '4action-nNwW___', '5action-_N_Wtlg', '7action-nNwWtlg', '1action-n______']
    # path_list = [path_action2, path_action3, path_action4, path_action5,
    #              path_8far_action7_s5, path_8far_action7_s10_4, path_action1]
    # title_list = ['2action-nN_____', '3action-nNw____', '4action-nNwW___', '5action-_N_Wtlg',
    #               '7action-nNwWtlg', '7action_2', '1action-n______']
    # path_list = [path_8far_action7_s10, path_8far_action7_s10_2, path_8far_action7_s10_3, path_8far_action7_s10_4]
    # title_list = ['s10_1', 's10_2', 's10_3', 's10_4']
    # path_list = [path_action2, path_action2_13, path_action2_24, path_action3, path_action4, path_action5, path_8far_action7_s10_4, path_action1]
    # title_list = ['2action-nN_____', '2action-n_w____', '2action-_N_W___', '3action-nNw____', '4action-nNwW___', '5action-_N_Wtlg',
    #               '7action-nNwWtlg', '1action-n______']

    # 4Mix
    # path_list = [path_action2, path_action2_2,
    #              path_action2_13_2, path_action2_13_3,
    #              path_action2_24, path_action2_24_2,
    #              path_action4, path_action4_2,
    #              path_action1, path_action1_2,
    #              path_action1_random, path_action1_random_2
    #              ]
    # title_list = ['2action-nN_____-1', '2action-nN_____-2',
    #               '2action-n_w____-2', '2action-n_w____-3',
    #               '2action-_N_W___-1', '2action-_N_W___-2',
    #               '4action-nNwW___-1', '4action-nNwW___-2',
    #               '1action-n______-1', '1action-n______-2',
    #               '1action-random-1', '1action-random-2']
    # separate = [2,
    #             2,
    #             2,
    #             2,
    #             2,
    #             2
    #             ]
    # separate_title_list = ['2action_12',
    #                        '2action_13',
    #                        '2action_24',
    #                        '4action',
    #                        '1action_1',
    #                        '1action_random',
    #                        ]

    # 8far
    # path_list = [path_8far_action7_s5, path_8far_action7_s5_2, path_8far_action7_s5_4,
    #              path_8far_action7_s10, path_8far_action7_s10_2, path_8far_action7_s10_3, path_8far_action7_s10_4,
    #              path_8far_action7_s15, path_8far_action7_s15_2,
    #              ]
    # title_list = ['s5_1', 's5_2', 's5_4',
    #               's10_1', 's10_2', 's10_3', 's10_4',
    #               's15_1', 's15_2',
    #               ]
    # separate = [3,
    #             4,
    #             2,
    #             ]
    # separate_title_list = ['s5',
    #                        's10',
    #                        's15',
    #                        ]

    # 8far_action2
    # path_list = [path_8far_action2_24_1, path_8far_action2_24_2,
    #              path_8far_action7_s10_2, path_8far_action7_s10_4,
    #              path_8faraction7_to_8faraction7_1, path_8faraction7_to_8faraction7_2,
    #              # path_8faraction7_to_8faraction7_similar_1, path_8faraction7_to_8faraction7_similar_2,
    #              # path_8faraction7_to_8faraction7_similar2_1
    #              ]
    # title_list = [
    #     '2action-_N_W___-1', '2action-_N_W___-2',
    #     '7action-nNwWtlg-1', '7action-nNwWtlg-2',
    #     '7action-model_transfer-1', '7action-model_transfer-2',
    #     # '7action-model_similar-1', '7action-model_similar-2',
    #     # '7action-model_similar2-1',
    # ]
    # separate = [2,
    #             2,
    #             2,
    #             # 2,
    #             # 1
    #             ]
    # separate_title_list = [
    #     '2action_24',
    #     '7action',
    #     '7action-model_transfer',
    #     # '7action-model_similar',
    #     # '7action-model_similar2'
    # ]


    # -defense
    # path_list = [path_action1_DEF,
    #              path_8far_1500_action1_s10_1, path_8far_1500_action1_s10_2,
    #              path_8far_1500_action2_24_s10_1, path_8far_1500_action2_24_s10_2,
    #              path_8far_1500_action7_s10_1, path_8far_1500_action7_s10_2,
    #              path_8far_action7_allreward_1, path_8far_action7_allreward_2,
    #              path_action8_8far_1, path_action8_8far_2
    #              ]
    # title_list = [
    #     '1action-def',
    #     '1action-atk-1', '1action-atk-2',
    #     '2action-atk-1', '2action-atk-2',
    #     '7action-atk-1', '7action-atk-2',
    #     '7action-atk/AR-1', '7action-atk/AR-2',
    #     '9action-1', '9action-2',
    # ]
    # separate = [1,
    #             2,
    #             2,
    #             2,
    #             2,
    #             2
    #             ]
    # separate_title_list = [
    #     '1action-def',
    #     '1action-atk',
    #     '2action-atk',
    #     '7action-atk',
    #     '7action-atk/AR',
    #     '9action',
    # ]

    # test 20240506
    # path_list = [path_test_1_1, path_test_1_2,
    #              path_test_1_test_1, path_test_1_test_2,
    #              path_test_1_mirror_2, path_test_1_mirror_3, path_test_1_mirror_4,
    #              path_test_1_mirror_test_1, path_test_1_mirror_test_2
    # # path_list = [path_action8_8far_1, path_action8_8far_2
    #              ]
    # title_list = [
    #     'test_1_1', 'test_1_2',
    #     'test_1_test_1', 'test_1_test_2',
    #     'test_1_mirror_1', 'test_1_mirror_2', 'test_1_mirror_3',
    #     'mirror_model_transfer_test_1', 'mirror_model_transfer_test_2',
    # ]
    # separate = [2,
    #             2,
    #             3,
    #             2
    #             ]
    # separate_title_list = [
    #     'original',
    #     'original_model_transfer',
    #     'mirror',
    #     'mirror_model_transfer',
    # ]

    # path_list = [path_8far_1,
    #              path_8far_mirror_1
    #              # path_unfair_test_2
    #              # path_list = [path_action8_8far_1, path_action8_8far_2
    #              ]
    # title_list = [
    #     '8far_1',
    #     '8far_mirror_1',
    #     # 'unfair_test_2',
    # ]
    # separate = [1, 1,
    #             # 1
    #             ]
    # separate_title_list = [
    #     '8far',
    #     '8far_mirror',
    #     # 'unfair_test',
    # ]

    # path_list = [path_unfair_2,
    #              # path_unfair_test_1, path_unfair_test_2
    #              # path_unfair_test_2
    #              # path_list = [path_action8_8far_1, path_action8_8far_2
    #              ]
    # title_list = [
    #     'unfair_1',
    #     # 'unfair_test_1', 'unfair_test_2',
    #     # 'unfair_test_2',
    # ]
    # separate = [1,
    #             # 2,
    #             # 1
    #             ]
    # separate_title_list = [
    #     'unfair',
    #     # 'unfair_model_transfer',
    #     # 'unfair_test',
    # ]

    # 宏动作
    # path_list = [path_8far_1500_action1_s10_1, path_8far_1500_action1_s10_2,
    #              path_8far_1500_action2_24_s10_1, path_8far_1500_action2_24_s10_2,
    #              path_8far_1500_action7_s10_1, path_8far_1500_action7_s10_2,
    #              path_8far_1, path_8far_2,
    #              path_8far_mirror_act12345_1, path_8far_mirror_act12345_2,
    #              path_8far_mirror_act123458_1, path_8far_mirror_act123458_2,
    #              path_8far_mirror_1,
    #              # path_8far_
    #              # path_8faraction7_to_8faraction7_1
    #              # path_unfair_test_1, path_unfair_test_2
    #              # path_unfair_test_2
    #              # path_list = [path_action8_8far_1, path_action8_8far_2
    #              ]
    # title_list = [
    #     'action1_1', 'action1_2',
    #     'action2_1', 'action2_2',
    #     'action7_1', 'action7_2',
    #     'action9_1', 'action9_2',
    #     'actionATK_mirror_1', 'actionATK_mirror_2',
    #     'actionMIX_mirror_1', 'actionMIX_mirror_2',
    #     'actionALL_mirror',
    #
    #     # 'unfair_test_1', 'unfair_test_2',
    #     # 'unfair_test_2',
    # ]
    # separate = [2, 2, 2, 2, 2, 2, 1,
    #             # 2,
    #             # 1
    #             ]
    # separate_title_list = [
    #     'action1',
    #     'action2',
    #     'action7',
    #     'action9',
    #     'actionATK_mirror',
    #     'actionMIX_mirror',
    #     'actionALL_mirror',
    #     # 'unfair_model_transfer',
    #     # 'unfair_test',
    # ]

    # 论文
    # jili-1 -
    wujiangli = './../datas/data_for_render/experiments_datas/parameters/LR10_RD10_nullReward/'
    wusuiji = './../datas/data_for_render/experiments_datas/others/M4vsM4_no_greedy/'
    jiandanstate1 = './../datas/data_for_render/experiments_datas/state_area_100/MM_Far_8/'
    jiandanstate2 = './../datas/data_for_render/experiments_datas/state_area_100/MM_Far_8_2/'
    # jili-2 - 单层s/单层聚类sc/多层不聚类m/多层聚类mc/多层聚类有局部奖励mcl
    clusters1 = './../datas/data_for_render/experiments_datas/problems/MM_Far_8/'
    clustersc1 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8/'
    clustersc2 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_2/'
    clustersc3= './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_3/'
    clustersc4 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_4/'
    clusterm1 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_e/'
    clusterm2 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Far_8_e_2/'
    clustermc1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_allreward_1/game_result1/'
    clustermc2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_allreward_2/game_result2/'
    clustermcl1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_s10_2/game_result2/'
    clustermcl2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/to500/8far_action7_s10_2/game_result1/'
    # jili-31 - 4v4 action_ATK_nearest/action_ATK_clu_nearest/action_ATK_nearest_weakest/
    #               action_ATK_clu_nearest_weakest/do_nothing
    xiaorong1____1 = './../datas/data_for_render/experiments_datas/shorttermR/action1/'
    xiaorong1____2 = './../datas/data_for_render/experiments_datas/shorttermR/action1_2/'
    xiaorong_2___1 = './../datas/data_for_render/experiments_datas/shorttermR/action1_clu/'
    xiaorong_2___2 = './../datas/data_for_render/experiments_datas/shorttermR/action1_clu_2/'
    xiaorong____n1 = './../datas/data_for_render/experiments_datas/shorttermR/action1_random/'
    xiaorong____n2 = './../datas/data_for_render/experiments_datas/shorttermR/action1_random_2/'
    xiaorong12___1 = './../datas/data_for_render/experiments_datas/shorttermR/action2/'
    xiaorong12___2 = './../datas/data_for_render/experiments_datas/shorttermR/action2_2/'
    xiaorong1_3__1 = './../datas/data_for_render/experiments_datas/shorttermR/action2_13/'
    xiaorong1_3__2 = './../datas/data_for_render/experiments_datas/shorttermR/action2_13_2/'
    xiaorong1_3__3 = './../datas/data_for_render/experiments_datas/shorttermR/action2_13_3/'
    xiaorong_2_4_1 = './../datas/data_for_render/experiments_datas/shorttermR/action2_24/'
    xiaorong_2_4_2 = './../datas/data_for_render/experiments_datas/shorttermR/action2_24_2/'
    xiaorong1234_1 = './../datas/data_for_render/experiments_datas/shorttermR/action4/'
    xiaorong5 = './../datas/data_for_render/experiments_datas/shorttermR/action5/'
    # jili-32 -
    # action1_D_ = './../datas/data_for_render/experiments_datas/defense/action1_DEF'
    action5A__1 = './../datas/data_for_render/experiments_datas/defense/action5_noDEF_1/'
    action5A__2 = './../datas/data_for_render/experiments_datas/defense/action5_noDEF_2/'
    action6AD_1 = './../datas/data_for_render/experiments_datas/defense/action6_1/'
    action6AD_2 = './../datas/data_for_render/experiments_datas/defense/action6_2/'
    far8action2A__1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action2_24_s10_1/'
    far8action2A__2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action2_24_s10_2/'
    far8action6AD_1 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action7_s10_1/'
    far8action6AD_2 = './../datas/data_for_render/experiments_datas/shorttermR_1500/8far_action7_s10_2/'
    far8action8ADM1 = './../datas/data_for_render/experiments_datas/defense/action8_8far_1/'
    far8action8ADM2 = './../datas/data_for_render/experiments_datas/defense/action8_8far_2/'
    far8action8ADM3 = './../datas/data_for_render/experiments_datas/defense/action8_8far_3/'
    # jili-4 -
    canshu1010 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_1010/'
    canshu1050 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_1050/'
    canshu1090 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_1090/'
    canshu5010 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_5010/'
    canshu5050 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_5050/'
    canshu5090 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_5090/'
    canshu9010 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_9010/'
    canshu9050 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_9050/'
    canshu9090 = './../datas/data_for_render/experiments_datas/parameters_new/MvsM4dist_9090/'
    # wenti-1 - dist4
    dist1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_1/'
    dist2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_2/'
    distchongyong1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_test_1/'
    distchongyong2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_test_2/'
    distmirror1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_1/'
    distmirror2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_2/'
    distmirror3 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_3/'
    distmirror4 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_4/'
    distmirrorchongyong1 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_test_1/'
    distmirrorchongyong2 = './../datas/data_for_render/experiments_datas/tests/MM4dist_1_mirror_test_2/'
    # mirror -
    # wenti-2 - far8
    far1 = './../datas/data_for_render/experiments_datas/tests/MM8far_1/'
    far2 = './../datas/data_for_render/experiments_datas/tests/MM8far_2/'
    farmirror1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_1/'
    farmirror2 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_2/'
    fartest1 = './../datas/data_for_render/experiments_datas/tests/MM8far_test_1/'
    fartest2 = './../datas/data_for_render/experiments_datas/tests/MM8far_test_2/'
    farmirrortest1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_test_1/'
    farmirrortest2 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_test_2/'
    farmirroractA__1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act12345_1/'
    farmirroractA__2 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act12345_2/'
    farmirroractA_M1 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act123458_1/'
    farmirroractA_M2 = './../datas/data_for_render/experiments_datas/tests/MM8far_mirror_act123458_2/'
    # nijing -
    weak = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak_8/'
    weak21 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak2_8/'
    weak22 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak2_8_2/'
    weak23 = './../datas/data_for_render/experiments_datas/parametric_clustering/MM_Weak2_8_e/'
    weak8vs91 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_1/'
    weak8vs92 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_2/'
    weak8vs93 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_3/'
    weak8vs94 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_4/'
    weak8vs9chongyong1 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_test_1/'
    weak8vs9chongyong2 = './../datas/data_for_render/experiments_datas/tests/MM8vs9_test_2/'
    # chongyong -
    chongyong_origin1 = './../datas/data_for_render/experiments_datas/parameters/offline_data_to_model/8faraction7_to_8faraction7_1/'
    chongyong_origin2 = './../datas/data_for_render/experiments_datas/parameters/offline_data_to_model/8faraction7_to_8faraction7_1/'
    chongyong_similar1 = './../datas/data_for_render/experiments_datas/parameters/offline_data_to_model/8faraction7_to_8faraction7_1/'
    chongyong_similar2 = './../datas/data_for_render/experiments_datas/parameters/offline_data_to_model/8faraction7_to_8faraction7_1/'
    chongyong_similar3 = './../datas/data_for_render/experiments_datas/parameters/offline_data_to_model/8faraction7_to_8faraction7_1/'

    # # jili-1
    # path_list = [
    #     wujiangli,
    #     wusuiji,
    #     jiandanstate1, jiandanstate2,
    # ]
    # title_list = [
    #     'RL1', 'RL2', 'RL3',
    #     'RL+cluster1',
    # ]
    # separate = [
    #     1,1,2
    # ]
    # separate_title_list = [
    #     'RL',
    #     'RL+cluster',
    #     'HRL',
    # ]

    # # jili-2
    # path_list = [
    #     clusters1, jiandanstate1, jiandanstate2,
    #     clustersc1, clustersc2, clustersc3, clustersc4,
    #     clusterm1, clusterm2,
    #     clustermc1, clustermc2,
    #     clustermcl1, clustermcl2
    # ]
    # title_list = [
    #     'RL1', 'RL2', 'RL3',
    #     'RL+cluster1', 'RL+cluster2', 'RL+cluster3', 'RL+cluster4',
    #     'HRL1', 'HRL2',
    #     'HRL+cluster1', 'HRL+cluster2',
    #     'HRL+cluster+lr1', 'HRL+cluster+lr2',
    # ]
    # separate = [
    #     3,4,2,2,2
    # ]
    # separate_title_list = [
    #     'RL',
    #     'RL+cluster',
    #     'HRL',
    #     'HRL+cluster',
    #     'HRL+cluster+local_reward',
    # ]
    # colors = ['#828282', '#464867', '#3C7C3D', '#DB7012', '#9D2B2B']

    # # jili-31
    # path_list = [
    #     xiaorong____n1, xiaorong____n2,
    #     xiaorong1____1, xiaorong1____2,
    #     xiaorong_2___1, xiaorong_2___2,
    #     xiaorong12___1, xiaorong12___2,
    #     xiaorong1_3__1, xiaorong1_3__2, xiaorong1_3__3,
    #     xiaorong_2_4_1, xiaorong_2_4_2,
    #     xiaorong1234_1, xiaorong5
    # ]
    # title_list = [
    #     'random1', 'random2',
    #     'An___1', 'An___2',
    #     'A_N__1', 'A_N__2',
    #     'AnN__1', 'AnN__2',
    #     'An_w_1', 'An_w_2', 'An_w_3',
    #     'A_N_W1', 'A_N_W2',
    #     'AnNwW1', 'AnNwW2',
    # ]
    # separate = [
    #     2, 2, 2, 2, 3, 2, 2
    # ]
    # separate_title_list = [
    #     'random',
    #     'n',
    #     'nw',
    #     'n+N',
    #     'n+nw',
    #     'N+NW',
    #     'n+N+nw+NW',
    # ]
    # colors = ['#828282', '#464867', '#1A37AB', '#5A3797', '#3C7C3D', '#DB7012', '#9D2B2B']

    # # jili-32
    # path_list = [
    #     far8action2A__1, far8action2A__2,
    #     far8action6AD_1, far8action6AD_2,
    #     # far8action8ADM1, far8action8ADM2, far8action8ADM3,
    #     far1, far2
    # ]
    # title_list = [
    #     'attack1', 'attack2',
    #     'attack+defense1', 'attack+defense2',
    #     # 'attack+defense+mix1', 'attack+defense+mix2', 'attack+defense+mix3',
    #     'mix1', 'mix2',
    # ]
    # separate = [
    #     2, 2, 2
    # ]
    # separate_title_list = [
    #     'attack',
    #     'attack+defense',
    #     # 'attack+defense+mix',
    #     'attack+defense+mix'
    # ]
    # colors = ['#9D2B2B', '#1A37AB', '#1F5B25']

    # # mirror -
    # path_list = [
    #     far1, far2,
    #     farmirror1, farmirror2,
    #     # farmirroractA__1, farmirroractA__2,
    #     # farmirroractA_M1, farmirroractA_M2,
    # ]
    # title_list = [
    #     'origin1', 'origin2',
    #     'mirror1', 'mirror2',
    #     # 'mix1', 'mix2',
    #     # 'mix1', 'mix2',
    # ]
    # separate = [
    #     2, 2,
    # ]
    # separate_title_list = [
    #     'origin_map',
    #     'mirror_map',
    #     # 'mirror2',
    #     # 'mirror3',
    # ]
    # colors = ['#464867', '#9D2B2B']

    # # jili-4 -
    # path_list = [
    #     canshu1010,
    #     canshu1050,
    #     canshu1090,
    #     canshu5010,
    #     canshu5050,
    #     canshu5090,
    #     canshu9010,
    #     canshu9050,
    #     canshu9090
    #     # farmirroractA__1, farmirroractA__2,
    #     # farmirroractA_M1, farmirroractA_M2,
    # ]
    # title_list = [
    #     'α0.1 γ0.1',
    #     'α0.1 γ0.5',
    #     'α0.1 γ0.9',
    #     'α0.5 γ0.1',
    #     'α0.5 γ0.5',
    #     'α0.5 γ0.9',
    #     'α0.9 γ0.1',
    #     'α0.9 γ0.5',
    #     'α0.9 γ0.9',
    #     # 'mix1', 'mix2',
    #     # 'mix1', 'mix2',
    # ]
    # separate = [
    #     1, 1, 1, 1, 1, 1, 1, 1, 1
    # ]
    # separate_title_list = [
    #     'α0.1 γ0.1',
    #     'α0.1 γ0.5',
    #     'α0.1 γ0.9',
    #     'α0.5 γ0.1',
    #     'α0.5 γ0.5',
    #     'α0.5 γ0.9',
    #     'α0.9 γ0.1',
    #     'α0.9 γ0.5',
    #     'α0.9 γ0.9',
    #     # 'mirror2',
    #     # 'mirror3',
    # ]
    # colors = ['#EEB4B4', '#D6494F', '#9D2B2B', '#D9E7CD', '#9CC184', '#669D62', '#9CE4D9', '#00C5CD', '#1A37AB']

    # # chongyong-4 -
    # path_list = [
    #     dist1, dist2,
    #     distchongyong1, distchongyong2,
    #     distmirror2, distmirror4,
    #     distmirrorchongyong1, distmirrorchongyong2
    #     # farmirroractA__1, farmirroractA__2,
    #     # farmirroractA_M1, farmirroractA_M2,
    # ]
    # title_list = [
    #     'origin1', 'origin2',
    #     're-origin1', 're-origin2',
    #     'mirror2', 'mirror4',
    #     're-mirror1', 're-mirror2',
    #     # 'mix1', 'mix2',
    #     # 'mix1', 'mix2',
    # ]
    # separate = [
    #     2, 2, 2, 2
    # ]
    # separate_title_list = [
    # 'origin_training',
    # 'origin_test',
    # 'mirror_training',
    # 'mirror_test',
    #     # 'mirror2',
    #     # 'mirror3',
    # ]
    # colors = ['#C02D06', '#FF6A6A', '#1F5B25', '#669D62']

    # chongyong-8 -
    path_list = [
        far1, far2,
        fartest1, fartest2,
        farmirror1, farmirror2,
        farmirrortest1, farmirrortest2
        # farmirroractA__1, farmirroractA__2,
        # farmirroractA_M1, farmirroractA_M2,
    ]
    title_list = [
        'origin1', 'origin2',
        're-origin1', 're-origin2',
        'mirror2', 'mirror4',
        're-mirror1', 're-mirror2',
        # 'mix1', 'mix2',
        # 'mix1', 'mix2',
    ]
    separate = [
        2, 2, 2, 2
    ]
    separate_title_list = [
        'origin_training',
        'origin_test',
        'mirror_training',
        'mirror_test',
        # 'mirror2',
        # 'mirror3',
    ]
    colors = ['#C02D06', '#FF6A6A', '#1F5B25', '#669D62']



    # colors = ['red', 'blue', 'green', 'orange', 'red', 'blue', 'green', 'orange', 'red']
    # colors = ['#828282', '#464867', '#1C1C1C',
    #           '#EEB4B4', '#FF6A6A', '#D6494F', '#C02D06', '#9D2B2B',
    #           '#FFEBCD', '#FFDAB9', '#E8A00F', '#CD8500', '#DB7012', '#8B4500',
    #           '#E7E5CC', '#fdf2a5', '#EFBE06',
    #           '#D9E7CD', '#C2D6A4', '#9CC184', '#669D62', '#3C7C3D', '#1F5B25', '#1E3D14', '#192813',
    #           '#7FFFD4', '#66CDAA', '#62959D',
    #           '#9CE4D9', '#8EE5EE', '#00C5CD', '#34AEEE', '#1A37AB',
    #           '#FAE6F2', '#FBCDFE', '#BF87C8', '#9B70CA', '#5A3797', '#37245F'
    #           ]
    # colors = ['#EEB4B4', '#D6494F', '#9D2B2B', '#D9E7CD', '#9CC184', '#669D62', '#9CE4D9', '#00C5CD', '#1A37AB']

    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', ]

    # draw_manage(path_8far_mirror_1)
    multi_draw_manage(path_list, title_list, separate, separate_title_list, colors, linestyles)
