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
    draw_clusters_result.drawClustersHealthResult(path, maxStep)
    print('局部奖励可视化输出完成')


def multi_draw_manage(path_list, title_list, separate, separate_title_list):
    draw_game_data_plot.drawWinRateLineChart(path_list, title_list, separate, separate_title_list)
    draw_game_data_plot.drawFitnessLineChart(path_list, title_list, separate, separate_title_list)
    draw_game_data_plot.drawParetoChart(path_list, title_list, separate, separate_title_list)


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
    path_list = [path_8far_1500_action1_s10_1, path_8far_1500_action1_s10_2,
                 path_8far_1500_action2_24_s10_1, path_8far_1500_action2_24_s10_2,
                 path_8far_1500_action7_s10_1, path_8far_1500_action7_s10_2,
                 path_8far_1, path_8far_2,
                 path_8far_mirror_act12345_1, path_8far_mirror_act12345_2,
                 path_8far_mirror_act123458_1, path_8far_mirror_act123458_2,
                 path_8far_mirror_1,
                 # path_8far_
                 # path_8faraction7_to_8faraction7_1
                 # path_unfair_test_1, path_unfair_test_2
                 # path_unfair_test_2
                 # path_list = [path_action8_8far_1, path_action8_8far_2
                 ]
    title_list = [
        'action1_1', 'action1_2',
        'action2_1', 'action2_2',
        'action7_1', 'action7_2',
        'action9_1', 'action9_2',
        'actionATK_mirror_1', 'actionATK_mirror_2',
        'actionMIX_mirror_1', 'actionMIX_mirror_2',
        'actionALL_mirror',

        # 'unfair_test_1', 'unfair_test_2',
        # 'unfair_test_2',
    ]
    separate = [2, 2, 2, 2, 2, 2, 1,
                # 2,
                # 1
                ]
    separate_title_list = [
        'action1',
        'action2',
        'action7',
        'action9',
        'actionATK_mirror',
        'actionMIX_mirror',
        'actionALL_mirror',
        # 'unfair_model_transfer',
        # 'unfair_test',
    ]

    # draw_manage(path_8far_mirror_1)
    multi_draw_manage(path_list, title_list, separate, separate_title_list)
