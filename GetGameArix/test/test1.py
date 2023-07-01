import numpy as np
import pandas as pd
import time

# 设置随机数池，确保每次运行程序随机数都一样以便观察
np.random.seed(2)
# 状态数量
N_STATES = 6
# 行为列表(0表示向左，1表示向右)
ACTIONS = [0, 1]
# 贪婪策略，90%选择最高分数，10%选择随机
EPSILON = 0.9
# 学习率
ALPHA = 0.1
# 未来奖励的衰减率
LAMBDA = 0.9
# 最大回合数
MAX_EPISODES = 13
# 刷新时间
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    """
    构建Q表，行表示状态(用行索引表示小人所在位置)，列表示行为
    :param n_states: 状态个数
    :param actions: 行为数组
    :return: Q表
    """
    q_table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return q_table


def choose_action(state, q_table):
    """
    根据当前状态和Q表选择要进行的动作
    :param state: 当前状态
    :param q_table: Q表
    :return: 动作名
    """
    # 从Q表中选取当前状态的所有行为及对应Q值
    state_actions = q_table.iloc[state, :]
    # 1.(1 - EPSILON)的概率随机选取行为
    # 2.所有行为Q值为0时(刚开始的时候)随机选取行为
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    # 其他情况按照Q值最大的行为进行选取
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(state, action):
    """
    从当前状态state采取了action行为之后，获得新的状态和当前这个行为的奖励
    :param state: 当前状态
    :param action: 当前行为
    :return: 当前行为的奖励
    """
    if action == 1:
        # 认为靠近T并且选择向右走时即到达T，游戏结束
        if state == N_STATES - 2:
            state_ = 'terminal'
            reward = 1
        else:
            state_ = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state_ = state
        else:
            state_ = state - 1
    return state_, reward


def update_env(state, episode, step_counter):
    """
    更新环境，一共N_STATES-2个-，1个o，1个T
    o表示小人所在的位置，T表示目标坐在位置，-表示小人可以移动到的地方
    当N_STATES=6的时候，打印效果为 --o--T
    :param state: 需要更新到的状态(小人的位置)
    :param episode: 第几回合
    :param step_counter: 当前回合中第几步
    """
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        # print('\r                     ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        # 设置了打印的间隔时间方便观看小人位置变化
        time.sleep(FRESH_TIME)


def q_learning():
    """
    模型训练的主循环
    """
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        # 每一回合先初始化环境
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)
        # 开始当前回合的步骤循环
        while not is_terminated:
            # 根据当前状态和Q表选择行为
            action = choose_action(state, q_table)
            # 根据当前状态和选择的行为在环境中获得执行后的状态和奖励值
            state_, reward = get_env_feedback(state, action)
            # 当前状态对应选择的行为的预测奖励值
            q_predict = q_table.iloc[state, action]
            # 当前状态对应选择的行为执行后的实际奖励值
            if state_ != 'terminal':
                q_target = reward + LAMBDA * q_table.iloc[state_, :].max()
            else:
                q_target = reward
                is_terminated = True
            # 预测奖励值的更新 = 学习率 * 实际值与预测值的差值
            q_table.iloc[state, action] += ALPHA * (q_target - q_predict)
            # 更新状态和环境
            state = state_
            update_env(state, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    table = q_learning()