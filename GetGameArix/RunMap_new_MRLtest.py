import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MultiAgentMarkovGame(gym.Env):
    def __init__(self, num_agents):
        super(MultiAgentMarkovGame, self).__init__()
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(2)  # 两个动作：0和1
        self.observation_space = spaces.Discrete(3)  # 三个状态：0、1和2

    def reset(self):
        self.state = self.observation_space.sample()  # 初始状态
        return self.state

    def step(self, actions):
        rewards = [0] * self.num_agents

        # 根据动作更新状态和奖励
        if actions[0] == actions[1]:  # 如果两个智能体的动作相同
            if actions[0] == 0:  # 如果动作为0
                rewards = [2, 2]  # 奖励为2
            else:
                rewards = [1, 1]  # 如果动作为1，则奖励为1
        else:
            rewards[0] = 0
            rewards[1] = 3

        done = False  # 游戏是否结束

        return self.state, rewards, done, {}

    def render(self):
        pass  # 实现可视化逻辑，例如打印当前状态等


# 创建环境
env = MultiAgentMarkovGame(num_agents=2)

# 重置环境
state = env.reset()

done = False
while not done:
    # 随机选择动作
    actions = [env.action_space.sample() for _ in range(env.num_agents)]

    # 执行动作
    next_state, rewards, done, _ = env.step(actions)

    # 更新状态
    state = next_state

    # 可视化
    env.render()