import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop, environment


class Agent(base_agent.BaseAgent):
    actions = ("do_nothing",
               "attack"
               )

    def get_my_units_by_type(self, obs, unit_type):
        # print(unit_type)
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def get_my_center_position(self, obs, unit_type):
        position = (0, 0)
        my_units = [unit for unit in obs.observation.raw_units
         if unit.unit_type == unit_type
         and unit.alliance == features.PlayerRelative.SELF]
        for unit in my_units:
            print(unit.x, unit.y)
            position += (unit.x, unit.y)
        return (position[0] / len(my_units), position[1] / len(my_units))

    def step(self, obs):
        super(Agent, self).step(obs)

        # if obs.first():
        #     np.save('dict_first.npy', obs)
        #     a = np.load('dict_first.npy', allow_pickle=True)
        # if obs.last():
        #     np.save('dict_end.npy', obs)
        #     b = np.load('dict_end.npy', allow_pickle=True)
        #     print(b[3].keys())
        # print(obs.observation)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        # marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        print(self.get_my_center_position(obs, units.Terran.Marine))
        # if len(marines) > 0:
        #     center_marines =
        #     # distances = self.get_distances(obs, marines, attack_xy)
        #     marine = marines[np.argmax(distances)]
        #     print(marine)
        #     return actions.RAW_FUNCTIONS.Attack_pt(
        #         "now", marine.tag, (attack_xy[0], attack_xy[1]))
        return actions.RAW_FUNCTIONS.no_op()

class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        # action = random.choice(self.actions)
        action = self.actions[1]
        return getattr(self, action)(obs)


def main(unused_argv):
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    # agent2 = SmartAgent()
    try:
        with sc2_env.SC2Env(
                map_name="MarineMicro",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         # sc2_env.Agent(sc2_env.Race.terran)],
                         sc2_env.Bot(sc2_env.Race.terran,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                realtime=True,
                step_mul=48,
                disable_fog=True,
                # save_replay_episodes=1,
                # replay_dir="D:/白春辉/实验平台/pysc2-tutorial/replay"
        ) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
