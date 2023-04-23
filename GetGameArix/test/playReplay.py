import io
import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units, replay, gfile
from pysc2.env import sc2_env, run_loop, environment
from pysc2.bin import *
import csv
import cv2

def main(unused_argv):
    with gfile.Open('./../replay/MarineMicro_MvsM_4_2023-04-05-08-46-53.SC2Replay', 'rb') as f:
        replay_data = f.read()
        # replay_io = io.BytesIO()
        # replay_io.write(replay_data)
        replay.get_replay_version(replay_data)


if __name__ == "__main__":
    app.run(main)