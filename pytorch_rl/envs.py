import os
import numpy
import gym
from gym import spaces

try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass

from multienv import MultiEnv

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = MultiEnv([
            'MiniGrid-GoToDoor-5x5-v0',
            'MiniGrid-PutNear-6x6-N2-v0'
        ])

        # TODO: add empty env as well?

        # DoorKey 5x5, 6x6

        #MiniGrid-Fetch-5x5-N2-v0
        #MiniGrid-Fetch-8x8-N3-v0

        #MiniGrid-PutNear-8x8-N3-v0

        #MiniGrid-GoToDoor-6x6-v0
        #MiniGrid-GoToDoor-8x8-v0





        env.seed(seed + rank)

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = FlatObsWrapper(env)

        return env

    return _thunk
