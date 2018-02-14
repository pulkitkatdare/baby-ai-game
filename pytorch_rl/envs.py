import os
import numpy
import gym
from gym import spaces

try:
    import gym_minigrid
    import teacher
    from gym_minigrid.wrappers import *
except:
    pass

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        
        if env_id=='MultiRoom-Teacher':
            env=gym.make('MiniGrid-MultiRoom-N6-v0')
            env=teacher.Teacher(env)
        else:
            env = gym.make(env_id)

        env.seed(seed + rank)

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = FlatObsWrapper(env)

        return env

    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
     
        return [observation['image'].transpose(2, 0, 1),observation['advice']]
    
    #def _observation(self, observation):
        #return observation['image'].transpose(2, 0, 1)
    
    
    
