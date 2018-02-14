import os
import gym
from gym import spaces

import sys
directory=os.getcwd()
directory=directory+'/gym_aigame/envs'
if not directory in sys.path:
    sys.path.insert(0,directory)
print("adding directory path")
import teacher
import gym_minigrid


def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        
        if env_id=='MultiRoom-Teacher':
            env=gym.make('MiniGrid-MultiRoom-N6-v0')
            env=teacher.Teacher(env)
        else:
            env = gym.make(env_id)

        env.seed(seed + rank)

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        #if isinstance(env.observation_space, spaces.Dict):
        #    print('dic state not supported. we use a Flat wrapper')
        #    env = FlatObsWrapper(env)
        #env=WrapPyTorch(env)
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
        print('observation', observation)
        return [observation['image'].transpose(2, 0, 1),observation['mission']]
    
    #def _observation(self, observation):
        #return observation['image'].transpose(2, 0, 1)
    
    
    
