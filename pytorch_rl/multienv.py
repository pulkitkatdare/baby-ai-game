import gym
import gym_minigrid

"""
Plan:
- Multi-env stores single instances of multiple envs
- Returns progress data through info object
- Separate component logs/graphs progress on each env
"""

class MultiEnv(gym.Env):
    """
    Environment which samples from multiple environments, for
    multi-taks learning
    """

    def __init__(self, env_names):
        self.env_names = env_names

        self.env_list = []

        self.action_space = None
        self.observation_space = None

        for env_name in env_names:
            print(env_name)

            env = gym.make(env_name)

            if self.action_space is None:
                self.action_space = env.action_space

            if self.observation_space is None:
                self.observation_space = env.observation_space

            assert env.action_space == self.action_space
            assert isinstance(env.observation_space, gym.spaces.Dict)

            self.env_list.append(env)

        self.reward_range = (0, 1)

        self.cur_env_idx = 0

    def seed(self, seed):
        for env in self.env_list:
            env.seed(seed)

        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        return [seed]

    def reset(self):
        env = self.env_list[self.cur_env_idx]
        return env.reset()

    def step(self, action):
        env = self.env_list[self.cur_env_idx]

        obs, reward, done, info = env.step(action)

        # TODO: normalize rewards based on reward range




        info['multi_env'] = {
            'env_name': self.env_names[self.cur_env_idx]
        }

        # If the episode is done, sample a new environment
        if done:
            #self.cur_env_idx = self.np_random.randint(0, len(self.env_list))
            self.cur_env_idx = (self.cur_env_idx + 1) % len(self.env_list)
            #print(self.cur_env_idx)

            # TODO: keep track of running reward per episode?





        return obs, reward, done, info

    def render(self, mode='human', close=False):
        env = self.env_list[self.cur_env_idx]
        return env.render(mode, close)

    def close(self):
        for env in self.env_list:
            env.close()

        self.cur_env_idx = 0
        self.env_names = None
        self.env_list = None
