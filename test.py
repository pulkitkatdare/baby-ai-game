#!/usr/bin/env python3

import random
import numpy as np
import gym
import gym_minigrid

class Rollout:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.total_reward = 0

#env = gym.make('MiniGrid-Empty-8x8-v0')
env = gym.make('MiniGrid-DoorKey-5x5-v0')
num_actions = env.action_space.n

num_steps = 0
num_episodes = 0

def random_rollout(env):
    global num_steps
    global num_episodes

    obs = env.reset()

    rollout = Rollout()

    while True:

        action = random.randint(0, num_actions - 1)

        rollout.obs.append(obs)
        rollout.action.append(action)

        obs, reward, done, info = env.step(action)
        num_steps += 1

        rollout.reward.append(reward)
        rollout.total_reward += reward

        if done:
            break

    num_episodes += 1
    return rollout

success_rollouts = []

while True:
    rollout = random_rollout(env)

    if rollout.total_reward > 0:
        print('success: reward=%s, length=%s' % (rollout.total_reward, len(rollout.obs)))
        success_rollouts.append(rollout)
