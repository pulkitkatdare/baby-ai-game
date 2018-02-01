#!/usr/bin/env python3

import time
import random
import numpy as np
import operator
from functools import reduce

import gym
import gym_minigrid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, input_size, obs_high, num_actions):
        super().__init__()

        self.num_inputs = reduce(operator.mul, input_size, 1)
        self.obs_high = obs_high

        self.a_fc1 = nn.Linear(self.num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.a_fc3 = nn.Linear(64, num_actions)
        self.v_fc3 = nn.Linear(64, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, inputs):
        # Reshape the input so that it is one-dimensional
        inputs = inputs.view(-1, self.num_inputs)

        # Rescale observation values in [0,1]
        inputs = inputs / self.obs_high

        # Don't put a relu on the last layer, because we want to avoid
        # zero probabilities
        x = F.relu(self.a_fc1(inputs))
        x = F.relu(self.a_fc2(x))
        action_scores = self.a_fc3(x)
        action_probs = F.softmax(action_scores, dim=1)

        state_value = self.v_fc3(x)

        return action_probs, state_value

    def select_action(self, obs):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action_probs, state_value = self(Variable(obs))

        # print(
        #     'action_probs: ',
        #     action_probs.volatile,
        #     'state_value: ',
        #     state_value.volatile)

        # print(action_probs)

        dist = Categorical(action_probs)
        action = dist.sample()

        #model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.data[0]

class Rollout:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.total_reward = 0

def random_rollout(env):
    global num_steps
    global num_episodes

    obs = env.reset()

    rollout = Rollout()

    while True:

        action = random.randint(0, env.action_space.n - 1)

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

def train_model(model, rollout):

    pass




#    optimizer.zero_grad()
#    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
#    loss.backward()
#    optimizer.step()







env = gym.make('MiniGrid-Empty-6x6-v0')
#env = gym.make('MiniGrid-DoorKey-5x5-v0')

model = Model(
    env.observation_space.shape,
    env.observation_space.high[0][0][0],
    env.action_space.n
)

optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.4)

num_steps = 0
num_episodes = 0
success_rollouts = []

while True:
    rollout = random_rollout(env)

    if rollout.total_reward > 0:
        print('success: reward=%s, length=%s' % (rollout.total_reward, len(rollout.obs)))
        success_rollouts.append(rollout)
