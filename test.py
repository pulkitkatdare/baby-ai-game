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

        dist = Categorical(action_probs)
        action = dist.sample()
        #log_prob = dist.log_prob(action)

        return action.data[0]

    def action_log_prob(self, obs, action):

        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action_probs, state_value = self(Variable(obs))
        dist = Categorical(action_probs)

        action = Variable(torch.LongTensor(1).fill_(action))
        #print(action)

        log_prob = dist.log_prob(action)

        return log_prob

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

def cross_entropy(yHat, y):
    if yHat == 1:
      return -log(y)
    else:
      return -log(1 - y)

def train_model(model, rollout):

    losses = []

    for step in range(0, len(rollout.obs)):
        rollout_action = rollout.action[step]
        #model_action = model.select_action(rollout.obs[step])
        #print(model_action)

        log_prob = model.action_log_prob(rollout.obs[step], rollout_action)

        #print(-log_prob)

        losses.append(-log_prob)

    optimizer.zero_grad()
    loss = torch.cat(losses).sum()

    print(loss)

    loss.backward()
    optimizer.step()

def eval_model(model, env, num_evals=64):
    sum_reward = 0
    obs = env.reset()

    for n in range(0, num_evals):
        while True:
            action = model.select_action(obs)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                break

    return sum_reward / num_evals

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

    if len(success_rollouts) > 0:
        #rollout = random.choice(success_rollouts)

        best_rollout = None
        for r in success_rollouts:
            if not best_rollout or len(r.obs) < len(best_rollout.obs):
                best_rollout = r

        train_model(model, best_rollout)
        r = eval_model(model, env)
        #print(len(success_rollouts))
        print(len(best_rollout.obs))
        print(r)
