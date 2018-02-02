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
    def __init__(self, input_size, num_actions):
        super().__init__()

        self.num_inputs = reduce(operator.mul, input_size, 1)
        self.num_actions = num_actions

        self.a_fc1 = nn.Linear(self.num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.a_fc3 = nn.Linear(64, num_actions)
        self.v_fc3 = nn.Linear(64, 1)

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.0005,
            momentum=0.4
        )

    def forward(self, obs):
        image = obs['image']

        # FIXME: what's the unsqueeze for?
        image = Variable(torch.from_numpy(image).float().unsqueeze(0))

        # Reshape the input so that it is one-dimensional
        image = image.view(-1, self.num_inputs)

        x = F.relu(self.a_fc1(image))
        x = F.relu(self.a_fc2(x))
        action_scores = self.a_fc3(x)
        action_probs = F.softmax(action_scores, dim=1)

        return action_probs

    def select_action(self, obs):
        action_probs = self.forward(obs)

        dist = Categorical(action_probs)
        action = dist.sample()
        #log_prob = dist.log_prob(action)

        return action.data[0]

    def action_log_prob(self, obs, action):
        action_probs = self.forward(obs)

        action = Variable(torch.LongTensor(1).fill_(action))

        dist = Categorical(action_probs)
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

    model.optimizer.zero_grad()
    loss = torch.cat(losses).sum()

    loss.backward()
    model.optimizer.step()

    return loss.cpu().data[0]

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
