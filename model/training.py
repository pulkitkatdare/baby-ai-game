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
    def __init__(self, input_size):
        super().__init__()

        self.num_inputs = reduce(operator.mul, input_size, 1)

        # Two output classes (true/false)
        self.fc1 = nn.Linear(self.num_inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.0005,
            momentum=0.4
        )

    def forward(self, image, string):
        # Reshape the input so that it is one-dimensional
        #image = Variable(torch.from_numpy(image).float().unsqueeze(0))
        image = Variable(torch.from_numpy(image).float())
        image = image.view(-1, self.num_inputs)

        # TODO: process string with RNN


        x = F.relu(self.fc1(image))
        x = F.relu(self.fc2(x))

        class_scores = self.fc3(x)
        class_probs = F.softmax(class_scores, dim=1)

        return class_probs

    # TODO: implement fn to get boolean val?
    #def predict(self, image, string):

    """
    def select_action(self, obs):
        action_probs = self.forward(obs)
        dist = Categorical(action_probs)
        action = dist.sample()
        #log_prob = dist.log_prob(action)

        return action.data[0]
    """




# TODO: lookup torch.nn.CrossEntropyLoss()




def train_model(model, rollout):
    losses = []

    for step in range(0, len(rollout.obs)):
        rollout_action = rollout.action[step]
        log_prob = model.action_log_prob(rollout.obs[step], rollout_action)
        losses.append(-log_prob)

    model.optimizer.zero_grad()
    loss = torch.cat(losses).sum()

    loss.backward()
    model.optimizer.step()

    return loss.cpu().data[0]
