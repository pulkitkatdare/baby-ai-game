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

        self.lossFn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.001,
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

    def train(image, string, labels):
        """
        Expects image, string and labels to be in tensor form
        """

        # TODO: need to sample batches from input data
        # To begin with, can start with batch size 1

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(inputs)
        loss = self.lossFn(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().data[0]



def encodeStr(string, maxLen=64, numCharCodes=27):
    string = string.lower()

    strArray = np.zeros(shape=(maxStrLen, numCharCodes), dtype='float32')

    for idx, ch in enumerate(str):
        if ch >= 'a' and ch <= 'z':
            chNo = ord(ch) - ord('a')
        elif ch == ' ':
            chNo = ord('z') - ord('a') + 1
        assert chNo < numCharCodes, '%s : %d' % (ch, chNo)
        strArray[idx, chNo] = 1

    return strArray
