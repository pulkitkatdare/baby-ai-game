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
    def __init__(self):
        super().__init__()

        #self.num_inputs = reduce(operator.mul, input_size, 1)
        self.img_size = 7 * 7 * 3

        # Two output classes (true/false)
        self.img_fc = nn.Linear(self.img_size, 64)

        self.rnn = nn.LSTM(27, 64, 1)

        self.fc2 = nn.Linear(64 + 64, 64)
        self.fc3 = nn.Linear(64, 2)

        self.lossFn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.4
        )

    def forward(self, image, string):
        batch_size = image.size(0)

        image = image.view(-1, self.img_size)

        # Pytorch expects string to have shape (len, batch, features)
        string = string.unsqueeze(0).transpose(0, 1)

        img_out = self.img_fc(image)

        rnn_out, (rnn_hidden, rnn_cell) = self.rnn(string)
        rnn_hidden = rnn_hidden.squeeze(0)

        x = torch.cat((rnn_hidden, img_out), 1)
        x = F.relu(self.fc2(x))
        class_scores = self.fc3(x)
        class_probs = F.softmax(class_scores, dim=1)

        return class_probs

    def train(self, image, string, label):
        """
        Expects image, string and labels to be in tensor form
        """

        image = Variable(torch.from_numpy(image).float())
        string = Variable(torch.from_numpy(string).float())
        label = Variable(torch.from_numpy(label).long())


        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(image, string)
        loss = self.lossFn(outputs, label)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().data[0]



def encodeStr(string, maxLen=64, numCharCodes=27):
    string = string.lower()

    strArray = np.zeros(shape=(maxLen, numCharCodes), dtype='float32')

    for idx, ch in enumerate(string):
        if ch >= 'a' and ch <= 'z':
            chNo = ord(ch) - ord('a')
        elif ch == ' ':
            chNo = ord('z') - ord('a') + 1
        assert chNo < numCharCodes, '%s : %d' % (ch, chNo)
        strArray[idx, chNo] = 1

    return strArray
