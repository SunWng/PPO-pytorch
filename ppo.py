#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class PPO(nn.Module):
    def __init__(self, dim_in, dim_out):
        self.fc1 = nn.Linear(dim_in, 256)
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU()

    def pi(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        action = self.fc2(x) # 6D output (3 positions + 3 positions)
        return action

    def get_data(self, data):
        pass

    def make_batch(self):
        pass

    def train(self):
        pass
