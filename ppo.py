#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class PPO(nn.Module):
    def __init__(self, dim_in, dim_out):
        # split dim_in dim_out
        self.data = []
        self.fc1 = nn.Linear(dim_in, 256)
        self.fc2 = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU()

    def pi(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        action = self.fc2(x) # 3D output (position of eef)
        # add normalization
        return action
    
    def value(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        value_hat = self.fc2(x)
        return value_hat

    def get_data(self, data):
        self.data.append(data)

    def make_batch(self):
        s_list, a_list, r_list = [], [], []
        for transition in self.data:
            s_i, a_i, r_i = transition
            s_list.append(s_i)
            a_list.append(a_i)
            r_list.append(r_i)
        
        s, a, r = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list, dtype=torch.float), torch.tensor(r_list, torch.float)
        return s, a, r

    def train(self):
        s, a, r = self.make_batch()
