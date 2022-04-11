#!/usr/bin/env python
#-*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np

## Hyperparatemers ##
k_epoch = 10
learning_rate = 0.001
eps_clip = 0.2

class PPO(nn.Module):
    def __init__(self):
        # split dim_in dim_out
        self.data = []
        self.fc1 = nn.Linear(32*64*64, 16384)
        self.fc2 = nn.Linear(16384, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.pi_mu = nn.Linear(256, 1)
        self.pi_std = nn.Linear(256, 1)
        self.v_hat = nn.Linear(256, 1)
        
        # input = 3 x 256 x 256
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten() # output = 32 * 64 * 64
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = self.conv(x)
        x = self.relu(self.fc1)
        x = self.relu(self.fc2)
        x = self.relu(self.fc3)
        x = self.relu(self.fc4)
        pi_mu = torch.tanh(self.pi_mu)
        pi_std = F.softplus(self.pi_std)
        
        return pi_mu, pi_std
    
    def value(self, x):
        x = self.conv(x)
        x = self.relu(self.fc1)
        x = self.relu(self.fc2)
        x = self.relu(self.fc3)
        x = self.relu(self.fc4)
        value_hat = self.v_hat(x)
        
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
        
        data = self.make_batch()
        
        advantage_list = []
        advantage = 0
        
        for s_i, a_i, r_i in s, a, r:
            advantage = r_i - self.value(s_i)
            advantage_list.append(advantage)
        
        advantage = torch.tensor(advantage, dtype=torch.float)
        
        # ratio calculation
        
        for i in k_epoch:
            for mini_batch in data:
                s, a, r, adv = mini_batch
                
                mu, std = self.pi(s)
                dist = Normal(mu, std)
                log_prob = dist.log_prob(a)
                ratio = torch.exp(log_prob - old_log_prob)
                obj_1 = ratio * adv
                obj_2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * adv
                loss = -torch.min(obj_1, obj_2)
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
