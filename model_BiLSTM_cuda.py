import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.io

from tqdm import tqdm

import random


class StructData():
    def __init__(self, file_name, batch_size=32, shuffle=False):
        self.data = scipy.io.loadmat(file_name)  # 读取mat文件
        self.sample = self.data['SAMPLE_Rf'][0]
        self.lb_input = self.data['Lb_input_RF'][0]
        self.ub_input = self.data['Ub_input_RF'][0]

        l = len(self.sample)
        l_train = int(np.floor(l * 0.8))
        l_test = l - l_train

        self.sample_batches = l

        self.train_batches = l_train
        self.test_batches = l_test

        self.train_data = self.sample[:l_train]
        self.test_data = self.sample[l_train:]

    def load_train_data(self):
        for i in range(self.train_batches):
            obs = self.train_data[i]['input']
            state = self.train_data[i]['output']

            obs = (obs - self.lb_input) / (self.ub_input - self.lb_input)

            obs = torch.FloatTensor(obs).cuda()
            state = torch.FloatTensor(state).cuda()

            yield obs, state

    def load_test_data(self):
        for i in range(self.test_batches):
            obs = self.test_data[i]['input']
            state = self.test_data[i]['output']

            obs = (obs - self.lb_input) / (self.ub_input - self.lb_input)

            obs = torch.FloatTensor(obs).cuda()
            state = torch.FloatTensor(state).cuda()

            yield obs, state

    def load_sample_data(self):
        self.sample_ = self.data['SAMPLE_Rf'][0]
        for i in range(len(self.sample)):
            obs = self.sample_[i]['input']
            state = self.sample_[i]['output']

            obs = (obs - self.lb_input) / (self.ub_input - self.lb_input)

            obs = torch.FloatTensor(obs).cuda()
            state = torch.FloatTensor(state).cuda()

            yield obs, state


'''
Model_LeakyReLU
'''


class Model_LeakyReLU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.num_directions = 1

        self.state_embedding = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.LeakyReLU(),
                                             nn.Linear(self.hidden_size,
                                                       self.num_directions * self.num_layers * self.hidden_size))
        self.obs_embedding = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.LeakyReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size))
        self.state_decoding = nn.Sequential(nn.Linear(self.num_directions * self.hidden_size, self.hidden_size),
                                            nn.LeakyReLU(), nn.Linear(self.hidden_size, self.output_size))

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=False)
        self.state0 = torch.zeros([1, self.input_size]).cuda()

    def init_state0(self, state0):
        self.state0 = state0

    def forward(self, obs):
        h0 = self.state_embedding(self.state0).view(-1, self.hidden_size)
        c0 = torch.zeros([self.num_directions * self.num_layers, self.hidden_size]).cuda()

        obs = self.obs_embedding(obs).view(-1, self.hidden_size)
        output, _ = self.lstm(obs, (h0, c0))

        state = self.state_decoding(output).cuda()

        return state


'''
Model_ReLU
'''


class Model_ReLU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.num_directions = 1

        self.state_embedding = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
                                             nn.Linear(self.hidden_size,
                                                       self.num_directions * self.num_layers * self.hidden_size))
        self.obs_embedding = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size))
        self.state_decoding = nn.Sequential(nn.Linear(self.num_directions * self.hidden_size, self.hidden_size),
                                            nn.ReLU(), nn.Linear(self.hidden_size, self.output_size))

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=False)
        self.state0 = torch.zeros([1, self.input_size]).cuda()

    def init_state0(self, state0):
        self.state0 = state0

    def forward(self, obs):
        h0 = self.state_embedding(self.state0).view(-1, self.hidden_size)
        c0 = torch.zeros([self.num_directions * self.num_layers, self.hidden_size]).cuda()

        obs = self.obs_embedding(obs).view(-1, self.hidden_size)
        output, _ = self.lstm(obs, (h0, c0))

        state = self.state_decoding(output).cuda()

        return state


'''
Model_Tanh
'''


class Model_Tanh(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.num_directions = 1

        self.state_embedding = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.Tanh(),
                                             nn.Linear(self.hidden_size,
                                                       self.num_directions * self.num_layers * self.hidden_size))
        self.obs_embedding = nn.Sequential(nn.Linear(self.input_size, self.hidden_size), nn.Tanh(),
                                           nn.Linear(self.hidden_size, self.hidden_size))
        self.state_decoding = nn.Sequential(nn.Linear(self.num_directions * self.hidden_size, self.hidden_size),
                                            nn.Tanh(), nn.Linear(self.hidden_size, self.output_size))

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=False)
        self.state0 = torch.zeros([1, self.input_size]).cuda()

    def init_state0(self, state0):
        self.state0 = state0

    def forward(self, obs):
        h0 = self.state_embedding(self.state0).view(-1, self.hidden_size)
        c0 = torch.zeros([self.num_directions * self.num_layers, self.hidden_size]).cuda()

        obs = self.obs_embedding(obs).view(-1, self.hidden_size)
        output, _ = self.lstm(obs, (h0, c0))

        state = self.state_decoding(output).cuda()

        return state