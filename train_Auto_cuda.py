import os
import numpy as np
# import matplotlib.pyplot as plt

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable

from tqdm import tqdm

from model_BiLSTM_cuda import StructData
from model_BiLSTM_cuda import Model_LeakyReLU
from model_BiLSTM_cuda import Model_ReLU
from model_BiLSTM_cuda import Model_Tanh


# model = Model().cuda()

for k1 in range(2):
    for k2 in range(4):
        hidden_size = int(16 * math.pow(2, k2))
        num_layers = int(k1 + 1)

        for k3 in range(3):
            print('%d' % hidden_size)

            if k3 == 0:
                # input_size hidden_size num_layers output_size
                model = Model_LeakyReLU(13, hidden_size, num_layers, 5).cuda()
                fun_name = 'LeakyReLU'

            if k3 == 1:
                # input_size hidden_size num_layers output_size
                model = Model_ReLU(13, hidden_size, num_layers, 5).cuda()
                fun_name = 'ReLU'

            if k3 == 2:
                # input_size hidden_size num_layers output_size
                model = Model_Tanh(13, hidden_size, num_layers, 5).cuda()
                fun_name = 'Tanh'

            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = ExponentialLR(optimizer, gamma=0.95)

            print('Load training examples')
            data = StructData('Sample_FL_Python_2_4.mat')

            print('Start training')

            path = './TrainingResults-V1/' + fun_name + '_%d_%d' % (num_layers, hidden_size)

            if os.path.exists(path) == False:
                os.makedirs(path)


            train_losses = []
            valid_losses = []
            best_loss = 1e100

            for epoch in range(300):
                print('Epoch', epoch)

                print('training %d %d' % (num_layers, hidden_size))

                model.train()

                train_loss = []

                pbar = tqdm(data.load_train_data(), total=data.train_batches)
                for obs, state in pbar:
                    pred = model(obs)
                    loss = F.cross_entropy(pred[-1, :], state[0, :])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_description('train_loss= %.4g' % loss.item())

                    train_loss.append(loss.item())

                train_loss = np.mean(train_loss)

                model.eval()

                print(train_loss)

                train_losses.append(train_loss)

                with torch.no_grad():
                    valid_loss = []

                    pbar = tqdm(data.load_test_data(), total=data.test_batches)
                    for obs, state in pbar:
                        # model.init_state0(obs[0])
                        pred = model(obs)
                        loss = F.cross_entropy(pred[-1, :], state[0, :])

                        pbar.set_description('valid_loss= %.4g' % loss.item())

                        valid_loss.append(loss.item())

                    valid_loss = np.mean(valid_loss)
                    print(valid_loss)

                    valid_losses.append(valid_loss)

                if valid_loss < best_loss:
                    NNname = path + '/best_' + fun_name + '_%d_%d.pt' % (num_layers, hidden_size)
                    torch.save(model.state_dict(), NNname)
                    best_loss = valid_loss


            NNname = path + '/models_' + fun_name + '_%d_%d.pth' % (num_layers, hidden_size)
            torch.save(model, NNname)
            NNname = path + '/train_loss_' + fun_name + '_%d_%d.txt' % (num_layers, hidden_size)
            np.savetxt(NNname, train_losses)
            NNname = path + '/valid_loss_' + fun_name + '_%d_%d.txt' % (num_layers, hidden_size)
            np.savetxt(NNname, valid_losses)
