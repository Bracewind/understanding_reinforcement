# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import glob
import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]


class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions):
        super(NNPolicy, self).__init__()
        self.nbState = channels
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 5 * 5, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 5 * 5)
        hx, cx = self.lstm(x, (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)

    def try_load(self, save_dir, checkpoint='*.tar'):
        paths = glob.glob(save_dir + checkpoint) ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class DQN(torch.nn.Module):
    def __init__(self, nbState, num_actions):
        super(DQN, self).__init__()
        self.nbState = nbState
        self.num_actions = num_actions
        self.input = nn.Linear(nbState, 24).cuda()
        self.deepFC1 = nn.Linear(24, 24).cuda()
        self.actionChosen = nn.Linear(24, num_actions).cuda()

        self.epsilon = 0.01

        self.initWeight()

    def initWeight(self):
        nn.init.constant_(self.input.weight, 0)
        nn.init.constant_(self.deepFC1.weight, 0)
        nn.init.constant_(self.actionChosen.weight, 0)


    def forward(self, inputs):
        inputs = inputs
        x = F.elu(self.input(inputs))
        x = F.elu(self.deepFC1(x))
        return self.actionChosen(x)

    def save(self, path):
        print("model saved at : ", path)
        torch.save(self.state_dict(), path)

    def try_load(self, save_file):
        self.load_state_dict(torch.load(save_file))

    def chooseAction(self, state):
        valueAction = self(state)
        action = (valueAction == max(valueAction)).nonzero()[0][0].cpu().numpy().tolist()

        if (self.epsilon > np.random.random()):
            action = np.random.random_integers(0, self.num_actions-1)

        return action
