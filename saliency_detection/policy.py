# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from torch import optim

import utils
import glob
import numpy as np
import random
import math
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

        self.steps_done = 0


    def forward(self, inputs):
        x = F.elu(self.input(inputs))
        x = F.elu(self.deepFC1(x))
        return self.actionChosen(x)

    def save(self, path):
        print("model saved at : ", path)
        torch.save(self.state_dict(), path)

    def try_load(self, save_file):
        self.load_state_dict(torch.load(save_file))

    def chooseAction(self, state):
        actions = self(state).cpu().tolist()

        randomValue = random.random()
        if randomValue < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            return actions.index(max(actions))

    def calculateLoss(self, state, action, next_state, reward, done, discountFactor):
        state_action_value = self(state)

        expected_state_action_value = reward
        if next_state is not None:
            expected_state_action_value += discountFactor * max(self(next_state))

        # copy the state action value given by the network
        action_values = utils.tensor_deepcopy(state_action_value)

        # change it to correspond to the state action value we want the network learn
        action_values[action] = expected_state_action_value

        # Compute Mean square error loss between what the network think and what we want
        loss = nn.MSELoss()
        value_loss = loss(state_action_value, action_values)
        return value_loss


class REINFORCE(torch.nn.Module):
    def __init__(self, nbState, num_actions):
        super(REINFORCE, self).__init__()
        self.nbState = nbState
        self.num_actions = num_actions
        self.input = nn.Linear(nbState, 24).cuda()
        self.deepFC1 = nn.Linear(24, 24).cuda()
        self.actionChosen = nn.Linear(24, num_actions).cuda()

        self.optimizer = optim.Adam(self.parameters(), 0.01)



    def forward(self, state):
        x = F.elu(self.input(state))
        x = F.elu(self.deepFC1(x))
        return F.softmax(self.actionChosen(x))

    def chooseAction(self, state):
        actionProbabilities = self(state).cpu().detach().numpy()

        value = np.random.random()
        currentProbaSum = 0
        currentAction = 0
        while value > actionProbabilities[currentAction] + currentProbaSum:
            currentProbaSum += actionProbabilities[currentAction]
            currentAction += 1
        return currentAction

    def trainModel(self, batch):
        state_tensor = torch.FloatTensor([elem.cpu().tolist() for elem in batch['states']]).cuda()
        reward_tensor = torch.FloatTensor(batch['rewards_discounted']).cuda()
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch['actions']).cuda()

        # Calculate loss
        logprob = torch.log(self(state_tensor))
        selected_logprobs = reward_tensor * \
                            logprob[np.arange(len(action_tensor)), action_tensor]
        loss = -selected_logprobs.mean()

        self.optimizer.zero_grad()
        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()

    def update_rule(self):
        pass