
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch import optim

import numpy as np


class Actor(torch.nn.Module):
    def __init__(self, nbState, dim_action):
        super(Actor, self).__init__()
        self.nb_state = nbState
        self.dim_action = dim_action

        self.init_actor()

    def init_actor(self):
        self.input_actor = nn.Linear(self.nb_state, 80).cuda()
        self.deepFC1_actor = nn.Linear(80, 40).cuda()
        self.output_actor = nn.Linear(40, self.dim_action).cuda()

    def forward(self, state):
        x = F.elu(self.input_actor(state))
        x = F.elu(self.deepFC1_actor(x))
        return F.softmax(self.output_actor(x))

    def chooseAction(self, state):
        actionProbabilities = self(state).cpu().detach().numpy()

        value = np.random.random()
        currentProbaSum = 0
        currentAction = 0
        while value > actionProbabilities[currentAction] + currentProbaSum:
            currentProbaSum += actionProbabilities[currentAction]
            currentAction += 1
        return currentAction

class Critic(torch.nn.Module):
    def __init__(self, nbState, dim_action):
        super(Critic, self).__init__()
        self.nb_state = nbState
        self.dim_action = dim_action

        self.init_critic()

        self.lr = 0.02

    def init_critic(self):
        self.input_critic = nn.Linear(self.nb_state, 40).cuda()
        self.deepFC1_critic = nn.Linear(40, 40).cuda()
        self.output_critic = nn.Linear(40, 1).cuda()

    def forward(self, state):
        x = F.elu(self.input_critic(state))
        x = F.elu(self.deepFC1_critic(x))
        return self.output_critic(x)

    def trainCritic(self, state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor, discount_factor):
        optimizer = optim.Adam(self.parameters(), self.lr)

        expected_reward = self(state_tensor)

        new_expected_reward = reward_tensor + (1-done_tensor)*discount_factor * self(next_state_tensor)

        loss = nn.MSELoss()
        value_loss = loss(expected_reward, new_expected_reward)

        # Optimize the model
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

class ActorCritic(torch.nn.Module):
    def __init__(self, nbState, dim_action):
        super(ActorCritic, self).__init__()
        self.nb_state = nbState
        self.dim_action = dim_action

        self.critic = Critic(nbState, dim_action)
        self.actor = Actor(nbState, dim_action)

        self.discount_factor = 0.99

    def forward(self, state):
        return self.actor(state)

    def chooseAction(self, state):
        return self.actor.chooseAction(state)

    def calculateLoss(self, batch):

        state_tensor = torch.FloatTensor([elem.cpu().tolist() for elem in batch['states']]).cuda()
        reward_tensor = torch.FloatTensor(batch['rewards']).cuda()
        # Actions are used as indices, must be LongTensor
        action_tensor = torch.LongTensor(batch['actions']).cuda()

        next_state_tensor = torch.FloatTensor([elem.cpu().tolist() for elem in batch['next_states']]).cuda()

        #we use the fact that done = 0 if false and done = 1 if true to train critic
        done_tensor = torch.FloatTensor([batch['done']]).cuda()

        self.critic.trainCritic(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor, self.discount_factor)

        # Calculate loss
        logprob = torch.log(self(state_tensor))

        advantage = self.discount_factor * (self.critic(next_state_tensor) - self.critic(state_tensor))*(1-done_tensor)

        selected_logprobs = (reward_tensor + self.discount_factor*self.critic(next_state_tensor)) * \
                             logprob[np.arange(len(action_tensor)), action_tensor]

        loss = -selected_logprobs.mean()
        return loss