import torch.optim as optim
import torch.nn as nn

import utils

import numpy as np


from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.memory[np.random.random_integers(0, len(self.memory)-1)])
        return batch

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(self, model, replayMemory):
        self.lr = 0.01
        self.discountFactor = 0.99
        self.stateMemory = []
        self.replayMemory = replayMemory

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.loss = nn.MSELoss()

    def train(self, batch_size):
        if len(self.replayMemory.memory) < batch_size:
            return
        transitions = self.replayMemory.sample(batch_size)

        for memory in transitions:
            state_action_value = self.model(memory.state)
            next_state_values = self.model(memory.next_state)
            expected_state_action_value = memory.reward + self.discountFactor*max(next_state_values)

            action_values = utils.tensor_deepcopy(state_action_value)
            action_values[memory.action] = expected_state_action_value

            # Compute Mean square error loss
            value_loss = self.loss(state_action_value, action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()


