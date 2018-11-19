import torch.optim as optim
import torch
import utils

import numpy as np
import random


from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(self, model, replayMemory):
        self.lr = 0.01
        self.discountFactor = 0.99
        self.stateMemory = []
        self.replayMemory = replayMemory
        self.model = model

        self.optimizer = optim.RMSprop(self.model.parameters())

    # history used for seeing loss
    def train(self, model, batch_size, history = []):
        optimizer = optim.Adam(model.parameters(), self.lr)
        if len(self.replayMemory.memory) < batch_size:
            return
        transitions = self.replayMemory.sample(batch_size)
        add_value_loss = 0

        for memory in transitions:

            value_loss = model.calculateLoss(memory.state, memory.action, memory.next_state, memory.reward, memory.done, self.discountFactor)

            add_value_loss += value_loss.item()

            # Optimize the model
            optimizer.zero_grad()
            value_loss.backward()
            optimizer.step()
        return add_value_loss/batch_size

    def trainDebug(self, model, batch_size):
        optimizer = optim.Adam(model.parameters(), self.lr)
        if len(self.replayMemory.memory) < batch_size:
            return
        transitions = self.replayMemory.sample(batch_size)
        add_value_loss = 0

        for memory in transitions:
            value_loss = model.calculateLoss(memory.state, memory.action, memory.next_state, memory.reward, memory.done,
                                             self.discountFactor)

            add_value_loss += value_loss.item()

            # Optimize the model
            optimizer.zero_grad()
            value_loss.backward()
            optimizer.step()
        return add_value_loss / batch_size
