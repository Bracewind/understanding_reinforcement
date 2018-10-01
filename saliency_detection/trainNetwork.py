import torch.optim as optim
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
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(self):
        self.lr = 0.0001
        self.discountFactor = 0.99
        self.stateMemory = []
        self.replayMemory = ReplayMemory(10000)

    def train(self, model, batch_size):
        if len(self.replayMemory.memory) < batch_size:
            return
        transitions = self.replayMemory.sample(batch_size)

        batch = []
        for transition in transitions:
            batch.append(Transition(*zip(*transitions)))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        for memory in transitions:
            state_action_values = model(memory.state)
            next_state_values = model(memory.next_state)
            expected_state_action_values = memory.reward + self.discountFactor*next_state_values
        '''
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()'''


