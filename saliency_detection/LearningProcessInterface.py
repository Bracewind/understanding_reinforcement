import torch

from trainNetwork import *
from matplotlib import pyplot
from saliency import *

import numpy as np
import matplotlib.pyplot as plt

import time

class LearningProcessInterface(object):
    def __init__(self, env, model):
        self.game = env
        self.playerModel = model
        self.memoryGame = ReplayMemory(10000)
        self.trainerModel = Trainer(self.playerModel, self.memoryGame)

    def oneEpisode(self, get_history=False, render=False):
        history = {'state': [], 'next_state': [], 'reward': [], 'action': [], 'image_saliency': [], 'done': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
        state = torch.Tensor(self.game.reset()).cuda()
        done = False
        while not done:

            action = self.playerModel.chooseAction(state)

            next_state, reward, done, expert_policy = self.game.step(action)
            next_state = torch.Tensor(next_state).cuda()

            self.memoryGame.push(state, action, next_state, reward, done)

            if render:
                self.game.render()

            history['state'].append(state)
            history['action'].append(action)
            history['reward'].append(reward)
            history['done'].append(done)
            history['next_state'].append(next_state)

            state = next_state
            # time.sleep(1)

        return history

    def oneEpisodeSaliency(self):
        history = {'ins': [], 'reward': [], 'image': [], 'done': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
        state = torch.Tensor(self.game.reset()).cuda()
        done = False
        while not done:
            action = self.playerModel.chooseAction(state)

            next_state, reward, done, expert_policy = self.game.step(action)
            next_state = torch.Tensor(next_state).cuda()

            self.memoryGame.push(state, action, next_state, reward, done)

            history['ins'].append(state)
            history['reward'].append(reward)
            history['done'].append(done)
            image = self.game.render(mode='rgb_array')
            history['image'].append(image)

            state = next_state
            # time.sleep(1)

        return history

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i]
                      for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def doBatch(self, batch_size):
        batch = {'rewards': [], 'rewards_discounted':[], 'states':[], 'next_states': [], 'actions':[], 'total_rewards':[], 'done': []}
        for episode in range(batch_size):
            history = self.oneEpisode()
            batch['rewards'].extend(history['reward'])
            batch['rewards_discounted'].extend(self.discount_rewards(history['reward']))
            batch['states'].extend(history['state'])
            batch['next_states'].extend(history['next_state'])
            batch['actions'].extend(history['action'])
            batch['total_rewards'].append(sum(history['reward']))
            batch['done'].extend(history['done'])
        return batch

    def trainModel(self, batch_size, nb_batch):
        loss_history = []
        for current_batch in range(nb_batch):
            self.doBatch(batch_size)
            history = self.trainerModel.train(self.playerModel, batch_size)
            if current_batch % 50 == 0:
                print("current_batch : ", current_batch)
            loss_history.append(history)

        plt.plot(range(len(loss_history)), loss_history)
        plt.show()


    def testModel(self, nbTestBeforeMean, nbTest):
        for i in range(nbTest):
            reward = 0
            for j in range(nbTestBeforeMean):
                history = self.oneEpisode(get_history=True, render=False)
                reward += sum(history['reward'])
            print(reward/nbTestBeforeMean)

    def playGameAndGetHistory(self):
        return self.oneEpisode(True)

    def rangeStats(self, nbGame):
        state = self.game.reset()
        nbStateParameter = len(state)
        rangeValues = [[state[indexState], state[indexState]] for indexState in range(nbStateParameter)]
        meanVariation = [0 for i in range(nbStateParameter)]

        numberState = [0, 0, 0, 0]

        for i in range(nbGame):
            lastState = None
            history = self.oneEpisode()
            for state in history['state']:
                for stateFeature in range(nbStateParameter):
                    newMaxRange = max(rangeValues[stateFeature][1], state[stateFeature])
                    newMinRange = min(rangeValues[stateFeature][0], state[stateFeature])
                    rangeValues[stateFeature] = [newMinRange, newMaxRange]

                    if (lastState is not None):
                        meanVariation[stateFeature] = (meanVariation[stateFeature]*numberState[stateFeature] + (state[stateFeature]-lastState[stateFeature])**2)/(numberState[stateFeature]+1)
                        numberState[stateFeature] += 1
                lastState = state

        return rangeValues, meanVariation



    def displayGameWithModel(self):
        self.oneEpisode(render=True)

    def calculate_saliency(self):
        return self.oneEpisodeSaliency()

    def __del__(self):
        self.game.close()
