import torch

from trainNetwork import *
from matplotlib import pyplot
from saliency import *

import numpy as np

import time

class LearningProcessInterface(object):
    def __init__(self, env, model):
        self.game = env
        self.playerModel = model
        self.memoryGame = ReplayMemory(10000)
        self.trainer = Trainer(self.playerModel, self.memoryGame)
        self.mean_ranges = [0,0,0,0]

    def oneEpisode(self, get_history=False, render=False, calculate_saliency=False):
        if calculate_saliency:
            get_history = True
        if get_history:
            history = {'ins': [], 'reward': [], 'image': [], 'image_saliency': [], 'done': [], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
        state = torch.Tensor(self.game.reset()).cuda()
        done = False
        while not done:

            action = self.playerModel.chooseAction(state)

            next_state, reward, done, expert_policy = self.game.step(action)
            next_state = torch.Tensor(next_state).cuda()

            self.memoryGame.push(state, action, next_state, reward, done)

            if get_history:
                history['ins'].append(state)
                history['reward'].append(reward)
                history['done'].append(done)

            if calculate_saliency:
                value_episode = {'ins':None, 'reward': None, 'done': False}
                value_episode['ins'] = state.cpu().detach().numpy()
                value_episode['reward'] = reward
                value_episode['done'] = done

                range_values = [4.8,100,80,100]
                values_saliency = score_state(self.playerModel, value_episode, apply_perturbation, range_values, totalReward)
                # TODO: no hardcode of meaning
                values_dict = {'CartPosition': values_saliency[0], 'CartVelocity': values_saliency[1], 'PoleAngle': values_saliency[2], 'PoleVelocityAtTip': values_saliency[3]}
                image = self.game.render(mode='rgb_array').tolist()
                length_image = len(image[0])

                saliency = create_image_representation(values_saliency, length_image)
                [image.append(saliency_value) for saliency_value in saliency]

                history['image_saliency'].append(np.array(image))

            state = next_state
            # time.sleep(1)

        if get_history:
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

    def trainModel(self, batchSize, epoch, seeAdvance):
        iter = 0
        history = []
        for j in range(epoch // batchSize):
            for i in range(batchSize):
                self.oneEpisode()
                self.trainer.train(batchSize, history)

                iter += 1
                if iter % seeAdvance == 0:
                    print(iter, "iteration done")

        pyplot.plot(range(len(history)), history)
        pyplot.show()

    def testModel(self, nbTestBeforeMean, nbTest):
        for i in range(nbTest):
            reward = 0
            for j in range(nbTestBeforeMean):
                history = self.oneEpisode(get_history=True, render=False)
                reward += sum(history['reward'])
            print(reward/nbTestBeforeMean)

    def playGameAndGetHistory(self):
        return self.oneEpisode(True)

    def displayGameWithModel(self):
        self.oneEpisode(render=True)

    def calculate_saliency(self):
        return self.oneEpisodeSaliency()

    def setMeanRanges(self):
        mean_ranges = [0,0,0,0]
        nbOfEpisodes = 10
        for episode in range(nbOfEpisodes):
            history = self.oneEpisodeSaliency()
            for state in range(len(history('ins'))-1):
                nbFeatures = 4
                for feature in range(nbFeatures):
                    mean_ranges[feature] = history('ins')[state][feature + 1] - history('ins')[state][feature]
            mean_ranges /= len(history('ins'))-1
        self.mean_ranges = mean_ranges
        print("mean_ranges = ", mean_ranges) 
            
    def __del__(self):
        self.game.close()
