import torch

from trainNetwork import *


class LearningProcessInterface(object):
    def __init__(self, env, model):
        self.game = env
        self.playerModel = model
        self.memoryGame = ReplayMemory(10000)
        self.trainer = Trainer(self.playerModel, self.memoryGame)


    def oneEpisode(self, render=False):
        state = torch.Tensor(self.game.reset()).cuda()
        done = False
        final_reward = 0
        while not done:

            action = self.playerModel.chooseAction(state)

            next_state, reward, done, expert_policy = self.game.step(action)
            if render:
                self.game.render()

            self.memoryGame.push(state, action, torch.Tensor(next_state).cuda(), reward)

            state = torch.Tensor(next_state).cuda()
            final_reward += reward


        return final_reward

    def trainModel(self, batchSize, epoch):
        for j in range(epoch // batchSize):
            for i in range(batchSize):
                self.oneEpisode()
                if (i>0):
                    self.trainer.train(batchSize)

    def testModel(self, nbTestBeforeMean, nbTest):
        for i in range(nbTest):
            reward = 0
            for j in range(nbTestBeforeMean):
                reward += self.oneEpisode()
            print(reward)

    def playGameWithModel(self):
        print(self.oneEpisode(render=True))

    def __del__(self):
        self.game.close()
