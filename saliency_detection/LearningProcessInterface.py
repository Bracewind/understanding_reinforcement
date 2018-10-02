import torch

from trainNetwork import *
from matplotlib import pyplot

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
            next_state = torch.Tensor(next_state).cuda()
            if render:
                self.game.render()

            self.memoryGame.push(state, action, next_state, reward, done)

            state = next_state
            final_reward += reward


        return final_reward

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
                reward += self.oneEpisode()
            print(reward/nbTestBeforeMean)

    def playGameWithModel(self):
        print(self.oneEpisode(render=True))

    def __del__(self):
        self.game.close()
