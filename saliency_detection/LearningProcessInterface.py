import torch

from trainNetwork import *
from matplotlib import pyplot

class LearningProcessInterface(object):
    def __init__(self, env, model):
        self.game = env
        self.playerModel = model
        self.memoryGame = ReplayMemory(10000)
        self.trainer = Trainer(self.playerModel, self.memoryGame)

    def oneEpisode(self, get_history=False, render=False):
        if get_history:
            history = {'ins': [], 'reward': [], 'image':[], 'logits': [], 'values': [], 'outs': [], 'hx': [], 'cx': []}
        state = torch.Tensor(self.game.reset()).cuda()
        done = False
        while not done:

            action = self.playerModel.chooseAction(state)

            next_state, reward, done, expert_policy = self.game.step(action)
            next_state = torch.Tensor(next_state).cuda()
            if render:
                print(self.game.metadata['render.modes'])
                self.game.render()

            self.memoryGame.push(state, action, next_state, reward, done)

            if get_history:
                history['ins'].append(state)
                history['reward'].append(reward)
                image = self.game.render('rgb_array')
                history['image'].append(image)

            state = next_state

        if get_history:
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
                history = self.oneEpisode(True)
                reward += sum(history['reward'])
            print(reward/nbTestBeforeMean)

    def playGameAndGetHistory(self):
        return self.oneEpisode(True)

    def displayGameWithModel(self):
        self.oneEpisode(render=True)

    def __del__(self):
        self.game.close()
