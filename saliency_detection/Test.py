from policy import *
from utils import *
from LearningProcessInterface import *

import gym
import time

def testUtils():
    t1 = torch.Tensor([1, 2]).cuda()
    t2 = tensor_deepcopy(t1)
    t2[0] = 8
    assert(t2[0] != t1[0])

def testDQN():
    agent = DQN(4, 2)
    torch.max(agent(torch.Tensor([4, 5, 6, 7]).cuda())).detach().cpu().numpy()


def testMemory():
    memory = ReplayMemory(10000)
    print(Transition(4, 5, 6, 7))
    for i in range(10):
        memory.push(i, 5, 6, 7)


def testTraining():
    agent = DQN(4, 2)
    memory = ReplayMemory(10000)
    for i in range(10):
        memory.push(torch.Tensor([i, 5, 6, 7]).cuda(), 0, torch.Tensor([i+1, 5, 6, 7]).cuda(), i+1)
        memory.push(torch.Tensor([i, 5, 6, 7]).cuda(), 1, torch.Tensor([i + 1, 5, 6, 7]).cuda(), i-1)
    trainer = Trainer(agent, memory)
    trainer.train(5)

def testLearning():
    model = DQN(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    learning_process.trainModel(32, 1000)
    learning_process.testModel(20, 20)
    time.sleep(1)


if __name__ == '__main__':
    testUtils()
    testDQN()
    testMemory()
    testTraining()
    testLearning()