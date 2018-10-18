from policy import *
from utils import *
from saliency import *
from visualize_concept import *

import gym
import time

PATH_TO_FOLDER = "/home/gregoire/Documents/Cours/CursusRecherche/Analogie/understanding_reinforcement/"

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
    print(Transition(4, 5, 6, 7, True))
    for i in range(10):
        memory.push(i, 5, 6, 7, True)


def testTraining():
    agent = DQN(4, 2)
    memory = ReplayMemory(10000)
    for i in range(10):
        memory.push(torch.Tensor([i, 5, 6, 7]).cuda(), 0, torch.Tensor([i+1, 5, 6, 7]).cuda(), i+1, True)
        memory.push(torch.Tensor([i, 5, 6, 7]).cuda(), 1, torch.Tensor([i + 1, 5, 6, 7]).cuda(), i-1, True)
    trainer = Trainer(agent, memory)
    trainer.train(5, [])


def testLearning():
    model = DQN(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    learning_process.trainModel(32, 2000, seeAdvance=100)
    learning_process.testModel(20, 20)
    for i in range(5):
        learning_process.displayGameWithModel()
    model.save(PATH_TO_FOLDER + "test_ckpt.pth.tar")


def test_saliency():
    model = DQN(4, 2)
    model.try_load(PATH_TO_FOLDER + "test_ckpt.pth.tar")
    env = gym.make('CartPole-v0')

    learning_process = LearningProcessInterface(env, model)
    learning_process.calculate_saliency()

def test_video():
    for alpha in range(1,20):
        make_movie('CartPole-v0', PATH_TO_FOLDER + "test_ckpt.pth.tar", alpha=alpha/100, suffix_name=alpha/100)
def test_mean_values():
    model = DQN(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    learning_process.trainModel(32, 2000, seeAdvance=100)
    learning_process.testModel(20, 20)
    for i in range(5):
        learning_process.displayGameWithModel()
    
    learnin_process.setMeanRanges()
    
if __name__ == '__main__':
    #testUtils()
    #testDQN()
    #testMemory()
    #testTraining()
    #testLearning()
    #test_saliency()
    #test_video()
    test_mean_values()
