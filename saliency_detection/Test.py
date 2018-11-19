from policy import *
from ActorCritic import *
from utils import *
from saliency import *
from visualize_concept import *

import gym
import time

PATH_TO_FOLDER = "/home/gregoire/DocPartages/Cours/CursusRecherche/Analogie/understanding_reinforcement/"

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

    nbTimeActionChosenEqual0BeforeTraining = 0
    for i in range(20):
        actionChosen = agent.chooseAction([2, 5, 6, 7])
        nbTimeActionChosenEqual0BeforeTraining += (actionChosen == 0)

    for i in range(10):
        # in the memory, the reward is positive when taking 0 and negative if taking 1, should be taking 0
        memory.push(torch.Tensor([2, 5, 6, 7]).cuda(), 0, torch.Tensor([i+1, 5, 6, 7]).cuda(), 1, True)
        memory.push(torch.Tensor([2, 5, 6, 7]).cuda(), 1, torch.Tensor([i + 1, 5, 6, 7]).cuda(), -1, True)
    trainer = Trainer(agent, memory)
    trainer.train(5, [])

    nbTimeActionChosenEqual0AfterTraining = 0
    for i in range(20):
        actionChosen = agent.chooseAction([2, 5, 6, 7])
        nbTimeActionChosenEqual0AfterTraining += (actionChosen == 0)

    if nbTimeActionChosenEqual0AfterTraining < nbTimeActionChosenEqual0BeforeTraining:
        print("possible error, the network does not take more action 0")

def testLearning():
    model = DQN(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    learning_process.trainModel(32, 300)
    learning_process.testModel(20, 20)
    for i in range(5):
        learning_process.displayGameWithModel()

def test_range_stats():
    model = DQN(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    learning_process.trainModel(32, 30)
    learning_process.testModel(20, 20)
    print(learning_process.rangeStats(100))


def test_saliency():
    model = DQN(4, 2)
    model.try_load(PATH_TO_FOLDER + "test_ckpt.pth.tar")
    env = gym.make('CartPole-v0')

    learning_process = LearningProcessInterface(env, model)
    history = learning_process.calculate_saliency()
    frameModified = calculate_saliency(4, 0, model, history, learning_process)
    print("ok")
    plt.figure()
    plt.imshow(frameModified)
    plt.show()

def test_video_based_on_temporal_difference():
    make_movie('CartPole-v0', PATH_TO_FOLDER + "test_ckpt.pth.tar", alpha=alpha / 100, suffix_name=alpha / 100)

def test_video():
    for alpha in range(1,20):
        make_movie('CartPole-v0', PATH_TO_FOLDER + "test_ckpt.pth.tar", alpha=alpha/100, suffix_name=alpha/100)

def test_REINFORCE():
    model = REINFORCE(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    NB_BATCH = 50
    BATCH_SIZE = 5

    for current_batch in range(NB_BATCH):
        batch_history = learning_process.doBatch(BATCH_SIZE)
        model.trainModel(batch_history)
        if current_batch % 50 == 0:
            print("current_batch : ", current_batch)

    learning_process.testModel(10, 10)

def test_actor_critic1():
    model = ActorCritic(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    NB_BATCH = 300
    BATCH_SIZE = 10

    optimizer = optim.Adam(model.parameters(), 0.01)

    for current_batch in range(NB_BATCH):
        batch_history = learning_process.doBatch(BATCH_SIZE)
        loss = model.calculateLoss(batch_history)
        optimizer.zero_grad()
        # Calculate gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        if current_batch % 50 == 0:
            print("current_batch : ", current_batch)

    learning_process.testModel(10, 10)

def test_actor_critic():
    model = ActorCritic(4, 2)
    env = gym.make('CartPole-v0')
    learning_process = LearningProcessInterface(env, model)

    NB_BATCH = 300
    BATCH_SIZE = 10

    learning_process.trainModel(BATCH_SIZE, NB_BATCH)

    learning_process.testModel(10, 10)


if __name__ == '__main__':
    #testUtils()
    #testDQN()
    #testMemory()
    #testTraining()
    #testLearning()
    #test_range_stats()
    #test_saliency()
    test_video_based_on_temporal_difference()
    #test_video()
    #test_REINFORCE()
    #test_actor_critic1()