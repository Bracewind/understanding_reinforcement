from __future__ import print_function
import warnings;

warnings.filterwarnings('ignore')  # mute warnings, live dangerously

import matplotlib.pyplot as plt
import matplotlib as mpl;

mpl.use("Agg")
import matplotlib.animation as manimation

import gym, os, sys, time, argparse

sys.path.append('..')
from policy import *
from LearningProcessInterface import *
from saliency import *
import numpy as np

alpha = 0.05

#all the function transform torch.Tensor in tab, we do the same here
def identity():
    return (lambda tab: [elem for elem in tab])

def create_one_perturbation(index, perturbation):
    #return tab, modifying only the index element with the perturbation
    return (lambda tab: [ [tab[currentIndex], perturbation(tab[currentIndex])][currentIndex == index] for currentIndex in range(len(tab))])

#those function change a parameter
#compute with those before to change multiple parameters
def high_range_based_perturbation(range_value):
    return (lambda x: x + alpha*range_value)

def low_range_based_perturbation(range_value):
    return (lambda x: x - alpha*range_value)

def high_temporal_variation_based_perturbation(temporal_variation):
    return (lambda x: x + temporal_variation)

def low_temporal_variation_based_perturbation(temporal_variation):
    return (lambda x: x - temporal_variation)

def calculate_saliency(nbOfState, numFrame, model, history, learning_process_interface):
    range_values, mean_temporal_difference = learning_process_interface.rangeStats(30)

    mean_values_saliencies = [0] * nbOfState
    scores = [0]*nbOfState
    ref_values = run_through_model_2(model, history, numFrame, identity())


    # calculate saliencies
    for parameter in range(nbOfState):
        high_change_in_parameter = high_temporal_variation_based_perturbation(mean_temporal_difference[parameter])
        low_change_in_parameter = low_temporal_variation_based_perturbation(mean_temporal_difference[parameter])
        values_saliency_high = run_through_model_2(model, history, numFrame,
                                                   create_one_perturbation(parameter, high_change_in_parameter))

        values_saliency_low = run_through_model_2(model, history, numFrame,
                                                  create_one_perturbation(parameter, low_change_in_parameter))

        mean_values_saliencies = (values_saliency_high + values_saliency_low) / 2
        scores[parameter] = (ref_values - mean_values_saliencies).pow(2).sum().mul(0.5).data[0]

    values_dict = {'CartPosition': scores[0], 'CartVelocity': scores[1],
                   'PoleAngle': scores[2], 'PoleVelocityAtTip': scores[3]}

    print("scores : ", scores)
    return scores

def make_movie(env_name, checkpoint='*.tar', num_frames=150, first_frame=0, resolution=75, \
               save_dir='./movies/', density=5, radius=5, prefix='default', overfit_mode=False, suffix_name="", alpha=0.01):
    # set up dir variables and environment
    load_dir = '{}{}/'.format('overfit-' if overfit_mode else '', env_name.lower())
    meta = get_env_meta(env_name)
    env = gym.make(env_name) if not overfit_mode else OverfitAtari(env_name, load_dir + 'expert/',
                                                                   seed=0)  # make a seeded env

    # set up agent
    nbOfState = env.observation_space.shape[0]
    model = DQN(nbOfState, num_actions=env.action_space.n)
    model.try_load(checkpoint)

    learning_process_interface = LearningProcessInterface(env, model)

    # get a rollout of the policy
    movie_title = "{}-{}-{}_saliencyConcept_Tempdiff_alpha_{}.mp4".format(prefix, num_frames, env_name.lower(), suffix_name)
    print('\tmaking movie "{}" using checkpoint at {}{}'.format(movie_title, load_dir, checkpoint))
    max_ep_len = first_frame + num_frames + 1
    torch.manual_seed(0)
    history = learning_process_interface.calculate_saliency()

    # make the movie!
    start = time.time()
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, artist='DQN', comment='mechanical-saliency-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)

    prog = '';
    total_frames = len(history['image'])
    f = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    with writer.saving(f, save_dir + movie_title, resolution):
        total_reward = 0

        for i in range(num_frames):
            # ix is the current frame
            ix = first_frame + i
            if ix < total_frames:  # prevent loop from trying to process a frame ix greater than rollout length

                image = history['image'][ix].copy()
                scores = calculate_saliency(nbOfState, ix, model, history, learning_process_interface)
                saliency = create_image_representation(scores, image.shape[1])
                print(saliency.shape)
                print(image.shape)
                finalFrame = np.concatenate((image, saliency))
                plt.imshow(finalFrame)
                plt.title(env_name.lower(), fontsize=15)
                writer.grab_frame()
                f.clear()

                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100 * i / min(num_frames, total_frames)), end='\r')
    print('\nfinished.')


# user might also want to access make_movie function from some other script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='Breakout-v0', type=str, help='gym environment')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=20, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=150, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str,
                        help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-c', '--checkpoint', default='*.tar', type=str,
                        help='checkpoint name (in case there is more than one')
    parser.add_argument('-o', '--overfit_mode', default=False, type=bool,
                        help='analyze an overfit environment (see paper)')
    args = parser.parse_args()

    make_movie(args.env, args.checkpoint, args.num_frames, args.first_frame, args.resolution,
               args.save_dir, args.density, args.radius, args.prefix, args.overfit_mode)
