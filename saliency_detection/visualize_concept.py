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


def make_movie(env_name, checkpoint='*.tar', num_frames=150, first_frame=0, resolution=75, \
               save_dir='./movies/', density=5, radius=5, prefix='default', overfit_mode=False, suffix_name="", alpha=0.01):
    # set up dir variables and environment
    load_dir = '{}{}/'.format('overfit-' if overfit_mode else '', env_name.lower())
    meta = get_env_meta(env_name)
    env = gym.make(env_name) if not overfit_mode else OverfitAtari(env_name, load_dir + 'expert/',
                                                                   seed=0)  # make a seeded env

    # set up agent
    model = DQN(4, num_actions=env.action_space.n)
    model.try_load(checkpoint)

    learning_process_interface = LearningProcessInterface(env, model)

    # get a rollout of the policy
    movie_title = "{}-{}-{}_saliencyConcept_alpha_{}.mp4".format(prefix, num_frames, env_name.lower(), suffix_name)
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
                length_image = image.shape[1]

                range_values = [4.8, 100, 80, 100]
                total_reward += history['reward'][ix]
                values_saliency = score_state(model, history, ix, apply_perturbation, range_values,
                                              alpha, total_reward)

                values_dict = {'CartPosition': values_saliency[0], 'CartVelocity': values_saliency[1],
                               'PoleAngle': values_saliency[2], 'PoleVelocityAtTip': values_saliency[3]}

                saliency = create_image_representation(values_saliency, length_image)
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
