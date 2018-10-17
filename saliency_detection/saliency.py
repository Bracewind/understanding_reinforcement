# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur


# parameters contains sign and alpha value : the add of alpha% of range max value
apply_gaussian = lambda x, parameters: x + parameters[0]*parameters[1]

def apply_perturbation(x, epsilon, alpha, range_value):
    return x + epsilon*alpha*range_value



def get_mask(center, size, r):
    y, x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size); mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()


def gaussian_perturbation(center, r):
    return np.random.normal(center, r)


def run_through_model(model, history, ix, interp_func=None, mask=None, blur_memory=None, mode='actor'):
    if mask is None:
        im = prepro(history['ins'][ix])
    else:
        assert(interp_func is not None, "interp func cannot be none")
        im = interp_func(prepro(history['ins'][ix]).squeeze(), mask).reshape(1,80,80) # perturb input I -> I'
    tens_state = torch.Tensor(im)
    state = Variable(tens_state.unsqueeze(0), volatile=True)
    hx = Variable(torch.Tensor(history['hx'][ix-1]).view(1,-1))
    cx = Variable(torch.Tensor(history['cx'][ix-1]).view(1,-1))
    if blur_memory is not None: cx.mul_(1-blur_memory) # perturb memory vector
    return model((state, (hx, cx)))[0] if mode == 'critic' else model((state, (hx, cx)))[1]


# ix is the numero of the current step
# history['ins'][ix] is the state at the ix step
def run_through_model_2(model, history, ix, input_to_change=None, perturbation=None, arg_perturbation=None, range_value=None):
    if input_to_change is None:
        state = history['ins'][ix]
    else:
        assert (perturbation is not None, "perturbation cannot be none when we want to change an input")
        state = perturbation(history['ins'][ix], *arg_perturbation)
    state = Variable(state, volatile=True).cuda()
    return model(state)


def score_frame(model, history, ix, r, d, interp_func, mode='actor'):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(80/d)+1,int(80/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,80,d):
        for j in range(0,80,d):
            mask = get_mask(center=[i,j], size=[80,80], r=r)
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode)
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).data[0]
    pmax = scores.max()
    scores = imresize(scores, size=[80,80], interp='bilinear').astype(np.float32)
    return pmax * scores / scores.max()


def score_state(model, history, ix, interp_func, range_values, alpha, total_reward):
    L = run_through_model_2(model, history, ix)
    scores = np.zeros(model.nbState)
    for input_feature in range(model.nbState):
        l_high = run_through_model_2(model, history, ix, input_feature, interp_func, [1, alpha, range_values[input_feature]])
        l_low = run_through_model_2(model, history, ix, input_feature, interp_func, [-1, alpha, range_values[input_feature]])
        l = (l_high + l_low)/2
        scores[input_feature] = (L-l).pow(2).sum().mul(0.5).data[0]
    return scores


def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    I = atari.astype('uint16')
    I[35:195,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I


def create_concept_saliency(saliency):
    S = saliency
    S -= S.min()
    return S

def create_image_representation(saliency, image_length):
    space_between_input = 40
    length_square = 40
    center = length_square//2
    currentCenter = center
    image = np.zeros([40, image_length, 3])
    for index in range(len(saliency)):
        score = saliency[index]
        nbcolumn = score // 255

        for x in range(length_square):
            for y in range(length_square):
                if abs(center-x) + abs(center-y) <= nbcolumn:
                    image[x, index*(length_square) + y, 0] = int(254)
                else:
                    image[x, index*(length_square) + y, 0] = 0

        currentCenter += 40
        image[0, index*(length_square)] = int(254)

    image = np.array(image, dtype="int")
    return image

def get_env_meta(env_name):
    meta = {}
    if env_name=="Pong-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 500
    elif env_name=="Breakout-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 300
    elif env_name=="SpaceInvaders-v0":
        meta['critic_ff'] = 400 ; meta['actor_ff'] = 400
    else:
        print('environment "{}" not supported'.format(env_name))
    return meta