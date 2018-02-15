import os
import cv2
import math
import random
import pickle

import numpy as np
from scipy.spatial.distance import jaccard

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.filters import sobel
from skimage.feature import canny
from skimage.transform import resize

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsample.transforms import RandomAffine, RandomFlip
from torch.utils.data.sampler import RandomSampler

def mean_iou(predictions, targets):
    predictions = predictions.data.cpu().numpy()
    targets = targets.data.cpu().numpy()

    num_img = predictions.shape[0]

    all_ious = []
    for img in range(num_img):
        pred = predictions[img]
        targ = targets[img]

        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
        for thresh in thresholds:
            # set pred threshold
            pred_thresh = np.where(pred > thresh, 1, 0)

            iou = calc_iou(pred_thresh, targ)

            # calculate IOU, add to list
            all_ious.append(iou)

    return np.mean(all_ious)

def calc_iou(msk_a, msk_b):
    msk_a = msk_a.astype(np.bool)
    msk_b = msk_b.astype(np.bool)

    intersection = np.logical_and(msk_a, msk_b)
    union = np.logical_or(msk_a, msk_b)

    return intersection.sum() / float(union.sum() + 1e-12) 

# plot the loss curves
def plot_losses(model_name, exp_name, train_losses, valid_losses, ious):
    plt.title('{} {}'.format(model_name, exp_name))
    plt.plot(train_losses, '--', label='train loss')
    plt.plot(valid_losses, label='valid loss')
    plt.ylim([0, 1.5])
    plt.legend(loc=3)
    plt.twinx()
    plt.plot(ious, label='IOU')
    plt.legend(loc=2)
    plt.savefig('../plots/{}_{}_train_curves.png'.format(model_name, exp_name), dpi=400)
    #plt.show()
