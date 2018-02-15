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

def random_crop(img, rnd_x, rnd_y, pad):
    orig_y = img.shape[0]
    orig_x = img.shape[1]
    img_ch = img.shape[2]

    img_padded = np.zeros((orig_y + pad*2, orig_x + pad*2, img_ch))
    for i in range(img_ch):
        img_padded[:,:,i] = np.pad(img[:,:,i], ((pad,pad),(pad,pad)), mode='constant')
    
    img_crop = img_padded[rnd_y:orig_y+rnd_y, rnd_x:orig_x+rnd_x]
    
    return img_crop

def transpose_img(img):
    for i in range(img.shape[2]):
        img[:,:,i] = np.transpose(img[:,:,i])
    return img

def rotate_img(img, num):
    img = np.rot90(img, num)
    return img

def train_transforms(img, msk, u=0.5, pad=8):
    # verticle flips
    if np.random.random() > u:
        img = np.flip(img, 0)
        msk = np.flip(msk, 0)
    
    # horizontal flips
    if np.random.random() > u:
        img = np.flip(img, 1)
        msk = np.flip(msk, 1)
    
    # random rotations
    if np.random.random() > u:
        num = np.random.randint(1,4)
        img = rotate_img(img, num)
        msk = rotate_img(msk, num)

    #if np.random.random() > u:
    #    img = transpose_img(img)
    #    msk = transpose_img(msk)

    # random crops
    if np.random.random() > u:
        # do the random cropping
        rnd_x = np.random.randint(0, pad*2)
        rnd_y = np.random.randint(0, pad*2)
        img = random_crop(img, rnd_x, rnd_y, pad)
        msk = random_crop(msk, rnd_x, rnd_y, pad)

    return img, msk

def train_transforms_multi_out(img, msk, dst, u=0.5, pad=8):
    # verticle flips
    if np.random.random() > u:
        img = np.flip(img, 0)
        msk = np.flip(msk, 0)
        dst = np.flip(dst, 0)
    
    # horizontal flips
    if np.random.random() > u:
        img = np.flip(img, 1)
        msk = np.flip(msk, 1)
        dst = np.flip(dst, 1)
    
    # random rotations
    if np.random.random() > u:
        num = np.random.randint(1,4)
        img = rotate_img(img, num)
        msk = rotate_img(msk, num)
        dst = rotate_img(dst, num)

    #if np.random.random() > u:
    #    img = transpose_img(img)
    #    msk = transpose_img(msk)

    # random crops
    if np.random.random() > u:
        # do the random cropping
        rnd_x = np.random.randint(0, pad*2)
        rnd_y = np.random.randint(0, pad*2)
        img = random_crop(img, rnd_x, rnd_y, pad)
        msk = random_crop(msk, rnd_x, rnd_y, pad)
        dst = random_crop(dst, rnd_x, rnd_y, pad)

    return img, msk, dst

def normalize_img(img_arr):
    img_arr = img_arr.astype(np.float32)
    img_min = np.min(img_arr)
    img_max = np.max(img_arr)
    img_mean = np.mean(img_arr)
    img_out = (img_arr - img_mean) / (img_max - img_min + 1e-12)
    return img_out

def standardize_img(img_arr):
    '''
    https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence
    '''
    bgr = img_arr[:,:,[2,1,0]].astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(lab[:,:,0])
    #print(out.shape)
    if out.mean() > 127:
        out = 255. - out
        
    return out