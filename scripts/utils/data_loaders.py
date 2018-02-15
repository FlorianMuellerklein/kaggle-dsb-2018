import os
import cv2
import glob
import math
import random
import pickle

import numpy as np
from scipy.spatial.distance import jaccard

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import peak_local_max

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchsample.transforms import RandomAffine, RandomFlip
from torch.utils.data.sampler import SubsetRandomSampler

from utils.image_transformations import train_transforms, normalize_img, standardize_img, train_transforms_multi_out

TRAIN_PATH = '../data/stage1_train/'
TEST_PATH = '../data/stage1_test/'

class CroppedImgMaskDataset(data.Dataset):
    def __init__(self, img_paths, norm_fn, std_fn, imsize, pad=16, u=0.5, transforms=None):
        self.transforms = transforms
        self.img_paths = img_paths
        self.norm_fn = norm_fn
        self.std_fn = std_fn
        self.imsize = imsize
        self.pad = pad
        self.u = u

    def __getitem__(self, index):
        np.random.seed()

        # get the image path each nuclei
        img_path = self.img_paths[index]

        # load the image and mask
        img = imread(img_path)[:,:,:3]
        msk = imread(img_path.replace('img', 'msk')).astype(np.bool)

        # preprocess the img
        if self.std_fn:
            img = self.std_fn(img)
        img = self.norm_fn(img)
        img = resize(img, (self.imsize, self.imsize), mode='constant', preserve_range=True)
        msk = resize(msk, (self.imsize, self.imsize), mode='constant', preserve_range=True)

        if self.std_fn:
            img = np.expand_dims(img, axis=-1)
        msk = np.expand_dims(msk, axis=-1)

        if self.transforms:
            img, msk = self.transforms(img, msk, **{'u': self.u, 'pad':self.pad})

        #plt.figure()
        #plt.subplot(121)
        #plt.imshow(img.squeeze())
        #plt.subplot(122)
        #plt.imshow(msk.squeeze())
        #plt.show()

        img = img.transpose((2,0,1)).astype(np.float32)
        msk = msk.transpose((2,0,1)).astype(np.float32)

        # convert to torch
        img_torch = torch.from_numpy(img)
        msk_torch = torch.from_numpy(msk)

        # create output dictionary
        out_dict = {'img': img_torch,
                    'msk': msk_torch}

        return out_dict

    def __len__(self):
        return len(self.img_paths)

class FullImgMaskDataset(data.Dataset):
    def __init__(self, img_ids, norm_fn, std_fn, imsize, pad=16, u=0.5, transforms=None):
        self.transforms = transforms
        self.img_ids = img_ids
        self.norm_fn = norm_fn
        self.std_fn = std_fn
        self.imsize = imsize
        self.pad = pad
        self.u = u

    def __getitem__(self, index):
        np.random.seed()
        # get the image id
        img_id = self.img_ids[index]

        # set up paths
        img_path = '../data/stage1_train/' + img_id
        msk_path = img_path + '/masks/'

        # load the image
        img = imread(img_path + '/images/' + img_id + '.png')[:,:,:3]
        # load the mask
        msk = np.zeros((self.imsize, self.imsize), dtype=np.int32)
        dst = np.zeros((self.imsize, self.imsize), dtype=np.float32)
        msk_files = next(os.walk(img_path + '/masks/'))[2]
        for msk_file in msk_files:
            msk_ = imread(img_path + '/masks/' + msk_file).astype(np.bool)
            msk_ = resize(msk_, (self.imsize, self.imsize), 
                           mode='constant', preserve_range=True)
            # get distance
            dst_ = ndimage.distance_transform_edt(msk_)
            # combine masks for each nuclei
            msk = np.maximum(msk, msk_.astype(np.int32))
            dst = np.maximum(dst, dst_.astype(np.float32))

        dst /= np.amax(dst)      

        # preprocess the img
        img = self.norm_fn(img)
        img = resize(img, (self.imsize, self.imsize), mode='constant', preserve_range=True)
        #msk = resize(msk, (self.imsize, self.imsize), mode='constant', preserve_range=True)
        #dst = resize(dst, (self.imsize, self.imsize), mode='constant', preserve_range=True)

        #img = np.expand_dims(img, axis=-1)
        msk = np.expand_dims(msk, axis=-1)
        dst = np.expand_dims(dst, axis=-1)

        if self.transforms:
            img, msk, dst = self.transforms(img, msk, dst, **{'u': self.u, 'pad':self.pad})

        #plt.figure()
        #plt.subplot(121)
        #plt.imshow(img.squeeze())
        #plt.subplot(122)
        #plt.imshow(msk.squeeze())
        #plt.show()

        #print(img.shape)

        img = img.transpose((2,0,1)).astype(np.float32)
        msk = msk.transpose((2,0,1)).astype(np.float32)
        dst = dst.transpose((2,0,1)).astype(np.float32)

        # convert to torch
        img_torch = torch.from_numpy(img)
        msk_torch = torch.from_numpy(msk)
        dst_torch = torch.from_numpy(dst)

        # create output dictionary
        out_dict = {'img': img_torch,
                    'msk': msk_torch,
                    'dst': dst_torch}

        return out_dict

    def __len__(self):
        return len(self.img_paths)

class TestDataset(data.Dataset):

    def __init__(self, test_ids, imsize, std_fn, norm_fn):
        self.test_ids = test_ids
        self.std_fn = std_fn
        self.norm_fn = norm_fn
        self.imsize = imsize

    def __getitem__(self, index):
        np.random.seed()

        # get the images and corresponding masks
        img_id = self.test_ids[index]

        img = imread('../data/stage1_test/' + img_id + '/images/' + img_id + '.png')[:,:,:3]

        im_h = img.shape[0]
        im_w = img.shape[1]

        img = self.std_fn(img)
        img = self.norm_fn(img)
        img = resize(img, (self.imsize, self.imsize), mode='constant', preserve_range=True)

        img = img.transpose((2,0,1)).astype(np.float32)

        # convert to torch
        img_torch = torch.from_numpy(img)

        return img_torch, im_h, im_w, img_id

    def __len__(self):
        return len(self.test_ids)

def get_cropimg_training_loaders(imsize=256, test_size=0.1, batch_size=32, 
                                 augment_prob=0.5, include_rnd=False):
    img_paths = glob.glob('../data/image_mask_crops/*img*.png')

    if include_rnd:
        img_paths += glob.glob('../data/random_crops_mask/*img*.png')

    print('Found {} images'.format(len(img_paths)))

    img_indices = list(range(len(img_paths)))

    train_idx, valid_idx = train_test_split(img_indices, test_size=test_size, random_state=666)

    train_dataset = CroppedImgMaskDataset(img_paths, norm_fn=normalize_img, std_fn=None, 
                                   imsize=imsize, pad=8, u=(1.0-augment_prob), transforms=train_transforms)

    valid_dataset = CroppedImgMaskDataset(img_paths, norm_fn=normalize_img, std_fn=None,
                                   imsize=imsize, pad=8, u=1.0, transforms=None)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   shuffle=False,
                                   num_workers=3,
                                   pin_memory=True)

    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   sampler=valid_sampler,
                                   shuffle=False,
                                   num_workers=3,
                                   pin_memory=True)

    return train_loader, valid_loader, len(train_idx), len(valid_idx)

def get_fullimg_training_loaders(imsize=256, test_size=0.1, batch_size=32, augment_prob=0.5):
    img_ids = next(os.walk('../data/stage1_train/'))[1]

    img_indices = list(range(len(img_ids)))

    train_idx, valid_idx = train_test_split(img_indices, test_size=test_size, random_state=666)

    train_dataset = FullImgMaskDataset(img_ids=img_ids, norm_fn=normalize_img, 
                                       std_fn=None, imsize=imsize, 
                                       pad=32, u=(1.0-augment_prob), transforms=train_transforms_multi_out)

    valid_dataset = FullImgMaskDataset(img_ids=img_ids, norm_fn=normalize_img, 
                                       std_fn=None, imsize=imsize, 
                                       pad=32, u=1.0, transforms=None)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   shuffle=False,
                                   num_workers=3,
                                   pin_memory=True)

    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   sampler=valid_sampler,
                                   shuffle=False,
                                   num_workers=3,
                                   pin_memory=True)

    return train_loader, valid_loader, len(train_idx), len(valid_idx)

def get_test_loader(imsize=256, batch_size=32):
    test_ids = next(os.walk(TEST_PATH))[1]
    
    test_dataset = TestDataset(test_ids, imsize=256, 
                               std_fn=standardize_img, norm_fn=normalize_img)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2)

    return test_loader