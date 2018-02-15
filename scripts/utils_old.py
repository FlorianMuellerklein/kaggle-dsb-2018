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

TRAIN_PATH = '../data/stage1_train/'
TEST_PATH = '../data/stage1_test/'

class ImgMaskDataset(data.Dataset):
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
        img = imread(img_path)
        msk = imread(img_path.replace('img', 'msk'))

        # preprocess the img
        img = self.std_fn(img)
        img = self.norm_fn(img)
        img = resize(img, (self.imsize, self.imsize), mode='constant', preserve_range=True)

        if self.transforms:
            img, msk = self.transforms(img, msk, **{'u': self.u, 'pad':self.pad})

        img = img.transpose((2,0,1)).astype(np.float32)
        msk = msk.transpoze((2,0,1)).astype(np.float32)

        # convert to torch
        img_torch = torch.from_numpy(img)
        msk_torch = torch.from_numpy(msk)

        # create output dictionary
        out_dict = {'img': img_torch,
                    'msk': msk_torch}

        return out_dict

    def __len__(self):
        return len(self.img_paths)

class NumpyMaskDataset(data.Dataset):

    def __init__(self, X_img, Y_msk, Y_edg, pad=32, u=0.5, transforms=None):
        self.X_img = X_img
        self.Y_msk = Y_msk
        self.Y_edg = Y_edg
        self.transforms = transforms
        self.pad = pad
        self.u = u

    def __getitem__(self, index):
        np.random.seed()

        # get the images and corresponding masks
        img = self.X_img[index]
        msk = self.Y_msk[index]
        edg = self.Y_edg[index]

        if self.transforms:
            img, msk, edg = self.transforms(img, msk, edg, **{'u': self.u, 'pad':self.pad})

        img = img.transpose((2,0,1)).astype(np.float32)
        msk = msk.transpose((2,0,1)).astype(np.int32)
        edg = edg.transpose((2,0,1)).astype(np.int32)

        # convert to torch
        img_torch = torch.from_numpy(img)
        msk_torch = torch.from_numpy(msk).float()
        edg_torch = torch.from_numpy(edg).float()

        # put data in a dictionary
        out_dict = {'img': img_torch,
                    'msk': msk_torch,
                    'edg': edg_torch,
                    'id': 'PLACEHOLDER'}

        return out_dict

    def __len__(self):
        return len(self.X_img)

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

def train_transforms(img, msk, u=0.5, pad=16):
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

def normalize_img(img_arr):
    img_min = np.min(img_arr)
    img_max = np.max(img_arr)
    img_mean = np.mean(img_arr)
    img_out = (img_arr - img_mean) / (img_max - img_min)
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

def get_training_loaders(imsize=256, test_size=0.1, batch_size=32, augment_prob=0.5):
    img_paths = glob.glob('../data/img_mask_crops/*img*.png')

    img_indices = list(range(len(img_paths)))

    train_idx, valid_idx = train_test_split(img_indices, test_size=test_size, random_state=666)

    train_dataset = ImgMaskDataset(img_paths, norm_fn=normalize_img, std_fn=standardize_img, 
                                   imsize=imsize, pad=16, u=(1.0-augment_prob), transforms=train_transforms)

    valid_dataset = ImgMaskDataset(img_paths, norm_fn=normalize_img, std_fn=standardize_img,
                                   imsize=imsize, pad=16,u=1.0, transforms=None)

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

    return train_loader, valid_loader, len(Y_msks_tr), len(Y_msks_tst)

def get_test_loader(imsize=256, batch_size=32):
    test_ids = next(os.walk(TEST_PATH))[1]
    
    test_dataset = TestDataset(test_ids, imsize=256, 
                               std_fn=standardize_img, norm_fn=normalize_img)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2)

    return test_loader

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

    return intersection.sum() / float(union.sum())

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

def load_train_data(tr_path, imsize):
    # get all of the image ids
    train_ids = next(os.walk(tr_path))[1]

    # set up matrices to hold values
    X_imgs = np.zeros((len(train_ids), imsize, imsize, 1), dtype=np.float32)
    Y_msks = np.zeros((len(train_ids), imsize, imsize, 1), dtype=np.int32)
    Y_edgs = np.zeros((len(train_ids), imsize, imsize, 1), dtype=np.int32)

    num_masks = []

    for n, im_id in enumerate(train_ids):
        im_path = tr_path + im_id
        img = imread(im_path + '/images/' + im_id + '.png')[:, :, :3] # some images have 4 channels
        img = resize(img, (imsize, imsize), mode='constant', preserve_range=True)
        img = standardize_img(img)
        img = normalize_img(img)
        X_imgs[n] = np.expand_dims(img, axis=-1)
        # concatenate the masks
        mask = np.zeros((imsize, imsize, 1), dtype=np.int32)
        edge = np.zeros((imsize, imsize, 1), dtype=np.int32)
        mask_files = next(os.walk(im_path + '/masks/'))[2]
        num_masks.append(len(mask_files))
        for i, mask_file in enumerate(mask_files):
            mask_ = imread(im_path + '/masks/' + mask_file)
            mask_ = resize(mask_, (imsize, imsize), 
                           mode='constant', preserve_range=True)
            # find the edges
            edge_ = canny(mask_)
            # combine masks for each nuclei
            edge_ = np.expand_dims(edge_, axis=-1)
            mask_ = np.expand_dims(mask_, axis=-1) / 255
            mask = np.add(mask, mask_.astype(np.int32))
            edge = np.maximum(edge, edge_.astype(np.int32))

        #plt.imshow(mask.squeeze())
        #plt.title('Mask')
        #plt.show()

        #plt.imshow(edge.squeeze())
        #plt.title('Instances')
        #plt.show()

        # add mask to Y values
        Y_msks[n] = mask
        Y_edgs[n] = edge

    with open('../data/class_weights.pkl', 'wb') as f:
        pickle.dump(num_masks, f)

    return X_imgs, Y_msks, Y_edgs