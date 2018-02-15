import os
import sys
import time
import math
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models import UNet_Multi
from utils.data_loaders import get_fullimg_training_loaders
from utils.evaluations import mean_iou, plot_losses

parser = argparse.ArgumentParser(description='Kaggle Cdiscounts Training')
parser.add_argument('--gpu', default=0, type=int, 
                    help='which gpu to run')
parser.add_argument('--batch_size', default=16, type=int, 
                    help='size of batches')
parser.add_argument('--aug_prob', default=0.5, type=float,
                    help='what probability of batches to augment')
parser.add_argument('--valid_size', default=0.1, type=float,
                    help='what proportion of training data to withold for validation')
parser.add_argument('--img_size', default=256, type=int,
                    help='height and width of images to use')
parser.add_argument('--epochs', default=150, type=int, 
                    help='number of epochs')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--l2', default=0.00001, type=float,
                    help='l2 regularization for model')
parser.add_argument('--lamb', default=0.5, type=float,
                    help='lambda value for class loss')
parser.add_argument('--es_patience', default=100, type=int, 
                    help='early stopping patience')
parser.add_argument('--model_name', default='UNet', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='fullimg_lossless_augmentations', type=str,
                    help='name of experiment for saving files')
args = parser.parse_args()

LR_SCHED = {0: args.lr,
            100: args.lr * 0.1,
            125: args.lr * 0.01}

torch.cuda.set_device(args.gpu)
cudnn.benchmark = True

net = UNet_Multi()
net.cuda()

# set model filenames
MODEL_CKPT = '../models-pytorch/best_{}_{}_full_img.pth'.format(args.model_name, 
                                                     args.exp_name)

train_loader, valid_loader, len_train, len_valid = get_fullimg_training_loaders(imsize=args.img_size, 
                                                                        test_size=args.valid_size, 
                                                                        batch_size=args.batch_size, 
                                                                        augment_prob=args.aug_prob)

msk_crit = nn.BCEWithLogitsLoss().cuda()
dst_crit = nn.BCEWithLogitsLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)

# training loop
def train():
     net.train(True)
     # keep track of losses
     iter_loss = 0.
     # keep track of corrects
     iter_correct = 0.
     for i, data in enumerate(train_loader):
          # get the inputs
          imgs = data['img']
          msks = data['msk']
          dsts = data['dst']

          #plt.imshow(imgs.numpy()[0].squeeze())
          #plt.show()

          imgs = Variable(imgs.cuda(async=True))
          msks = Variable(msks.cuda(async=True))
          dsts = Variable(dsts.cuda(async=True))

          optimizer.zero_grad()

          # forward + backward + optimize
          pred_msk, pred_dst = net(imgs)
          loss = msk_crit(pred_msk, msks) + args.lamb * dst_crit(pred_dst, dsts)
          # optimize network
          loss.backward()

          optimizer.step()

          # get training stats
          iter_loss += loss.data[0]

          sys.stdout.write('\r')
          sys.stdout.write('B: {:>3}/{:<3} | {:.4}'.format(i, 
                                                  len_train // args.batch_size,
                                                  loss.data[0]))
          sys.stdout.flush()

     epoch_loss = iter_loss / (len_train // args.batch_size)
     print('\n' + 'Avg Train Loss: {:.4}'.format(epoch_loss))

     return epoch_loss

# validation loop
def validate():
     ious = []
     net.eval() 
     # keep track of losses
     val_iter_loss = 0.
     # keep track of corrects
     val_iter_correct = 0
     for j, data in enumerate(valid_loader):
          val_imgs = data['img']
          val_msks = data['msk']
          val_dsts = data['dst']

          val_imgs = Variable(val_imgs.cuda(async=True), volatile=True)
          val_msks = Variable(val_msks.cuda(async=True), volatile=True)
          val_dsts = Variable(val_dsts.cuda(async=True), volatile=True)

          vpred_msk, vpred_dst = net(val_imgs)
          v_l = msk_crit(vpred_msk, val_msks) + args.lamb * dst_crit(vpred_dst, val_dsts)

          # get validation stats
          val_iter_loss += v_l.data[0]

          ious.append(mean_iou(vpred_msk, val_msks))

     epoch_vloss = val_iter_loss / (len_valid // args.batch_size)
     print('Avg Eval Loss: {:.4} | Avg Eval IOU: {:.4}'.format(epoch_vloss, np.mean(ious)))
     return epoch_vloss, np.mean(ious)

# train the model
try:
     print('Training ...')
     train_losses = []
     valid_losses = []
     valid_ious = []
     best_val_loss = 10.0
     for e in range(args.epochs):
          print('\n' + 'Epoch {}/{}'.format(e, args.epochs))
          start = time.time()

          t_l = train()
          v_l, viou = validate()
          train_losses.append(t_l)
          valid_losses.append(v_l)
          valid_ious.append(viou)

          # write the losses to a text file
          #with open('../logs/losses_{}_{}_{}.txt'.format(args.model_name, 
          #                                               args.exp_name, 
          #                                               k), 'a') as logfile:
          #    logfile.write('{},{},{},{}'.format(e, t_l, v_l, v_a) + "\n")

          # save the model everytime we get a new best valid loss
          if v_l < best_val_loss:
               torch.save(net.state_dict(), MODEL_CKPT)
               best_val_loss = v_l

          # if the validation loss gets worse increment 1 to the patience values
          #if v_l > best_val_loss:
          #     valid_patience += 1
          #     lr_patience += 1

          # if the model stops improving by a certain number epochs, stop
          #if valid_patience == args.es_patience:
          #     break
          if e in LR_SCHED:
               for params in optimizer.param_groups:
                    params['lr'] = LR_SCHED[e]

          print('Time: {}'.format(time.time()-start))

     print('Finished Training')

except KeyboardInterrupt:
     pass

plot_losses(args.model_name, args.exp_name, train_losses, valid_losses, valid_ious)