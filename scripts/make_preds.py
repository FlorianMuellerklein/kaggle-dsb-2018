import argparse
import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from skimage.morphology import label, watershed
from skimage.feature import peak_local_max
from skimage.transform import resize
 
from utils import get_test_loader
from models import UNet, contour_SEResUNet

parser = argparse.ArgumentParser(description='Kaggle Cdiscounts Training')
parser.add_argument('--gpu', default=1, type=int, 
                    help='which gpu to run')
parser.add_argument('--batch_size', default=16, type=int, 
                    help='size of batches')
parser.add_argument('--img_size', default=448, type=int,
                    help='height and width of images to use')
args = parser.parse_args()

net = contour_SEResUNet().cuda()
net.load_state_dict(torch.load('../models-pytorch/best_SEResUNet_Contour_flips_l1_lamb0.5.pth'))
net.eval()

test_loader = get_test_loader(imsize=args.img_size)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x):
    # watershed instance generation
    #x = np.where(x > 0.5, 1, 0)
    #distance = ndimage.distance_transform_edt(x)
    #local_max = peak_local_max(distance, indices=False, footprint=np.ones((3,3)), labels=x)
    #markers = ndimage.label(local_max)[0]
    #lab_img = watershed(-distance, markers, mask=x)
    # regular morphology label
    lab_img = label(x > 0.5)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

try:
    test_ids = []
    rle_preds = []
    for data in test_loader:
        tst_img, im_h, im_w, im_id = data

        #plt.imshow(tst_img.numpy()[0].squeeze())
        #plt.show()

        tst_img = Variable(tst_img.cuda(), volatile=True)

        pred_msk, pred_edg = net(tst_img)
        pred_msk = F.sigmoid(pred_msk)

        pred_msk = pred_msk.data.cpu().numpy()

        for i, id_ in enumerate(im_id):
            split_mask = resize(pred_msk[i].squeeze(), (im_h[i], im_w[i]), mode='constant')

            rle = list(prob_to_rles(split_mask))
            rle_preds.extend(rle)
            test_ids.extend([id_] * len(rle))

except KeyboardInterrupt:
    pass

sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rle_preds).apply(lambda x: ' '.join(str(y) for y in x))
print(sub.head())
sub.to_csv('../subm/SEResUNet_Contour_flips_l1_lamb0.5_no_std.csv', index=False)