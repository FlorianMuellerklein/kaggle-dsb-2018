import math

import torch
import torchvision
import torch.nn as nn
import torchvision as vsn
import torch.nn.functional as F

class SELayer(nn.Module):
    '''
    Squeeze and Excitation layer, borrowed from moskomule/senet.pytorch
    '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(channel=planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResUNet(nn.Module):
    def __init__(self, block, layers):
        super(SEResUNet, self).__init__()
        self.inplanes = 16
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # encoder layers
        self.d_selayer_1 = self._make_layer(block, 32, layers[0])
        self.d_selayer_2 = self._make_layer(block, 64, layers[1], stride=2)
        self.d_selayer_3 = self._make_layer(block, 128, layers[2], stride=2)
        self.d_selayer_4 = self._make_layer(block, 256, layers[3], stride=2)
        # decoder layers
        self.u_selayer_1 = self._make_layer(block, 384, layers[3])
        self.u_selayer_2 = self._make_layer(block, 128, layers[2])
        self.u_selayer_3 = self._make_layer(block, 64, layers[1])
        # deconv layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # upsampling layers
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.lrelu(x)            # 32x256x256
        
        # encoding layers
        e1 = self.d_selayer_1(x)     # 64x256x256
        e2 = self.d_selayer_2(e1)    # 128x128x128
        e3 = self.d_selayer_3(e2)    # 256x64x64
        # bridge layer
        e4 = self.d_selayer_4(e3)    # 512x32x32
        # decoding
        u1 = self.upsample(e4)       # 256x64x64
        c1 = torch.cat((u1, e3), 1)  # 512x64x64
        print(c1.size())
        d1 = self.u_selayer_1(c1)    # 256x64x64
        u2 = self.upsample(d1)        # 128x128x128
        c2 = torch.cat((u2, e2), 1)  # 256x128x128
        print(c2.size())
        d2 = self.u_selayer_2(c2)    # 128x128x128
        u3 = self.upsample(d2)        # 64x256x256
        c3 = torch.cat((u3, e1), 1)  # 128x256x256
        print(c3.size())
        d3 = self.u_selayer_3(c3)    # 64x256x256
        # output
        out = self.out_conv(d3)
        
        return out

class SEResUNet_Contour(nn.Module):
    def __init__(self, block, layers):
        super(SEResUNet_Contour, self).__init__()
        self.inplanes = 16
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # encoder layers
        self.d_selayer_1 = self._make_layer(block, 32, layers[0])
        self.d_selayer_2 = self._make_layer(block, 64, layers[1], stride=2)
        self.d_selayer_3 = self._make_layer(block, 128, layers[2], stride=2)
        self.d_selayer_4 = self._make_layer(block, 256, layers[3], stride=2)
        # decoder layers
        self.u_selayer_1 = self._make_layer(block, 128, layers[3])
        self.u_selayer_2 = self._make_layer(block, 64, layers[2])
        self.u_selayer_3a = self._make_layer(block, 32, layers[1])
        self.u_selayer_3b = self._make_layer(block, 32, layers[1])
        # upsampling layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # output layer
        self.out_mask = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.out_cont = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # nonlinearity
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        # dropout
        self.dropout = nn.Dropout(0.25)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.lrelu(x)                               # 32x256x256
        
        # encoding layers
        e1 = self.dropout(self.d_selayer_1(x))          # 64x256x256
        e2 = self.dropout(self.d_selayer_2(e1))         # 128x128x128
        e3 = self.dropout(self.d_selayer_3(e2))         # 256x64x64
        # bridge layer
        e4 = self.d_selayer_4(e3)                       # 512x32x32
        # decoding
        u1 = self.leaky_relu(self.deconv1(e4))          # 256x64x64
        c1 = torch.cat((u1, e3), 1)                     # 512x64x64
        d1 = self.dropout(self.u_selayer_1(c1))         # 256x64x64
        u2 = self.leaky_relu(self.deconv2(d1))          # 128x128x128
        c2 = torch.cat((u2, e2), 1)                     # 256x128x128
        d2 = self.dropout(self.u_selayer_2(c2))         # 128x128x128
        u3 = self.leaky_relu(self.deconv3(d2))          # 64x256x256
        c3 = torch.cat((u3, e1), 1)                     # 128x256x256
        d3a = self.dropout(self.u_selayer_3a(c3))       # 64x256x256
        #d3b = self.u_selayer_3b(c3)                    # 64x256x256
        # output
        out_msk = self.out_mask(d3a)                    # 1x256x256
        out_con = self.out_cont(d3a)                    # 1x256x256
        
        return out_msk, out_con

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # encoding layers
        self.conv_1a = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_1b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_2a = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_3a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        self.conv_4a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)
        self.conv_5a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_5b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # decoding layers
        self.conv_6a = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1)
        self.conv_6b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_7a = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv_7b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_8a = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.conv_8b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_9a = nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1)
        self.conv_9b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        # decong layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # upsampling layers
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)  
        # output layer
        self.out_mask = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        # activation layers
        self.elu = nn.ELU(inplace=True)
        # dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # encoding layers
        e1a = self.dropout(self.elu(self.conv_1a(x)))
        e1b = self.elu(self.conv_1b(e1a))
        e1 = self.max_pool_1(e1b)
        e2a = self.dropout(self.elu(self.conv_2a(e1)))
        e2b = self.elu(self.conv_2b(e2a))
        e2 = self.max_pool_1(e2b)
        e3a = self.dropout(self.elu(self.conv_3a(e2)))
        e3b = self.elu(self.conv_3b(e3a))
        e3 = self.max_pool_1(e3b)
        e4a = self.dropout(self.elu(self.conv_4a(e3)))
        e4b = self.elu(self.conv_4b(e4a))
        e4 = self.max_pool_1(e4b)
        e5a = self.dropout(self.elu(self.conv_5a(e4)))
        e5b = self.elu(self.conv_5b(e5a))
        # decoding layers
        u1 = self.upsample(e5b)
        c1 = torch.cat((e4b, u1), 1)
        d1a = self.dropout(self.elu(self.conv_6a(c1)))
        d1b = self.elu(self.conv_6b(d1a))

        u2 = self.upsample(d1b)
        c2 = torch.cat((e3b, u2), 1)
        d2a = self.dropout(self.elu(self.conv_7a(c2)))
        d2b = self.elu(self.conv_7b(d2a))

        u3 = self.upsample(d2b)
        c3 = torch.cat((e2b, u3), 1)
        d3a = self.dropout(self.elu(self.conv_8a(c3)))
        d3b = self.elu(self.conv_8b(d3a))

        u4 = self.upsample(d3b)
        c4 = torch.cat((e1b, u4), 1)
        d4a = self.dropout(self.elu(self.conv_9a(c4)))
        d4b = self.elu(self.conv_9b(d4a))

        # output
        out_msk = self.out_mask(d4b)   # 1x256x256
        
        return out_msk

class UNet_Multi(nn.Module):
    def __init__(self):
        super(UNet_Multi, self).__init__()
        # encoding layers
        self.conv_1a = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv_1b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_2a = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_3a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        self.conv_4a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)
        self.conv_5a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_5b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # decoding layers
        self.conv_6a = nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1)
        self.conv_6b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_7a = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.conv_7b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_8a = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.conv_8b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv_9a = nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1)
        self.conv_9b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        # decong layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # upsampling layers
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)  
        # output layer
        self.out_mask = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        # output distances
        self.out_dist = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        # activation layers
        self.elu = nn.ELU(inplace=True)
        # dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # encoding layers
        e1a = self.dropout(self.elu(self.conv_1a(x)))
        e1b = self.elu(self.conv_1b(e1a))
        e1 = self.max_pool_1(e1b)
        e2a = self.dropout(self.elu(self.conv_2a(e1)))
        e2b = self.elu(self.conv_2b(e2a))
        e2 = self.max_pool_1(e2b)
        e3a = self.dropout(self.elu(self.conv_3a(e2)))
        e3b = self.elu(self.conv_3b(e3a))
        e3 = self.max_pool_1(e3b)
        e4a = self.dropout(self.elu(self.conv_4a(e3)))
        e4b = self.elu(self.conv_4b(e4a))
        e4 = self.max_pool_1(e4b)
        e5a = self.dropout(self.elu(self.conv_5a(e4)))
        e5b = self.elu(self.conv_5b(e5a))
        # decoding layers
        u1 = self.upsample(e5b)
        c1 = torch.cat((e4b, u1), 1)
        d1a = self.dropout(self.elu(self.conv_6a(c1)))
        d1b = self.elu(self.conv_6b(d1a))

        u2 = self.upsample(d1b)
        c2 = torch.cat((e3b, u2), 1)
        d2a = self.dropout(self.elu(self.conv_7a(c2)))
        d2b = self.elu(self.conv_7b(d2a))

        u3 = self.upsample(d2b)
        c3 = torch.cat((e2b, u3), 1)
        d3a = self.dropout(self.elu(self.conv_8a(c3)))
        d3b = self.elu(self.conv_8b(d3a))

        u4 = self.upsample(d3b)
        c4 = torch.cat((e1b, u4), 1)
        d4a = self.dropout(self.elu(self.conv_9a(c4)))
        d4b = self.elu(self.conv_9b(d4a))

        # output
        out_msk = self.out_mask(d4b)   # 1x256x256
        out_dst = self.out_dist(d4b)
        
        return out_msk, out_dst

def basic_SEResUNet():
    model = SEResUNet(SEBasicBlock, layers=[1, 1, 1, 1])
    return model

def contour_SEResUNet():
    model = SEResUNet_Contour(SEBasicBlock, layers=[1, 1, 1, 1])
    return model
