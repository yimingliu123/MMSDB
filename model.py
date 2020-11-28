# ====================================================
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/10/9                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import selayer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5)
        self.conv1_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)

        self.conv1_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv1_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv1_6 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5)
        # -------------------------
        #self.shuffle = nn.PixelShuffle()          # N x Cr^2 x H x W -->  C x rH x rW

    def forward(self, x):
        x1_1 = self.conv1_1(x)  # 3 - 128
        x1_2 = self.conv1_2(x1_1)  # 128 - 64
        x1_3 = self.conv1_3(x1_2)  # 64 - 32
        x1_4 = self.conv1_4(x1_3)  # 32 - 64
        x1_5 = self.conv1_5(x1_4 + x1_2)  # 64 - 128     res
        x1_6 = self.conv1_6(x1_5 + x1_1)  # 128 - 3      res

        return x1_6

class DualModel(nn.Module):
    '''
    back loop
    '''
    def __init__(self):
        super(DualModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x