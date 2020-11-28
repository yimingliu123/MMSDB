# ====================================================
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur                          
# Date: 2020/11/3                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================


import torch
import torch.nn as nn
import block_zoo as B
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1)
        self.encoder_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=2)