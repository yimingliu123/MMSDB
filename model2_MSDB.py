# ====================================================
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/10/12                                     
# Description:   4 MSDB
#  << National University of Defense Technology >>  
# ====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import block_zoo as Block
from selayer import SELayer

class Model(nn.Module):
    def __init__(self, feature_channel, num_msdb=4, PFA=False):
        super(Model, self).__init__()
        self.conv_feature_expand = Block.same_conv(in_channels=3, out_channels=feature_channel, kernel_size=3)    # maybe 64
        self.msdb1 = Block.MSDB(in_channels=feature_channel, PFA=PFA)
        self.msdb2 = Block.MSDB(in_channels=feature_channel, PFA=PFA)
        self.msdb3 = Block.MSDB(in_channels=feature_channel, PFA=PFA)            # in = out
        self.msdb4 = Block.MSDB(in_channels=feature_channel, PFA=PFA)

        self.conv_msdb_merge = nn.Conv2d(in_channels=feature_channel * num_msdb, out_channels=feature_channel, kernel_size=1, stride=1)
        self.selayer = SELayer(channel=feature_channel)
        self.conv_mix1 = Block.same_conv(in_channels=feature_channel, out_channels=feature_channel, kernel_size=3)
        self.conv_mix2 = Block.same_conv(in_channels=feature_channel, out_channels=feature_channel // 2, kernel_size=3)
        self.conv_mix3 = Block.same_conv(in_channels=feature_channel // 2, out_channels=feature_channel // 4, kernel_size=3)
        self.conv_result = Block.same_conv(in_channels=feature_channel // 4, out_channels=3, kernel_size=1)
        # self.pixelshuffle = nn.PixelShuffle(2)
        # self.adp_pool = nn.AdaptiveMaxPool2d((128, 128))


    def forward(self, x):

        out_feature = self.conv_feature_expand(x)
        msdb_out_1 = self.msdb1(out_feature)
        msdb_out_2 = self.msdb2(msdb_out_1)
        msdb_out_3 = self.msdb3(msdb_out_2)
        msdb_out_4 = self.msdb4(msdb_out_3)
        msdb_merge = F.leaky_relu(self.conv_msdb_merge(torch.cat([msdb_out_1, msdb_out_2, msdb_out_3, msdb_out_4], dim=1)))
        msdb_merge = self.selayer(msdb_merge)
        mix1 = F.leaky_relu(self.conv_mix1(msdb_merge)) + out_feature
        mix2 = F.leaky_relu(self.conv_mix2(mix1))
        mix3 = F.relu(self.conv_mix3(mix2))
        result = self.conv_result(mix3)

        return result
