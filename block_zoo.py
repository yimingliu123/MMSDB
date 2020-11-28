# ====================================================
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/10/11                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from selayer import SELayer

def same_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):                       # same conv
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

# def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
#     conv = same_conv(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
#     pixel_shuffle = nn.PixelShuffle(upscale_factor)
#     return sequential(conv, pixel_shuffle)
# for reference


class EnhanceSpatialAware(nn.Module):
    # down + pixelshuffle
    pass


class ProminentFeatureAttention(nn.Module):
    def __init__(self, in_channel):
        super(ProminentFeatureAttention, self).__init__()
        self.conv1 = same_conv(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.conv_max = same_conv(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.conv_feature_refine = same_conv(in_channels=in_channel, out_channels=in_channel, kernel_size=3)        # larger kernel size? 5 7?
        self.conv2 = same_conv(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.conv3 = same_conv(in_channels=in_channel, out_channels=in_channel, kernel_size=1)

    def forward(self, x):
        x_conv = F.leaky_relu(self.conv1(x))                           # in=out k=1        original size

        max_out = F.max_pool2d(x_conv, kernel_size=7, stride=3)          # Max pool  - downsample
        max_feature = F.leaky_relu(self.conv_max(max_out))          # in=out k=1     maybe we could ignore this 1x1 conv for saving computation
        max_refine = F.leaky_relu(self.conv_feature_refine(max_feature))    # feature refine  in=out k=3x3
        max_refine = F.leaky_relu(self.conv2(max_refine))           # in=out k=1  also ignore

        max_refine = F.interpolate(max_refine, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)        # interpolate to original size
        res_refine = self.conv3(x_conv + max_refine)
        weight = torch.sigmoid(res_refine)
        return x * weight


class MSDB(nn.Module):
    '''

    Multi Scale Distillation Block
    consist of 4 distillation step
    '''
    def __init__(self, in_channels, PFA=False):
        super(MSDB, self).__init__()
        self.PFA = PFA
        self.coarse_channels = self.remain_channel = in_channels
        self.distilled_channels_4 = in_channels // 4
        self.distilled_channels_3 = in_channels // 3
        self.distilled_channels_2 = in_channels // 2
        # ----------------------------------------------------------------------------------------------------------------------------
        self.conv_coarse_1 = same_conv(in_channels=in_channels, out_channels=self.remain_channel, kernel_size=3)
        self.conv_distill_1_4 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_4, kernel_size=3)
        self.conv_distill_1_3 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_3, kernel_size=3)
        self.conv_distill_1_2 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_2, kernel_size=3)
        self.fusion_distill_1 = nn.Conv2d(in_channels=self.distilled_channels_2 + self.distilled_channels_3 + self.distilled_channels_4,
                                          out_channels=in_channels, kernel_size=1, stride=1)
        # ----------------------------------------------------------------------------------------------------------------------------
        self.conv_coarse_2 = same_conv(in_channels=in_channels, out_channels=self.remain_channel, kernel_size=3)
        self.conv_distill_2_4 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_4, kernel_size=3)
        self.conv_distill_2_3 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_3, kernel_size=3)
        self.conv_distill_2_2 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_2, kernel_size=3)
        self.fusion_distill_2 = nn.Conv2d(in_channels=self.distilled_channels_2 + self.distilled_channels_3 + self.distilled_channels_4,
                                          out_channels=in_channels, kernel_size=1, stride=1)
        # ----------------------------------------------------------------------------------------------------------------------------
        self.conv_coarse_3 = same_conv(in_channels=in_channels, out_channels=self.remain_channel, kernel_size=3)
        self.conv_distill_3_4 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_4, kernel_size=3)
        self.conv_distill_3_3 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_3, kernel_size=3)
        self.conv_distill_3_2 = same_conv(in_channels=in_channels, out_channels=self.distilled_channels_2, kernel_size=3)
        self.fusion_distill_3 = nn.Conv2d(in_channels=self.distilled_channels_2 + self.distilled_channels_3 + self.distilled_channels_4,
                                          out_channels=in_channels, kernel_size=1, stride=1)
        # ----------------------------------------------------------------------------------------------------------------------------
        #                                              dis_channel_2  = in_channel // 2
        # these is no specified distillation in level 4
        self.conv_distill_4_2 = same_conv(in_channels=in_channels, out_channels=self.remain_channel, kernel_size=3)   # kernel size unlike above
        # ----------------------------------------------------------------------------------------------------------------------------
        self.selayer1 = SELayer(channel=self.remain_channel * 4)
        self.conv_distill_all = same_conv(in_channels=self.remain_channel * 4, out_channels=self.remain_channel, kernel_size=1)
        self.selayer2 = SELayer(channel=self.remain_channel)
        self.PFA = ProminentFeatureAttention(in_channel=in_channels) if self.PFA else None

    def forward(self, x):
        coarse1 = F.leaky_relu(self.conv_coarse_1(x)) + x           # shallow res
        # 64
        dis1_2 = F.leaky_relu(self.conv_distill_1_2(x))             # maybe I should activate the output after fusion?
        # 32
        dis1_3 = F.leaky_relu(self.conv_distill_1_3(x))
        # 21
        dis1_4 = F.leaky_relu(self.conv_distill_1_4(x))
        # 16
        dis_fusion_1 = self.fusion_distill_1(torch.cat([dis1_2, dis1_3, dis1_4], dim=1))        # dim=1 BxCxHxW  / channel wise  channel=32(64/2)
        # 32
        coarse2 = F.leaky_relu(self.conv_coarse_2(dis_fusion_1)) + dis_fusion_1           # changed
        dis2_2 = F.leaky_relu(self.conv_distill_2_2(dis_fusion_1))
        dis2_3 = F.leaky_relu(self.conv_distill_2_3(dis_fusion_1))
        dis2_4 = F.leaky_relu(self.conv_distill_2_4(dis_fusion_1))
        dis_fusion_2 = self.fusion_distill_2(torch.cat([dis2_2, dis2_3, dis2_4], dim=1))            # fusion conv output_channel is in_channel // 2

        coarse3 = F.leaky_relu(self.conv_coarse_3(dis_fusion_2)) + dis_fusion_2
        dis3_2 = F.leaky_relu(self.conv_distill_3_2(dis_fusion_2))
        dis3_3 = F.leaky_relu(self.conv_distill_3_3(dis_fusion_2))
        dis3_4 = F.leaky_relu(self.conv_distill_3_4(dis_fusion_2))
        dis_fusion_3 = self.fusion_distill_3(torch.cat([dis3_2, dis3_3, dis3_4], dim=1))

        coarse4 = F.leaky_relu(self.conv_distill_4_2(dis_fusion_3)) + dis_fusion_3       # only dis4  no

        distilled_feature = self.selayer1(torch.cat([coarse1, coarse2, coarse3, coarse4], dim=1))
        output = self.conv_distill_all(distilled_feature)                      # change the dimension
        output = self.selayer2(output)
        if self.PFA:
            output = self.PFA(output)
        return output


class JCSBoost(nn.Module):

    def __init__(self, in_channels=64, upscale_factor=2):
        super(JCSBoost, self).__init__()
        self.forward = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels ** upscale_factor, kernel_size=1, stride=upscale_factor),
                                     SELayer(channel=in_channels ** upscale_factor),
                                     nn.PixelShuffle(upscale_factor=upscale_factor),
                                     nn.Conv2d(in_channels=128, out_channels=in_channels, kernel_size=1, stride=1),
                                     nn.LeakyReLU())

    def forward(self, x):
        return self.forward(x)


class SFDB(nn.Module):
    def __init__(self, in_channels):
        super(SFDB, self).__init__()
        self.coarse_channels = self.remain_channel = in_channels
        