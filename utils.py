# ====================================================
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur                          
# Date: 2020/10/16                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================


import torch
import numpy as np
from torchvision import transforms

from glob import glob
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr




def sorted_glob(pattern):
    return sorted(glob(pattern))


def calculate_psnr(real, sharpen):
    psnr_score = compare_psnr(real, sharpen)
    return psnr_score

def calculate_ssim(real, sharpen):
    ssim_score = compare_ssim(real, sharpen, multichannel=True)
    return ssim_score


def tensor2np(tensor):
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy().astype('uint8')
    return tensor


def calculate_all(real, sharpen):
    real = np.asarray(real)
    sharpen = np.asarray(sharpen)
    psnr = compare_psnr(real, sharpen)
    ssim = compare_ssim(real, sharpen, multichannel=True)

    return psnr, ssim




class InverseTensor(object):
    '''
    1: inverse normalization
    2: to [0 - 255]
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = torch.squeeze(tensor, dim=0)                  # tensor[0, :, :, :]
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        tensor = tensor.mul(255).detach().cpu().round().byte()                 # miracle sentence
        tensor = transforms.ToPILImage()(tensor)
        return tensor

