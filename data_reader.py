# ====================================================>>
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/9/21                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================>>


import torch.utils.data as data
import numpy as np
from glob import glob
from torchvision import transforms
from PIL import Image
import random


class GiveMeData(data.Dataset):
    def __init__(self, data_path, label_path, crop_size=256):
        self.data_path = data_path
        self.label_path = label_path
        self.data, self.target = map(lambda x: sorted(glob(x, recursive=True)), (self.data_path, self.label_path))
        # self.random_seed = np.random.randint(1234567)
        self.transform = self._get_transform(crop_size)

    def __getitem__(self, item):
        img = Image.open(self.data[item])           # RGB
        target = Image.open(self.target[item])

        # random.seed(1234567)       # processing data & label at same time by using fixed random seed
        img = self.transform(img)
        # random.seed(1234567)            # not helpful
        target = self.transform(target)

        return {'data': img, 'target': target}

    def __len__(self):
        return len(self.data)

    def _get_transform(self, crop_size):
        data_transform = transforms.Compose([
                                             transforms.CenterCrop(size=(crop_size, crop_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                             ])
        return data_transform


