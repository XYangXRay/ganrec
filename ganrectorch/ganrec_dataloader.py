import torch
import skimage.io as io
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from utils import *

import sys
sys.stdout = open('prints.txt', 'w')

from utils import *

def normalize_transform(image, mean = None, std = None):
    if mean is not None and std is not None:
        image = (image - mean) / std
        image = image / np.max(image)
    image = torch.from_numpy(image)
    if len(image.shape) == 2:
        image = image.unsqueeze(0).float()
    elif len(image.shape) == 3:
        image = image.unsqueeze(1).float()
    return image

class Ganrec_Dataloader(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs.update(get_all_info(**kwargs))
        keys = self.kwargs.keys()
        [self.__setattr__(key, self.kwargs[key]) for key in keys]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((self.mean,), (self.std,)),
            ])
        self.dims = (self.ND, self.shape_x, self.shape_y)
        print("dimensions of the dataset is {}".format(self.dims))
        self.tranformed_images = None

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx = None):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            kwargs.update(get_all_info(**kwargs))
            keys = kwargs.keys()
            [self.__setattr__(key, kwargs[key]) for key in keys]
            
            if type(idx) is not list:    
                self.tranformed_images = normalize_transform(self.image, self.mean, self.std)
            else:
                images = [normalize_transform(self.image[i], self.mean[i], self.std[i]) for i in range(len(self.idx))]
                self.tranformed_images= torch.stack(images)
        else:
            if type(self.idx) is not list:
                self.tranformed_images = normalize_transform(self.image, self.mean, self.std)
            else:
                self.tranformed_images = torch.stack([normalize_transform(self.image[i], self.mean[i], self.std[i]) for i in range(len(self.idx))])
        return self.tranformed_images
    def normalize(self, idx = None):
        image = self.__getitem__(idx)
        image = normalize_transform(image)
        return image
    
    def get_kwargs(self):
        return self.__dict__
    
    def visualize(self, idx = None, random = False):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            kwargs.update(get_all_info(**kwargs))
            keys = kwargs.keys()
            [self.__setattr__(key, kwargs[key]) for key in keys]
            images = self.image
        if type(images) is not list:
            images = [images]

        rows = int(np.sqrt(len(images)))
        cols = rows + 1
        print("rows: {}, cols: {}".format(rows, cols))
        if random == False:
            visualize(images, rows = rows, cols = cols)
        else:
            visualize(images, rows = rows, cols = cols, random=True)

    def normal_visualize(self, idx = None, random = False):
        if self.tranformed_images is None:
            self.__getitem__(idx)
        print(self.tranformed_images.shape)
        images = [self.tranformed_images[i, 0, :, :].numpy() for i in range(self.tranformed_images.shape[0])]
        rows = int(np.sqrt(len(images)))
        cols = rows + 1
        if random == False:    
            visualize(images, rows = rows, cols = cols, random = False )
        else:
            visualize(images, random=True)

args = {
        "path": "/asap3/petra3/gpfs/p05/2023/data/11016663//processed/thomas_001_d150/flat_corrected/rawBin2",
        "idx": 89,
        "energy_kev": 18.0,
        "detector_pixel_size": 2.57 * 1e-6,
        "distance_sample_detector": 0.15,
        "alpha": 1e-8,
        "delta_beta": 1,
        "pad": 1,
        "method": 'TIE',
        'iter_num': 500,
        'conv_num': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'l1_ratio': 10,
        'abs_ratio': 1.0,
        'g_learning_rate': 1e-3,
        'd_learning_rate': 1e-5,
        'phase_only': False,
        'save_wpath': None,
        'init_wpath': None,
        'init_model': False,
        'recon_monitor': True,
        'seed': 42,
    }
