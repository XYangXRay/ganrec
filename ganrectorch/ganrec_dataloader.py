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


def normalize_transform(image, mean = None, std = None):
    if mean is not None and std is not None:
        image = (image - mean) / std
        image = image / np.max(image)
    return expand(image)

def expand(image):
    image = torch.from_numpy(image)
    if len(image.shape) == 2:
        image = image.unsqueeze(0).float()
    elif len(image.shape) == 3:
        image = image.unsqueeze(1).float()
    return image



def torch_reshape(image):
    if type(image) is list:
        image = torch.stack(image)

    if type(image) is not torch.Tensor:
            image = torch.from_numpy(image)
    
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        image = image.float()
        return image
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
        image = image.float()
        return image
    elif len(image.shape) == 4:
        image = image.float()
        return image
    else:
        print("Image shape not supported")
        return None

def torchnor_phase(image):
    image = torch_reshape(image)
    image = image - torch.min(image)
    if torch.max(image) != 0:
        image = image / torch.max(image)
    return image

def save_path_generator(**kwargs):
    try:
        file_name = os.path.splitext(os.path.basename(kwargs['image_path']))[0]
        #the folder name of the 'image_path'
        folder = os.path.basename(os.path.dirname(kwargs['image_path']))
    except:
        file_name = os.path.splitext(os.path.basename(kwargs['image_path'][0]))[0]
        folder = os.path.basename(os.path.dirname(kwargs['image_path'][0]))
        
    init_path = os.getcwd() + '/data/saved_weights/' + folder + '/'
    save_wpath = os.getcwd() + '/data/retrieved/' + folder + '/'
    if not os.path.exists(init_path):
        os.makedirs(init_path)
    if not os.path.exists(save_wpath):
        os.makedirs(save_wpath)
    kwargs['file_name'] = file_name
    kwargs['save_wpath'] = save_wpath
    kwargs['init_wpath'] = init_path
    return kwargs 

def get_all_info(path = None, images = None, idx = 1000, energy_kev = 18.0, detector_pixel_size = 2.57 * 1e-6, distance_sample_detector = 0.15, alpha = 1e-8, delta_beta = 1e1, pad = 1, method = 'TIE', image = None, phase_path= None, attenuation_path = None, phase_image = None, attenuation_image = None, **kwargs):
    """
    make sure that the unit of energy is in keV, the unit of detector_pixel_size is in meter, and the unit of distance_sample_detector is in meter
    """

    if path is not None:
        images = list(io.imread_collection(path + '/*.tif').files)
        # images = sorted(glob(path + '/*.tif'))
        if type(idx) is list:
            image_path = [images[i] for i in idx]
            image = load_images_parallel(image_path)
            shape_x = image[0].shape[0]
            shape_y = image[0].shape[1]
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
            ND = len(idx)
            print("images are of leng", len(image))
        else:
            image_path = images[idx]
            image = load_image(image_path)
            shape_x = image.shape[0]
            shape_y = image.shape[1]
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
            ND = 1
        assert images is not None, "either image or path should be provided"
        
    else:
        assert images is not None, "either image or path should be provided"
        image_path = images[idx]
        image = load_image(images[idx])
        Fx, Fy = grid_generator(image.shape[0], image.shape[1], upscale=1, ps = detector_pixel_size)
        shape_x = image.shape[0]
        shape_y = image.shape[1]
        ND = 1
    
    
    if phase_path is not None and attenuation_path is not None:
        phase_images = io.imread_collection(phase_path + '/*.tif').files
        attenuation_images = io.imread_collection(attenuation_path + '/*.tif').files
        if type(idx) is list:
            phase_image_path = []
            attenuation_image_path = []
            for i in idx:
                phase_image_path.append(phase_images[i])
                attenuation_image_path.append(attenuation_images[i])
            phase_image = load_images_parallel(phase_image_path)
            attenuation_image = load_images_parallel(attenuation_image_path)
        else:
            phase_image_path = phase_images[idx]
            attenuation_image_path = attenuation_images[idx]
            phase_image = load_image(phase_image_path)
            attenuation_image = load_image(attenuation_image_path)

    kwargs = {
        "path": path,
        "output_path" : os.getcwd(),
        "idx": idx,
        "column_name": 'path',
        "energy_J": energy_kev_to_joule(energy_kev),
        "energy_kev": energy_kev,
        "lam": wavelength_from_energy(energy_kev),
        "detector_pixel_size": detector_pixel_size,
        "distance_sample_detector": detector_pixel_size,
        "fresnel_number": fresnel_calculator(energy_kev = energy_kev, detector_pixel_size = detector_pixel_size, distance_sample_detector = distance_sample_detector),
        "wave_number": wave_number(energy_kev),

        "shape_x": shape_x,
        "shape_y": shape_y,
        "pad_mode": 'symmetric',
        'shape': [int(shape_x), int(shape_y)],
        'nx': int(shape_x), 'ny': int(shape_y),
        'distance': [distance_sample_detector], 
        'energy': energy_kev, 
        'alpha': alpha, 
        'pad': pad,
        'nfx': int(shape_x) * pad, 
        'nfy': int(shape_y) * pad,
        'pixel_size': [detector_pixel_size, detector_pixel_size],
        'sample_frequency': [1.0/detector_pixel_size, 1.0/detector_pixel_size],
        'fx': Fx, 'fy': Fy,
        'method': method, 
        'delta_beta': delta_beta,
        "fresnel_factor":  fresnel_operator(int(shape_x) * pad, int(shape_y) * pad, detector_pixel_size, distance_sample_detector, energy_kev),
        "image_path": image_path,
        "image": image,
        "ND": ND,
        "phase_path": phase_path,
        "attenuation_path": attenuation_path,
        'phase_image': phase_image,
        'attenuation_image': attenuation_image,
    } 
    kwargs.update(save_path_generator(**kwargs))
    return kwargs

def FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None):
    """  Parameters: 
            E0 - initial complex field in x-y source plane
            detector_pixel_size - pixel size in microns
            lambda0 - wavelength in nm
            distance_sample_detector - distance_sample_detector-value (distance from sensor to object)
            background - optional background image to divide out from
        
        Returns: E1 - propagated complex field in x-y sensor plane"""  
    import os
    dtype = torch.complex64
    H = torch.from_numpy(ff).type(dtype)
    detector_wavefield = torch.exp(torch.complex(-absorption, phase))
    detector_wavefield = detector_wavefield.type(dtype)
    
    # Compute FFT centered about 0
    E0fft = (torch.fft.fft2(detector_wavefield)).type(dtype)

    # Multiply spectrum with fresnel phase-factor
    print("E0fft shape: ", E0fft.shape, "H shape: ", H.shape)
    G = H * E0fft
    # Ef = torch.signal.ifft2d(torch.signal.ifftshift(G)) # Output after deshifting Fourier transform
    I = (torch.abs(torch.fft.ifft2(G))**2).type(dtype)

    if dark_image is not None and ref_image is not None:
        I = I * (ref_image - dark_image) + dark_image
    
    I = torch_reshape(I) #without normalizing
    # I = torchnor_phase(torch.reshape(I, [1, I.shape[0], I.shape[1], 1]))
    return I


class Ganrec_Dataloader():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.kwargs.update(get_all_info(**kwargs))
        keys = self.kwargs.keys()
        [self.__setattr__(key, self.kwargs[key]) for key in keys]
        self.dims = (self.ND, self.shape_x, self.shape_y)
        self.transformed_images = None

        super(Ganrec_Dataloader, self).__init__()

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
                self.transformed_images = torchnor_phase(self.image)
            else:
                images = [torchnor_phase(self.image[i]) for i in range(len(self.idx))]
                self.transformed_images= torch.stack(images)
        else:
            if type(self.idx) is not list:
                self.transformed_images = torchnor_phase(self.image)
            else:
                self.transformed_images = torch.stack([torchnor_phase(self.image[i]).squeeze(0) for i in range(len(self.idx))])
        return self.transformed_images
    
    def normalize(self, idx = None):
        image = self.__getitem__(idx)
        image = torchnor_phase(image)
        return image
    
    def get_kwargs(self):
        return self.__dict__
    
    def input_visualize(self, idx = None, random = False, show_or_plot = 'show'):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            kwargs.update(get_all_info(**kwargs))
            keys = kwargs.keys()
            [self.__setattr__(key, kwargs[key]) for key in keys]
            images = self.image
        else:
            images = self.image
        if type(images) is not list:
            images = [images]

        rows = int(np.sqrt(len(images)))
        if rows ==1:
            cols = len(images)
        else:
            cols = rows + 1
        print("rows: {}, cols: {}".format(rows, cols))
        visualize(images, rows = rows, cols = cols, random = random, show_or_plot = show_or_plot)

    def normal_visualize(self, idx = None, random = False, show_or_plot = 'show'):
        if self.transformed_images is None:
            self.__getitem__(idx)
        print(self.transformed_images.shape)
        images = [self.transformed_images[i, 0, :, :].numpy() for i in range(self.transformed_images.shape[0])]
        rows = int(np.sqrt(len(images)))
        cols = rows + 1
        visualize(images, rows = rows, cols = cols, random = random, show_or_plot = show_or_plot)

    
    def forward_propagate(self, distance = None):
        if distance is None:
            distance = self.distance_sample_detector
        self.propagated_forward = FresnelPropagator(self.phase, self.attenuation, self.fresnel_factor, distance)
        self.propagated_forward = torch_reshape(self.propagated_forward)
        return self.propagated_forward
