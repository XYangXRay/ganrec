import torch
import skimage.io as io
import numpy as np
from utils import *
import sys
# sys.stdout = open('prints.txt', 'w')

from utils import *
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from joblib import Parallel, delayed
from torchsummary import summary

    
import pytorch_lightning as pl
def L1_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

def L2_error(y_pred, y_true):
    return torch.mean(torch.pow(y_pred - y_true, 2))

def L1_L2_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) + torch.pow(y_pred - y_true, 2))

def tik_reg(y_pred, y_true, alpha = 1e-8):
    return torch.mean(torch.abs(y_pred - y_true) + alpha * torch.pow(y_pred, 2))

args = {
        "path": None,
        "idx": 0,
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
        'pad': 1,
        'alpha': 1e-8,
        'iter_num': 100,
        'output_num': 2,
        'init_model': False,
        'transform_type': 'reshape',
        'transform_factor': 0.5,
        'file_type': 'tif',
    }

def get_args():
    return args

def tensor_to_np(tensor):
    if type(tensor) is list:
        if len(tensor[0].shape) <= 2:
            try:
                return [t.detach().cpu().numpy() for t in tensor]
            except:
                return [t.numpy()(t) for t in tensor]
        elif len(tensor[0].shape) == 3:
            try:
                return [t.detach().cpu().numpy()[0,:,:] for t in tensor]
            except:
                return [t.numpy()[0,:,:] for t in tensor]
        else:
            try:
                return [t.detach().cpu().numpy()[0,0,:,:] for t in tensor]
            except:
                return [t.numpy()[0,0,:,:] for t in tensor]
    else:
        if len(tensor.shape) <= 2:
            try:
                return tensor.detach().cpu().numpy()
            except:
                return tensor.numpy()
        elif len(tensor.shape) == 3:
            try:
                return tensor.detach().cpu().numpy()[0,:,:]
            except:
                return tensor.numpy()[0,:,:]
        else:
            try:
                return tensor.detach().cpu().numpy()[0,0,:,:]
            except:
                return tensor.numpy()[0,0,:,:]
    
def torch_reshape(image, complex = False):
    #if it's tensor and of shape 4, return the image
    if type(image) is torch.Tensor and len(image.shape) == 4:
        return image
    
    if type(image) is list:
        #use joblib to parallelize the process
        if len(image) > 30:
            n_jobs = 30
        # else:
        #     n_jobs = 5
            image = Parallel(n_jobs=n_jobs)(delayed(torch.from_numpy)(image[i]) for i in range(len(image)))
        else:
            try:
                image = [torch.from_numpy(image[i]) for i in range(len(image))]
            except:
                image = [image[i] for i in range(len(image))]
        image = torch.stack(image)
    if type(image) is not torch.Tensor:
            image = torch.from_numpy(image)
    
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)    
    elif len(image.shape) > 5:
        image = image.squeeze(1)
        print("Image shape not supported")
        return image
    
    #rearrange the dimension of the image [batch, channel, x, y]
    image = image.permute(1, 0, 2, 3)
    if complex or image.dtype == torch.complex64 or image.dtype == torch.complex128:
        image = image.type(torch.complex64)
    else:
        image = image.type(torch.float32)
    return image

def torch_norm(image):
    image = torch_reshape(image)
    image = (image - torch.mean(image))/torch.std(image)
    return image

def torchnor_phase(image):
    image = torch_reshape(image)
    image = (image - torch.mean(image))/torch.std(image)
    if torch.max(image) != 0:
        image = image / torch.max(image)
    return image

def torch_contrast(image, contrast_factor = 0.02):
    if type(image) is not torch.Tensor:
        image = torch_reshape(image)
    image = transforms.functional.adjust_contrast(image, contrast_factor) 
    return image

def torch_brightness(image, brightness_factor = 0.02):
    image = torch_reshape(image)
    image = transforms.functional.adjust_brightness(image, brightness_factor)
    return image

def torch_detector(image):
    image = torch_reshape(image, complex = True)
    image = torch.abs(image)**2
    return image

def live_plot(self):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    for i in range(self.iter_num):
        clear_output(wait=True)
        plt.figure(figsize=(20,10))
        plt.subplot(1,3,3)
        plt.plot(tensor_to_np(self.gen_loss_list), label='gen_loss')
        plt.plot(tensor_to_np(self.dis_loss_list), label='dis_loss')
        plt.title('iteration: '+str(i))
        plt.legend()
        plt.subplot(1,3,1)
        plt.title('propagated_intensity')
        plt.imshow(tensor_to_np(self.propagated_intensity_list[i]), cmap='gray')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.title('phase')
        plt.imshow(tensor_to_np(self.phase_list[i]), cmap='coolwarm')
        plt.colorbar()
        plt.gca()
        plt.show()

def FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None):
    """  
    H: Real -> Complex space
    
    Parameters: 
          - initial complex field in x-y source plane
            ff - Fresnel factor constructed!
            distance_sample_detector - distance_sample_detector-value (distance from sensor to object)
            ref_image and dark_image - optional background image to divide out from
        
    Returns: propagated complex field in x-y sensor plane"""  
    
    import os
    dtype = torch.complex64
    H = torch.from_numpy(ff).type(dtype)
    H = torch.reshape(H, phase.shape)
    detector_wavefield = torch.exp(torch.complex(-absorption, phase))
    detector_wavefield = detector_wavefield.type(dtype)
    
    # Compute FFT centered about 0
    E0fft = (torch.fft.fft2(detector_wavefield)).type(dtype)
    G = H * E0fft
    I = (torch.abs(torch.fft.ifft2(G))**2).type(dtype)

    if dark_image is not None and ref_image is not None:
        I = I * (ref_image - dark_image) + dark_image
    I = torch_reshape(I) #without normalizing
    return I

def ssim(img1, img2):
    """
    img1 and img2 are torch tensors of shape (batch_size, 1, x, y)
    """
    img1 = torch_reshape(img1)
    img2 = torch_reshape(img2)
    img1 = img1.squeeze(1)
    img2 = img2.squeeze(1)
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    from skimage.metrics import structural_similarity
    ssim = structural_similarity(img1, img2, data_range=img2.max() - img2.min())
    return ssim

def forward_propagate(shape_x = None, shape_y = None, pad = None, energy_kev = None, detector_pixel_size = None, distance_sample_detector = None, phase_image = None, attenuation_image = None, fresnel_factor  = None, wavefield = None, distance = None, mode = 'constant', value = 1, **kwargs):
    assert phase_image is not None or attenuation_image is not None, "phase_image and attenuation_image are not provided"
    if fresnel_factor is None:
        if distance is None:
            distance = distance_sample_detector
        fresnel_factor = ffactor(shape_x*pad, shape_y*pad, energy_kev, distance, detector_pixel_size) if type(distance) is not list else ffactors(shape_x*pad, shape_y*pad, energy_kev, distance, detector_pixel_size)
    # print("fresnel_factor shape", fresnel_factor.shape) if type(distance) is not list else print("fresnel_factor shape", fresnel_factor[0].shape)
    
    if wavefield is None:
        if phase_image is None and attenuation_image is None:
            assert 'phase_image' in kwargs.keys() and 'attenuation_image' in kwargs.keys(), "phase_image and attenuation_image are not provided"
            phase_image = phase_image
            attenuation_image = attenuation_image

        if type(phase_image) is not list:
            phase_image = [phase_image]
        if type(attenuation_image) is not list:
            attenuation_image = [attenuation_image]

        if type(phase_image) is list and type(phase_image[0]) is torch.Tensor and len(phase_image[0].shape) == 4:
            phase_image = torch.stack(phase_image)
            attenuation_image = torch.stack(attenuation_image)
            #squeeze the first dimension
            if len(phase_image.shape) == 5:
                phase_image = phase_image.squeeze(0)
                attenuation_image = attenuation_image.squeeze(0)
        else:
            phase_image = torch_reshape(phase_image)
            attenuation_image = torch_reshape(attenuation_image)
        
        fresnel_factor = torch_reshape(fresnel_factor, complex = True)
                
        # print("phase_image shape, attenuation_image shape, fresnel_factor shape", phase_image.shape, attenuation_image.shape, fresnel_factor.shape)
    
        odd_case = False
        px, py = phase_image.shape[2], phase_image.shape[3]
        if py % 2 != 0:
            phase_image = F.pad(phase_image, (0, 1, 0, 0), mode, value)
            attenuation_image = F.pad(attenuation_image, (0, 1, 0, 0), mode, value)
            odd_case = True
        if px % 2 != 0:
            phase_image = F.pad(phase_image, (0, 0, 0, 1), mode, value)
            attenuation_image = F.pad(attenuation_image, (0, 0, 0, 1),mode, value)
            odd_case = True
        if odd_case:
            new_px, new_py = phase_image.shape[2], phase_image.shape[3]
            if distance is None:
                distance = distance_sample_detector
            fresnel_factor = torch_reshape(ffactor(new_px*pad, new_py*pad, energy_kev, distance, detector_pixel_size), complex = True) if type(distance) is not list else torch_reshape(ffactors(new_px*pad, new_py*pad, energy_kev, distance, detector_pixel_size), complex = True)
            print("odd case", px, py, phase_image.shape, attenuation_image.shape, fresnel_factor.shape)
        
        # print("phase_image shape, attenuation_image shape, fresnel_factor shape", phase_image.shape, attenuation_image.shape, fresnel_factor.shape)
        
        
        #pad the phase and attenuation image to the same size as fresnel_factor by adding ones to the end of the image
        if type(fresnel_factor) is not list and fresnel_factor.shape != phase_image.shape or type(fresnel_factor) is list and fresnel_factor[0].shape != phase_image.shape:
            if mode == 'constant':
                if value == 'mean':
                        phase_value = tensor_to_np(torch.mean(phase_image)).item()
                        attenuation_value = tensor_to_np(torch.mean(attenuation_image)).item()
                else:
                    phase_value = value
                    attenuation_value = value
                phase_image = F.pad(phase_image, (int((fresnel_factor.shape[3] - phase_image.shape[3])/2), int((fresnel_factor.shape[3] - phase_image.shape[3])/2), int((fresnel_factor.shape[2] - phase_image.shape[2])/2), int((fresnel_factor.shape[2] - phase_image.shape[2])/2)), mode = mode, value = phase_value)
                attenuation_image = F.pad(attenuation_image, (int((fresnel_factor.shape[3] - attenuation_image.shape[3])/2), int((fresnel_factor.shape[3] - attenuation_image.shape[3])/2), int((fresnel_factor.shape[2] - attenuation_image.shape[2])/2), int((fresnel_factor.shape[2] - attenuation_image.shape[2])/2)), mode = mode, value = attenuation_value)
            else:
                phase_image = F.pad(phase_image, (int((fresnel_factor.shape[3] - phase_image.shape[3])/2), int((fresnel_factor.shape[3] - phase_image.shape[3])/2), int((fresnel_factor.shape[2] - phase_image.shape[2])/2), int((fresnel_factor.shape[2] - phase_image.shape[2])/2)), mode)
                attenuation_image = F.pad(attenuation_image, (int((fresnel_factor.shape[3] - attenuation_image.shape[3])/2), int((fresnel_factor.shape[3] - attenuation_image.shape[3])/2), int((fresnel_factor.shape[2] - attenuation_image.shape[2])/2), int((fresnel_factor.shape[2] - attenuation_image.shape[2])/2)), mode)
        # print("phase_image shape, attenuation_image shape, fresnel_factor shape", phase_image.shape, attenuation_image.shape, fresnel_factor.shape)
        wavefield = torch.exp(torch.complex(-attenuation_image, phase_image))
    
    else:
        wavefield = torch_reshape(wavefield, complex = True)
        fresnel_factor = torch_reshape(fresnel_factor, complex = True)
        px, py = fresnel_factor.shape[2], fresnel_factor.shape[3]
    # visualize([tensor_to_np(phase_image)])
    I = (torch.abs(torch.fft.ifft2(fresnel_factor * torch.fft.fft2(wavefield)))**2)
    I = I[:, :, int((I.shape[2] - shape_x)/2):int((I.shape[2] + shape_x)/2), int((I.shape[3] - shape_y)/2):int((I.shape[3] + shape_y)/2)]
    I = torch_reshape(I)
    return I
    
class Ganrec_Dataloader(torch.utils.data.Dataset):
    def __init__(self,**kwargs):
        self.kwargs = get_args()
        self.kwargs.update(kwargs)
        self.kwargs.update(get_all_info(**kwargs))
        keys = self.kwargs.keys()
        [self.__setattr__(key, self.kwargs[key]) for key in keys]
        self.dims = (self.ND, self.shape_x, self.shape_y)
        if self.kwargs['transform_type'] == 'reshape':
            self.transformed_images = torch_reshape(self.image)
        elif self.kwargs['transform_type'] == 'normalize':
            self.transformed_images = torchnor_phase(self.image)
        elif self.kwargs['transform_type'] == 'contrast':
            self.transformed_images = torch_reshape(self.image)
            self.transformed_images = torch_contrast(self.transformed_images, kwargs['transform_factor'])
        elif self.kwargs['transform_type'] == 'contrast_normalize':
            self.transformed_images = torch_reshape(self.image)
            self.transformed_images = torch_contrast(self.transformed_images, kwargs['transform_factor'])
            self.transformed_images = torchnor_phase(self.transformed_images)
        elif self.kwargs['transform_type'] == 'brightness':
            self.transformed_images = torch_reshape(self.image)
            self.transformed_images = torch_brightness(self.transformed_images,  kwargs['transform_factor'])
        elif self.kwargs['transform_type'] == 'norm':
            self.transformed_images = torch_norm(self.image)

        else:
            self.transformed_images = torch_reshape(self.image)
    
        self.batch_size = self.transformed_images.shape[0]
        super(Ganrec_Dataloader, self).__init__()

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx = None, transform_type = None): 
            
        if idx is not None:
            kwargs = self.kwargs
            if transform_type is not None:
                kwargs['transform_type'] = transform_type
            kwargs["idx"] = idx
            new = Ganrec_Dataloader(**kwargs)
            return new.transformed_images
        else:
            return self.transformed_images
    
    def __getbatchimages__(self, batch_size = None, transform_type = None):
        """
        Returns a batch of transformed images:
        ********************************
        batch_images, idx_batches = dataloader.__getbatchimages__(batch_size = 100, transform_type = 'reshape')
        for i, image in enumerate(batch_images):
            print(image.shape)
        ********************************
        """
        if batch_size is None:
            batch_size = self.batch_size
        if transform_type is None:
            transform_type = self.kwargs['transform_type']
        all_images = self.all_images
        n = len(all_images)
        seed = self.seed
        import random
        random.seed(seed)
        all_idx = list(range(n))
        random.shuffle(all_idx)
        idx_batches = [all_idx[i:i + batch_size] for i in range(0, n, batch_size)]
        batch_images = []
        from joblib import Parallel, delayed
        n_jobs = max(30, len(idx_batches))
        batch_images = Parallel(n_jobs=n_jobs)(delayed(self.__getitem__)(idx, transform_type) for idx in idx_batches)
        return batch_images, idx_batches
    
        
    def normalize(self, idx = None, transform_type = None):
        image = self.__getitem__(idx, transform_type)
        # image = torchnor_phase(image)
        return image
    
    def get_kwargs(self):
        return self.__dict__
    
    def visualize(self, idx = None, random = False, show_or_plot = 'show', tranformed = False):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            new = Ganrec_Dataloader(**kwargs)
        else:
            new = self
        if tranformed:
            images = new.transformed_images
            try:
                images = images[:,0,:,:].detach().cpu().numpy()
            except:
                images = images[0,:,:].numpy()
        else:
            images = new.image

        if type(images) is not list:
            # images = [images]
            images = list(images)
        rows = int(np.sqrt(len(images)))
        if rows ==1:
            cols = len(images)
        else:
            cols = rows + 1
        fig = visualize(images, rows=rows, cols = cols, random=random, show_or_plot = show_or_plot)
        return fig
    
    def forward_propagate(self, distance = None):
        distance = self.distance_sample_detector if distance is None else distance
        wavefield = None if 'wavefield' not in self.kwargs.keys() else self.kwargs['wavefield']
        print(distance)

        if type(distance) is not list:
            self.propagated_forward = forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = distance, phase_image = self.phase, attenuation_image = self.attenuation, fresnel_factor  = self.fresnel_factor, wavefield = wavefield, distance = distance)
        else:
            propagated_forward = [forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = distance[i], phase_image = self.phase, attenuation_image = self.attenuation, fresnel_factor  = None, wavefield = wavefield, distance = distance[i]) for i in range(len(distance))]
            self.propagated_forward = torch.stack(propagated_forward, dim = 0)[:,0,:,:,:]
            print(self.propagated_forward.shape)
        return self.propagated_forward

class unet_Module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depth, bias, batch_norm, activation, device = None):
        super(unet_Module, self).__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.batch_norm = batch_norm
        self.activation = activation
        
        self.down_list = [2**(i+5) for i in range(depth)]
        self.up_list = [2**(4+depth-i) for i in range(depth)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_list = [in_channels] + self.down_list
        self.up_list = self.up_list + [out_channels]
        
        self.down_conv = nn.ModuleList([nn.Conv2d(self.down_list[i], self.down_list[i+1], kernel_size, stride, padding, bias=bias) for i in range(depth) if i < depth])
        self.up_conv = nn.ModuleList([nn.ConvTranspose2d(self.up_list[i], self.up_list[i+1], kernel_size, stride, padding, bias=bias) for i in range(depth)])
        self.down_bn = nn.ModuleList([nn.BatchNorm2d(self.down_list[i+1]) for i in range(depth)])
        self.up_bn = nn.ModuleList([nn.BatchNorm2d(self.up_list[i+1]) for i in range(depth)])
        self.concat_conv = nn.ModuleList([nn.ConvTranspose2d(self.up_list[i+1] + self.up_list[i], self.up_list[i+1], kernel_size, stride, padding, bias=bias) for i in range(depth-1)])
        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.batch_norm = batch_norm
        if device is not None:
            self.device = device
        else:
            self.device = None
        
        self.alpha = 1e-8
    def forward(self, x):
        """
        *********************************************************************************************************************************************
            first stage: convolution: 1 -> 32 -> 64 | then downsampling 
            second stage: convolution: 64 -> 128 | then downsampling 128 -> 256 | 
            third stage: conv: 256 -> 512 | downsampling: 512 -> 1024 | 
            fourthstage: deconv: 1024 -> 512 | upsampling: 512 -> 256 | 
            thirdstage: combine previous thirdstage and deconv: 512+256 -> 128 | 
            second stage: combine with prev second stage and deconv:  256 + 128 -> 64 | 
            first stage: combine with prev first stage and deconv: 64 -> 32 | 32 -> 2   where 2 is the number of output channels   
        *********************************************************************************************************************************************     
        """
        self.input_size = x.detach().cpu().numpy().shape
        out = []
        x = self.down_conv[0](x)
        x = nn.ReLU()(x)
        # x = nn.SiLU()(x)

        
        for i in range(1,self.depth):          
            x = self.down_conv[i](x)
            if self.batch_norm:
                x = self.down_bn[i](x)
            # x = nn.SiLU()(x)
            x = nn.ReLU()(x)
            x = self.downsample(x)
            if i != self.depth-1:
                out.append(x)
        for i in range(self.depth - 1):
            x = self.up_conv[i](x)
            if i != 0:
                x = torch.cat([x, out[-i]], dim=1)
                x = self.concat_conv[i](x)
            x = self.upsample(x)
            x = nn.SiLU()(x)
            if self.batch_norm:
                x = self.up_bn[i](x)
            x = nn.ReLU()(x)
            # x = nn.SiLU()(x)

        x = self.up_conv[self.depth-1](x)
        self.out = x
        return x

    def propagate(self, x, fresnel_factor):
        #the error has two parts:
        #1. the error between the propagated wave intensity and the input wave
        #2. the error between phase,phase2 and attenuation,attenuation2
        
        out = self.forward(x)
        phase = out[:,0,:,:]
        attenuation = out[:,1,:,:]

        dtype = torch.complex64
        wave = torch.exp(1j * phase - attenuation).type(dtype)
        fresnel_factor.type(dtype).to(self.device)
        # fresnel_factor = torch.reshape(torch.from_numpy(ff), (1,1,ff.shape[0],ff.shape[1])).type(dtype)

        propagated_wave = torch.fft.ifft2(torch.fft.fft2(wave) * fresnel_factor)
        propagated_intensity = torch.abs(propagated_wave)**2
        return propagated_intensity, phase, attenuation
        
        # propagate = FresnelPropagator(phase, attenuation, ff=ff)
        # intensity = torch_brightness(torch.abs(propagate)**2)
        # return intensity, phase, attenuation

    def size(self) -> int:
        return super().__sizeof__()
    
    def summary(self, input_size):
        "example: self.summary((1,128,128))"
        from torchsummary import summary
        overall = summary(self, input_size=input_size)
        return overall

    def viz_model(self, input_size):
        """example: self.viz_model((3,128,128))"""
        from torchviz import make_dot
        random_input = torch_reshape(torch.randn(input_size))
        return make_dot(self(random_input), params=dict(list(self.named_parameters())))
    
    def train(self, ff, epochs, dataloader, optimizer, loss_function):
        for epoch in range(epochs):
            intensity, phase, attenuation = self.propagate(dataloader.transformed_images, ff)
            loss = loss_function(intensity, dataloader.images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: {}, loss: {}'.format(epoch, loss))
        return intensity, phase, attenuation

criterionGAN = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

class unet_light(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depth, bias, batch_norm, activation):
        super(unet_light, self).__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def setup(self, stage=None):
        self.unet = unet_Module(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.depth, self.bias, self.batch_norm, self.activation)
        return self.unet
    
    def forward(self, x, plot=False):
        unet = self.setup()
        if plot:
            yield unet.viz_model(x.shape)
        return unet(x)

    def training_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        y_hat = self.forward(x)
        phase = y_hat[:,0,:,:]
        attenuation = y_hat[:,1,:,:]
        ff = torch_reshape(kwargs["ff"])
        projection = FresnelPropagator(phase, attenuation, ff)
        intensity = torch.pow(torch.abs(projection), 2)
        loss = L1_error(intensity, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer