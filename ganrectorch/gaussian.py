from setup import *
from utils import load_images_parallel, fun_images_parallel
# from ganrec_dataloader import *
from models import *
from gaussian_blur import *
from visualize import visualize
from unet import UNet
import random
import time


from scipy.ndimage import gaussian_filter, median_filter
from skimage.data import shepp_logan_phantom, camera
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as mSSIM
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM


from torch import optim
import torchvision.transforms as T

def get_shape(image):
    if image is None:
        return None, None, None, None
    if type(image) is not list:
        image = [image]
    n_of_images= len(image)
    if len(image[0].shape) == 2:
        px, py = image[0].shape
        ND = n_of_images
        return ND, 1, px, py
    elif len(image[0].shape) == 3:
        shape = image[0].shape
        ND = min(shape)
        shape = [shape[i] for i in range(len(shape)) if shape[i] != ND]
        px, py = shape
        return n_of_images, ND, px, py
    elif len(image[0].shape) == 4:
        shape = image[0].shape
        if 1 in shape:
            shape = [shape[i] for i in range(len(shape)) if shape[i] != 1]
        ND = min(shape)
        shape = [shape[i] for i in range(len(shape)) if shape[i] != ND]
        channels = min(shape)
        shape = [shape[i] for i in range(len(shape)) if shape[i] != channels]
        px, py = shape
        return ND, channels, px, py
    else:
        raise ValueError("The input image has to be 2D, 3D or 4D")

def fresnel_calc(energy, z, pv):
    """z and pv have to in meters"""
    if energy is None or z is None or pv is None:
        return None
    if type(energy) is not list or type(z) is not list or type(pv) is not list:
        wavelength = wavelength_from_energy(eneryg_J(energy)).magnitude
        fresnel_number = pv**2/(wavelength*z) 
    else:
        energy = [energy] if type(energy) is not list else energy
        wavelength = [wavelength_from_energy(eneryg_J(ener)).magnitude for ener in energy]
        z = [z] if type(z) is not list else z
        pv = [pv] if type(pv) is not list else pv
        fresnel_number = []
        for i in range(len(energy)):
            for j in range(len(z)):
                for k in range(len(pv)):
                    fresnel_number.append(pv[k]**2/(wavelength[i]*z[j]))
    return  fresnel_number

def base_coeff(px = None, py=None, image=None):
    if [px, py] == [None, None]:
        ND, channels, px, py = get_shape(image)
    freq_1 = fftfreq(px)
    freq_2 = fftfreq(py)
    xi, eta = np.meshgrid(freq_1, freq_2)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    return np.exp((xi ** 2 + eta ** 2) / 2).T

def ffactor(energy=None, z=None, pv=None, px = None, py = None, image = None, fresnel_number = None):
    if px== None and py==None and image is not None:
        ND, channel, px, py = get_shape(image)
    if fresnel_number == None:
        fresnel_number = fresnel_calc(energy, z, pv)
    else:
        if [energy, z, pv] != [None, None, None]:
            if fresnel_number != fresnel_calc(energy, z, pv): 
                print("fresnel_number is not consistent with energy, z, and pv")
    
    basecoeff = base_coeff(px, py, image)
    if type(fresnel_number) is not list:
        ffs = basecoeff**(-1j*2*np.pi/fresnel_number)
    else:
        ffs = [basecoeff**(-1j*2*np.pi/fres) for fres in fresnel_number]
    return ffs

def get_image(path, idx = None, file_type=None, **kwargs):
    if type(path) is not list:
        if type(path) is str:
            if os.path.isdir(path):
                images = list(io.imread_collection(path + '/*.' + file_type).files)
                
                if idx is None:
                    image_path = path
                    
                else:
                    image_path = [path[i] for i in idx]
                    
                image = load_images_parallel(image_path)
                
            elif os.path.isfile(path):
                image = io.imread(path)
                image_path = path
            
                if len(images.shape) == 2:
                    image = images
                elif len(images.shape) == 3:
                    image = images[idx,:,:] if idx is not None else images
                else:
                    image = images[idx,:,:,:] if idx is not None else images
                image = [image]
        elif 'numpy' in str(type(path)) or 'torch' in str(type(path)) or 'jax' in str(type(path)):
            if len(path.shape) == 2:
                image = path
            elif len(path.shape) == 3:
                image = path[idx,:,:] if idx is not None else path
            else:
                image = path[idx,:,:,:] if idx is not None else path
            image = [image]
            image_path = None
        elif 'ImageCollection' in str(type(path)):
            image_path = path.files
            image_path = [image_path[i] for i in idx] if idx is not None else image_path
            image = load_images_parallel(image_path)
        
        else:
            try:
                image_path = None
                image = path
            except:
                image_path = None
                image = None
                print("couldn't load image from path")
            pass
    else:
        image = []
        image_path = []
        for path in kwargs['path']:
            image_, image_path_ = get_image(path, kwargs['idx'], kwargs['file_type'])
            image += image_
            image_path += image_path_
        kwargs['image'] = image
        kwargs['image_path'] = image_path
    return image, image_path

def prepare_dict(**kwargs):
    similar_terms = [
        ['path', 'paths','images', 'i_inputs', 'image', 'i_input', 'hologram', 'intensity'],
        ['file_type', 'file_types', 'filetype', 'filetypes'],
        ['idx', 'indices', 'index'],
        ['energy', 'energy_kev'], 
        ['detector_pixel_size', 'pv'],
        ['distance_sample_detector', 'z'],
        ['fresnel_number', 'fresnel_number', 'fresnelnumbers', 'fresnelnumbers'],
        ['fresnel_factor', 'ffs', 'frensel_factors', 'fresnelfactor'],
        ['lam', 'lamda', 'wavelength', 'wave_length'],
        ['phase', 'phase_image'],
        ['attenuation', 'attenuation_image'],
        ['pad', 'pad_value', 'magnification_factor', 'upscale'],
        ['downsampling_factor'],
        ['mode', 'pad_mode'],
        ['experiment_name'],
        ['task', 'method'],
        ['alpha', 'alpha_value'],
        ['abs_ratio'],
        ['delta_beta', 'delta_beta_value'],
        ['shape_x', 'px'],
        ['shape_y', 'py'],
        ]
    for i, terms in enumerate(similar_terms):
        for term in terms:
            if term in kwargs.keys():
                kwargs[similar_terms[i][0]] = kwargs[term]
                break

    if kwargs['idx'] is not None:
        kwargs['idx'] = [kwargs['idx']] if type(kwargs['idx']) is not list else kwargs['idx']

    keys_to_search = ['pad', 'mode',   'task',     'alpha', 'delta_beta', 'idx', 'file_type',       'save_path',              'save', 'save_format', 'save_all', 'downsampling_factor', 'fresnel_number', 'fresnel_factor']
    when_none =     [  1,   'reflect', 'learn_phase',  1e-8,     1e1,      None,     'tif',      os.getcwd() + '/results/',    False,    'tif',         False,          1,                 None,             None]
    for i in range(len(keys_to_search)):
        if keys_to_search[i] not in kwargs.keys() or kwargs[keys_to_search[i]] is None:
            kwargs[keys_to_search[i]] = when_none[i]
    
    assert kwargs['path'] is not None or kwargs['phase'] is not None or kwargs['attenuation'] is not None, "path, phase or attenuation are not provided"
    kwargs['image'], kwargs['image_path'] = get_image(kwargs['path'], kwargs['idx'], kwargs['file_type'])
    if kwargs['downsampling_factor'] > 1:
        tensor_image = torch_reshape(kwargs['image'])
        tensor_image = T.Resize((tensor_image.shape[2]//kwargs['downsampling_factor'], tensor_image.shape[3]//kwargs['downsampling_factor']))(tensor_image)
        kwargs['image'] = [tensor_to_np(tensor_image)]
        kwargs['distance_sample_detector'] = kwargs['distance_sample_detector'] * kwargs['downsampling_factor'] if kwargs['distance_sample_detector'] is not None else None
        kwargs['fresnel_number'] = kwargs['fresnel_number'] * kwargs['downsampling_factor']**2 if kwargs['fresnel_number'] is not None else None
    
    kwargs['ND'], kwargs['channels'], kwargs['shape_x'], kwargs['shape_y'] = get_shape(kwargs['image'][0])
    kwargs['shape'] = [kwargs['shape_x'], kwargs['shape_y']]
    kwargs['lam'] = wavelength_from_energy(eneryg_J(kwargs['energy'])).magnitude if kwargs['energy'] is not None else None
    kwargs['fresnel_number'] = fresnel_calc(kwargs['energy'], kwargs['distance_sample_detector'], kwargs['detector_pixel_size']) if kwargs['fresnel_number'] is None else kwargs['fresnel_number']
    kwargs['fresnel_factor'] = ffactor(energy = kwargs['energy'], z = kwargs['distance_sample_detector'], pv = kwargs['detector_pixel_size'], image = kwargs['image'], fresnel_number = kwargs['fresnel_number']) if kwargs['fresnel_factor'] is None else kwargs['fresnel_factor']
    return kwargs

class tensor_to_numpy(nn.Module):
    def __init__(self):
        super(tensor_to_numpy, self).__init__()
    def forward(self, x):
        return tensor_to_np(x)

class numpy_to_torch(nn.Module):
    def __init__(self, complex = False):
        super(numpy_to_torch, self).__init__()
        self.complex = complex
    def forward(self, x):
        return torch_reshape(x, complex = self.complex)
    
class gaussian_filtering(nn.Module):
    def __init__(self):
        super(gaussian_filtering, self).__init__()
    def forward(self, x, sigma):
        x = tensor_to_np(x)

        blurred = gaussian_filter(x, sigma)
        
        return torch_reshape(blurred, complex = False)
     
class Ganrec_Dataloader(torch.utils.data.Dataset):
    """
    The dataloader takes the different arguements and arranges it in a way that it can be used for the training and other purposes.
    ********************************
    Transform_type can be: 'reshape', 'normalize', 'contrast', 'contrast_normalize', 'brightness', 'norm'
    The image is then saved in the transformed_images variable. This will be used for the training.

    If there are multiple images, the idx variable is used to choose the image. 
    Batch_size is the number of images in the batch, which can be used for the training. 
    The batch_images variable is used to get the batch of images. The idx_batches variable is used to get the
    
    get_all_info() is used to get the information about the image and the fresnel factor.

    If phase and attenuation are provided, the forward propagation is done and saved in the propagated_forward variable.
    The fresnel factor is also saved in the fresnel_factor variable.
    ********************************
    """
    def __init__(self,**kwargs):
        self.kwargs = get_args()
        self.kwargs.update(kwargs)
        
        if 'downsampling_factor' not in self.kwargs.keys():
            self.kwargs['downsampling_factor'] = 1
        else:
            from skimage.transform import resize
            
        self.kwargs.update(prepare_dict(**kwargs))
        if 'phase' not in self.kwargs.keys() and 'attenuation' not in self.kwargs.keys():
            kwargs['phase'] = None
            kwargs['attenuation'] = None
        keys = self.kwargs.keys()
        [self.__setattr__(key, self.kwargs[key]) for key in keys]
        self.dims = (self.ND, self.shape_x, self.shape_y)
        self.transformed_images = transform(self.image, self.kwargs['transform_type'], self.kwargs['transform_factor']).requires_grad_(True)
        self.kwargs['transform_type'] = self.transform_type
        self.kwargs['transform_factor'] = self.transform_factor
        self.kwargs['transformed_images'] = self.transformed_images
        self.batch_size = self.transformed_images.shape[0]
        super(Ganrec_Dataloader, self).__init__()
        if 'mode' in self.kwargs.keys():
            self.mode = self.kwargs['mode']
        else:
            self.mode = 'reflect'
        if 'value' in self.kwargs.keys():
            self.value = self.kwargs['value']

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
    
    def update_values(self, change_all = False, **info):
        """
        This function is used to update the values of the dataloader. 
        If the change is a shouldn't affect values from the get_all_info() function, then change_all should be False.
        Otherwise, it should be True.
        """

        if change_all:
            kwargs = self.kwargs
            kwargs.update(info)
            kwargs.update(get_all_info(**kwargs))
            [self.__setattr__(key, kwargs[key]) for key in kwargs.keys()]
            self.transformed_images = transform(self.image, kwargs['transform_type'], kwargs['transform_factor'])
            self.dims = (self.ND, self.shape_x, self.shape_y)
            print(self.kwargs['transform_type'])
        else:
            kwargs = self.kwargs
            kwargs.update(info)
            [self.__setattr__(key, info[key]) for key in info.keys()]
            # if 'transform_type' in info.keys() or 'transform_factor' in info.keys():
            #     self.transformed_images = transform(self.image, self.kwargs['transform_type'], self.kwargs['transform_factor'])
            #     self.dims = (self.ND, self.shape_x, self.shape_y)
            #     print(self.kwargs['transform_type'])
        
        return self
            
    def normalize(self, idx = None, transform_type = None):
        image = self.__getitem__(idx, transform_type)
        # image = torchnor_phase(image)
        return image
    
    def get_kwargs(self):
        return self.__dict__
    
    def visualize(self, idx = None, show_or_plot = 'show', tranformed = False):
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
        fig = visualize(images, show_or_plot = show_or_plot)
        return fig
    
    def forward_propagate(self, distance = None):
        distance = self.distance_sample_detector if distance is None else distance
        wavefield = None if 'wavefield' not in self.kwargs.keys() else self.kwargs['wavefield']

        if type(distance) is not list:
            self.propagated_forward = forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = distance, phase_image = self.phase, attenuation_image = self.attenuation, fresnel_factor  = self.fresnel_factor, wavefield = wavefield, distance = distance, mode = self.mode, value = self.value)
        else:
            propagated_forward = [forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = distance[i], phase_image = self.phase, attenuation_image = self.attenuation, fresnel_factor  = None, wavefield = wavefield, distance = distance[i], mode=self.mode, value=self.value) for i in range(len(distance))]
            self.propagated_forward = torch.stack(propagated_forward, dim = 0)[:,0,:,:,:]
            print(self.propagated_forward.shape)
        return self.propagated_forward
   
class make_ganrec_model(nn.Module):
    def __init__(self, shape_x, shape_y, conv_num, conv_size, dropout, output_num, fresnel_factor, transformed_images=None, device=None, **kwargs):
        super(make_ganrec_model, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        
        self.base_coeff = to_device(torch_reshape(self.base_coeff, complex=True), self.device) if 'base_coeff' in self.__dict__.keys() else None
        self.fresnel_factor = to_device(torch_reshape(fresnel_factor, complex=True), self.device) if fresnel_factor is not None else None
        self.transformed_images = to_device(transformed_images, self.device) 
        self.image = to_device(self.image, self.device)
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.conv_num = conv_num
        self.conv_size = conv_size
        self.dropout = dropout
        self.output_num = output_num
        self.units = kwargs['units'] if 'units' in kwargs.keys() else 128
        self.fc_size = shape_x * shape_y
        self.task = kwargs['task'] if 'task' in kwargs.keys() else 'phase_retrieval'
        self.input_channels = kwargs['input_channels'] if 'input_channels' in kwargs.keys() and kwargs['input_channels'] is not None else 1
        
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        if 'mode' in kwargs.keys():
            self.mode = kwargs['mode']
        else:
            self.mode = 'constant'
        if 'value' in kwargs.keys():
            self.value = kwargs['value']
        else:
            self.value = 'mean'
        if 'abs_ratio' in kwargs.keys():
            self.abs_ratio = kwargs['abs_ratio']
        else:
            self.abs_ratio = 1
        ##################################################################################################
        # We first define the generator model
        ##################################################################################################
        
        
        if self.fc_depth == 0:
            self.fc_stack = []
        else:
            units = [self.units]*self.fc_depth
            self.fc_submodule = nn.ModuleList([
                dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout) for i in range(len(units))
            ])
            self.fc_stack = nn.ModuleList([
                nn.Flatten(),
                Transpose(),
                dense_layer(in_features=1, out_features=self.units, dropout=self.dropout, transpose=False),
                *self.fc_submodule,
                dense_layer(in_features=self.units, out_features=1, dropout=0),
                Reshape((-1, 1, self.shape_x, self.shape_y)),
            ])
        
    

        if self.cnn_depth == 0:
            self.cnn_stack = []
        else:
            conv_size_add_list = list(np.arange(1,self.cnn_depth+1))
            deconv_size_add_list = list(np.arange(self.cnn_depth, 0, -1))
            self.conv_stack = nn.ModuleList([
                conv2d_layer(in_channels=conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+conv_size_add_list[i], stride=1, apply_batchnorm=True, normal_init=True) for i in range(len(conv_size_add_list))
            ])

            self.dconv_stack = nn.ModuleList([
                deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+deconv_size_add_list[i], stride=1, apply_batchnorm=True, normal_init=True) for i in range(len(deconv_size_add_list))
            ])
            self.cnn_stack = nn.ModuleList([      
                conv2d_layer(in_channels=1, out_channels=self.conv_num, kernel_size=self.conv_size, stride=1), 
                *self.conv_stack[0:],
                *self.dconv_stack,
                deconv2d_layer(in_channels=self.conv_num, out_channels=self.output_num, kernel_size=self.conv_size, stride=1),
    
            ])
        
        if self.depth == 0:
            self.generator_model = to_device(nn.Sequential(
                *self.fc_stack,
                *self.cnn_stack,
            ), self.device)
        else:
            self.generator_model = to_device(nn.Sequential(*self.fc_stack, UNet(n_channels=self.input_channels, n_classes=self.output_num, bilinear=True)), self.device)

        if 'init_model' in kwargs.keys():
            if kwargs['init_model']:
                # Load the model
                init_model_path = kwargs.get('init_model_path', 'model/ganrec_model')
                self.generator_model.load_state_dict(torch.load(init_model_path))

        else:
            self.init_weights()

        ##################################################################################################
        # We then define the discriminator model
        ##################################################################################################
        in_channels_list = list(2**np.arange(self.dis_depth)) # [1, 16, 16, 32, 32]
        out_channels_list =list(2**np.arange(1,self.dis_depth+1)) #[16, 16, 32, 32, 64]
        kernel_size_list = [self.conv_num]*len(in_channels_list) #[3,3,3,3,3]
        stride_list = [1]*len(in_channels_list) #[2,2,2,2,2]
        discriminator_stack = nn.ModuleList([
            conv2d_layer(in_channels=in_channels_list[i], out_channels=out_channels_list[i], kernel_size=kernel_size_list[i], stride=stride_list[i]) for i in range(len(in_channels_list))
        ])

        self.discriminator_model =  to_device(nn.Sequential(
            *discriminator_stack,
            nn.Flatten(),
        ), self.device)

        if 'task' not in kwargs.keys():
            self.task = 'phase_retrieval'

        if self.task == 'learn_gaussian':
            self.gaussian_filter = to_device(Gaussian_challenge(self.transformed_images, kernel_size=self.gaussian_kernel_size, sigma=self.sigma), self.device)
            # self.gaussian_filter = to_device(T.GaussianBlur(kernel_size=self.gaussian_kernel_size, sigma=self.sigma), self.device)

        self.ssim = to_device(SSIM(), self.device)
        self.psnr = to_device(PSNR(), self.device)
        self.mssim = to_device(mSSIM(), self.device)
        # self.fid.update(self.transformed_images, real=True)
        

        if 'ground_truth' in kwargs.keys() and kwargs['ground_truth'] is not None:
            self.ground_truth = to_device(self.transform(kwargs['ground_truth']), self.device)
        else:
            self.ground_truth = None

        self.reshaped = to_device(transform(self.transformed_images, 'reshape') , self.device)
        self.normalized = transform(self.reshaped, 'normalize')
        self.norm = transform(self.reshaped, 'norm')
        self.contrast = transform(self.reshaped, 'contrast', self.transform_factor)
        self.contrast_normalize = transform(self.reshaped, 'constrast_normalize', self.transform_factor)
        self.brightness = transform(self.reshaped, 'brightness', self.transform_factor)
        self.brightness_normalize = transform(self.reshaped, 'brightness_normalize', self.transform_factor)
        self.fourier = torch.fft.fft2(self.reshaped)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def run_different_scenarios(model, df , i):
        df.loc[i, 'iter_num'] = len(model.gen_loss_list)
        df.loc[i, 'gen_loss'] = model.gen_loss_list[-1]
        df.loc[i, 'dis_loss'] = model.dis_loss_list[-1]
        df.loc[i, 'main_diff'] = model.main_diff_list[-1]
        df.loc[i, 'ssim_list'] = model.ssim_list[-1]
        df.loc[i, 'psnr_list'] = model.psnr_list[-1]
        df.loc[i, 'mssim_list'] = model.mssim_list[-1]
        df.loc[i, 'setup_info'] = get_file_nem(model.__dict__)
        return df
    
    def refine_parameters_using_condition(self, condition = None, values = None, change_from_soure = False, info = None, ratio = None, plot_pd=False, show_images=False, cmap = 'gray'):
        if ratio is None:
            ratio = {'l1_ratio': 10, 'contrast_ratio': 0, 'normalized_ratio': 0, 'brightness_ratio': 0, 'reg_l1_ratio': 0.001, 'reg_l2_ratio': 0.001, 'contrast_normalize_ratio': 0, 'brightness_normalize_ratio': 0, 'l2_ratio': 0, 'fourier_ratio': 0, 'norm_ratio': 0, 'entropy_ratio': 1, 'real_loss_ratio': 1, 'fake_loss_ratio': 1}
        
        
        last_unblurred_images = []
        last_phases = []
        last_atts = []
        last_props = []
        df = pd.DataFrame(columns=['iter_num', 'gen_loss', 'dis_loss', 'main_diff', 'ssim_list', 'psnr_list', 'mssim_list', 'setup_info'])
        if type(ratio) is list:
            df.index.name = 'ratios'
            
            for i, r in enumerate(ratio):
                model = make_ganrec_model(**self.__dict__)
                if self.task == 'learn_gaussian':
                    model.train(save_model = False, save_model_path = 'model/ganrec_model', **r)
                    last_unblurred_images.append(model.unblurred_list[-1])
                    last_props.append(model.propagated_intensity_list[-1])
                    
                else:
                    model.train(save_model = False, save_model_path = 'model/ganrec_model', **r)
                    last_phases.append(model.phase_list[-1])
                    last_atts.append(model.attenuation_list[-1])
                    last_props.append(model.propagated_intensity_list[-1])

                print('gen_loss: ', model.gen_loss_list[-1], 'dis_loss: ', model.dis_loss_list[-1], 'main_diff: ', model.main_diff_list[-1])
                df.loc[i, 'iter_num'] = len(model.gen_loss_list)
                df.loc[i, 'gen_loss'] = model.gen_loss_list[-1]
                df.loc[i, 'dis_loss'] = model.dis_loss_list[-1]
                df.loc[i, 'main_diff'] = model.main_diff_list[-1]
                df.loc[i, 'ssim_list'] = model.ssim_list[-1]
                df.loc[i, 'psnr_list'] = model.psnr_list[-1]
                df.loc[i, 'mssim_list'] = model.mssim_list[-1]
                df.loc[i, 'setup_info'] = get_file_nem(model.__dict__)
            #replace nan values with 0
            df = df.fillna(0)
            column_range = ['main_diff', 'gen_loss', 'dis_loss', 'ssim_list', 'psnr_list', 'mssim_list']
            #list_element.index(element)
            min_vals = [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]
            max_vals = [df[column].max() for column in column_range], [df[column].idxmax() for column in column_range]
            
            if plot_pd:         
                plot_pandas(df, column_range=column_range)
            if show_images:
                if self.task == 'learn_gaussian':
                    visualize(last_unblurred_images, title = ['unblurred_' + str(i) for i in range(len(last_unblurred_images))], cmap = cmap)
                    visualize(last_props, title = ['propagated_intensity_' + str(i) for i in range(len(last_props))], cmap = cmap)
                else:
                    visualize(last_props, title = ['propagated_intensity_' + str(i) for i in range(len(last_props))], cmap=cmap)
                    visualize(last_phases, title = ['phase_' + str(i) for i in range(len(last_phases))], cmap=cmap)

            if self.task == 'learn_gaussian':
                return df, last_unblurred_images, last_props, min_vals, max_vals
            else:
                return df, last_props, last_phases, last_atts, min_vals, max_vals
        else:
            df.index.name = condition

            if type(values) is not list:
                values = [values]

            for i, value in enumerate(values):    
                if change_from_soure == True:
                    info[condition] = value if condition in info.keys() else info.update({condition: value})
                    new_dataloader = Ganrec_Dataloader(**info)
                    model = make_ganrec_model(**new_dataloader.__dict__)
                else:
                    kwargs = self.__dict__
                    kwargs[condition] = value if condition in kwargs.keys() else kwargs.update({condition: value})
                    model = make_ganrec_model(**kwargs)

                if self.task == 'learn_gaussian':
                    model.train(save_model = False, save_model_path = 'model/ganrec_model', **ratio)
                    last_unblurred_images.append(model.unblurred_list[-1])
                    last_props.append(model.propagated_intensity_list[-1])

                else:
                    model.train(save_model = False, save_model_path = 'model/ganrec_model', **ratio)
                    last_phases.append(model.phase_list[-1])
                    last_atts.append(model.attenuation_list[-1])
                    last_props.append(model.propagated_intensity_list[-1])

                print('gen_loss: ', model.gen_loss_list[-1], 'dis_loss: ', model.dis_loss_list[-1], 'main_diff: ', model.main_diff_list[-1])
                df.loc[i, 'iter_num'] = len(model.gen_loss_list)
                df.loc[i, 'gen_loss'] = model.gen_loss_list[-1]
                df.loc[i, 'dis_loss'] = model.dis_loss_list[-1]
                df.loc[i, 'main_diff'] = model.main_diff_list[-1]
                df.loc[i, 'ssim_list'] = model.ssim_list[-1]
                df.loc[i, 'psnr_list'] = model.psnr_list[-1]
                df.loc[i, 'mssim_list'] = model.mssim_list[-1]
                df.loc[i, 'setup_info'] = get_file_nem(model.__dict__)
            #replace nan values with 0
            df = df.fillna(0)
            column_range = ['main_diff', 'gen_loss', 'dis_loss', 'ssim_list', 'psnr_list', 'mssim_list']
            #list_element.index(element)
            min_vals = [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]
            max_vals = [df[column].max() for column in column_range], [df[column].idxmax() for column in column_range]

            if plot_pd:
                plot_pandas(df, column_range=column_range)
            if show_images:
                if self.task == 'learn_gaussian':
                    visualize(last_unblurred_images, title = ['unblurred_' + str(i) for i in range(len(last_unblurred_images))], cmap = cmap)
                    visualize(last_props, title = ['propagated_intensity_' + str(i) for i in range(len(last_props))], cmap = cmap)
                else:
                    visualize(last_props, title = ['propagated_intensity_' + str(i) for i in range(len(last_props))], cmap=cmap)
                    visualize(last_phases, title = ['phase_' + str(i) for i in range(len(last_phases))], cmap=cmap)

            if self.task == 'learn_gaussian':
                return df, last_unblurred_images, last_props, min_vals, max_vals
            else:
                return df, last_props, last_phases, last_atts, min_vals, max_vals

    def make_model(self):
        # self.generator = self.generator_model
        # self.discriminator = self.discriminator_model
        
        # self.generator_optimizer = torch.optim.Adam(self.generator_model.parameters(), lr=1e-1)
        # self.discriminator_optimizer = torch.optim.Adam(self.discriminator_model.parameters(), lr=1e-2)

        # if 'task' in self.__dict__.keys() and self.task == 'learn_gaussian':
        if 'g_learning_rate' not in self.__dict__.keys():
            self.g_learning_rate = 1e-5
        if 'd_learning_rate' not in self.__dict__.keys():
            self.d_learning_rate = 1e-8
        if 'weight_decay' not in self.__dict__.keys():
            self.weight_decay = 1e-8
        if 'momentum' not in self.__dict__.keys():
            self.momentum = 0.9
        if 'amp' not in self.__dict__.keys():
            self.amp = False

        self.generator_optimizer = optim.RAdam(self.generator_model.parameters(),lr=self.g_learning_rate, weight_decay=self.weight_decay)
        self.discriminator_optimizer = optim.Adam(self.discriminator_model.parameters(), lr=self.d_learning_rate, weight_decay=self.weight_decay, amsgrad=True, maximize=True)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.generator_optimizer, 'min', patience=5)  # goal: maximize Dice score
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)    
        
    def forward_generator(self, x):
        self.pred = self.generator_model(x)
        return self.pred

    def forward_discriminator(self, x):
        self.likelihood = self.discriminator_model(x)
        return self.likelihood
    
    def transform(self, x):
        if 'transform_type' in self.kwargs.keys():
            transform_type = self.transform_type
        else:
            transform_type = 'normalize'

        if 'transform_factor' in self.kwargs.keys():
            transform_factor = self.transform_factor
        else:
            transform_factor = 0.1
        return transform(x, transform_type, transform_factor)

    def propagator(self):
        phase = transform(self.pred[:,0,:,:], 'normalize') #*2 -1) * torch.pi
        attenuation =  (1 - transform(self.pred[:,1,:,:], 'normalize'))*self.abs_ratio
        propagated_intensity = transform(forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = self.distance_sample_detector, phase_image = phase, attenuation_image = attenuation, fresnel_factor  = self.fresnel_factor, wavefield = None, distance =  self.distance_sample_detector, mode = self.mode, value = self.value), self.transform_type, self.transform_factor)
        self.difference = propagated_intensity - self.transformed_images
        self.main_diff = torch.mean(torch.abs(self.difference))

        return propagated_intensity, phase, attenuation
    
    def gaussian_conv(self, x = None):
        if x is None:
            x = self.transform(self.pred)
        else:
            if self.output_num == 1:
                x = transform(x, 'reshape')
            elif self.output_num == 2:
                self.phase = x[:,0,:,:]
                self.attenuation = x[:,1,:,:]
                self.modulus = torch_reshape(torch.abs(torch.exp(1j*self.phase) * torch.exp(-self.attenuation))**2)
                x = torch_reshape(self.phase)
            # x = self.transform(x)
        self.blurred = to_device((self.gaussian_filter(x)), self.device)
        self.difference = self.blurred - self.transformed_images
        if self.l1_ratio != 0:
            self.main_diff = torch.mean(torch.abs(self.difference))
        else:
            self.main_diff = torch.mean(torch.square(self.difference))
        return self.blurred, x
        
    
    def forward(self, x = None):
        x = self.transformed_images if x is None else x
        self.pred = self.generator_model(x)
        propagated_intensity, phase, attenuation = self.propagator()
        self.fake_output = self.discriminator_model(propagated_intensity)
        self.real_output = self.discriminator_model(x)
        return self.fake_output, self.real_output, propagated_intensity, phase, attenuation
    
    def forward_gaussian(self, x):
        self.pred = self.generator_model(x)
        self.propagated_intensity, self.unblurred = self.gaussian_conv(self.pred)
        self.fake_output = self.discriminator_model(self.propagated_intensity)
        self.real_output = self.discriminator_model(x)
        return self.fake_output, self.real_output, self.propagated_intensity, self.unblurred
    

    def generator_loss(self, fake_output, x, propagated_intensity):
        cross_entropy = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output)))
        if self.l1_ratio == 0:
            l2_loss = self.main_diff if self.l2_ratio != 0 else 0
            l1_loss = 0
        else:
            l1_loss =self.main_diff if self.l1_ratio != 0 else 0
            l2_loss = torch.mean(torch.square(self.difference)) if self.l2_ratio != 0 else 0
        
        contrast_diff = torch.mean(torch.abs(self.contrast - transform(propagated_intensity, 'contrast', self.transform_factor))) if self.contrast_ratio != 0 else 0
        contrast_noramlize_difference = torch.mean(torch.square(self.contrast_normalize - transform(propagated_intensity, 'contrast_normalize', self.transform_factor))) if self.contrast_normalize_ratio != 0 else 0
        brightness_noramlize_difference = torch.mean(torch.square(self.brightness_normalize - transform(propagated_intensity, 'brightness_normalize', 1 - self.transform_factor))) if self.brightness_normalize_ratio != 0 else 0
        brightness_diff = torch.mean(torch.square(self.brightness - transform(propagated_intensity, 'brightness', 1 - self.transform_factor))) if self.brightness_ratio != 0 else 0
        
        norm_diff = torch.mean(torch.abs(self.norm - transform(propagated_intensity, 'norm', self.transform_factor))) if self.norm_ratio != 0 else 0
        normalized_diff = torch.mean(torch.abs(self.normalized - transform(propagated_intensity, 'normalize', self.transform_factor))) if self.normalized_ratio != 0 else 0
        
        #use logaritmic loss for the fourier difference
        fourier_diff_real = torch.mean(torch.square(torch.log(torch.abs(torch.fft.fft2(propagated_intensity).real + 1e-10)))) if self.fourier_ratio != 0 else 0
        fourier_diff_imag = torch.mean(torch.square(torch.log(torch.abs(torch.fft.fft2(propagated_intensity).imag + 1e-10)))) if self.fourier_ratio != 0 else 0
        fourier_diff = fourier_diff_real + fourier_diff_imag
        #we use a truncated - quadratic gradient regularization
        reg_l2_loss = torch.mean(torch.square(self.pred[:, :, 1:, :] - self.pred[:, :, :-1, :])) + torch.mean(torch.square((self.pred[:, :, :, 1:] - self.pred[:, :, :, :-1]))) if self.reg_l2_ratio != 0 else 0
        if reg_l2_loss > 0: 
            reg_l2_loss = 1 if reg_l2_loss >1 else reg_l2_loss
            
        reg_l1_loss = torch.mean(torch.abs(self.pred[:, :, 1:, :] - self.pred[:, :, :-1, :])) + torch.mean(torch.abs((self.pred[:, :, :, 1:] - self.pred[:, :, :, :-1]))) if self.reg_l1_ratio != 0 else 0
        if self.epoch > self.iter_num//10:
            self.reg_l1_ratio = 0
            self.reg_l2_ratio = 0
        #if self.modulus exists, then make sure
        self.final_loss =  self.entropy_ratio * cross_entropy + self.l1_ratio * l1_loss + self.contrast_ratio * contrast_diff + self.norm_ratio * norm_diff + self.normalized_ratio * normalized_diff + self.brightness_ratio * brightness_diff + self.contrast_normalize_ratio * contrast_noramlize_difference + self.brightness_normalize_ratio * brightness_noramlize_difference + self.l2_ratio * l2_loss + self.fourier_ratio * fourier_diff
        return self.final_loss + reg_l2_loss * self.reg_l2_ratio + reg_l1_loss * self.reg_l1_ratio
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)))
        fake_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)))
        self.total_loss = self.real_loss_ratio * real_loss + self.fake_loss_ratio * fake_loss
        return self.total_loss
    
    def train_step(self, x):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        if self.task == 'learn_gaussian':
            self.fake_output, self.real_output, self.propagated_intensity, self.unblurred = self.forward_gaussian(x)
        else:
            self.fake_output, self.real_output, self.propagated_intensity, self.phase, self.attenuation = self.forward(x)
        self.gen_loss = self.generator_loss(self.fake_output, x, self.propagated_intensity)
        self.dis_loss = self.discriminator_loss(self.real_output, self.fake_output)
        self.gen_loss.backward(retain_graph=True)
        self.dis_loss.backward()
        self.generator_optimizer.step()
        self.discriminator_optimizer.step()
        if self.task == 'learn_gaussian':
            return self.gen_loss, self.dis_loss, self.propagated_intensity, self.unblurred
        else:
            return self.gen_loss, self.dis_loss, self.propagated_intensity, self.phase, self.attenuation
        
    def train(self, iter_num = None, save_model = None, save_model_path = None, **ratio): # l1_ratio = 10, contrast_ratio = 0, normalized_ratio = 0, norm_ratio = 0, brightness_ratio = 0, contrast_normalize_ratio = 0, brightness_normalize_ratio = 0, l2_ratio = 0, fourier_ratio = 0):
        for key, value in ratio.items():
            setattr(self, key, value)
        iter_num = self.iter_num if iter_num is None else iter_num
        save_model = self.save_model if save_model is None else save_model
        save_model_path = self.save_model_path if save_model_path is None else save_model_path

        self.make_model()
        self.gen_loss_list, self.dis_loss_list, self.propagated_intensity_list, self.phase_list, self.attenuation_list, self.main_diff_list, self.epoch_time, self.ssim_list, self.psnr_list, self.mssim_list = [], [], [], [], [], [], [], [], [], []
        self.ground_main_diff_list, self.ground_ssim_list, self.ground_psnr_list, self.ground_mssim_list = [], [], [], []
        if self.task == 'learn_gaussian':
            self.unblurred_list = []

        self.start_time = time.time()  

        stop_training = False
        print('start training')
        for i in range(iter_num):
            self.epoch = i
            if self.task == 'learn_gaussian':
                self.g_loss, self.d_loss, self.propagated_intensity, self.unblurred = self.train_step(self.transformed_images)
                self.unblurred_list.append(tensor_to_np(self.unblurred))
                self.scheduler.step(self.g_loss)

            else:
                self.g_loss, self.d_loss, self.propagated_intensity, self.phase, self.attenuation = self.train_step(self.transformed_images)
                self.phase_list.append(tensor_to_np(self.phase))
                self.attenuation_list.append(tensor_to_np(self.attenuation))
                self.scheduler.step(self.g_loss)

            self.ssim_list.append(tensor_to_np(self.ssim(self.transformed_images, self.propagated_intensity)))
            self.psnr_list.append(tensor_to_np(self.psnr(self.transformed_images, self.propagated_intensity)))
            try:
                self.mssim_list.append(tensor_to_np(self.mssim(self.transformed_images, self.propagated_intensity))) 
            except:
                self.mssim_list.append(None)

            if self.ground_truth is not None:
                if self.task == 'learn_gaussian':
                    self.ground_truth_difference = tensor_to_np(self.transform(self.unblurred) - self.transform(self.ground_truth))
                    self.ground_ssim_list.append(tensor_to_np(self.ssim(self.ground_truth, self.unblurred)))
                    self.ground_psnr_list.append(tensor_to_np(self.psnr(self.ground_truth, self.unblurred)))
                    try:
                        self.ground_mssim_list.append(tensor_to_np(self.mssim(self.ground_truth, self.unblurred)))
                    except:
                        self.ground_mssim_list.append(None)
                    self.ground_main_diff_list.append(np.mean(np.abs(self.ground_truth_difference)))
                else:
                    self.ground_truth_difference = tensor_to_np(self.phase - self.ground_truth)
                    self.ground_ssim_list.append(tensor_to_np(self.ssim(self.ground_truth, self.phase)))
                    self.ground_psnr_list.append(tensor_to_np(self.psnr(self.ground_truth, self.phase)))
                    self.ground_mssim_list.append(tensor_to_np(self.mssim(self.ground_truth, self.phase)))
                    self.ground_main_diff_list.append(np.mean(np.abs(self.ground_truth_difference)))

            self.gen_loss_list.append(tensor_to_np(self.gen_loss))
            self.dis_loss_list.append(tensor_to_np(self.dis_loss))
            self.main_diff_list.append(tensor_to_np(self.main_diff))
            self.propagated_intensity_list.append(tensor_to_np(self.propagated_intensity))

            
            #no better imporvement for 10 epochs
            if i > 50:
                if self.task == 'learn_gaussian':
                    #if the std of the last 10 propagated_intensity is less than 1e-3, stop the training
                    if np.std(self.main_diff_list[-50:]) < 1e-8:
                        print('std of propagated_intensity is less than 1e-3, stop the training')
                        stop_training = True

            self.epoch_time.append(time.time() - self.start_time)

            if i ==0 or (i+1) % (self.iter_num//2) == 0 or stop_training is True:
                print('epoch', i, 'gen_loss: ', self.g_loss, 'dis_loss: ', self.d_loss, 'main_diff: ', self.main_diff, "t_epoch: ", self.epoch_time[-1], "remaining time: ", time_to_string(self.epoch_time[0] * iter_num - self.epoch_time[0] * i), 'ssim: ', self.ssim_list[-1], 'psnr: ', self.psnr_list[-1])
            
            if stop_training is True:
                break
        if save_model is True and save_model_path is not None:
            torch.save(self.generator_model.state_dict(), save_model_path)
        self.total_time = time.time() - self.start_time

        if self.task == 'learn_gaussian':
            return self.gen_loss_list, self.dis_loss_list, self.propagated_intensity_list, self.unblurred_list, None
        else:
            return self.gen_loss_list, self.dis_loss_list, self.propagated_intensity_list, self.phase_list, self.attenuation_list
        
    def visualize(self, show_or_plot = 'show', cmap = 'coolwarm', dict = None, axis = 'off', plot_axis = 'half'):
        if self.task == 'learn_gaussian':
            learned_image = self.unblurred_list[-1]
        else:
            learned_image = self.phase_list[-1]

        if self.ground_truth is not None:
            df = pd.DataFrame(columns=['gen_loss', 'dis_loss', 'main_diff', 'ssim_list', 'psnr_list', 'mssim_list', 'setup_info', 'ground_ssim_list', 'ground_psnr_list', 'ground_mssim_list', 'ground_main_diff_list'])

        else:
            df = pd.DataFrame(columns=['gen_loss', 'dis_loss', 'main_diff', 'ssim_list', 'psnr_list', 'mssim_list', 'setup_info'])
        df.index.name = 'iter_num'
        df['gen_loss'] = self.gen_loss_list
        df['dis_loss'] = self.dis_loss_list
        df['main_diff'] = self.main_diff_list   
        df['ssim_list'] = self.ssim_list
        df['psnr_list'] = self.psnr_list
        df['mssim_list'] = self.mssim_list
        df['setup_info'] = get_file_nem(self.__dict__)
        if self.ground_truth is not None:
            df['ground_ssim_list'] = self.ground_ssim_list
            df['ground_psnr_list'] = self.ground_psnr_list
            df['ground_mssim_list'] = self.ground_mssim_list
            df['ground_main_diff_list'] = self.ground_main_diff_list

        if self.ground_truth is not None:
            ground_images = [tensor_to_np(self.ground_truth), tensor_to_np(self.ground_truth) - learned_image]
            ground_image_titles = ['ground_truth', 'diff b/n \n GT and learned']
        else:
            ground_images = []
            ground_image_titles = []
        images = [tensor_to_np(self.transformed_images), self.propagated_intensity_list[-1], tensor_to_np(self.difference), learned_image]
        image_titles = ['given image to model', 'reconstructed: iter'+str(self.iter_num), 'diff b/n \n input and generated', 'backpropagated/learned image']

        [images.append(ground_images[i]) for i in range(len(ground_images)) if ground_images != []]
        [image_titles.append(ground_image_titles[i]) for i in range(len(ground_image_titles)) if ground_image_titles != []]

        fig_1 = visualize(images, show_or_plot = show_or_plot, title = image_titles, cmap = cmap, dict = dict, axis = axis, plot_axis = plot_axis)
       
        if self.ground_truth is not None:
            plot_pandas(df, column_range=['gen_loss', 'dis_loss', 'main_diff', 'ssim_list', 'psnr_list', 'mssim_list', 'ground_ssim_list', 'ground_psnr_list', 'ground_mssim_list', 'ground_main_diff_list'])
        else:
            plot_pandas(df, column_range=['gen_loss', 'dis_loss', 'main_diff', 'ssim_list', 'psnr_list', 'mssim_list'])
        
        return fig_1
    
    def live_plot(self, iter_num = None, rate = None, cmap = 'gray'):

        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        if iter_num is None:
            iter_num = self.iter_num
        if rate is None:
            rate = iter_num//7
        for i in range(iter_num):
            clear_output(wait=True) 
            if i % rate == 0:       
                plt.figure(figsize=(20,10))                            
                plt.subplot(1,5,4)
                plt.plot(self.gen_loss_list, label='gen_loss')
                plt.plot(self.dis_loss_list, label='dis_loss')
                plt.plot(self.main_diff_list, label='main_diff')
                plt.plot(self.ssim_list, label='ssim')
                plt.plot(self.psnr_list, label='psnr')
                plt.plot(self.mssim_list, label='mssim')
                plt.title('iteration: '+str(i))
                plt.legend()
                plt.subplot(1,5,1)
                plt.title('input_image')
                plt.imshow(tensor_to_np(self.transformed_images), cmap=cmap)
                plt.colorbar()
                plt.subplot(1,5,2)
                plt.title('propagated_intensity')
                plt.imshow(self.propagated_intensity_list[i], cmap=cmap)
                plt.colorbar()
                plt.subplot(1,5,5)
                if self.task == 'learn_gaussian':
                    plt.title('unblurred')
                    plt.imshow(self.unblurred_list[i], cmap=cmap)
                    plt.colorbar()
                else:
                    plt.title('phase')
                    plt.imshow(self.phase_list[i], cmap=cmap)
                    plt.colorbar()
                plt.gca()
                plt.subplot(1,5,3)
                plt.title('difference')
                plt.imshow(self.propagated_intensity_list[i] - tensor_to_np(self.transformed_images), cmap='gist_earth')
                plt.colorbar()

            plt.show()
            

    def save(self, path = None, name = None):
        if path is None:
            path = self.save_path
        if name is None:
            name = get_file_nem(self.__dict__)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + 'propagated/'):
            os.makedirs(path + 'propagated/')
        if not os.path.exists(path + 'phase/'):
            os.makedirs(path + 'phase/')
        if not os.path.exists(path + 'abs/'):
            os.makedirs(path + 'abs/')
        if path[-1] != '/':
            path += '/'
            
        torch.save(self.generator_model.state_dict(), path + 'generator_'+name+'.pth')
        torch.save(self.discriminator_model.state_dict(), path + 'discriminator_' + name + '.pth')
        np.save(path + 'gen_loss_' + name + '.npy', self.gen_loss_list)
        np.save(path + 'dis_loss_' + name + '.npy', self.dis_loss_list)
        io.imsave(path + 'propagated/propagated_intensity_' + name + '.npy', self.propagated_intensity_list[-1])
        io.imsave.save(path + 'phase/phase_' + name + '.npy', self.phase_list[-1])
        io.imsave(path + 'abs/attenuation_' + name + '.npy', self.attenuation_list[-1])

        return path + name

  