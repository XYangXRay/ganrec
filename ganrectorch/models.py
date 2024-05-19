import torch
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
from utils import *
from ganrec_dataloader import *

def initializer(m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
                
def discriminator_loss(real_output, fake_output):
    real_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)))
    fake_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, img_output, pred, l1_ratio = 10):
    #with autograd
    return torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))) + \
              l1_ratio * torch.mean(torch.abs(img_output - pred))

def dense_layer(in_features= 128, out_features = 128, dropout = 0.25, apply_batchnorm=True, transpose = False):
    initializer = torch.nn.init.xavier_uniform_(torch.empty(out_features, in_features))
    if apply_batchnorm:
        result = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
    else:
        result = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    if transpose:
        result = nn.Sequential(
            nn.Linear(in_features, out_features),
            Transpose(),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
    return result

def conv2d_layer(in_channels, out_channels, kernel_size, stride, apply_batchnorm=True, normal_init = True):
    def initializer(m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    if apply_batchnorm:
        result = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, 
                kernel_size, 
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU())
    else:
        result = nn.Sequential(
            nn.Conv2d(in_channels,
                out_channels, 
                kernel_size, 
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.LeakyReLU())
    if normal_init:
        result.apply(initializer)
    return result

def deconv2d_layer(in_channels, out_channels, kernel_size, stride, apply_batchnorm=True, normal_init = True):
    def initializer(m):
        if type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
    if apply_batchnorm:
        result = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                out_channels,
                kernel_size,
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU())
        
    else:
        result = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                out_channels,
                kernel_size,
                stride=stride, 
                padding_mode='zeros',
                bias=False),
            nn.LeakyReLU())
    if normal_init:
        result.apply(initializer)
    return result

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

def to_device(x, device):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        return x.to(device)
    except:
        return torch_reshape(x).to(device)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)
    
class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.view(-1, 1)
  
class Print_module(nn.Module):
    def __init__(self, word = ''):
        super(Print_module, self).__init__()
        self.word = word
    def forward(self, x):
        print(self.word, x.shape)
        return x
    
class make_generator(nn.Module):
    def __init__(self, shape_x, shape_y, conv_num, conv_size, dropout, output_num, units= 128, device=None, **kwargs):
        super(make_ganrec_model, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = device
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.conv_num = conv_num
        self.conv_size = conv_size
        self.dropout = dropout
        self.output_num = output_num
        self.units = units
        self.fc_size = shape_x * shape_y

        ##################################################################################################
        # We first define the generator model
        ##################################################################################################
        
        self.fc_stack = nn.ModuleList([
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
        ])
        self.conv_stack = nn.ModuleList([
            conv2d_layer(in_channels=1, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            conv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            conv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            conv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size, stride=1),
        ])
        self.dconv_stack = nn.ModuleList([
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.output_num, kernel_size=self.conv_size+2, stride=1),
            
        ])
        self.last = conv2d_layer(in_channels=self.output_num, out_channels=self.output_num, kernel_size=self.conv_size, stride=1)
        
        self.generator_model = to_device(nn.Sequential(
            nn.Flatten(),
            Transpose(),
            dense_layer(in_features=1, out_features=self.units, dropout=self.dropout, transpose=False),
            *self.fc_stack,
            dense_layer(in_features=self.units, out_features=1, dropout=0),
            Reshape((-1, 1, self.shape_x, self.shape_y)),
            *self.conv_stack,
            *self.dconv_stack,
            self.last,
        ), self.device)

        if 'init_model' in kwargs.keys():
            if kwargs['init_model']:
                # Load the model
                init_model_path = kwargs.get('init_model_path', 'model/ganrec_model')
                self.generator_model.load_state_dict(torch.load(init_model_path))

        else:
            self.init_weights()

    def forward_generator(self, x):
        self.pred = self.generator_model(x)
        return self.pred
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

class make_discriminator(nn.Module):
    def __init__(self, device=None, **kwargs):
        super(make_ganrec_model, self).__init__()
        self.device = device
        ##################################################################################################
        # We then define the discriminator model
        ##################################################################################################

        discriminator_stack = nn.ModuleList([
            conv2d_layer(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            conv2d_layer(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            conv2d_layer(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            conv2d_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            conv2d_layer(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            conv2d_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            conv2d_layer(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            conv2d_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Flatten(),
        ])

        self.discriminator_model =  to_device(nn.Sequential(
            *discriminator_stack
        ), self.device)

    def forward_discriminator(self, x):
        self.likelihood = self.discriminator_model(x)
        return self.likelihood
    
class transform_images(nn.Module):
    def __init__(self, image) -> None:
        super().__init__()
        self.transformed_images = to_device(image, self.device)
        self.reshaped = to_device(transform(self.transformed_images, 'reshape') , self.device)
        self.normalized = to_device(transform(self.transformed_images, 'normalize'), self.device)
        self.norm = to_device(transform(self.transformed_images, 'norm'), self.device)
        self.contrast = to_device(transform(self.transformed_images, 'contrast', self.transform_factor), self.device)
        self.brightness = to_device(transform(self.transformed_images, 'brightness', self.transform_factor), self.device)

        #a function that returns the transformed images
    def transform(self, x):
        if 'transform_type' in self.kwargs.keys():
            transform_type = self.kwargs['transform_type']
        else:
            transform_type = 'normalize'

        if 'transform_factor' in self.kwargs.keys():
            transform_factor = self.kwargs['transform_factor']
        else:
            transform_factor = 0.5
        return transform(x, transform_type, transform_factor)

def run_model_with_condition(dataloader, condition, value, df, last_props, last_phases, last_atts, ratio):
    if type(condition) is not str:
        condition = str(condition)
    if condition in dataloader.kwargs.keys():
        dataloader.kwargs[condition] = value
    else:
        dataloader.kwargs.update({condition: value})
    model = make_ganrec_model(**dataloader.kwargs)
    gen_loss_list, dis_loss_list, propagated_intensity_list, phase_list, attenuation_list = model.train(save_model = False, save_model_path = 'model/ganrec_model', **ratio)
    df.loc[value, condition] = value
    df.loc[value, 'iter_num'] = len(gen_loss_list)
    df.loc[value, 'gen_loss'] = gen_loss_list[-1]
    df.loc[value, 'dis_loss'] = dis_loss_list[-1]
    df.loc[value, 'main_diff'] = model.main_diff_list[-1]
    df.loc[value, 'setup_info'] = get_file_nem(model.__dict__)
    last_props.append(propagated_intensity_list[-1][-1])
    last_phases.append(phase_list[-1][-1])
    last_atts.append(attenuation_list[-1][-1])
    return gen_loss_list, dis_loss_list, propagated_intensity_list, phase_list, attenuation_list


class make_ganrec_model(nn.Module):
    def __init__(self, shape_x, shape_y, conv_num, conv_size, dropout, output_num, fresnel_factor, transformed_images=None, device=None, **kwargs):
        super(make_ganrec_model, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        
        self.fresnel_factor = to_device(torch_reshape(fresnel_factor, complex=True), self.device)
        self.transformed_images = to_device(transformed_images, self.device)
        self.image = to_device(self.image, self.device)
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.conv_num = conv_num
        self.conv_size = conv_size
        self.dropout = dropout
        self.output_num = output_num
        self.units = 128
        self.fc_size = shape_x * shape_y
        
        self.possible_distance_gap = None if 'possible_distance_gap' not in kwargs.keys() or kwargs['possible_distance_gap'] == 0 else kwargs['possible_distance_gap']
        self.number_of_distances = None if 'number_of_distances' not in kwargs.keys() else kwargs['number_of_distances']
        self.possible_distances = list(np.linspace(self.distance_sample_detector - self.possible_distance_gap*self.number_of_distances, self.distance_sample_detector + self.possible_distance_gap*self.number_of_distances, self.number_of_distances+1)) if self.possible_distance_gap is not None else [self.distance_sample_detector]
        self.possible_fresnels_factors = torch_reshape([ffactors(self.shape_x, self.shape_y, self.energy_kev, distance, self.detector_pixel_size) for distance in self.possible_distances], complex = True)

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
        
        self.fc_stack = nn.ModuleList([
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
            dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
            # dense_layer(in_features=self.units, out_features=self.units, dropout=self.dropout),
        ])
        self.conv_stack = nn.ModuleList([
            conv2d_layer(in_channels=1, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            conv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            # conv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            conv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size, stride=1),
        ])
        self.dconv_stack = nn.ModuleList([
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            # deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.conv_num, kernel_size=self.conv_size+2, stride=1),
            deconv2d_layer(in_channels=self.conv_num, out_channels=self.output_num, kernel_size=self.conv_size+2, stride=1),
            
        ])
        self.last = conv2d_layer(in_channels=self.output_num, out_channels=self.output_num, kernel_size=self.conv_size, stride=1)
        
        self.generator_model = to_device(nn.Sequential(
            nn.Flatten(),
            Transpose(),
            dense_layer(in_features=1, out_features=self.units, dropout=self.dropout, transpose=False),
            *self.fc_stack,
            dense_layer(in_features=self.units, out_features=1, dropout=0),
            Reshape((-1, 1, self.shape_x, self.shape_y)),
            *self.conv_stack,
            *self.dconv_stack,
            self.last,
        ), self.device)

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

        discriminator_stack = nn.ModuleList([
            conv2d_layer(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            conv2d_layer(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            conv2d_layer(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            conv2d_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            conv2d_layer(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            conv2d_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            conv2d_layer(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            conv2d_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Flatten(),
        ])

        self.discriminator_model =  to_device(nn.Sequential(
            *discriminator_stack
        ), self.device)

        self.reshaped = to_device(transform(self.image, 'reshape') , self.device)
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

    def refine_parameters_using_condition(self, condition, values, change_from_soure = False, info = None, ratio = None, plot_pandas=False, show_images=False):
        if ratio is None:
            ratio = {'l1_ratio': 10, 'contrast_ratio': 0, 'normalized_ratio': 5, 'brightness_ratio': 0, 'contrast_normalize_ratio': 0, 'brightness_normalize_ratio': 0, 'l2_ratio': 0, 'fourier_ratio': 1}

        df = pd.DataFrame(columns=[condition, 'iter_num', 'gen_loss', 'dis_loss', 'main_diff',  'setup_info'])
        df.index.name = condition
        last_phases = []
        last_atts = []
        last_props = []
        
        if type(values) is not list:
            values = [values]
        
        
        for value in values:
            if change_from_soure == True:
                info[condition] = value if condition in info.keys() else info.update({condition: value})
                new_dataloader = Ganrec_Dataloader(**info)
                model = make_ganrec_model(**new_dataloader.__dict__)
            else:
                kwargs = self.__dict__
                kwargs[condition] = value if condition in kwargs.keys() else kwargs.update({condition: value})
                model = make_ganrec_model(**kwargs)
            
            gen_loss_list, dis_loss_list, propagated_intensity_list, phase_list, attenuation_list = model.train(save_model = False, save_model_path = 'model/ganrec_model', **ratio)
            print('gen_loss: ', gen_loss_list[-1], 'dis_loss: ', dis_loss_list[-1], 'main_diff: ', model.main_diff_list[-1])
            df.loc[value, condition] = value
            df.loc[value, 'iter_num'] = len(gen_loss_list)
            df.loc[value, 'gen_loss'] = gen_loss_list[-1]
            df.loc[value, 'dis_loss'] = dis_loss_list[-1]
            df.loc[value, 'main_diff'] = model.main_diff_list[-1]
            df.loc[value, 'setup_info'] = get_file_nem(model.__dict__)
            last_props.append(propagated_intensity_list[-1])
            last_phases.append(phase_list[-1])
            last_atts.append(attenuation_list[-1])
        column_range = ['main_diff', 'gen_loss', 'dis_loss']
        #list_element.index(element)
        min_vals = [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]
        max_vals = [df[column].max() for column in column_range], [df[column].idxmax() for column in column_range]
        indices_in_values = [values.index(min_vals[1][i]) for i in range(len(min_vals[1]))] 
        if plot_pandas:         
            plot_pandas(df, column_range, x_column = condition)
        if show_images:
            visualize([last_props[-1], last_phases[-1], last_atts[-1]], title = ['propagated_intensity', 'phase', 'attenuation'])
        return df, last_props, last_phases, last_atts, min_vals, max_vals, indices_in_values

    def make_model(self):
        self.generator = self.generator_model
        self.discriminator = self.discriminator_model
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5)
        
    def forward_generator(self, x):
        self.pred = self.generator_model(x)
        return self.pred

    def forward_discriminator(self, x):
        self.likelihood = self.discriminator_model(x)
        return self.likelihood
    
    def transform(self, x):
        if 'transform_type' in self.kwargs.keys():
            transform_type = self.kwargs['transform_type']
        else:
            transform_type = 'normalize'

        if 'transform_factor' in self.kwargs.keys():
            transform_factor = self.kwargs['transform_factor']
        else:
            transform_factor = 0.5
        return transform(x, transform_type, transform_factor)
    
    def propagate_and_difference(self, distance, index):
        fresnel_factor = to_device(self.possible_fresnels_factors[index, 0], self.device)
        propagated_intensity = forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = distance, phase_image = self.phase, attenuation_image = self.attenuation, fresnel_factor  = fresnel_factor, wavefield = None, distance =  distance, mode = self.mode, value = self.value)
        difference = propagated_intensity - self.transformed_images
        main_diff = torch.mean(torch.abs(difference))
        return main_diff

    def find_best_distance(self, possible_distances):
        results = torch.zeros(len(possible_distances))
        for i, distance in enumerate(possible_distances):
            results[i] = self.propagate_and_difference(distance, i)
        if torch.min(results) < self.main_diff:
            self.distance_sample_detector = possible_distances[torch.argmin(results)]
            self.fresnel_factor = to_device(self.possible_fresnels_factors[torch.argmin(results)], self.device)
            print("The distance is updated to: ", self.distance_sample_detector)
            print("old difference: ", self.main_diff, " and new difference: ", torch.min(results))
    
    def propagator(self):
        phase = transform(self.pred[:,0,:,:], 'normalize') #*2 -1) * torch.pi
        attenuation =  (1 - transform(self.pred[:,1,:,:], 'normalize'))*self.abs_ratio
        propagated_intensity = transform(forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = self.distance_sample_detector, phase_image = phase, attenuation_image = attenuation, fresnel_factor  = self.fresnel_factor, wavefield = None, distance =  self.distance_sample_detector, mode = self.mode, value = self.value), self.transform_type, self.transform_factor)
        self.difference = propagated_intensity - self.transformed_images
        self.main_diff = torch.mean(torch.abs(self.difference))

        if self.possible_distance_gap is not None and self.epoch == 1:
            self.find_best_distance(self.possible_distances)

        if self.possible_distance_gap is not None and self.epoch > 1:
            if np.mean(self.main_diff_list[-10:]) < 0.001:
                self.find_best_distance(self.possible_distances)
        
        return propagated_intensity, phase, attenuation
    
    def forward(self, x = None):
        x = self.transformed_images if x is None else x
        self.pred = self.generator(x)
        propagated_intensity, phase, attenuation = self.propagator()
        self.fake_output = self.discriminator(propagated_intensity)
        self.real_output = self.discriminator(x)
        return self.fake_output, self.real_output, propagated_intensity, phase, attenuation
    
    def generator_loss(self, fake_output, x, propagated_intensity, l1_ratio = 10, contrast_ratio = 0, normalized_ratio = 0, brightness_ratio = 0, contrast_normalize_ratio = 10, brightness_normalize_ratio = 0, l2_ratio = 0, fourier_ratio = 0):
        x = self.transformed_images if x is None else x
        if self.epoch > self.iter_num//10 + 1:
            contrast_ratio = 0
            contrast_normalize_ratio = 0
            brightness_ratio = 0
            brightness_normalize_ratio = 0
        if self.epoch < self.iter_num//90:
            fourier_ratio = 0

        cross_entropy = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output)))
        l2_loss = torch.mean(torch.square(self.difference)) if l2_ratio != 0 else 0

        contrast_diff = torch.mean(torch.abs(self.contrast - transform(propagated_intensity, 'contrast', self.transform_factor))) if contrast_ratio != 0 else 0
        contrast_noramlize_difference = torch.mean(torch.square(self.contrast_normalize - transform(propagated_intensity, 'contrast_normalize', self.transform_factor))) if contrast_normalize_ratio != 0 else 0
        brightness_noramlize_difference = torch.mean(torch.square(self.brightness_normalize - transform(propagated_intensity, 'brightness_normalize', 1 - self.transform_factor))) if brightness_normalize_ratio != 0 else 0
        brightness_diff = torch.mean(torch.square(self.brightness - transform(propagated_intensity, 'brightness', 1 - self.transform_factor))) if brightness_ratio != 0 else 0
        
        norm_diff = torch.mean(torch.abs(self.norm - transform(propagated_intensity, 'norm', self.transform_factor))) if normalized_ratio != 0 else 0
        normalized_diff = torch.mean(torch.abs(self.normalized - transform(propagated_intensity, 'normalize', self.transform_factor))) if normalized_ratio != 0 else 0
        
        #use logaritmic loss for the fourier difference
        fourier_diff_real = torch.mean(torch.square(torch.log(torch.abs(torch.fft.fft2(propagated_intensity).real + 1)) - torch.log(torch.abs(self.fourier.real) + 1))) if fourier_ratio != 0 else 0
        fourier_diff_imag = torch.mean(torch.square(torch.log(torch.abs(torch.fft.fft2(propagated_intensity).imag + 1)) - torch.log(torch.abs(self.fourier.imag) + 1))) if fourier_ratio != 0 else 0
        fourier_diff = fourier_diff_real + fourier_diff_imag
        return cross_entropy + l1_ratio * self.main_diff + contrast_ratio * contrast_diff + normalized_ratio * norm_diff + normalized_ratio * normalized_diff + brightness_ratio * brightness_diff + contrast_normalize_ratio * contrast_noramlize_difference + brightness_normalize_ratio * brightness_noramlize_difference + l2_ratio * l2_loss + fourier_ratio * fourier_diff

    def discriminator_loss(self, real_output, fake_output):
        real_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)))
        fake_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)))
        total_loss = real_loss + fake_loss
        return total_loss
    
    def train_step(self, x, l1_ratio = 10, contrast_ratio = 0, normalized_ratio = 0, brightness_ratio = 0, contrast_normalize_ratio = 0, brightness_normalize_ratio = 0, l2_ratio = 0, fourier_ratio = 0):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        self.fake_output, self.real_output, self.propagated_intensity, self.phase, self.attenuation = self.forward(x)
        self.gen_loss = self.generator_loss(self.fake_output, x, self.propagated_intensity, l1_ratio = l1_ratio, l2_ratio = l2_ratio, contrast_ratio = contrast_ratio, normalized_ratio = normalized_ratio, brightness_ratio = brightness_ratio, contrast_normalize_ratio = contrast_normalize_ratio, brightness_normalize_ratio = brightness_normalize_ratio)
        self.dis_loss = self.discriminator_loss(self.real_output, self.fake_output)
        self.gen_loss.backward(retain_graph=True)
        self.dis_loss.backward()
        self.generator_optimizer.step()
        self.discriminator_optimizer.step()
        return self.gen_loss, self.dis_loss, self.propagated_intensity, self.phase, self.attenuation
    
    def train(self, iter_num = None, save_model = None, save_model_path = None, l1_ratio = 10, contrast_ratio = 0, normalized_ratio = 0, brightness_ratio = 0, contrast_normalize_ratio = 0, brightness_normalize_ratio = 0, l2_ratio = 0, fourier_ratio = 0):
        
        iter_num = self.iter_num if iter_num is None else iter_num
        save_model = self.save_model if save_model is None else save_model
        save_model_path = self.save_model_path if save_model_path is None else save_model_path
        self.make_model()
        self.gen_loss_list = []
        self.dis_loss_list = []
        self.propagated_intensity_list = []
        self.phase_list = []
        self.attenuation_list = []
        self.main_diff_list = []
        self.epoch_time = []

        import time
        self.start_time = time.time()  
        for i in range(iter_num):
            self.epoch = i
            self.g_loss, self.d_loss, self.propagated_intensity, self.phase, self.attenuation = self.train_step(None, l1_ratio = l1_ratio, contrast_ratio = contrast_ratio, normalized_ratio = normalized_ratio, brightness_ratio = brightness_ratio, contrast_normalize_ratio = contrast_normalize_ratio, brightness_normalize_ratio = brightness_normalize_ratio, l2_ratio = l2_ratio, fourier_ratio = fourier_ratio)
            self.gen_loss_list.append(tensor_to_np(self.gen_loss))
            self.dis_loss_list.append(tensor_to_np(self.dis_loss))
            self.main_diff_list.append(tensor_to_np(self.main_diff))
            self.propagated_intensity_list.append(tensor_to_np(self.propagated_intensity))
            self.phase_list.append(tensor_to_np(self.phase))
            self.attenuation_list.append(tensor_to_np(self.attenuation))
            self.epoch_time.append(time.time() - self.start_time)
            if i % self.iter_num//2 == 0:
                print('gen_loss: ', self.g_loss, 'dis_loss: ', self.d_loss, 'main_diff: ', self.main_diff, "t_epoch: ", self.epoch_time[-1], "remaining time: ", time_to_string(self.epoch_time[0] * iter_num - self.epoch_time[0] * i))
            
            # if self.epoch > self.iter_num//90 and torch.abs(self.main_diff - np.mean(self.main_diff_list[-100:])) < 0.00001:
            #     for param_group in self.generator_optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * 0.9
            #     for param_group in self.discriminator_optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * 0.9  
            # if self.epoch > self.iter_num//90 and torch.abs(self.main_diff - np.mean(self.main_diff_list[-100:])) < 0.000001:
            #     print('Training stopped')
            #     self.iter_num = self.epoch
            #     self.main_diff_list.append(tensor_to_np(self.main_diff))    
            #     break
        if save_model is True and save_model_path is not None:
            torch.save(self.generator.state_dict(), save_model_path)
        self.total_time = time.time() - self.start_time
        return self.gen_loss_list, self.dis_loss_list, self.propagated_intensity_list, self.phase_list, self.attenuation_list
    
    def test(self, iter_num, testloader, save_model = False, save_model_path = None):
        from torch.autograd import Variable
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.eval()
        self.discriminator.eval()
        self.gen_loss_list = []
        self.dis_loss_list = []
        self.propagated_intensity_list = []
        self.phase_list = []
        self.attenuation_list = []
        for i in range(iter_num):
            self.epoch = i
            self.gen_loss, self.dis_loss, self.propagated_intensity, self.phase, self.attenuation = self.train_step(testloader)
            self.gen_loss_list.append(tensor_to_np(self.gen_loss))
            self.dis_loss_list.append(tensor_to_np(self.dis_loss))
            self.propagated_intensity_list.append(tensor_to_np(self.propagated_intensity))
            self.phase_list.append(tensor_to_np(self.phase))
            self.attenuation_list.append(tensor_to_np(self.attenuation))
            if i % 100 == 0:
                print('gen_loss: ', self.gen_loss, 'dis_loss: ', self.dis_loss, 'main_diff: ', self.main_diff)
        if save_model:
            torch.save(self.generator.state_dict(), save_model_path)
        return self.gen_loss_list, self.dis_loss_list, self.propagated_intensity_list, self.phase_list, self.attenuation_list

    def live_plot(self, iter_num = None, rate = 1):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        if iter_num is None:
            iter_num = self.iter_num
        for i in range(iter_num):
            clear_output(wait=True) 
            if i % rate == 0:       
                plt.figure(figsize=(20,10))                            
                plt.subplot(1,3,3)
                plt.subplot(1,3,3)
                plt.plot(self.gen_loss_list, label='gen_loss')
                plt.plot(self.dis_loss_list, label='dis_loss')
                plt.plot(self.main_diff_list, label='main_diff')
                plt.title('iteration: '+str(i))
                plt.legend()
                plt.subplot(1,3,1)
                plt.title('propagated_intensity')
                plt.imshow(self.propagated_intensity_list[i][-1], cmap='gray')
                plt.colorbar()
                plt.subplot(1,3,2)
                plt.title('phase')
                plt.imshow(self.phase_list[i], cmap='coolwarm')
                plt.colorbar()
                plt.gca()
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
            
        torch.save(self.generator.state_dict(), path + 'generator_'+name+'.pth')
        torch.save(self.discriminator.state_dict(), path + 'discriminator_' + name + '.pth')
        np.save(path + 'gen_loss_' + name + '.npy', self.gen_loss_list)
        np.save(path + 'dis_loss_' + name + '.npy', self.dis_loss_list)
        io.imsave(path + 'propagated/propagated_intensity_' + name + '.npy', self.propagated_intensity_list[-1])
        io.imsave.save(path + 'phase/phase_' + name + '.npy', self.phase_list[-1])
        io.imsave(path + 'abs/attenuation_' + name + '.npy', self.attenuation_list[-1])

        return path + name

#Example
"""
info = {
    'phase': given_phase,
    'attenuation': given_absorption,
    'pv': pv.magnitude,
    'wavenumber': 2 * np.pi / lam.magnitude,
    'distance_sample_detector': z.magnitude,
    'fresnel_number': fresnel_number.magnitude,
    'lam': lam.magnitude,
    'energy_kev': energy.magnitude,
    'z': different_distances[33],
    'pad':2,
    'idx': list(np.arange(0, len(different_distances), len(different_distances)//6)),
    'transform_factor': 0.7,
    'transform_type': 'norm',
    'mode': 'reflect',
    'value': 'mean',
}
dataloader = Ganrec_Dataloader(**info)
dataloader.update_values(iter_num = 2000, change_all=True, transform_type = 'normalize', transform_factor = 0.7, idx = 0)
model = make_ganrec_model(**dataloader.get_kwargs())
gen_loss_list, dis_loss_list, propagated_intensity_list, phase_list, attenuation_list = model.train(save_model=True, save_model_path='model.pth', l1_ratio = 0.1, contrast_ratio = 0.1, normalized_ratio = 10, brightness_ratio = 0.0004, contrast_normalize_ratio = 0.004, brightness_normalize_ratio = 0)
fig = visualize([tensor_to_np(dataloader.transformed_images), propagated_intensity_list[-1], phase_list[-1], attenuation_list[-1]])
"""
# class Generator(nn.Module):
#     def __init__(self, img_h, img_w, conv_num, conv_size, dropout, output_num):
#         super(Generator, self).__init__()
#         units_out = 128
#         self.img_w = img_w
#         self.img_h = img_h

#         # Calculate the size after the fully connected layers
#         fc_size = img_w * img_h

#         self.fc_stack = nn.Sequential(
#             self.dense_norm(fc_size, units_out, dropout),
#             self.dense_norm(units_out, units_out, dropout),
#             self.dense_norm(units_out, units_out, dropout),
#             self.dense_norm(units_out, fc_size, 0)
#         )

#         self.conv_stack = nn.Sequential(
#             self.conv2d_norm(1, conv_num, conv_size+2, 1),
#             self.conv2d_norm(conv_num, conv_num, conv_size+2, 1),
#             self.conv2d_norm(conv_num, conv_num, conv_size, 1),
#         )

#         self.dconv_stack = nn.Sequential(
#             self.dconv2d_norm(conv_num, conv_num, conv_size+2, 1),
#             self.dconv2d_norm(conv_num, conv_num, conv_size+2, 1),
#             self.dconv2d_norm(conv_num, conv_num, conv_size, 1),
#         )

#         self.last = nn.Sequential(
#             nn.Conv2d(conv_num, output_num, 3, 1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten
#         print(x.shape)
#         x = self.fc_stack(x)
#         x = x.view(-1, 1, self.img_h, self.img_w)  # Reshape to (batch_size, channels, height, width)
#         x = self.conv_stack(x)
#         x = self.dconv_stack(x)
#         x = self.last(x)
#         return x

#     def dense_norm(self, units_in, units_out, dropout):
#         return nn.Sequential(
#             nn.Linear(units_in, units_out),
#             nn.BatchNorm1d(units_out),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#     def conv2d_norm(self, in_channels, out_channels, kernel_size, stride):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )

#     def dconv2d_norm(self, in_channels, out_channels, kernel_size, stride):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, output_padding=stride-1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )



# class Discriminator(nn.Module):
#     def __init__(self, nang, px):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(1, 16, (5, 5), stride=(2, 2)),
#             nn.Conv2d(16, 16, (5, 5), stride=(1, 1)),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2),

#             nn.Conv2d(16, 32, (5, 5), stride=(2, 2)),
#             nn.Conv2d(32, 32, (5, 5), stride=(1, 1)),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2),

#             nn.Conv2d(32, 64, (3, 3), stride=(2, 2)),
#             nn.Conv2d(64, 64, (3, 3), stride=(1, 1)),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2),

#             nn.Conv2d(64, 128, (3, 3), stride=(2, 2)),
#             nn.Conv2d(128, 128, (3, 3), stride=(1, 1)),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.2),

#             nn.Flatten(),
#             nn.Linear(nang * px * 128, 256),
#             nn.Linear(256, 128),
#         )

#     def forward(self, input):
#         return self.main(input)