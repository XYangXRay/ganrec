import torch
import torch.nn as nn
import torch.nn.init as init

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
    
class make_gen(nn.Module):
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

class make_dis(nn.Module):
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
    def __init__(self) -> None:
        super().__init__()
        self.transformed_images = to_device(transformed_images, self.device)
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


class make_ganrec_model(nn.Module):
    def __init__(self, shape_x, shape_y, conv_num, conv_size, dropout, output_num, fresnel_factor, transformed_images=None, device=None, **kwargs):
        super(make_ganrec_model, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = device
        self.fresnel_factor = to_device(torch_reshape(fresnel_factor.T, complex=True), self.device)
        self.transformed_images = to_device(transformed_images, self.device)
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.conv_num = conv_num
        self.conv_size = conv_size
        self.dropout = dropout
        self.output_num = output_num
        self.units = 128
        self.fc_size = shape_x * shape_y
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

        self.reshaped = to_device(transform(self.transformed_images, 'reshape') , self.device)
        self.normalized = to_device(transform(self.transformed_images, 'normalize'), self.device)
        self.norm = to_device(transform(self.transformed_images, 'norm'), self.device)
        self.contrast = to_device(transform(self.transformed_images, 'contrast', self.transform_factor), self.device)
        self.contrast_normalize = to_device(transform(self.transformed_images, 'constrast_normalize', self.transform_factor), self.device)
        self.brightness = to_device(transform(self.transformed_images, 'brightness', self.transform_factor), self.device)
        self.brightness_normalize = to_device(transform(self.transformed_images, 'brightness_normalize', self.transform_factor), self.device)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

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
    
    def propagator(self):
        phase = transform(self.pred[:,0,:,:])
        attenuation =  (1 - transform(self.pred[:,1,:,:]))*self.abs_ratio
        propagated_intensity = transform(forward_propagate(shape_x = self.shape_x, shape_y = self.shape_y, pad = self.pad, energy_kev = self.energy_kev, detector_pixel_size = self.detector_pixel_size, distance_sample_detector = self.distance_sample_detector, phase_image = phase, attenuation_image = attenuation, fresnel_factor  = self.fresnel_factor, wavefield = None, distance =  self.distance_sample_detector, mode = self.mode, value = self.value))
        return propagated_intensity, phase, attenuation
    
    def forward(self, x):
        self.pred = self.generator(x)
        propagated_intensity, phase, attenuation = self.propagator()
        self.fake_output = self.discriminator(propagated_intensity)
        self.real_output = self.discriminator(self.transformed_images)
        return self.fake_output, self.real_output, propagated_intensity, phase, attenuation
    
    def generator_loss(self, fake_output, image_output, propagated_intensity, l1_ratio = 10):
        return torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))) + \
                l1_ratio * torch.mean(torch.abs(self.reshaped - transform(propagated_intensity, 'reshape', 0.5))) + \
                l1_ratio * torch.mean(torch.abs(self.contrast_normalize - transform(propagated_intensity, 'contrast_normalize', self.transform_factor))) + \
                    1/2*l1_ratio * torch.mean(torch.abs(self.brightness_normalize - transform(propagated_intensity, 'brightness_normalize', self.transform_factor))) + \
                    1/2*l1_ratio * torch.mean(torch.abs(self.norm - transform(propagated_intensity, 'norm', 0.5)))
                    # l1_ratio * torch.mean(torch.abs(self.normalized - transform(propagated_intensity, 'normalize', 0.5))) +\
                
    
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)))
        fake_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)))
        total_loss = real_loss + fake_loss
        return total_loss
    
    def train_step(self, x):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        self.fake_output, self.real_output, self.propagated_intensity, self.phase, self.attenuation = self.forward(x)
        self.gen_loss = self.generator_loss(self.fake_output, self.transformed_images, self.propagated_intensity)
        self.dis_loss = self.discriminator_loss(self.real_output, self.fake_output)
        self.gen_loss.backward(retain_graph=True)
        self.dis_loss.backward()
        self.generator_optimizer.step()
        self.discriminator_optimizer.step()
        return self.gen_loss, self.dis_loss, self.propagated_intensity, self.phase, self.attenuation
    
    def train(self, iter_num = None, save_model = None, save_model_path = None):
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.transformed_images = to_device(self.transformed_images, self.device)
            self.fresnel_factor = to_device(self.fresnel_factor, self.device)
            self.generator.to(self.device)
            self.discriminator.to(self.device)
    
        iter_num = self.iter_num if iter_num is None else iter_num
        save_model = self.save_model if save_model is None else save_model
        save_model_path = self.save_model_path if save_model_path is None else save_model_path

        self.make_model()
        self.gen_loss_list = []
        self.dis_loss_list = []
        self.propagated_intensity_list = []
        self.phase_list = []
        self.attenuation_list = []
        for i in range(iter_num):
            self.g_loss, self.d_loss, self.propagated_intensity, self.phase, self.attenuation = self.train_step(self.transformed_images)
            self.gen_loss_list.append(tensor_to_np(self.gen_loss))
            self.dis_loss_list.append(tensor_to_np(self.dis_loss))
            self.propagated_intensity_list.append(tensor_to_np(self.propagated_intensity))
            self.phase_list.append(tensor_to_np(self.phase))
            self.attenuation_list.append(tensor_to_np(self.attenuation))
            if i % 100 == 0:
                print('gen_loss: ', self.g_loss, 'dis_loss: ', self.d_loss)
        if save_model:
            torch.save(self.generator.state_dict(), save_model_path)
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
            self.gen_loss, self.dis_loss, self.propagated_intensity, self.phase, self.attenuation = self.train_step(testloader)
            self.gen_loss_list.append(tensor_to_np(self.gen_loss))
            self.dis_loss_list.append(tensor_to_np(self.dis_loss))
            self.propagated_intensity_list.append(tensor_to_np(self.propagated_intensity))
            self.phase_list.append(tensor_to_np(self.phase))
            self.attenuation_list.append(tensor_to_np(self.attenuation))
            if i % 10 == 0:
                print('gen_loss: ', self.gen_loss, 'dis_loss: ', self.dis_loss)
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
                plt.title('iteration: '+str(i))
                plt.legend()
                plt.subplot(1,3,1)
                plt.title('propagated_intensity')
                plt.imshow(self.propagated_intensity_list[i], cmap='gray')
                plt.colorbar()
                plt.subplot(1,3,2)
                plt.title('phase')
                plt.imshow(self.phase_list[i], cmap='coolwarm')
                plt.colorbar()
                plt.gca()
            plt.show()