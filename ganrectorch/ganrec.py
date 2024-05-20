import os
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from ganrectorch.models import Generator, Discriminator
from ganrectorch.propagators import RadonTransform
from ganrectorch.utils import RECONmonitor, to_device, tensor_to_np

# Load the configuration from the JSON file
def load_config(filename):
        # Get the directory of the script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the full path to the config file
    config_path = os.path.join(dir_path, filename)
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Use the configuration
config = load_config('config.json')

def discriminator_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss

def l1_loss(img1, img2):
    return torch.mean(torch.abs(img1 - img2))

def l2_loss(img1, img2):
    return torch.pow(torch.mean(torch.abs(img1-img2)), 2)

def generator_loss(fake_output, img_output, pred, l1_ratio):
    gen_loss = F.binary_cross_entropy_with_logits(fake_output, 
                                                  torch.ones_like(fake_output)) + l1_loss(img_output, pred) * l1_ratio
    return gen_loss

def tfnor_phase(img):
    img = (img - img.mean()) / img.std()
    img = img / torch.max(img)
    return img


class GANtomo:
    def __init__(self, prj_input, angle, **kwargs):
        super(GANtomo, self).__init__()
        tomo_args = config['GANtomo']
        tomo_args.update(**kwargs)
        self.nang, self.px = prj_input.shape
        self.prj_input = torch.from_numpy(prj_input)
        self.prj_input = self.prj_input.view(-1, 1, self.nang, self.px)
        self.prj_input = to_device(self.prj_input)
        self.angle = torch.from_numpy(angle)
        self.angle = to_device(self.angle)
        self.iter_num = tomo_args['iter_num']
        self.conv_num = tomo_args['conv_num']
        self.conv_size = tomo_args['conv_size']
        self.dropout = tomo_args['dropout']
        self.l1_ratio = tomo_args['l1_ratio']
        self.g_learning_rate = tomo_args['g_learning_rate']
        self.d_learning_rate = tomo_args['d_learning_rate']
        self.save_wpath = tomo_args['save_wpath']
        self.init_wpath = tomo_args['init_wpath']
        self.init_model = tomo_args['init_model']
        self.recon_monitor = tomo_args['recon_monitor']
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = Generator(self.prj_input.shape[2],
                                   self.prj_input.shape[3],
                                   self.conv_num,
                                   self.conv_size,
                                   self.dropout,
                                   1)
        self.discriminator = Discriminator(self.prj_input.shape[2],
                                           self.prj_input.shape[3])
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

    def nor_tomo(self, data):
        min_val = torch.min(data)
        max_val = torch.max(data)
    
    # Apply min-max normalizatio
        normalized_data = (data - min_val) / (max_val - min_val)
    
        return normalized_data
        # img = transforms.Normalize((0.5,), (0.5,))(img)
        # return img

    def recon_step(self, prj, ang):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        recon = self.generator(prj)
        recon = self.nor_tomo(recon)
        tomo_radon_obj = RadonTransform(recon, ang)
        prj_rec = tomo_radon_obj.forward()
        prj_rec = self.nor_tomo(prj_rec)
        real_output = self.discriminator(prj)
        fake_output = self.discriminator(prj_rec)
        g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
        d_loss = discriminator_loss(real_output, fake_output)
        g_loss.backward(retain_graph=True)
        d_loss.backward()
        self.generator_optimizer.step()
        self.discriminator_optimizer.step()

        return {'recon': recon,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    def recon(self):
        
        self.prj_input = self.nor_tomo(self.prj_input)
        
        self.make_model()
        if self.init_wpath:
            self.generator.load_state_dict(torch.load(self.init_wpath+'generator.pth'))
            print('generator is initialized')
            self.discriminator.load_state_dict(torch.load(self.init_wpath+'discriminator.pth'))
        recon = torch.zeros((self.iter_num, 1, self.px, self.px))
        gen_loss = torch.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo', self.prj_input.cpu())
            # recon_monitor.display()
            # recon_monitor.initial_plot(self.prj_input.view(self.nang, self.px).cpu())
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            step_result = self.recon_step(self.prj_input, self.angle)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            ###########################################################################
            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                if self.recon_monitor:
                    prj_rec = step_result['prj_rec'].view(self.nang, self.px)
                    prj_diff = torch.abs(prj_rec - self.prj_input.view((self.nang, self.px))).cpu()
                    rec_plt = recon[epoch].view(self.px, self.px).cpu()
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss.cpu())
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].item()))
        if self.save_wpath != None:
            torch.save(self.generator.state_dict(), self.save_wpath+'generator.pth')
            torch.save(self.discriminator.state_dict(), self.save_wpath+'discriminator.pth')
        if self.recon_monitor:
            recon_monitor.close_plot()
        return tensor_to_np(recon[epoch].cpu())

