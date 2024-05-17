import os
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from ganrectf.models import make_discriminator, make_generator
from ganrectf.propagators import TomoRadon
from ganrectf.utils import RECONmonitor

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
    gen_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output)) \
               + l1_loss(img_output, pred) * l1_ratio
    return gen_loss

def tfnor_phase(img):
    img = (img - img.mean()) / img.std()
    img = img / torch.max(img)
    return img

class GANtomo(nn.Module):
    def __init__(self, prj_input, angle, **kwargs):
        super(GANtomo, self).__init__()
        tomo_args = config['GANtomo']
        tomo_args.update(**kwargs)
        self.prj_input = prj_input
        self.angle = angle
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
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        1)
                 
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.filter_optimizer = optim.Adam(self.filter.parameters(), lr=5e-5)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

    def tfnor_tomo(self, img):
        img = transforms.Normalize((0.5,), (0.5,))(img)
        return img

    def recon_step(self, prj, ang):      
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        recon = self.generator(prj)
        recon = self.tfnor_tomo(recon)
        tomo_radon_obj = TomoRadon(recon, ang)
        prj_rec = tomo_radon_obj.compute()
        prj_rec = self.tfnor_tomo(prj_rec)
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
        nang, px = self.prj_input.shape
        prj = self.prj_input.view(1, nang, px, 1) 
        prj = prj.float()
        prj = self.tfnor_tomo(prj)
        ang = self.angle.float()
        self.make_model()
        if self.init_wpath:
            self.generator.load_state_dict(torch.load(self.init_wpath+'generator.pth'))
            print('generator is initialized')
            self.discriminator.load_state_dict(torch.load(self.init_wpath+'discriminator.pth'))
        recon = torch.zeros((self.iter_num, px, px, 1))
        gen_loss = torch.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo')
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            
            step_result = self.recon_step(prj, ang)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            ###########################################################################
            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                if recon_monitor:
                    prj_rec = step_result['prj_rec'].view(nang, px)
                    prj_diff = torch.abs(prj_rec - self.prj_input.view((nang, px)))
                    rec_plt = recon[epoch].view(px, px)
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].item()))
        if self.save_wpath != None:
            torch.save(self.generator.state_dict(), self.save_wpath+'generator.pth')
            torch.save(self.discriminator.state_dict(), self.save_wpath+'discriminator.pth')
        recon_monitor.close_plot()
        return recon[epoch].float()