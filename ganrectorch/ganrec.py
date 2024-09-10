import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn, optim
from torch.amp import GradScaler, autocast
from ganrectorch.models import Generator, Discriminator
from ganrectorch.propagators import RadonTransform, PhaseFresnel
from ganrectorch.utils import (RECONmonitor, 
                               to_device, 
                               tensor_to_np, 
                               pad_to_power_of_2_square, 
                               unpad_image, 
                               next_power_of_2, 
                               )

def torch_configures():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 32
    # torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True


# Load the configuration from the JSON file
def load_config(filename):
    # Get the directory of the script
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the full path to the config file
    config_path = os.path.join(dir_path, filename)

    with open(config_path, "r") as file:
        config = json.load(file)
    return config


# Use the configuration
config = load_config("config.json")


# @torch.compile()
def discriminator_loss(real_output, fake_output):
    real_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output)))
    fake_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output)))
    total_loss = real_loss + fake_loss
    return total_loss


# @torch.compile()
def l1_loss(img1, img2):
    return torch.mean(torch.abs(img1 - img2))


# @torch.compile()
def l2_loss(img1, img2):
    return torch.pow(torch.mean(torch.abs(img1 - img2)), 2)


# @torch.compile()
def generator_loss(fake_output, img_output, pred, l1_ratio):
    # with autograd
    return torch.mean(
        torch.nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))
    ) + l1_ratio * l1_loss(img_output, pred)


# @torch.compile()
def tfnor_phase(img):
    img = (img - img.mean()) / img.std()
    img = img / torch.max(img)
    return img


class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, data):
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data


# class GANtomo:
#     def __init__(self, prj_input, angle, **kwargs):
#         super(GANtomo, self).__init__()
#         tomo_args = config["GANtomo"]
#         tomo_args.update(**kwargs)
#         torch_configures()
#         self.scaler = GradScaler()
#         self._input_data(prj_input, angle)
#         self.iter_num = tomo_args["iter_num"]
#         self.conv_num = tomo_args["conv_num"]
#         self.conv_size = tomo_args["conv_size"]
#         self.dropout = tomo_args["dropout"]
#         self.l1_ratio = tomo_args["l1_ratio"]
#         self.g_learning_rate = tomo_args["g_learning_rate"]
#         self.d_learning_rate = tomo_args["d_learning_rate"]
#         self.save_wpath = tomo_args["save_wpath"]
#         self.init_wpath = tomo_args["init_wpath"]
#         self.init_model = tomo_args["init_model"]
#         self.recon_monitor = tomo_args["recon_monitor"]
#         self.generator = None
#         self.discriminator = None
#         self.generator_optimizer = None
#         self.discriminator_optimizer = None

#     def _input_data(self, prj_input, angle):
#         """
#         Prepare and move input data to the GPU.
#         """
#         # Convert and reshape prj_input
#         self.nang, self.px = prj_input.shape
#         self.prj_input = torch.from_numpy(prj_input)
#         self.prj_input = self.prj_input.view(-1, 1, self.nang, self.px)

#         # Convert angle
#         self.angle = torch.from_numpy(angle)
#         # self.prj_input, self.angle = to_device([self.prj_input, self.angle])

#     def make_model(self):
#         self.generator = Generator(
#             self.prj_input.shape[2], self.prj_input.shape[3], self.conv_num, self.conv_size, self.dropout, 1
#         )
#         self.discriminator = Discriminator()
#         self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
#         self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

#     # @torch.compile()
#     def nor_tomo(self, data):

#         # Calculate the mean and standard deviation of the data
#         mean = torch.mean(data)
#         std = torch.std(data)

#         # Standardize the data (z-score normalization)
#         standardized_data = (data - mean) / std

#         # Find the minimum value in the standardized data
#         standardized_min = torch.min(standardized_data)

#         # Shift the data to start from 0
#         shifted_data = standardized_data - standardized_min

#         return shifted_data

#     @torch.compile()
#     def recon_step(self, prj, ang):
#         self.generator_optimizer.zero_grad()
#         self.discriminator_optimizer.zero_grad()
#         with autocast():
#             recon = self.generator(prj)
#             recon = self.nor_tomo(recon)
#             prj_rec = self.radon(recon, ang)
#             prj_rec = self.nor_tomo(prj_rec)
#             # print(f"prj shape: {prj.shape}")
#             # print(f"prj_rec shape: {prj_rec.shape}")
#             real_output = self.discriminator(prj)
#             fake_output = self.discriminator(prj_rec)
#             g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
#             d_loss = discriminator_loss(real_output, fake_output)

#         # Backward pass with gradient scaling
#         self.scaler.scale(g_loss).backward(retain_graph=True)
#         self.scaler.scale(d_loss).backward()

#         # Optimizer step with gradient scaling
#         self.scaler.step(self.generator_optimizer)
#         self.scaler.step(self.discriminator_optimizer)
#         self.scaler.update()

#         return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

#     def recon(self):
#         self.make_model()
#         self.radon = RadonTransform(torch.empty(1, 1, self.px, self.px), self.angle)
#         self.prj_input, self.angle, self.generator, self.discriminator, self.radon = to_device(
#             [self.prj_input, self.angle, self.generator, self.discriminator, self.radon]
#         )
#         self.prj_input = self.nor_tomo(self.prj_input)
#         if self.init_wpath:
#             self.generator.load_state_dict(torch.load(self.init_wpath + "generator.pth"))
#             print("generator is initialized")
#             self.discriminator.load_state_dict(torch.load(self.init_wpath + "discriminator.pth"))
#         recon = torch.zeros((self.iter_num, 1, self.px, self.px))
#         gen_loss = torch.zeros((self.iter_num))

#         ###########################################################################
#         # Reconstruction process monitor
#         if self.recon_monitor:
#             plot_x, plot_loss = [], []
#             recon_monitor = RECONmonitor("tomo", self.prj_input.cpu())
#             pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

#         ###########################################################################
#         for epoch in range(self.iter_num):

#             ###########################################################################
#             ## Call the rconstruction step

#             step_result = self.recon_step(self.prj_input, self.angle)
#             recon = step_result["recon"]
#             gen_loss[epoch] = step_result["g_loss"]
#             ###########################################################################
#             if self.recon_monitor:
#                 plot_x.append(epoch)
#                 plot_loss = gen_loss[: epoch + 1]
#                 pbar.set_postfix(G_loss=gen_loss[epoch].item(), D_loss=step_result["d_loss"].item())
#                 pbar.update(1)
#             if (epoch + 1) % 100 == 0:
#                 if self.recon_monitor:
#                     prj_rec = step_result["prj_rec"].view(self.nang, self.px)
#                     prj_diff = torch.abs(prj_rec - self.prj_input.view((self.nang, self.px))).cpu()
#                     rec_plt = recon.view(self.px, self.px).cpu()
#                     recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss.cpu())
#                 # print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
#                 #                                                            gen_loss[epoch],
#                 #                                                            step_result['d_loss'].item()))
#         if self.save_wpath != None:
#             torch.save(self.generator.state_dict(), self.save_wpath + "generator.pth")
#             torch.save(self.discriminator.state_dict(), self.save_wpath + "discriminator.pth")
#         if self.recon_monitor:
#             recon_monitor.close_plot()
#         return tensor_to_np(recon.cpu())
 


class GANtomo(nn.Module):
    def __init__(self, prj_input, angle, **kwargs):
        super(GANtomo, self).__init__()
        tomo_args = config["GANtomo"]
        tomo_args.update(**kwargs)
        self.scaler = GradScaler()
        self.iter_num = tomo_args["iter_num"]
        self.conv_num = tomo_args["conv_num"]
        self.conv_size = tomo_args["conv_size"]
        self.dropout = tomo_args["dropout"]
        self.l1_ratio = tomo_args["l1_ratio"]
        self.g_learning_rate = tomo_args["g_learning_rate"]
        self.d_learning_rate = tomo_args["d_learning_rate"]
        self.save_wpath = tomo_args["save_wpath"]
        self.init_wpath = tomo_args["init_wpath"]
        self.recon_monitor = tomo_args["recon_monitor"]

        self._input_data(prj_input, angle)
        self.make_model()

    def _input_data(self, prj_input, angle):
        """
        Prepare and move input data to the GPU.
        """
        self.nang, self.px = prj_input.shape
        self.prj_input = torch.tensor(self.nor_tomo(prj_input), device='cuda').view(-1, 1, self.nang, self.px)
        self.angle = torch.tensor(angle, device='cuda')

    def make_model(self):
        self.generator = Generator(
            self.prj_input.shape[2], self.prj_input.shape[3], self.conv_num, self.conv_size, self.dropout, 1
        ).to('cuda')
        self.discriminator = Discriminator().to('cuda')
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

    def nor_tomo(self, data):
        mean = data.mean()
        std = data.std()
        normalized_data = (data - mean) / std
        return normalized_data - normalized_data.min()

    @torch.compile()  # Optional: Ensure compatibility if torch.compile() is available.
    def recon_step(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            recon = self.generator(self.prj_input)
            recon = self.nor_tomo(recon)
            prj_rec = self.radon(recon, self.angle)
            prj_rec = self.nor_tomo(prj_rec)
            real_output = self.discriminator(self.prj_input)
            fake_output = self.discriminator(prj_rec)
            g_loss = generator_loss(fake_output, self.prj_input, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        
        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.scale(d_loss).backward()

        self.scaler.step(self.generator_optimizer)
        self.scaler.step(self.discriminator_optimizer)
        self.scaler.update()

        return {"recon": recon.detach(), "prj_rec": prj_rec.detach(), "g_loss": g_loss, "d_loss": d_loss}

    def recon(self):
        self.radon = RadonTransform(torch.empty(1, 1, self.px, self.px, device='cuda'), self.angle)
        
        if self.init_wpath:
            self.generator.load_state_dict(torch.load(self.init_wpath + "generator.pth"))
            self.discriminator.load_state_dict(torch.load(self.init_wpath + "discriminator.pth"))

        recon = torch.zeros((self.iter_num, 1, self.px, self.px), device='cuda')
        gen_loss = torch.zeros((self.iter_num), device='cuda')

        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo", self.prj_input.cpu())
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

        for epoch in range(self.iter_num):
            step_result = self.recon_step()
            recon = step_result["recon"]
            gen_loss[epoch] = step_result["g_loss"]

            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch].item(), D_loss=step_result["d_loss"].item())
                pbar.update(1)

                if (epoch + 1) % 100 == 0:
                    prj_rec = step_result["prj_rec"].view(self.nang, self.px)
                    prj_diff = torch.abs(prj_rec - self.prj_input.view((self.nang, self.px))).cpu()
                    rec_plt = recon.view(self.px, self.px).cpu()
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss.cpu())

        if self.save_wpath:
            torch.save(self.generator.state_dict(), self.save_wpath + "generator.pth")
            torch.save(self.discriminator.state_dict(), self.save_wpath + "discriminator.pth")

        if self.recon_monitor:
            recon_monitor.close_plot()

        return tensor_to_np(recon.cpu())

    
    
# class GANphase:
#     def __init__(self, i_input, energy, z, pv, **kwargs):
#         super(GANphase, self).__init__()
#         phase_args = config["GANphase"]
#         phase_args.update(**kwargs)
#         torch_configures()
#         self.scaler = GradScaler()
#         self._input_data(i_input, energy, z, pv)
#         self.iter_num = phase_args["iter_num"]
#         self.conv_num = phase_args["conv_num"]
#         self.conv_size = phase_args["conv_size"]
#         self.dropout = phase_args["dropout"]
#         self.l1_ratio = phase_args["l1_ratio"]
#         self.abs_ratio = phase_args["abs_ratio"]
#         self.g_learning_rate = phase_args["g_learning_rate"]
#         self.d_learning_rate = phase_args["d_learning_rate"]
#         self.phase_only = phase_args["phase_only"]
#         self.save_wpath = phase_args["save_wpath"]
#         self.init_wpath = phase_args["init_wpath"]
#         self.init_model = phase_args["init_model"]
#         self.recon_monitor = phase_args["recon_monitor"]
#         self.generator = None
#         self.discriminator = None
#         self.generator_optimizer = None
#         self.discriminator_optimizer = None

#     def _input_data(self, i_input, energy, z, pv):
#         """
#         Prepare and move input data to the GPU.
#         """
#         # Convert and reshape prj_input
#         self.img_h, self.img_w = i_input.shape
#         self.i_input = torch.from_numpy(pad_to_power_of_2_square(i_input))
#         self.padded_dim = next_power_of_2(max(self.img_h, self.img_w))
#         self.i_input = self.i_input.view(-1, 1, self.padded_dim, self.padded_dim)
#         self.energy = torch.tensor(energy)
#         self.z = torch.tensor(z)
#         self.pv = torch.tensor(pv)
#         self.ff = ffactor(self.padded_dim, self.energy, self.z, self.pv)
#         # self.prj_input, self.angle = to_device([self.prj_input, self.angle])

#     def make_model(self):
#         self.generator = Generator(
#             self.padded_dim, self.padded_dim, self.conv_num, self.conv_size, self.dropout, 2
#         )
#         self.discriminator = Discriminator()
#         self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
#         self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

#     # @torch.compile()
#     def nor_phase(self, data):

#         # Calculate the mean and standard deviation of the data
#         mean = torch.mean(data)
#         std = torch.std(data)

#         # Standardize the data (z-score normalization)
#         data = (data - mean) / std

#         return data/torch.max(data)

#     # @torch.compile()
#     def recon_step(self, i_input):
#         self.generator_optimizer.zero_grad()
#         self.discriminator_optimizer.zero_grad()
#         with autocast():
#             recon = self.generator(i_input)
#             phase = self.nor_phase(recon[:, 0, :, :].squeeze())
#             absorption = (1- self.nor_phase(recon[:, 1, :, :].squeeze()))*self.abs_ratio
#             if self.phase_only:
#                 absorption = torch.zeros_like(phase)
#             i_rec = self.fresnel(phase, absorption)
#             # print(i_rec.dtype, i_rec.shape)
#             # print(i_input.dtype, i_input.shape)
#             real_output = self.discriminator(i_input)
#             fake_output = self.discriminator(i_rec.half())
#             g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
#             d_loss = discriminator_loss(real_output, fake_output)

#         # Backward pass with gradient scaling
#         self.scaler.scale(g_loss).backward(retain_graph=True)
#         self.scaler.scale(d_loss).backward()

#         # Optimizer step with gradient scaling
#         self.scaler.step(self.generator_optimizer)
#         self.scaler.step(self.discriminator_optimizer)
#         self.scaler.update()

#         return {"phase": phase, "absorption": absorption, "i_rec": i_rec, "g_loss": g_loss, "d_loss": d_loss}

#     def recon(self):
#         self.make_model()  
#         self.fresnel = PhaseFresnel(torch.empty(1, 1, self.padded_dim, self.padded_dim),
#                                     torch.empty(1, 1, self.padded_dim, self.padded_dim),
#                                     self.ff, self.padded_dim)
#         self.i_input, self.generator, self.discriminator, self.fresnel = to_device(
#             [self.i_input, self.generator, self.discriminator, self.fresnel]
#         )
#         self.i_input = self.nor_phase(self.i_input)
#         if self.init_wpath:
#             self.generator.load_state_dict(torch.load(self.init_wpath + "generator.pth"))
#             print("generator is initialized")
#             self.discriminator.load_state_dict(torch.load(self.init_wpath + "discriminator.pth"))
#         recon = torch.zeros((self.iter_num, 1, self.padded_dim, self.padded_dim))
#         gen_loss = torch.zeros((self.iter_num))

#         ###########################################################################
#         # Reconstruction process monitor
#         if self.recon_monitor:
#             plot_x, plot_loss = [], []
#             recon_monitor = RECONmonitor("phase", self.i_input.cpu())
#             pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

#         ###########################################################################
#         for epoch in range(self.iter_num):

#             ###########################################################################
#             ## Call the rconstruction step

#             step_result = self.recon_step(self.i_input)
#             recon = step_result["phase"]
#             gen_loss[epoch] = step_result["g_loss"]
#             ###########################################################################
#             if self.recon_monitor:
#                 plot_x.append(epoch)
#                 plot_loss = gen_loss[: epoch + 1]
#                 pbar.set_postfix(G_loss=gen_loss[epoch].item(), D_loss=step_result["d_loss"].item())
#                 pbar.update(1)
#             if (epoch + 1) % 100 == 0:
#                 if self.recon_monitor:
#                     i_rec = step_result["i_rec"]
#                     i_diff = torch.abs(i_rec - self.i_input).cpu()
#                     rec_plt = recon.cpu()
#                     recon_monitor.update_plot(epoch, i_diff, rec_plt, plot_x, plot_loss.cpu())
#         if self.save_wpath != None:
#             torch.save(self.generator.state_dict(), self.save_wpath + "generator.pth")
#             torch.save(self.discriminator.state_dict(), self.save_wpath + "discriminator.pth")
#         if self.recon_monitor:
#             recon_monitor.close_plot()
#         return tensor_to_np(recon.cpu())
    
    




class GANphase(nn.Module):
    def __init__(self, i_input, energy, z, pv, **kwargs):
        super(GANphase, self).__init__()
        phase_args = config["GANphase"]
        phase_args.update(**kwargs)
        torch_configures()  # Assuming this sets up necessary Torch configurations

        self.scaler = GradScaler()
        self._input_data(i_input, energy, z, pv)
        self.iter_num = phase_args["iter_num"]
        self.conv_num = phase_args["conv_num"]
        self.conv_size = phase_args["conv_size"]
        self.dropout = phase_args["dropout"]
        self.l1_ratio = phase_args["l1_ratio"]
        self.abs_ratio = phase_args["abs_ratio"]
        self.g_learning_rate = phase_args["g_learning_rate"]
        self.d_learning_rate = phase_args["d_learning_rate"]
        self.phase_only = phase_args["phase_only"]
        self.save_wpath = phase_args["save_wpath"]
        self.init_wpath = phase_args["init_wpath"]
        self.init_model = phase_args["init_model"]
        self.recon_monitor = phase_args["recon_monitor"]
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def _input_data(self, i_input, energy, z, pv):
        """
        Prepare and move input data to the GPU.
        """
        self.img_h, self.img_w = i_input.shape
        self.i_input = torch.from_numpy(pad_to_power_of_2_square(i_input)).float().to('cuda')
        self.padded_dim = next_power_of_2(max(self.img_h, self.img_w))
        self.i_input = self.i_input.view(-1, 1, self.padded_dim, self.padded_dim)
        self.energy = torch.tensor(energy).to('cuda')
        self.z = torch.tensor(z).to('cuda')
        self.pv = torch.tensor(pv).to('cuda')
        self.ff = self.ffactor(self.padded_dim*2, self.energy, self.z, self.pv).to('cuda')
    
    def make_model(self):
        self.generator = Generator(
            self.padded_dim, self.padded_dim, self.conv_num, self.conv_size, self.dropout, 2
        ).to('cuda')
        self.discriminator = Discriminator().to('cuda')
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)
        
    def ffactor(self, px, energy, z, pv, device='cuda'):
    # Calculate the wavelength in meters
        lambda_p = 1.23984122e-09 / energy

    # Calculate the frequency prefactor
        frequ_prefactor = 2 * torch.pi * lambda_p * z / pv**2

    # Generate the frequency components using torch
        freq = torch.fft.fftfreq(px, device=device)

    # Create the xi and eta grids
        xi, eta = torch.meshgrid(freq, freq, indexing='ij')

    # Calculate the phase factor
        h = torch.exp(-1j * frequ_prefactor * (xi**2 + eta**2) / 2)

        return h


    def nor_recon(self, data):
        mean = data.mean()
        std = data.std()
        data = (data - mean) / std
        return data / data.max()
    
    def nor_input(self, data):
        mean = data.mean()
        std = data.std()
        normalized_data = (data - mean) / std
        return normalized_data - normalized_data.min()
    
    @torch.compile()
    def recon_step(self, i_input):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        with autocast(device_type='cuda'):
            recon = self.generator(i_input)
            recon = self.nor_recon(recon)
            # phase = self.nor_phase(recon[:, 0, :, :].squeeze())
            # absorption = (1 - self.nor_phase(recon[:, 1, :, :].squeeze())) * self.abs_ratio
            if self.phase_only:
                recon[:, 1, :, :] = 0
            i_rec = self.fresnel(recon)
            i_rec = self.nor_input(i_rec)
            real_output = self.discriminator(i_input)
            fake_output = self.discriminator(i_rec)
            g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.scale(d_loss).backward()

        self.scaler.step(self.generator_optimizer)
        self.scaler.step(self.discriminator_optimizer)
        self.scaler.update()

        return {"phase": recon[:,0,:,:], "absorption": recon[:,1,:,:], "i_rec": i_rec, "g_loss": g_loss, "d_loss": d_loss}

    def recon(self):
        self.make_model()
        self.fresnel = PhaseFresnel(
            torch.empty(1, 2, self.padded_dim, self.padded_dim, device='cuda'),
            self.ff, self.padded_dim, self.abs_ratio).to('cuda')
        # self.i_input = transforms.Normalize(mean=[0.0], std=[1.0])(self.i_input)
        # self.i_input = self.nor_input(self.i_input)
        
        if self.init_wpath:
            self.generator.load_state_dict(torch.load(self.init_wpath + "generator.pth"))
            print("generator is initialized")
            self.discriminator.load_state_dict(torch.load(self.init_wpath + "discriminator.pth"))

        recon = torch.zeros((self.iter_num, 1, self.padded_dim, self.padded_dim), device='cuda')
        gen_loss = torch.zeros((self.iter_num), device='cuda')

        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("phase", self.i_input.cpu())
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

        for epoch in range(self.iter_num):
            step_result = self.recon_step(self.i_input)
            recon = step_result["phase"]
            gen_loss[epoch] = step_result["g_loss"]

            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch].item(), D_loss=step_result["d_loss"].item())
                pbar.update(1)

                if (epoch + 1) % 100 == 0:
                    if self.recon_monitor:
                        i_rec = step_result["i_rec"]
                        i_diff = torch.abs(i_rec - self.i_input).cpu()
                        rec_plt = recon.cpu()
                        recon_monitor.update_plot(epoch, i_diff, rec_plt, plot_x, plot_loss.cpu())

        if self.save_wpath is not None:
            torch.save(self.generator.state_dict(), self.save_wpath + "generator.pth")
            torch.save(self.discriminator.state_dict(), self.save_wpath + "discriminator.pth")

        if self.recon_monitor:
            recon_monitor.close_plot()

        return tensor_to_np(recon.cpu())


