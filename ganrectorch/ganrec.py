import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from ganrectorch.models import Generator, Discriminator
from ganrectorch.propagators import RadonTransform
from ganrectorch.utils import RECONmonitor, to_device, tensor_to_np


def torch_configures():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 32


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


class GANtomo:
    def __init__(self, prj_input, angle, **kwargs):
        super(GANtomo, self).__init__()
        tomo_args = config["GANtomo"]
        tomo_args.update(**kwargs)
        torch_configures()
        self.scaler = GradScaler()
        self._input_data(prj_input, angle)
        self.iter_num = tomo_args["iter_num"]
        self.conv_num = tomo_args["conv_num"]
        self.conv_size = tomo_args["conv_size"]
        self.dropout = tomo_args["dropout"]
        self.l1_ratio = tomo_args["l1_ratio"]
        self.g_learning_rate = tomo_args["g_learning_rate"]
        self.d_learning_rate = tomo_args["d_learning_rate"]
        self.save_wpath = tomo_args["save_wpath"]
        self.init_wpath = tomo_args["init_wpath"]
        self.init_model = tomo_args["init_model"]
        self.recon_monitor = tomo_args["recon_monitor"]
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def _input_data(self, prj_input, angle):
        """
        Prepare and move input data to the GPU.
        """
        # Convert and reshape prj_input
        self.nang, self.px = prj_input.shape
        self.prj_input = torch.from_numpy(prj_input)
        self.prj_input = self.prj_input.view(-1, 1, self.nang, self.px)

        # Convert angle
        self.angle = torch.from_numpy(angle)
        # self.prj_input, self.angle = to_device([self.prj_input, self.angle])

    def make_model(self):
        self.generator = Generator(
            self.prj_input.shape[2], self.prj_input.shape[3], self.conv_num, self.conv_size, self.dropout, 1
        )
        self.discriminator = Discriminator()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

    # @torch.compile()
    def nor_tomo(self, data):

        # Calculate the mean and standard deviation of the data
        mean = torch.mean(data)
        std = torch.std(data)

        # Standardize the data (z-score normalization)
        standardized_data = (data - mean) / std

        # Find the minimum value in the standardized data
        standardized_min = torch.min(standardized_data)

        # Shift the data to start from 0
        shifted_data = standardized_data - standardized_min

        return shifted_data

    @torch.compile()
    def recon_step(self, prj, ang):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        with autocast():
            recon = self.generator(prj)
            recon = self.nor_tomo(recon)
            prj_rec = self.radon(recon, ang)
            prj_rec = self.nor_tomo(prj_rec)
            # print(f"prj shape: {prj.shape}")
            # print(f"prj_rec shape: {prj_rec.shape}")
            real_output = self.discriminator(prj)
            fake_output = self.discriminator(prj_rec)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)

        # Backward pass with gradient scaling
        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.scale(d_loss).backward()

        # Optimizer step with gradient scaling
        self.scaler.step(self.generator_optimizer)
        self.scaler.step(self.discriminator_optimizer)
        self.scaler.update()

        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    def recon(self):
        self.make_model()
        self.radon = RadonTransform(torch.empty(1, 1, self.px, self.px), self.angle)
        self.prj_input, self.angle, self.generator, self.discriminator, self.radon = to_device(
            [self.prj_input, self.angle, self.generator, self.discriminator, self.radon]
        )
        self.prj_input = self.nor_tomo(self.prj_input)
        if self.init_wpath:
            self.generator.load_state_dict(torch.load(self.init_wpath + "generator.pth"))
            print("generator is initialized")
            self.discriminator.load_state_dict(torch.load(self.init_wpath + "discriminator.pth"))
        recon = torch.zeros((self.iter_num, 1, self.px, self.px))
        gen_loss = torch.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo", self.prj_input.cpu())
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            step_result = self.recon_step(self.prj_input, self.angle)
            recon = step_result["recon"]
            gen_loss[epoch] = step_result["g_loss"]
            ###########################################################################
            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch].item(), D_loss=step_result["d_loss"].item())
                pbar.update(1)
            if (epoch + 1) % 100 == 0:
                if self.recon_monitor:
                    prj_rec = step_result["prj_rec"].view(self.nang, self.px)
                    prj_diff = torch.abs(prj_rec - self.prj_input.view((self.nang, self.px))).cpu()
                    rec_plt = recon.view(self.px, self.px).cpu()
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss.cpu())
                # print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                #                                                            gen_loss[epoch],
                #                                                            step_result['d_loss'].item()))
        if self.save_wpath != None:
            torch.save(self.generator.state_dict(), self.save_wpath + "generator.pth")
            torch.save(self.discriminator.state_dict(), self.save_wpath + "discriminator.pth")
        if self.recon_monitor:
            recon_monitor.close_plot()
        return tensor_to_np(recon.cpu())
