"""GAN-based reconstruction driven by Hugging Face ``diffusers`` UNet models.

This mirrors :mod:`ganrectorch.ganrec` but replaces the hand-written CNN
generator with :class:`ganrecdiff.models.DiffusersGenerator`.  The physics
forward models and plotting utilities are reused from ``ganrectorch``.
"""

import os
import json
import copy

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ganrecdiff.models import DiffusersGenerator, Discriminator
from ganrectorch.propagators import RadonTransform
from ganrectorch.utils import RECONmonitor, to_device, tensor_to_np, ffactor


def _load_config(filename="config.json"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, filename), "r") as file:
        return json.load(file)


config = _load_config()


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def discriminator_loss(real_output, fake_output):
    bce = torch.nn.BCEWithLogitsLoss()
    real_loss = bce(real_output, torch.ones_like(real_output))
    fake_loss = bce(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


def l1_loss(img1, img2):
    return torch.mean(torch.abs(img1 - img2))


def generator_loss(fake_output, img_output, pred, l1_ratio):
    bce = torch.nn.BCEWithLogitsLoss()
    return bce(fake_output, torch.ones_like(fake_output)) + l1_ratio * l1_loss(img_output, pred)


def nor_tomo(data):
    """Z-score normalize then shift to start at zero."""
    data = (data - torch.mean(data)) / torch.std(data)
    return data - torch.min(data)


def nor_phase(img):
    img = (img - img.mean()) / img.std()
    return img / torch.max(img)


def fresnel_propagate(phase, absorption, ff_complex, px):
    """Free-space (Fresnel) intensity from a phase/absorption object.

    ``phase`` and ``absorption`` are 2-D ``(px, px)`` tensors; ``ff_complex`` is
    the ``(2*px, 2*px)`` complex Fresnel transfer function.
    """
    pad = px // 2
    phase = torch.nn.functional.pad(phase[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
    absorption = torch.nn.functional.pad(absorption[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
    wavefield = torch.exp(torch.complex(-absorption, phase))
    intensity = torch.abs(torch.fft.ifft2(ff_complex * torch.fft.fft2(wavefield))) ** 2
    return intensity[pad:pad + px, pad:pad + px]


# --------------------------------------------------------------------------- #
# Stability helpers
# --------------------------------------------------------------------------- #
class DeviceEMA:
    """Exponential moving average of model parameters (on device)."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def swap_in(self):
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters()}
        for n, p in self.model.named_parameters():
            p.data.copy_(self.shadow[n])

    def swap_out(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self.backup[n])
        self.backup = {}


class DeviceSnapshot:
    """On-device snapshot/restore of model state dicts."""

    def __init__(self, models):
        self.models = models
        self.copies = [copy.deepcopy(m.state_dict()) for m in models]

    @torch.no_grad()
    def snapshot(self):
        self.copies = [copy.deepcopy(m.state_dict()) for m in self.models]

    @torch.no_grad()
    def restore(self):
        for m, sd in zip(self.models, self.copies):
            m.load_state_dict(sd)


class ScalarEMA:
    def __init__(self, decay=0.95):
        self.decay = decay
        self.value = None

    def update(self, x):
        x = float(x)
        self.value = x if self.value is None else self.decay * self.value + (1.0 - self.decay) * x
        return self.value


# --------------------------------------------------------------------------- #
# Base reconstruction driver
# --------------------------------------------------------------------------- #
class _GANBase:
    """Shared training loop for diffusers-backed GAN reconstruction."""

    def __init__(self, args):
        self.iter_num = args["iter_num"]
        self.block_out_channels = tuple(args["block_out_channels"])
        self.layers_per_block = args["layers_per_block"]
        self.l1_ratio = args["l1_ratio"]
        self.g_learning_rate = args["g_learning_rate"]
        self.d_learning_rate = args["d_learning_rate"]
        self.save_wpath = args["save_wpath"]
        self.init_wpath = args["init_wpath"]
        self.recon_monitor = args["recon_monitor"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(self.device.type, enabled=self.device.type == "cuda")

    # --- to be provided by subclasses ------------------------------------- #
    def _output_num(self):
        return 1

    def _recon_shape(self):
        """(out_h, out_w) of the generator output."""
        raise NotImplementedError

    def forward_model(self, gen_output):
        """Map generator output to measurement space.

        Returns a dict with at least ``predicted`` (same shape as the input
        measurement) and ``recon`` (final reconstruction to return).
        """
        raise NotImplementedError

    # --- model construction ----------------------------------------------- #
    def make_model(self):
        out_h, out_w = self._recon_shape()
        self.generator = DiffusersGenerator(
            out_h,
            out_w,
            in_channels=1,
            output_num=self._output_num(),
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
        )
        self.discriminator = Discriminator(in_channels=1)
        self.generator_optimizer = optim.AdamW(
            self.generator.parameters(), lr=self.g_learning_rate, weight_decay=1e-4, betas=(0.9, 0.99)
        )
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), lr=self.d_learning_rate, weight_decay=1e-4, betas=(0.9, 0.99)
        )

    def recon_step(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        with autocast(self.device.type, enabled=self.device.type == "cuda"):
            gen_output = self.generator(self.input_tensor)
            result = self.forward_model(gen_output)
            pred = result["predicted"]
            real_output = self.discriminator(self.input_tensor)
            fake_output = self.discriminator(pred)
            g_loss = generator_loss(fake_output, self.input_tensor, pred, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)

        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.scale(d_loss).backward()
        self.scaler.unscale_(self.generator_optimizer)
        self.scaler.unscale_(self.discriminator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.scaler.step(self.generator_optimizer)
        self.scaler.step(self.discriminator_optimizer)
        self.scaler.update()

        result.update(g_loss=g_loss, d_loss=d_loss)
        return result

    # --- main loop -------------------------------------------------------- #
    def _run(self, monitor_target="tomo"):
        self.make_model()
        self._to_device()
        if self.init_wpath:
            self.generator.load_state_dict(torch.load(os.path.join(self.init_wpath, "generator.pth")))
            self.discriminator.load_state_dict(torch.load(os.path.join(self.init_wpath, "discriminator.pth")))

        ema_decay = float(getattr(self, "ema_decay", 0.99))
        snapshot_every = int(getattr(self, "snapshot_every", 50))
        log_every = int(getattr(self, "log_every", 10))
        spike_factor = float(getattr(self, "spike_factor", 1.5))
        warmup_steps = int(getattr(self, "warmup_steps", max(5, self.iter_num // 20)))
        lr_backoff = float(getattr(self, "lr_backoff", 0.5))
        lr_floor = float(getattr(self, "lr_floor", 1e-7))
        freeze_disc_max = int(getattr(self, "freeze_disc_max", 10))

        ema = DeviceEMA(self.generator, decay=ema_decay)
        snap = DeviceSnapshot([self.generator, self.discriminator])
        snap.snapshot()
        g_ema, d_ema = ScalarEMA(0.95), ScalarEMA(0.95)
        d_lr0 = self.d_learning_rate
        freeze_disc_steps = 0

        gen_loss = torch.zeros(self.iter_num)
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor(monitor_target, self.input_tensor.cpu())
        pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

        recon = None
        for epoch in range(self.iter_num):
            if freeze_disc_steps > 0:
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = 0.0
                freeze_disc_steps -= 1
            else:
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = d_lr0

            result = self.recon_step()
            g_loss_val = result["g_loss"].item()
            d_loss_val = result["d_loss"].item()
            recon = result["recon"]

            if not (np.isfinite(g_loss_val) and np.isfinite(d_loss_val)):
                snap.restore()
                for pg in self.generator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                freeze_disc_steps = freeze_disc_max
                pbar.set_postfix_str("NaN/Inf->rollback")
                pbar.update(1)
                continue

            if epoch % log_every == 0 and torch.var(recon).item() < 1e-6:
                snap.restore()
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                pbar.set_postfix_str("flat->rollback")
                pbar.update(1)
                continue

            g_bar = g_ema.update(g_loss_val)
            d_bar = d_ema.update(d_loss_val)

            if epoch > warmup_steps and epoch % log_every == 0:
                if g_loss_val > spike_factor * max(1e-8, g_bar) or d_loss_val > spike_factor * max(1e-8, d_bar):
                    snap.restore()
                    for pg in self.generator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    for pg in self.discriminator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    freeze_disc_steps = freeze_disc_max
                    pbar.set_postfix_str("spike->rollback")
                    pbar.update(1)
                    continue

            ema.update()
            if epoch % snapshot_every == 0:
                snap.snapshot()
            gen_loss[epoch] = g_loss_val

            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss.append(g_loss_val)
                pbar.set_postfix(G_loss=f"{g_loss_val:.4f}", D_loss=f"{d_loss_val:.4f}")
            pbar.update(1)

            if (epoch + 1) % 100 == 0 and self.recon_monitor:
                self._update_monitor(recon_monitor, epoch, result, recon, plot_x, plot_loss)

        pbar.close()
        if self.save_wpath is not None:
            torch.save(self.generator.state_dict(), os.path.join(self.save_wpath, "generator.pth"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.save_wpath, "discriminator.pth"))
        if self.recon_monitor:
            recon_monitor.close_plot()

        ema.swap_in()
        try:
            with torch.no_grad():
                final = self.forward_model(self.generator(self.input_tensor))["recon"]
        finally:
            ema.swap_out()
        return tensor_to_np(final.cpu())

    def _update_monitor(self, monitor, epoch, result, recon, plot_x, plot_loss):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Tomography
# --------------------------------------------------------------------------- #
class GANtomo(_GANBase):
    """Tomographic reconstruction with a diffusers UNet generator."""

    def __init__(self, prj_input, angle, **kwargs):
        args = dict(config["GANtomo"])
        args.update(kwargs)
        super().__init__(args)
        self.nang, self.px = prj_input.shape
        self.prj_input = torch.from_numpy(prj_input).view(-1, 1, self.nang, self.px)
        self.angle = torch.from_numpy(angle)

    def _recon_shape(self):
        return self.px, self.px

    def _to_device(self):
        self.radon = RadonTransform(torch.empty(1, 1, self.px, self.px), self.angle)
        self.prj_input, self.angle, self.generator, self.discriminator, self.radon = to_device(
            [self.prj_input, self.angle, self.generator, self.discriminator, self.radon]
        )
        self.prj_input = nor_tomo(self.prj_input)
        self.input_tensor = self.prj_input

    def forward_model(self, gen_output):
        recon = nor_tomo(gen_output)
        prj_rec = nor_tomo(self.radon(recon, self.angle))
        return {"recon": recon, "predicted": prj_rec}

    def _update_monitor(self, monitor, epoch, result, recon, plot_x, plot_loss):
        prj_rec = result["predicted"].view(self.nang, self.px)
        prj_diff = torch.abs(prj_rec - self.input_tensor.view(self.nang, self.px)).cpu()
        rec_plt = recon.view(self.px, self.px).cpu()
        monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, torch.tensor(plot_loss))

    def recon(self):
        return self._run(monitor_target="tomo")


# --------------------------------------------------------------------------- #
# Phase retrieval (Fresnel)
# --------------------------------------------------------------------------- #
class GANphase(_GANBase):
    """Fresnel phase retrieval with a diffusers UNet generator."""

    def __init__(self, i_input, energy, z, pv, **kwargs):
        args = dict(config["GANphase"])
        args.update(kwargs)
        super().__init__(args)
        self.px, _ = i_input.shape
        self.i_input = torch.from_numpy(i_input).view(-1, 1, self.px, self.px)
        self.energy = energy
        self.z = z
        self.pv = pv
        self.abs_ratio = args["abs_ratio"]
        self.phase_only = args["phase_only"]
        ff = ffactor(self.px * 2, self.px * 2, energy, z, pv)
        self.ff = torch.from_numpy(np.stack([ff.real, ff.imag], axis=0)).float()

    def _output_num(self):
        return 1 if self.phase_only else 2

    def _recon_shape(self):
        return self.px, self.px

    def _to_device(self):
        self.i_input, self.generator, self.discriminator, self.ff = to_device(
            [self.i_input, self.generator, self.discriminator, self.ff]
        )
        self.i_input = nor_phase(self.i_input)
        self.input_tensor = self.i_input
        self._ff_complex = torch.complex(self.ff[0], self.ff[1])

    def forward_model(self, gen_output):
        phase = nor_phase(gen_output[:, 0:1, :, :])
        if self.phase_only:
            absorption = torch.zeros_like(phase)
        else:
            absorption = nor_phase(gen_output[:, 1:2, :, :]) * self.abs_ratio
        i_rec = fresnel_propagate(phase[0, 0], absorption[0, 0], self._ff_complex, self.px)
        i_rec = nor_phase(i_rec.view(-1, 1, self.px, self.px))
        return {"recon": phase, "absorption": absorption, "predicted": i_rec}

    def _update_monitor(self, monitor, epoch, result, recon, plot_x, plot_loss):
        i_rec = result["predicted"].view(self.px, self.px)
        i_diff = torch.abs(i_rec - self.input_tensor.view(self.px, self.px)).cpu()
        rec_plt = recon.view(self.px, self.px).cpu()
        monitor.update_plot(epoch, i_diff, rec_plt, plot_x, torch.tensor(plot_loss))

    def recon(self):
        return self._run(monitor_target="phase")


# --------------------------------------------------------------------------- #
# Generic reconstruction with a user-supplied forward model
# --------------------------------------------------------------------------- #
class GANrec(_GANBase):
    """General-purpose diffusers-backed GAN reconstruction.

    Parameters
    ----------
    input_data : ndarray (2-D)
        Measured data.
    forward_fn : callable
        ``(gen_output, input_tensor) -> dict`` returning at least
        ``"predicted"`` (same shape as the input) and ``"recon"``.  Must use
        differentiable torch ops only.
    output_num : int, optional
        Number of generator output channels (default 1).
    shape_output : tuple, optional
        ``(out_h, out_w)`` of the reconstruction. Defaults to a square of the
        input's last dimension.
    monitor_type : str, optional
        ``"tomo"`` or ``"phase"`` for the live plot (default ``"tomo"``).
    """

    def __init__(self, input_data, forward_fn, output_num=1, shape_output=None,
                 monitor_type="tomo", **kwargs):
        args = dict(config["GANrec"])
        args.update(kwargs)
        super().__init__(args)
        self._forward_fn = forward_fn
        self._out_num = output_num
        h, w = input_data.shape
        self._shape_output = shape_output or (w, w)
        self._monitor_type = monitor_type
        self.input_data = torch.from_numpy(input_data).view(-1, 1, h, w)

    def _output_num(self):
        return self._out_num

    def _recon_shape(self):
        return self._shape_output

    def _to_device(self):
        self.input_data, self.generator, self.discriminator = to_device(
            [self.input_data, self.generator, self.discriminator]
        )
        self.input_tensor = self.input_data

    def forward_model(self, gen_output):
        result = self._forward_fn(gen_output, self.input_tensor)
        if "recon" not in result:
            result["recon"] = gen_output
        return result

    def _update_monitor(self, monitor, epoch, result, recon, plot_x, plot_loss):
        pred = result["predicted"]
        diff = torch.abs(pred.view(*self.input_tensor.shape[-2:]) - self.input_tensor.view(*self.input_tensor.shape[-2:])).cpu()
        rec_plt = recon.view(*self._shape_output).cpu()
        monitor.update_plot(epoch, diff, rec_plt, plot_x, torch.tensor(plot_loss))

    def recon(self):
        return self._run(monitor_target=self._monitor_type)
