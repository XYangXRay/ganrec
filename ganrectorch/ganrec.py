import os
import json
import copy
from tqdm import tqdm
import numpy as np
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


class DeviceEMA:
    """Exponential moving average of model parameters (on device)."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def swap_in(self):
        """Load EMA weights into the model (backup current weights)."""
        self.backup = {name: p.data.clone() for name, p in self.model.named_parameters()}
        for name, p in self.model.named_parameters():
            p.data.copy_(self.shadow[name])

    def swap_out(self):
        """Restore original (non-EMA) weights."""
        for name, p in self.model.named_parameters():
            p.data.copy_(self.backup[name])
        self.backup = {}


class DeviceSnapshot:
    """On-device snapshot/restore of model state dicts."""
    def __init__(self, models):
        self.models = models
        self.copies = [copy.deepcopy(m.state_dict()) for m in models]

    @torch.no_grad()
    def snapshot(self):
        for i, m in enumerate(self.models):
            self.copies[i] = copy.deepcopy(m.state_dict())

    @torch.no_grad()
    def restore(self):
        for m, sd in zip(self.models, self.copies):
            m.load_state_dict(sd)


class ScalarEMA:
    """Exponential moving average of a scalar value."""
    def __init__(self, decay=0.95):
        self.decay = decay
        self.value = None

    def update(self, x):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x
        return self.value


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
        self.generator_optimizer = optim.AdamW(
            self.generator.parameters(), lr=self.g_learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.99)
        )
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), lr=self.d_learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.99)
        )

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

    def recon_step(self, prj, ang):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        with autocast():
            recon = self.generator(prj)
            recon = self.nor_tomo(recon)
            prj_rec = self.radon(recon, ang)
            prj_rec = self.nor_tomo(prj_rec)
            real_output = self.discriminator(prj)
            fake_output = self.discriminator(prj_rec)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)

        # Backward pass with gradient scaling
        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.scale(d_loss).backward()

        # Unscale before clipping
        self.scaler.unscale_(self.generator_optimizer)
        self.scaler.unscale_(self.discriminator_optimizer)

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

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

        # ---------- Stability tunables ----------
        ema_decay       = float(getattr(self, "ema_decay", 0.99))
        snapshot_every  = int(getattr(self, "snapshot_every", 50))
        log_every       = int(getattr(self, "log_every", 10))
        spike_factor    = float(getattr(self, "spike_factor", 1.5))
        warmup_steps    = int(getattr(self, "warmup_steps", max(5, self.iter_num // 20)))
        lr_backoff      = float(getattr(self, "lr_backoff", 0.5))
        lr_floor        = float(getattr(self, "lr_floor", 1e-6))
        freeze_disc_max = int(getattr(self, "freeze_disc_max", 10))

        # ---------- EMA & Snapshots ----------
        ema = DeviceEMA(self.generator, decay=ema_decay)
        snap = DeviceSnapshot([self.generator, self.discriminator])
        snap.snapshot()

        g_ema = ScalarEMA(0.95)
        d_ema = ScalarEMA(0.95)

        g_lr0 = self.g_learning_rate
        d_lr0 = self.d_learning_rate
        freeze_disc_steps = 0

        # ---------- Monitor ----------
        gen_loss = torch.zeros(self.iter_num)
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo", self.prj_input.cpu())
        pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)

        recon = None
        for epoch in range(self.iter_num):
            # Apply temporary D freeze by setting LR to 0
            if freeze_disc_steps > 0:
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = 0.0
                freeze_disc_steps -= 1
            else:
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = d_lr0

            # ---------- Reconstruction step ----------
            step_result = self.recon_step(self.prj_input, self.angle)
            g_loss_val = step_result["g_loss"].item()
            d_loss_val = step_result["d_loss"].item()
            recon = step_result["recon"]

            # ---------- NaN / Inf guard ----------
            if not (np.isfinite(g_loss_val) and np.isfinite(d_loss_val)):
                snap.restore()
                # Backoff learning rates
                for pg in self.generator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                freeze_disc_steps = freeze_disc_max
                pbar.set_postfix_str("NaN/Inf→rollback")
                pbar.update(1)
                continue

            # ---------- Flat output guard ----------
            if epoch % log_every == 0:
                recon_var = torch.var(recon).item()
                if recon_var < 1e-4:
                    snap.restore()
                    for pg in self.discriminator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    pbar.set_postfix_str("flat→rollback")
                    pbar.update(1)
                    continue

            # ---------- Update scalar EMAs ----------
            g_bar = g_ema.update(g_loss_val)
            d_bar = d_ema.update(d_loss_val)

            # ---------- Spike detection ----------
            if epoch > warmup_steps and epoch % log_every == 0:
                if (g_loss_val > spike_factor * max(1e-8, g_bar) or
                        d_loss_val > spike_factor * max(1e-8, d_bar)):
                    snap.restore()
                    for pg in self.generator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    for pg in self.discriminator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    freeze_disc_steps = freeze_disc_max
                    pbar.set_postfix_str("spike→rollback")
                    pbar.update(1)
                    continue

            # ---------- Good step: update EMA and snapshot ----------
            ema.update()
            if epoch % snapshot_every == 0:
                snap.snapshot()

            gen_loss[epoch] = g_loss_val

            # ---------- Logging / monitor ----------
            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=f"{g_loss_val:.4f}", D_loss=f"{d_loss_val:.4f}")
            pbar.update(1)

            if (epoch + 1) % 100 == 0:
                if self.recon_monitor:
                    prj_rec = step_result["prj_rec"].view(self.nang, self.px)
                    prj_diff = torch.abs(prj_rec - self.prj_input.view((self.nang, self.px))).cpu()
                    rec_plt = recon.view(self.px, self.px).cpu()
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss.cpu())

        pbar.close()

        if self.save_wpath is not None:
            torch.save(self.generator.state_dict(), self.save_wpath + "generator.pth")
            torch.save(self.discriminator.state_dict(), self.save_wpath + "discriminator.pth")
        if self.recon_monitor:
            recon_monitor.close_plot()

        # Final output: use EMA weights for the result
        ema.swap_in()
        try:
            with torch.no_grad():
                final_recon = self.generator(self.prj_input)
                final_recon = self.nor_tomo(final_recon)
        finally:
            ema.swap_out()

        return tensor_to_np(final_recon.cpu())


class GANrec:
    """
    General-purpose GAN-based reconstruction with full stability features.

    Unlike specialized classes (GANtomo, etc.), GANrec is not restricted
    to a specific forward model.  The user provides a ``forward_fn`` callable
    that defines the physics mapping from reconstruction to measurement space.

    Stability features:
    EMA of generator weights, on-device rollback snapshots, NaN/Inf guards,
    loss-spike detection, discriminator freezing, gradient clipping, and
    learning-rate backoff.

    Parameters
    ----------
    input_data : ndarray (2-D)
        Measured data (sinogram, intensity image, diffraction pattern, …).
    forward_fn : callable
        ``(gen_output, input_tensor) -> dict``
        Must return a dict with at least ``"predicted"`` (simulated measurement,
        same shape as *input_tensor*).  May include extra keys (``"recon"``,
        ``"phase"``, …) carried through for monitoring / output extraction.
    output_num : int, optional
        Number of generator output channels (default 1).
    output_key : str, optional
        Key in ``forward_fn`` result for the final output (default ``"recon"``).
    shape_output : tuple, optional
        Output reshape target. Defaults to ``(input_data.shape[-1],) * 2``.
    **kwargs
        Override config values (``iter_num``, ``l1_ratio``, ``g_learning_rate``,
        ``d_learning_rate``, ``conv_num``, …).

    Examples
    --------
    **Tomography**::

        from ganrectorch.propagators import RadonTransform

        radon = RadonTransform(torch.empty(1, 1, px, px), angle)
        radon = radon.to(device)

        def tomo_forward(gen_output, input_tensor):
            recon = nor_tomo(gen_output)
            prj_rec = radon(recon, angle)
            prj_rec = nor_tomo(prj_rec)
            return {"recon": recon, "predicted": prj_rec}

        gan = GANrec(sinogram, tomo_forward)
        result = gan.recon()

    **Phase retrieval (Fresnel)**::

        from ganrectorch.propagators import PhaseFresnel

        def phase_forward(gen_output, input_tensor):
            phase = normalize_phase(gen_output[:, 0:1, :, :])
            absorption = normalize_phase(gen_output[:, 1:2, :, :]) * abs_ratio
            i_rec = PhaseFresnel(phase[0,0], absorption[0,0], ff, px).compute()
            return {"phase": phase, "absorption": absorption, "predicted": i_rec}

        gan = GANrec(intensity, phase_forward, output_num=2, output_key="phase")
        result = gan.recon()
    """

    def __init__(self, input_data, forward_fn, output_num=1, output_key="recon",
                 shape_output=None, **kwargs):
        base_args = config["GANtomo"].copy()
        base_args.update(kwargs)
        super().__init__()
        torch_configures()
        self.scaler = GradScaler()
        self.forward_fn = forward_fn
        self.output_num = output_num
        self.output_key = output_key

        # Store and prepare input
        self.input_data = input_data
        self.input_shape = input_data.shape
        self.shape_output = shape_output or (self.input_shape[-1], self.input_shape[-1])
        self.input_tensor = (
            torch.from_numpy(input_data).float().unsqueeze(0).unsqueeze(0)
        )  # [1, 1, H, W]

        self.iter_num = base_args["iter_num"]
        self.conv_num = base_args["conv_num"]
        self.conv_size = base_args["conv_size"]
        self.dropout = base_args["dropout"]
        self.l1_ratio = base_args["l1_ratio"]
        self.g_learning_rate = base_args["g_learning_rate"]
        self.d_learning_rate = base_args["d_learning_rate"]
        self.save_wpath = base_args["save_wpath"]
        self.init_wpath = base_args["init_wpath"]
        self.init_model = base_args["init_model"]
        self.recon_monitor = base_args["recon_monitor"]
        self.generator = None
        self.discriminator = None

    def make_model(self):
        h, w = self.input_shape
        self.generator = Generator(h, w, self.conv_num, self.conv_size,
                                   self.dropout, self.output_num)
        self.discriminator = Discriminator()
        self.generator_optimizer = optim.AdamW(
            self.generator.parameters(), lr=self.g_learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.99),
        )
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), lr=self.d_learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.99),
        )

    def recon_step(self, input_tensor):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        with autocast():
            gen_output = self.generator(input_tensor)
            fwd = self.forward_fn(gen_output, input_tensor)
            predicted = fwd["predicted"]
            real_output = self.discriminator(input_tensor)
            fake_output = self.discriminator(predicted)
            g_loss = generator_loss(fake_output, input_tensor, predicted,
                                    self.l1_ratio)
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

        fwd["g_loss"] = g_loss
        fwd["d_loss"] = d_loss
        return fwd

    def recon(self):
        """Run the full reconstruction loop with stability safeguards.

        Returns
        -------
        ndarray
            Reconstruction reshaped to ``shape_output``.
        """
        self.make_model()
        self.input_tensor, self.generator, self.discriminator = to_device(
            [self.input_tensor, self.generator, self.discriminator]
        )

        if self.init_wpath:
            self.generator.load_state_dict(
                torch.load(self.init_wpath + "generator.pth"))
            self.discriminator.load_state_dict(
                torch.load(self.init_wpath + "discriminator.pth"))
            print("Models are initialized")

        # ---------- Stability tunables ----------
        ema_decay       = float(getattr(self, "ema_decay", 0.99))
        snapshot_every  = int(getattr(self, "snapshot_every", 50))
        log_every       = int(getattr(self, "log_every", 10))
        spike_factor    = float(getattr(self, "spike_factor", 1.5))
        warmup_steps    = int(getattr(self, "warmup_steps",
                                      max(5, self.iter_num // 20)))
        lr_backoff      = float(getattr(self, "lr_backoff", 0.5))
        lr_floor        = float(getattr(self, "lr_floor", 1e-6))
        freeze_disc_max = int(getattr(self, "freeze_disc_max", 10))

        # ---------- EMA & Snapshots ----------
        ema = DeviceEMA(self.generator, decay=ema_decay)
        snap = DeviceSnapshot([self.generator, self.discriminator])
        snap.snapshot()

        g_ema_s = ScalarEMA(0.95)
        d_ema_s = ScalarEMA(0.95)
        d_lr0 = self.d_learning_rate
        freeze_disc_steps = 0

        # ---------- Monitor ----------
        pbar = tqdm(total=self.iter_num, desc="Reconstruction", leave=True)

        step_result = {}
        for step in range(self.iter_num):
            # D freeze management
            if freeze_disc_steps > 0:
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = 0.0
                freeze_disc_steps -= 1
            else:
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = d_lr0

            # ---- step ----
            step_result = self.recon_step(self.input_tensor)
            g_loss_val = step_result["g_loss"].item()
            d_loss_val = step_result["d_loss"].item()

            # NaN / Inf guard
            if not (np.isfinite(g_loss_val) and np.isfinite(d_loss_val)):
                snap.restore()
                for pg in self.generator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                for pg in self.discriminator_optimizer.param_groups:
                    pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                freeze_disc_steps = freeze_disc_max
                pbar.set_postfix_str("NaN/Inf→rollback")
                pbar.update(1)
                continue

            # Flat-output guard
            if self.output_key in step_result and step % log_every == 0:
                if torch.var(step_result[self.output_key]).item() < 1e-4:
                    snap.restore()
                    for pg in self.discriminator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    pbar.set_postfix_str("flat→rollback")
                    pbar.update(1)
                    continue

            # Update scalar EMAs
            g_bar = g_ema_s.update(g_loss_val)
            d_bar = d_ema_s.update(d_loss_val)

            # Spike detection
            if step > warmup_steps and step % log_every == 0:
                if (g_loss_val > spike_factor * max(1e-8, g_bar) or
                        d_loss_val > spike_factor * max(1e-8, d_bar)):
                    snap.restore()
                    for pg in self.generator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    for pg in self.discriminator_optimizer.param_groups:
                        pg["lr"] = max(lr_floor, pg["lr"] * lr_backoff)
                    freeze_disc_steps = freeze_disc_max
                    pbar.set_postfix_str("spike→rollback")
                    pbar.update(1)
                    continue

            # Good step
            ema.update()
            if step % snapshot_every == 0:
                snap.snapshot()

            if step % log_every == 0:
                pbar.set_postfix(G_loss=f"{g_loss_val:.4f}",
                                 D_loss=f"{d_loss_val:.4f}")
            pbar.update(1)

        pbar.close()

        if self.save_wpath is not None:
            torch.save(self.generator.state_dict(),
                       self.save_wpath + "generator.pth")
            torch.save(self.discriminator.state_dict(),
                       self.save_wpath + "discriminator.pth")

        # ---------- Final output with EMA weights ----------
        ema.swap_in()
        try:
            with torch.no_grad():
                gen_output = self.generator(self.input_tensor)
                fwd = self.forward_fn(gen_output, self.input_tensor)
                if self.output_key in fwd:
                    final = fwd[self.output_key]
                else:
                    final = fwd["predicted"]
        finally:
            ema.swap_out()

        return tensor_to_np(final.cpu()).reshape(self.shape_output)
