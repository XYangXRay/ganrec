import os
import json
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler, autocast
from ganrectorch.models import Generator, Discriminator
from ganrectorch.propagators import RadonTransform, PhaseFresnel, PhaseFraunhofer
from ganrectorch.loss import generator_loss, discriminator_loss
from ganrectorch.utils import RECONmonitor, to_device, tensor_to_np, ffactor


def torch_configures():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64


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
                 shape_output=None, config_key="GANtomo", monitor_type="tomo",
                 **kwargs):
        base_args = config[config_key].copy()
        base_args.update(kwargs)
        super().__init__()
        torch_configures()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Default to fp32 (like the TF path); fp16 autocast destabilizes the
        # SSIM/physics loss and triggers NaN/Inf rollbacks.
        self.amp_enabled = bool(base_args.get("amp", False))
        self.use_compile = bool(base_args.get("use_compile", False))
        self.scaler = GradScaler(self.device.type, enabled=self.amp_enabled)
        self.forward_fn = forward_fn
        self.output_num = output_num
        self.output_key = output_key
        self.monitor_type = monitor_type

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
                                   self.dropout, self.output_num).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        # Materialize LazyLinear layers before building optimizers / compiling
        with torch.no_grad():
            self.discriminator(torch.zeros(1, 1, h, w, device=self.device))
        fused = self.device.type == "cuda"
        self.generator_optimizer = optim.AdamW(
            self.generator.parameters(), lr=self.g_learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.99), fused=fused,
        )
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(), lr=self.d_learning_rate,
            weight_decay=1e-4, betas=(0.9, 0.99), fused=fused,
        )
        if self.use_compile:
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)

    def recon_step(self, input_tensor, train_disc=True):
        dev = self.device.type

        # ---- Generator forward (mixed precision for the network only) ----
        with autocast(dev, enabled=self.amp_enabled):
            gen_output = self.generator(input_tensor)
        # Physics forward model runs in fp32 for numerical stability
        fwd = self.forward_fn(gen_output.float(), input_tensor)
        predicted = fwd["predicted"]

        # ---- Discriminator update (detached fake -> no retained graph) ----
        if train_disc:
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            with autocast(dev, enabled=self.amp_enabled):
                real_output = self.discriminator(input_tensor)
                fake_output = self.discriminator(predicted.detach())
                d_loss = discriminator_loss(real_output, fake_output)
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.discriminator_optimizer)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.scaler.step(self.discriminator_optimizer)
        else:
            with torch.no_grad(), autocast(dev, enabled=self.amp_enabled):
                real_output = self.discriminator(input_tensor)
                fake_output = self.discriminator(predicted.detach())
                d_loss = discriminator_loss(real_output, fake_output)

        # ---- Generator update (freeze D grads to save backward compute) ----
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.discriminator.requires_grad_(False)
        recon = fwd.get("recon", predicted)
        with autocast(dev, enabled=self.amp_enabled):
            fake_output = self.discriminator(predicted)
            g_loss = generator_loss(fake_output, input_tensor, predicted,
                                    recon, self.l1_ratio)
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.generator_optimizer)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.scaler.step(self.generator_optimizer)
        self.discriminator.requires_grad_(True)
        self.scaler.update()

        fwd["g_loss"] = g_loss.detach()
        fwd["d_loss"] = d_loss.detach()
        return fwd

    def recon(self):
        """Run the full reconstruction loop with stability safeguards.

        Returns
        -------
        ndarray
            Reconstruction reshaped to ``shape_output``.
        """
        self.make_model()
        self.input_tensor = self.input_tensor.to(self.device, non_blocking=True)

        if self.init_wpath:
            self.generator.load_state_dict(
                torch.load(self.init_wpath + "generator.pth", weights_only=True))
            self.discriminator.load_state_dict(
                torch.load(self.init_wpath + "discriminator.pth", weights_only=True))
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
        recon_monitor = None
        plot_x, plot_loss = [], []
        if self.recon_monitor and self.monitor_type in ("tomo", "phase"):
            recon_monitor = RECONmonitor(self.monitor_type, self.input_tensor)
        pbar = tqdm(total=self.iter_num, desc="Reconstruction", leave=True)

        step_result = {}
        for step in range(self.iter_num):
            # D freeze management (skip D update during recovery windows)
            train_disc = freeze_disc_steps == 0
            if freeze_disc_steps > 0:
                freeze_disc_steps -= 1

            # ---- step ----
            step_result = self.recon_step(self.input_tensor, train_disc=train_disc)
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

            # ---------- Live monitor ----------
            if recon_monitor is not None:
                plot_x.append(step)
                plot_loss.append(g_loss_val)
                if (step + 1) % 100 == 0:
                    pred_2d = step_result["predicted"].detach().reshape(self.input_shape)
                    inp_2d = self.input_tensor.detach().reshape(self.input_shape)
                    img_diff = torch.abs(pred_2d - inp_2d).cpu()
                    rec_2d = step_result[self.output_key].detach().reshape(self.shape_output)
                    rec_plt = self._display_recon(rec_2d).cpu()
                    recon_monitor.update_plot(step, img_diff, rec_plt,
                                              plot_x, torch.tensor(plot_loss))

        pbar.close()
        if recon_monitor is not None:
            recon_monitor.close_plot()

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

        out = tensor_to_np(self._display_recon(final).cpu()).reshape(self.shape_output)
        return out

    def _display_recon(self, t):
        """Prepare a reconstruction tensor for display/output.

        For tomography the valid reconstruction lives inside the inscribed
        circle; the unconstrained corners can blow up and dominate the color
        scale.  Masking to the disk (then min-max) keeps the result viewable.
        """
        t = t.detach().float().reshape(self.shape_output)
        if self.monitor_type == "tomo" and t.shape[-1] == t.shape[-2]:
            t = t * _disk_mask_t(t)
            # Clip rare hot pixels so a single outlier can't compress the scale.
            hi = torch.quantile(t.flatten(), 0.999)
            t = torch.clamp(t, max=hi)
        return _minmax_t(t)


# ---------------------------------------------------------------------------
# Torch normalization helpers used by the specialized forward models
# ---------------------------------------------------------------------------


def _nor_tomo_t(data, eps=1e-8):
    """Standardize then min-max to [0, 1] (mirrors TF ``tfnor_tomo``)."""
    data = (data - torch.mean(data)) / (torch.std(data) + eps)
    data = data - torch.min(data)
    return data / (torch.max(data) + eps)


def _normalize_to_target_range_t(generated, target, eps=1e-8):
    """Linearly rescale ``generated`` so its [min, max] matches ``target`` (mirrors TF ``normalize_to_target_range``)."""
    gen_min = torch.min(generated)
    gen_max = torch.max(generated)
    tar_min = torch.min(target)
    tar_max = torch.max(target)
    gen_range = gen_max - gen_min
    tar_range = tar_max - tar_min
    if gen_range < eps:
        return torch.full_like(generated, (tar_min + tar_max) * 0.5)
    return (generated - gen_min) / gen_range * tar_range + tar_min

def _minmax_t(data):
    """Min-max normalize a tensor to the range [0, 1]."""
    dmin = torch.min(data)
    dmax = torch.max(data)
    return (data - dmin) / (dmax - dmin + 1e-8)


def _disk_mask_t(t):
    """Boolean mask of the inscribed circle for a square 2-D tensor."""
    h, w = t.shape[-2], t.shape[-1]
    yy = torch.arange(h, device=t.device).view(-1, 1).float()
    xx = torch.arange(w, device=t.device).view(1, -1).float()
    r = torch.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
    return (r <= (min(h, w) / 2.0 - 1)).float()


# ---------------------------------------------------------------------------
# Specialized reconstruction classes (thin subclasses of GANrec)
# ---------------------------------------------------------------------------
class GANtomo(GANrec):
    """GAN-based parallel-beam tomographic reconstruction.

    Parameters
    ----------
    prj_input : ndarray (nang, px)
        Measured sinogram.
    angle : ndarray (nang,)
        Projection angles in radians.
    **kwargs
        Override any value from the ``GANtomo`` config section.
    """

    def __init__(self, prj_input, angle, **kwargs):
        self._angle = torch.from_numpy(angle).float()

        def tomo_forward(gen_output, input_tensor):
            recon = _nor_tomo_t(gen_output)
            ang = self._angle.to(gen_output.device)
            prj_rec = RadonTransform(recon, ang)(recon, ang)
            prj_rec = _normalize_to_target_range_t(prj_rec, input_tensor)
            return {"recon": recon, "predicted": prj_rec}

        super().__init__(prj_input, tomo_forward, output_num=1,
                         output_key="recon", config_key="GANtomo",
                         monitor_type="tomo", **kwargs)


class GANphase(GANrec):
    """GAN-based near-field (Fresnel) phase retrieval.

    Parameters
    ----------
    i_input : ndarray (px, px)
        Measured (flat-field corrected) intensity image.
    energy : float
        X-ray energy in keV.
    z : float
        Sample-to-detector distance (propagation distance).
    pv : float
        Detector pixel size.
    **kwargs
        Override any value from the ``GANphase`` config section.
    """

    def __init__(self, i_input, energy, z, pv, **kwargs):
        px = i_input.shape[-1]
        self._px = px
        phase_cfg = config["GANphase"]
        self._phase_only = kwargs.pop("phase_only", phase_cfg["phase_only"])
        self._abs_ratio = kwargs.pop("abs_ratio", phase_cfg["abs_ratio"])
        ff_np = ffactor(px * 2, px * 2, energy, z, pv)
        self._ff = torch.from_numpy(ff_np).to(torch.complex64)

        def phase_forward(gen_output, input_tensor):
            phase = tfnor_phase(gen_output[:, 0:1, :, :])
            if self._phase_only:
                absorption = torch.zeros_like(phase)
            else:
                absorption = tfnor_phase(gen_output[:, 1:2, :, :]) * self._abs_ratio
            ff = self._ff.to(gen_output.device)
            i_rec = PhaseFresnel(phase[0, 0], absorption[0, 0], ff, self._px).compute()
            i_rec = _minmax_t(i_rec)
            return {"phase": phase, "absorption": absorption, "predicted": i_rec}

        super().__init__(i_input, phase_forward, output_num=2,
                         output_key="phase", config_key="GANphase",
                         monitor_type="phase", **kwargs)


class GANdiffraction(GANrec):
    """GAN-based far-field (Fraunhofer) diffraction phase retrieval.

    Parameters
    ----------
    i_input : ndarray (px, px)
        Measured diffraction intensity pattern.
    **kwargs
        Override any value from the ``GANdiffraction`` config section.
    """

    def __init__(self, i_input, **kwargs):
        phase_cfg = config["GANdiffraction"]
        self._phase_only = kwargs.pop("phase_only", phase_cfg["phase_only"])
        self._abs_ratio = kwargs.pop("abs_ratio", phase_cfg["abs_ratio"])

        def diff_forward(gen_output, input_tensor):
            phase = tfnor_phase(gen_output[:, 0:1, :, :])
            if self._phase_only:
                absorption = torch.zeros_like(phase)
            else:
                absorption = tfnor_phase(gen_output[:, 1:2, :, :]) * self._abs_ratio
            i_rec = PhaseFraunhofer(phase[0, 0], absorption[0, 0]).compute()
            return {"phase": phase, "absorption": absorption, "predicted": i_rec}

        super().__init__(i_input, diff_forward, output_num=2,
                         output_key="phase", config_key="GANdiffraction",
                         monitor_type="phase", **kwargs)
