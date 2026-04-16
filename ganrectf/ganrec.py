import os
import numpy as np
import json
from tqdm import tqdm
import tensorflow as tf
from ganrectf.propagators import TomoRadon, TensorRadon, PhaseFresnel, PhaseFraunhofer
from ganrectf.models import make_generator, make_discriminator
from ganrectf.utils import RECONmonitor, ffactor
from ganrectf.tfutils import (
    normalize_to_target_range, tfnor_tomo, tfnor_phase,
    flat_output, DeviceEMA, DeviceSnapshot, ScalarEMA,
)
from ganrectf.loss import generator_loss, discriminator_loss


def tf_configures():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _try_xla():
    """Return True if XLA JIT compilation is available."""
    try:
        tf.function(lambda x: x + 1, jit_compile=True)(tf.constant(0.0))
        return True
    except Exception:
        return False


_XLA_OK = None  # lazy singleton


def xla_available():
    global _XLA_OK
    if _XLA_OK is None:
        _XLA_OK = _try_xla()
    return _XLA_OK


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


class GANrec:
    """
    General-purpose GAN-based reconstruction with full stability features.

    Unlike specialized classes (GANtomo, GANphase, etc.), GANrec is not restricted
    to a specific forward model.  The user provides a ``forward_fn`` callable that
    defines the physics mapping from reconstruction space to measurement space.

    Stability features:
    EMA of generator weights, on-device rollback snapshots, NaN/Inf guards,
    loss-spike detection, discriminator freezing, learning-rate backoff,
    cosine-decay learning-rate schedule, best-loss auto-restore.

    Parameters
    ----------
    input_data : ndarray (2-D)
        Measured data (sinogram, intensity image, diffraction pattern, ...).
    forward_fn : callable
        ``(gen_output, input_tensor) -> dict``
        Called inside ``@tf.function``; **must use TF operations only**.
        The returned dict must contain at least:

        * ``"predicted"`` -- simulated measurement, same shape as *input_tensor*.

        It may include additional keys (``"recon"``, ``"phase"``, ...) that are
        carried through for monitoring and final-output extraction.
    output_num : int, optional
        Number of generator output channels (default 1).
    output_key : str, optional
        Key in the ``forward_fn`` result dict to use as the final
        reconstruction output (default ``"recon"``).
    shape_output : tuple, optional
        Shape to reshape the final output into.
        Defaults to ``(input_data.shape[-1], input_data.shape[-1])``.
    monitor_type : str, optional
        Passed to ``RECONmonitor``: ``"tomo"``, ``"phase"``, or ``"tensor"``
        (default ``"tomo"``).
    **kwargs
        Override any default config value (``iter_num``, ``l1_ratio``,
        ``g_learning_rate``, ``d_learning_rate``, ``conv_num``, ...).

    Examples
    --------
    **Tomography**::

        angle_tf = tf.cast(angles, tf.float32)

        def tomo_forward(gen_output, input_tensor):
            recon = tfnor_tomo(gen_output)
            prj_rec = TomoRadon(recon, angle_tf).compute()
            prj_rec = normalize_to_target_range(prj_rec, input_tensor)
            return {"recon": recon, "predicted": prj_rec}

        gan = GANrec(sinogram, tomo_forward)
        result = gan.recon()

    **Phase retrieval (Fresnel)**::

        ff = ffactor(img_w * 2, energy, z, pv)

        def phase_forward(gen_output, input_tensor):
            phase = tfnor_phase(gen_output[:, :, :, 0])
            phase = tf.reshape(phase, [img_h, img_w])
            absorption = (1 - tfnor_phase(gen_output[:, :, :, 1])) * abs_ratio
            absorption = tf.reshape(absorption, [img_h, img_w])
            i_rec = PhaseFresnel(phase, absorption, ff, img_w).compute()
            return {"phase": phase, "absorption": absorption, "predicted": i_rec}

        gan = GANrec(intensity, phase_forward, output_num=2,
                     output_key="phase", monitor_type="phase")
        result = gan.recon()
    """

    def __init__(self, input_data, forward_fn, output_num=1, output_key="recon",
                 shape_output=None, monitor_type="tomo", **kwargs):
        base_args = config["GANrec"].copy()
        base_args.update(kwargs)
        super().__init__()
        tf_configures()
        self.input_data = input_data
        self.forward_fn = forward_fn
        self.output_num = output_num
        self.output_key = output_key
        self.shape_input = input_data.shape
        self.shape_output = shape_output or (self.shape_input[-1], self.shape_input[-1])
        self.monitor_type = monitor_type

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
        self.ema_decay = base_args.get("ema_decay", 0.99)
        self.snapshot_every = base_args.get("snapshot_every", 50)
        self.disk_save_every = base_args.get("disk_save_every", 100)
        self.log_every = base_args.get("log_every", 10)
        self.spike_factor = base_args.get("spike_factor", 1.5)
        self.warmup_ratio = base_args.get("warmup_ratio", 0.05)
        self.lr_backoff = base_args.get("lr_backoff", 0.5)
        self.lr_floor = base_args.get("lr_floor", 1e-6)
        self.freeze_disc_max = base_args.get("freeze_disc_max", 10)
        self.rescale_range = base_args.get("rescale_range", None)
        self.auto_scale = base_args.get("auto_scale", True)
        self._data_min = float(np.min(input_data))
        self._data_max = float(np.max(input_data))
        # Pre-scale input to [0,1] for well-conditioned training
        if self.auto_scale:
            drange = self._data_max - self._data_min
            if drange > 1e-12:
                self._scale = drange
                self._offset = self._data_min
                self.input_data = (input_data - self._offset) / self._scale
            else:
                self._scale = 1.0
                self._offset = 0.0
        else:
            self._scale = 1.0
            self._offset = 0.0
        self._make_model()

    def _make_model(self):
        self.generator = make_generator(
            self.shape_input, self.conv_num, self.conv_size,
            self.dropout, self.output_num
        )
        self.discriminator = make_discriminator(self.shape_input)

        # Cosine-decay LR schedules for smoother convergence
        g_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.g_learning_rate,
            decay_steps=int(self.iter_num),
            alpha=self.lr_floor / max(self.g_learning_rate, 1e-12),
        )
        d_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.d_learning_rate,
            decay_steps=int(self.iter_num),
            alpha=self.lr_floor / max(self.d_learning_rate, 1e-12),
        )

        self.generator_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=g_lr_schedule,
            weight_decay=1e-4, beta_2=0.99, clipnorm=1.0,
        )
        self.discriminator_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=d_lr_schedule,
            weight_decay=1e-4, beta_2=0.99, clipnorm=1.0,
        )
        self.generator.compile(jit_compile=xla_available())
        self.discriminator.compile(jit_compile=xla_available())
        self.g_optimizer = self.generator_optimizer
        self.d_optimizer = self.discriminator_optimizer
        # On-device flag: when True, discriminator gradients are not applied
        self._freeze_disc = tf.Variable(False, trainable=False, dtype=tf.bool)

    def _rescale_output(self, arr):
        """Rescale reconstruction output to a target value range.

        Parameters
        ----------
        arr : ndarray
            Raw (normalized) reconstruction output.

        Returns
        -------
        ndarray
            Rescaled array.  Unchanged when ``rescale_range`` is *None*.

        Notes
        -----
        ``rescale_range`` accepts:

        * ``None``              – no rescaling (backward-compatible default).
        * ``"input"``           – linearly map to ``[input_min, input_max]``.
        * ``[vmin, vmax]``      – linearly map to a user-specified range.
        """
        mode = self.rescale_range
        if mode is None:
            return arr
        if isinstance(mode, str) and mode == "input":
            target_min, target_max = self._data_min, self._data_max
        elif isinstance(mode, (list, tuple)) and len(mode) == 2:
            target_min, target_max = float(mode[0]), float(mode[1])
        else:
            return arr
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max - arr_min < 1e-12:
            return np.full_like(arr, (target_min + target_max) / 2.0)
        return ((arr - arr_min) / (arr_max - arr_min)
                * (target_max - target_min) + target_min)

    @tf.function(reduce_retracing=True)
    def recon_step(self, input_tensor):
        # Single persistent tape avoids building two independent tape contexts
        with tf.GradientTape(persistent=True) as tape:
            gen_output = self.generator(input_tensor, training=True)
            fwd = self.forward_fn(gen_output, input_tensor)
            predicted = fwd["predicted"]
            real_output = self.discriminator(input_tensor, training=True)
            fake_output = self.discriminator(predicted, training=True)
            recon = fwd.get("recon", gen_output)
            g_loss = generator_loss(fake_output, input_tensor, predicted,
                                    recon, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)

        gen_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        disc_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        del tape  # free persistent tape memory immediately

        # Clip gradients by global norm for extra stability
        gen_grads, g_gnorm = tf.clip_by_global_norm(gen_grads, 5.0)
        disc_grads, d_gnorm = tf.clip_by_global_norm(disc_grads, 5.0)

        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        # Skip discriminator update when frozen (spike/NaN recovery)
        def _apply_disc():
            self.discriminator_optimizer.apply_gradients(
                zip(disc_grads, self.discriminator.trainable_variables)
            )
            return tf.constant(0)

        tf.cond(
            self._freeze_disc,
            lambda: tf.constant(0),
            _apply_disc,
        )

        fwd["g_loss"] = g_loss
        fwd["d_loss"] = d_loss
        fwd["g_gnorm"] = g_gnorm
        fwd["d_gnorm"] = d_gnorm
        return fwd

    @tf.function(reduce_retracing=True)
    def _check_health(self, g_loss, d_loss):
        """On-device NaN/Inf check — avoids per-step .numpy() host sync."""
        finite = tf.math.is_finite(g_loss) & tf.math.is_finite(d_loss)
        return finite

    def recon(self, input_data=None):
        """Run the full reconstruction loop with stability safeguards.

        Parameters
        ----------
        input_data : ndarray, optional
            Override the input data provided at construction time.

        Returns
        -------
        ndarray
            Reconstruction reshaped to ``shape_output``.
        """
        # ---------- Optional initial load ----------
        if getattr(self, "init_wpath", None):
            try:
                self.generator.load_weights(
                    os.path.join(self.init_wpath, "generator.keras"))
                self.discriminator.load_weights(
                    os.path.join(self.init_wpath, "discriminator.keras"))
                print("Models are initialized")
            except Exception as e:
                print(f"[init] load failed: {e}")

        # ---------- Inputs ----------
        if input_data is not None:
            self.input_data = input_data
        # tf.constant avoids re-tracing: the tensor is baked as a graph constant
        input_tensor = tf.constant(
            self.input_data.astype(np.float32)[None, ..., None]
        )

        # ---------- Tunables ----------
        ema_decay       = float(self.ema_decay)
        snapshot_every  = int(self.snapshot_every)
        disk_save_every = int(self.disk_save_every)
        log_every       = int(self.log_every)
        spike_factor    = float(self.spike_factor)
        warmup_steps    = max(5, int(self.iter_num * self.warmup_ratio))
        lr_backoff      = float(self.lr_backoff)
        lr_floor        = float(self.lr_floor)
        freeze_disc_max = int(self.freeze_disc_max)
        weights_dir     = self.save_wpath
        total_steps     = int(self.iter_num)

        # ---------- EMA & Snapshots ----------
        ema = DeviceEMA(self.generator, decay=ema_decay)
        snap = DeviceSnapshot([
            self.generator.trainable_variables,
            self.discriminator.trainable_variables,
        ])
        snap.snapshot()

        # Best-loss tracking: auto-restore to the best-seen weights
        best_snap = DeviceSnapshot([
            self.generator.trainable_variables,
            self.discriminator.trainable_variables,
        ])
        best_g_loss = float("inf")
        best_snap.snapshot()

        g_ema = ScalarEMA(0.95)
        d_ema = ScalarEMA(0.95)

        # ---------- Monitor (TensorBoard) ----------
        recon_monitor = None
        if self.recon_monitor:
            recon_monitor = RECONmonitor(self.monitor_type, self.input_data)
            recon_monitor.total_steps = total_steps

        # Use tqdm only when monitor is not handling inline notebook display
        use_tqdm = not (recon_monitor is not None and recon_monitor._in_notebook)
        if use_tqdm:
            pbar = tqdm(total=total_steps, desc="Reconstruction", leave=True)
        else:
            pbar = None

        freeze_disc_steps = 0
        step_result = {}

        for step in range(total_steps):
            # D freeze management
            if freeze_disc_steps > 0:
                self._freeze_disc.assign(True)
            else:
                self._freeze_disc.assign(False)

            # ---- forward / backward / update ----
            step_result = self.recon_step(input_tensor)

            g_loss_t = step_result["g_loss"]
            d_loss_t = step_result["d_loss"]

            # On-device finiteness check (only sync to host on failure)
            finite = self._check_health(g_loss_t, d_loss_t)
            if not bool(finite):
                snap.restore()
                freeze_disc_steps = freeze_disc_max
                if pbar and step % log_every == 0:
                    pbar.set_postfix_str("NaN/Inf->rollback")
                if recon_monitor is not None:
                    recon_monitor.step += 1
                if pbar:
                    pbar.update(1)
                continue

            # Periodic host sync for diagnostics (every log_every steps)
            if step % log_every == 0:
                g_val = float(g_loss_t.numpy())
                d_val = float(d_loss_t.numpy())

                # Flat-output guard
                if self.output_key in step_result:
                    if flat_output(step_result[self.output_key]):
                        snap.restore()
                        if pbar:
                            pbar.set_postfix_str("flat->rollback")
                            pbar.update(1)
                        if recon_monitor is not None:
                            recon_monitor.step += 1
                        continue

                # Spike detection (after warmup)
                g_ema_v = g_ema.update(g_loss_t)
                d_ema_v = d_ema.update(d_loss_t)
                if step > warmup_steps:
                    g_bar = float(g_ema_v.numpy())
                    d_bar = float(d_ema_v.numpy())
                    if (g_val > spike_factor * max(1e-8, g_bar) or
                            d_val > spike_factor * max(1e-8, d_bar)):
                        snap.restore()
                        freeze_disc_steps = freeze_disc_max
                        if pbar:
                            pbar.set_postfix_str("spike->rollback")
                            pbar.update(1)
                        if recon_monitor is not None:
                            recon_monitor.step += 1
                        continue

                # Best-loss tracking
                if g_val < best_g_loss:
                    best_g_loss = g_val
                    best_snap.snapshot()

                if pbar:
                    pbar.set_postfix(
                        G=f"{g_val:.4f}",
                        D=f"{d_val:.4f}",
                        gn=f"{float(step_result['g_gnorm'].numpy()):.1f}",
                    )
            else:
                # Still update scalar EMAs on non-log steps (cheap, stays on device)
                g_ema.update(g_loss_t)
                d_ema.update(d_loss_t)

            # Good step -- EMA & snapshot
            ema.update()
            if step % snapshot_every == 0:
                snap.snapshot()

            # Disk save (rare)
            if weights_dir and (step % disk_save_every == 0 or
                                step == total_steps - 1):
                try:
                    os.makedirs(weights_dir, exist_ok=True)
                    self.generator.save_weights(
                        os.path.join(weights_dir, "generator.weights.h5"))
                    self.discriminator.save_weights(
                        os.path.join(weights_dir, "discriminator.weights.h5"))
                except Exception as e:
                    if pbar and step % log_every == 0:
                        pbar.set_postfix_str(f"save err: {e}")

            # Monitor (TensorBoard summaries)
            if recon_monitor is not None:
                recon_monitor.update_plot(step_result)

            if freeze_disc_steps > 0:
                freeze_disc_steps -= 1
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        if recon_monitor is not None:
            try:
                recon_monitor.close_plot(last_step_result=step_result)
            except Exception:
                pass

        # ---------- Final output: restore best weights, then apply EMA ----------
        best_snap.restore()
        ema.swap_in()
        try:
            gen_output = self.generator(input_tensor, training=False)
            fwd = self.forward_fn(gen_output, input_tensor)
            self._last_fwd = fwd
            if self.output_key in fwd:
                final = fwd[self.output_key].numpy()
            else:
                final = fwd["predicted"].numpy()
        finally:
            ema.swap_out()

        final = np.reshape(final.astype(np.float32), self.shape_output)
        # Undo auto-scaling to restore original data range
        if self.auto_scale:
            final = final * self._scale + self._offset
        return self._rescale_output(final)


class GANtomo(GANrec):
    """GAN-based tomographic reconstruction.

    Wraps ``GANrec`` with a Radon-transform forward model.

    Parameters
    ----------
    prj_input : ndarray (2-D)
        Sinogram (angles x pixels).
    angle : array-like
        Projection angles in radians.
    **kwargs
        Forwarded to ``GANrec``.
    """

    def __init__(self, prj_input, angle, **kwargs):
        angle_tf = tf.cast(angle, tf.float32)

        def forward_fn(gen_output, input_tensor):
            recon = tfnor_tomo(gen_output)
            prj_rec = TomoRadon(recon, angle_tf).compute()
            prj_rec = normalize_to_target_range(prj_rec, input_tensor)
            return {"recon": recon, "predicted": prj_rec}

        # Use normalization for training stability; rescale output to original range
        kwargs.setdefault("rescale_range", "input")
        super().__init__(prj_input, forward_fn, output_num=1,
                         output_key="recon", monitor_type="tomo", **kwargs)


class GANphase(GANrec):
    """GAN-based phase retrieval (Fresnel propagation).

    Wraps ``GANrec`` with a Fresnel forward model.

    Parameters
    ----------
    i_input : ndarray (2-D)
        Intensity image.
    energy : float
        Beam energy (keV).
    z : float
        Propagation distance.
    pv : float
        Pixel size.
    **kwargs
        Forwarded to ``GANrec``.
    """

    def __init__(self, i_input, energy, z, pv, **kwargs):
        phase_cfg = config.get("GANphase", {}).copy()
        phase_cfg.update(kwargs)

        abs_ratio = phase_cfg.get("abs_ratio", 1.0)
        phase_only = phase_cfg.get("phase_only", False)
        img_h, img_w = i_input.shape
        ff = ffactor(img_w * 2, energy, z, pv)

        # Pre-standardize input to match PhaseFresnel.compute() output.
        # PhaseFresnel applies tf.image.per_image_standardization (zero-mean,
        # unit-variance).  Without this, the discriminator sees different
        # distributions for real vs fake, and SSIM / mean_match / L1 all
        # compare mismatched value ranges.
        mean = float(np.mean(i_input))
        adj_std = max(float(np.std(i_input)),
                      1.0 / np.sqrt(float(i_input.size)))
        i_input_std = (i_input - mean) / adj_std

        def forward_fn(gen_output, input_tensor):
            phase = tfnor_phase(gen_output[:, :, :, 0])
            phase = tf.reshape(phase, [img_h, img_w])
            if phase_only:
                absorption = tf.zeros_like(phase)
            else:
                absorption = (1 - tfnor_phase(gen_output[:, :, :, 1])) * abs_ratio
                absorption = tf.reshape(absorption, [img_h, img_w])
            i_rec = PhaseFresnel(phase, absorption, ff, img_w).compute()
            # Pass normalized phase as "recon" so generator_loss computes
            # TV regularization on the phase (matching the original code),
            # not on the raw multi-channel generator output.
            recon = tf.reshape(phase, [1, img_h, img_w, 1])
            return {"phase": phase, "absorption": absorption,
                    "predicted": i_rec, "recon": recon}

        # Disable auto_scale: we already standardized the input above to
        # match PhaseFresnel's per_image_standardization output.
        phase_cfg.setdefault("auto_scale", False)

        super().__init__(i_input_std, forward_fn, output_num=2,
                         output_key="phase", monitor_type="phase",
                         **phase_cfg)


class GANdiffraction(GANrec):
    """GAN-based diffraction pattern reconstruction (Fraunhofer propagation).

    Wraps ``GANrec`` with a Fraunhofer forward model.

    Parameters
    ----------
    i_input : ndarray (2-D)
        Measured diffraction pattern.
    mask : ndarray or None
        Binary mask applied to the simulated pattern.
    **kwargs
        Forwarded to ``GANrec``.
    """

    def __init__(self, i_input, mask, **kwargs):
        diff_cfg = config.get("GANdiffraction", {}).copy()
        diff_cfg.update(kwargs)

        abs_ratio = diff_cfg.get("abs_ratio", 1.0)
        phase_only = diff_cfg.get("phase_only", False)
        px = i_input.shape[0]
        use_mask = mask is not None
        if use_mask:
            mask_tf = tf.constant(mask, dtype=tf.float32)

        def _tfnor_diff(img):
            return (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))

        def forward_fn(gen_output, input_tensor):
            phase = _tfnor_diff(gen_output[:, :, :, 0])
            phase = tf.reshape(phase, [px // 2, px // 2])
            phase = tf.pad(phase, [[64, 64], [64, 64]])
            absorption = (1 - _tfnor_diff(gen_output[:, :, :, 1])) * abs_ratio
            absorption = tf.reshape(absorption, [px // 2, px // 2])
            absorption = tf.pad(absorption, [[64, 64], [64, 64]])
            if phase_only:
                absorption = tf.zeros_like(phase)
            i_rec = PhaseFraunhofer(phase, absorption).compute()
            if use_mask:
                mask_r = tf.reshape(mask_tf, [1, mask_tf.shape[0], mask_tf.shape[1], 1])
                i_rec = tf.multiply(i_rec, mask_r)
            i_rec = _tfnor_diff(i_rec)
            return {"phase": phase, "absorption": absorption, "predicted": i_rec}

        super().__init__(i_input, forward_fn, output_num=2,
                         output_key="phase", monitor_type="phase",
                         **diff_cfg)

    def recon(self, input_data=None):
        """Run reconstruction and return ``(absorption, phase)``."""
        phase = super().recon(input_data)
        absorption = np.reshape(
            self._last_fwd["absorption"].numpy().astype(np.float32),
            self.shape_output
        )
        absorption = self._rescale_output(absorption)
        return absorption, phase


class GANtensor(GANrec):
    """GAN-based tensor tomographic reconstruction.

    Wraps ``GANrec`` with a tensor Radon-transform forward model.

    Parameters
    ----------
    prj_input : ndarray (2-D)
        Sinogram (angles x pixels).
    angle : array-like
        Projection angles in radians.
    psi : array-like
        Rotation angles for tensor tomography.
    **kwargs
        Forwarded to ``GANrec``.
    """

    def __init__(self, prj_input, angle, psi, **kwargs):
        tensor_cfg = config.get("GANtensor", {}).copy()
        tensor_cfg.update(kwargs)

        angle_tf = tf.cast(angle, tf.float32)
        px = prj_input.shape[1]

        def _tensor_norm(img):
            img = tf.image.per_image_standardization(img)
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            return img

        def forward_fn(gen_output, input_tensor):
            recon = _tensor_norm(gen_output)
            prj_rec = TensorRadon(recon, angle_tf, psi).compute()
            prj_rec = _tensor_norm(prj_rec)
            return {"recon": recon, "predicted": prj_rec}

        super().__init__(prj_input, forward_fn, output_num=3,
                         output_key="recon",
                         shape_output=(px, px, 3),
                         monitor_type="tensor",
                         **tensor_cfg)

    def recon(self, input_data=None):
        """Run reconstruction and return tensor components as ``(C, H, W)``."""
        result = super().recon(input_data)
        return np.transpose(result, axes=(2, 0, 1))


class GANtomo3D:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config["GANrec"].copy()
        tomo_args.update(**kwargs)
        super(GANtomo3D, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
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
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(
            self.prj_input.shape[0], self.prj_input.shape[1], self.conv_num, self.conv_size, self.dropout, 1
        )
        self.discriminator = make_discriminator(self.prj_input.shape[0], self.prj_input.shape[1])
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    def tfnor_tomo(data):
        # Calculate the mean and standard deviation of the data
        mean = tf.reduce_mean(data)
        std = tf.math.reduce_std(data)

        # Standardize the data (z-score normalization)
        standardized_data = (data - mean) / std

        # Find the minimum value in the standardized data
        standardized_min = tf.reduce_min(standardized_data)

        # Shift the data to start from 0
        shifted_data = standardized_data - standardized_min

        return shifted_data

    @tf.function
    def recon_step(self, prj, ang):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(prj)
            recon = self.tfnor_tomo(recon)
            prj_rec = self.tomo_radon(recon, ang)
            prj_rec = self.tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, recon, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    def recon_step_filter(self, prj, ang):
        with tf.GradientTape() as filter_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            prj_filter = self.filter(prj)
            prj_filter = self.tfnor_data(prj_filter)
            recon = self.generator(prj_filter)
            recon = self.tfnor_data(recon)
            prj_rec = TomoRadon(recon, ang).compute
            prj_rec = self.tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            filter_output = self.discriminator(prj_filter, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, recon, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_filter": prj_filter, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        # prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath + "generator.h5")
            print("generator is initilized")
            self.discriminator.load_weights(self.init_wpath + "discriminator.h5")
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            recon_monitor = RECONmonitor("tomo", self.prj_input)
        else:
            recon_monitor = None
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.recon_step(prj, ang)
            step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step_filter(prj, ang)
            recon[epoch, :, :, :] = step_result["recon"]
            gen_loss[epoch] = step_result["g_loss"]
            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.train_step_filter(prj, ang)
            ###########################################################################

            if recon_monitor is not None:
                recon_monitor.update_plot(step_result)

            if (epoch + 1) % 100 == 0:
                print(
                    "Iteration {}: G_loss is {} and D_loss is {}".format(
                        epoch + 1, gen_loss[epoch], step_result["d_loss"].numpy()
                    )
                )
            # plt.close()
        if recon_monitor is not None:
            recon_monitor.close_plot()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath + "generator.h5")
            self.discriminator.save(self.save_wpath + "discriminator.h5")
        return recon[epoch]
        # return avg_results(recon, gen_loss)
