import os
import numpy as np
import json
from tqdm import tqdm
import tensorflow as tf
from ganrectf.propagators import TomoRadon, TensorRadon, PhaseFresnel, PhaseFraunhofer
from ganrectf.models import make_generator, make_discriminator
from ganrectf.utils import RECONmonitor, ffactor


def tf_configures():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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

@tf.function
def normalize_to_target_range(
    generated: tf.Tensor,
    target: tf.Tensor,
    axis=None,
    epsilon: float = 1e-8
) -> tf.Tensor:
    """
    Linearly rescales `generated` so that its [min, max] equals
    the [min, max] of `target`, even if those values are negative.
    If `generated` is constant (min == max), it will be mapped
    to the midpoint of the `target` range.

    Args:
        generated: tf.Tensor of any shape.
        target:    tf.Tensor broadcastable to generated.
        axis:      Dimensions to reduce over when computing min/max
                   (e.g. [1,2,3] to keep batch dim). None = all elements.
        epsilon:   Small constant to avoid divide-by-zero.

    Returns:
        A tf.Tensor same shape as `generated`, but with
        min→min(target) and max→max(target).
    """
    # 1) compute per-sample minima and maxima
    gen_min = tf.reduce_min(generated, axis=axis, keepdims=True)
    gen_max = tf.reduce_max(generated, axis=axis, keepdims=True)
    tar_min = tf.reduce_min(target,    axis=axis, keepdims=True)
    tar_max = tf.reduce_max(target,    axis=axis, keepdims=True)

    # 2) compute ranges
    gen_range = gen_max - gen_min
    tar_range = tar_max - tar_min

    # 3) avoid zero division: if gen_range is too small, replace it with 1
    safe_gen_range = tf.where(
        gen_range < epsilon,
        tf.ones_like(gen_range),
        gen_range
    )

    # 4) normalize to [0,1], then scale to [tar_min, tar_max]
    normalized = (generated - gen_min) / safe_gen_range
    scaled = normalized * tar_range + tar_min

    # 5) for constant inputs, map to target midpoint
    target_mid = (tar_min + tar_max) * 0.5
    scaled = tf.where(gen_range < epsilon, target_mid, scaled)
    return scaled

@tf.function
def tfnor_tomo(img: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """
    Standardizes input (zero-mean, unit-variance) and shifts to non-negative range.
    Supports both 2D images and 3D volumes.

    Args:
        img: 4D Tensor [B, H, W, C] for images or
             5D Tensor [B, D, H, W, C] for volumes.
        eps: Small constant to avoid divide-by-zero.

    Returns:
        Tensor of same shape as img, with per-sample standardized values shifted
        so that the minimum is 0.
    """
    # Determine reduction axes based on tensor rank
    ndim = img.shape.ndims
    if ndim == 4:
        # [batch, height, width, channels]
        axes = [1, 2, 3]
    elif ndim == 5:
        # [batch, depth, height, width, channels]
        axes = [1, 2, 3, 4]
    else:
        # fallback: normalize over all elements
        axes = None

    # Compute per-sample mean and std
    mean = tf.reduce_mean(img, axis=axes, keepdims=True)
    std = tf.math.reduce_std(img, axis=axes, keepdims=True)
    # Standardize
    img_norm = (img - mean) / (std + eps)

    # Shift so minimum becomes 0
    min_val = tf.reduce_min(img_norm, axis=axes, keepdims=True)
    img_pos = (img_norm - min_val)/ (tf.reduce_max(img_norm - min_val) + eps)

    return img_pos

# @tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))
    )
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    )
    total_loss = real_loss + fake_loss
    return total_loss


def l1_loss(img1, img2):
    return tf.reduce_mean(tf.abs(img1 - img2))


def l2_loss(img1, img2):
    return tf.square(tf.reduce_mean(tf.abs(img1 - img2)))


# @tf.function
def generator_loss(fake_output, img_output, pred, l1_ratio):
    gen_loss = (
        tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output))
        )
        + l1_loss(img_output, pred) * l1_ratio
    )
    return gen_loss


def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img


def avg_results(recon, loss):
    sort_index = np.argsort(loss)
    recon_tmp = recon[sort_index[:10], :, :, :]
    return np.mean(recon_tmp, axis=0)

# compute every log_every steps
def flat_output(y, thr=1e-4):
    # y in [-1,1]; thr is very small variance
    v = tf.math.reduce_variance(y)
    return bool((v < thr).numpy())

def set_lr(opt, new_lr: float):
    try:
        opt.learning_rate.assign(new_lr)
    except Exception:
        opt.learning_rate = new_lr

class DeviceEMA:
    """EMA on device: shadow Variables match model.trainable_variables."""
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = tf.constant(decay, tf.float32)
        self.src    = list(model.trainable_variables)
        self.shadow = [tf.Variable(v, trainable=False) for v in self.src]
        self.backup = [tf.Variable(v, trainable=False) for v in self.src]

    @tf.function(reduce_retracing=True)   # keep this compiled, it's called every step
    def update(self):
        for s, v in zip(self.shadow, self.src):
            s.assign(self.decay * s + (1.0 - self.decay) * v)

    # NOTE: no @tf.function here — avoid retracing warnings
    def swap_in(self):
        # backup current to self.backup, then load shadow into model
        for b, v in zip(self.backup, self.src):
            b.assign(v)
        for v, s in zip(self.src, self.shadow):
            v.assign(s)

    # NOTE: no @tf.function here — avoid retracing warnings
    def swap_out(self):
        # restore from backup into model
        for v, b in zip(self.src, self.backup):
            v.assign(b)

class DeviceSnapshot:
    """On-device snapshot/restore of variables (no host copies, no tf.function)."""
    def __init__(self, var_lists):
        # var_lists is e.g. [gen.trainable_variables, disc.trainable_variables]
        self.targets = [list(vs) for vs in var_lists]
        self.copies  = [[tf.Variable(v, trainable=False) for v in vs]
                        for vs in self.targets]

    # No @tf.function here: avoid retracing
    def snapshot(self):
        for copies, targets in zip(self.copies, self.targets):
            for c, v in zip(copies, targets):
                c.assign(v)

    # No @tf.function here: avoid retracing
    def restore(self):
        for copies, targets in zip(self.copies, self.targets):
            for v, c in zip(targets, copies):
                v.assign(c)

class ScalarEMA:
    """Scalar EMA without host sync."""
    def __init__(self, decay=0.95):
        self.decay = tf.constant(decay, tf.float32)
        self.value = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.has   = tf.Variable(False, trainable=False, dtype=tf.bool)

    @tf.function
    def update(self, x):
        x = tf.cast(x, tf.float32)
        def _first(): 
            self.value.assign(x); self.has.assign(True)
            return self.value
        def _upd():
            self.value.assign(self.decay * self.value + (1.0 - self.decay) * x)
            return self.value
        return tf.cond(self.has, _upd, _first)


class GANtomo:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config["GANtomo"]
        tomo_args.update(**kwargs)
        super(GANtomo, self).__init__()
        tf_configures()
        self.prj_input = prj_input
        self.shape_input = self.prj_input.shape
        self.shape_output = (self.shape_input[1], self.shape_input[1])  
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
        self._make_model()

    def _make_model(self):
        self.generator = make_generator(self.shape_input, 
                                        self.conv_num, 
                                        self.conv_size, 
                                        self.dropout, 1)
        self.discriminator = make_discriminator(self.shape_input)
        self.generator_optimizer = tf.keras.optimizers.AdamW(self.g_learning_rate,
                                                             weight_decay=1e-4,
                                                             beta_2=0.99,
                                                             clipnorm=1.0)
        self.discriminator_optimizer = tf.keras.optimizers.AdamW(self.d_learning_rate,
                                                                weight_decay=1e-4,
                                                                beta_2=0.99,
                                                                clipnorm=1.0)
        self.generator.compile()
        self.discriminator.compile()

    # @tf.function
    # def tfnor_tomo(self, img):
    #     img = tf.image.per_image_standardization(img)
    #     img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    #     return img

    @tf.function
    def recon_step(self, prj, ang):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(prj)
            recon = tfnor_tomo(recon)
            tomo_radon_obj = TomoRadon(recon, ang)
            prj_rec = tomo_radon_obj.compute()
            prj_rec = normalize_to_target_range(prj_rec, prj)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}
    
    def recon(self, prj_input=None):
        """
        Fast & stable reconstruction loop:
        - On-device EMA & rollback snapshots (no host copies)
        - Rare disk saves (weights-only) instead of per-step saves
        - Minimal host sync (log every k steps)
        - Brief D 'freeze' by zeroing LR (no retracing)
        Expects: self.generator, self.discriminator, self.recon_step(prj, ang),
                self.iter_num, self.angle, self.prj_input, self.shape_output.
        Optional: self.g_optimizer, self.d_optimizer, self.recon_monitor, self.save_wpath, self.init_wpath.
        """
        # ---------- Optional initial load ----------
        if getattr(self, "init_wpath", None):
            try:
                self.generator.load_weights(os.path.join(self.init_wpath, "generator.keras"))
                self.discriminator.load_weights(os.path.join(self.init_wpath, "discriminator.keras"))
                print("Models are initialized")
            except Exception as e:
                print(f"[init] load failed: {e}")

        # ---------- Inputs ----------
        if prj_input is not None:
            self.prj_input = prj_input
        prj = tf.cast(self.prj_input, tf.float32)[None, ..., None]
        ang = tf.cast(self.angle, tf.float32)
        self.t = tf.random.normal((1, 500))

        # ---------- Tunables (you can override on self.*) ----------
        ema_decay        = float(getattr(self, "ema_decay",        0.99))
        snapshot_every   = int(getattr(self,  "snapshot_every",    50))     # on-device RAM snapshot
        disk_save_every  = int(getattr(self,  "disk_save_every",   100))   # rare weights-only to disk
        log_every        = int(getattr(self,  "log_every",         10))     # host sync for logs/plots
        monitor_every    = int(getattr(self,  "monitor_every",     100))     # plotting interval
        spike_factor     = float(getattr(self, "spike_factor",     1.5))
        warmup_steps     = int(getattr(self,  "warmup_steps",      max(5, int(self.iter_num)//20)))
        lr_backoff       = float(getattr(self, "lr_backoff",       0.5))
        lr_floor         = float(getattr(self, "lr_floor",         1e-6))
        freeze_disc_max  = int(getattr(self,  "freeze_disc_max",   10))
        weights_dir      = getattr(self,     "save_wpath",         None)

        # ---------- EMA & Snapshots (on device) ----------
        ema = DeviceEMA(self.generator, decay=ema_decay)
        snap = DeviceSnapshot([self.generator.trainable_variables,
                            self.discriminator.trainable_variables])
        snap.snapshot()  # initial good state

        # Scalar EMAs for anomaly detection (no host sync)
        g_ema = ScalarEMA(0.95)
        d_ema = ScalarEMA(0.95)

        # Keep original learning rates to restore after temporary freezes
        g_lr0 = (float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
                if hasattr(self, "g_optimizer") else None)
        d_lr0 = (float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))
                if hasattr(self, "d_optimizer") else None)

        # ---------- Monitor ----------
        pbar = tqdm(total=int(self.iter_num), desc="Reconstruction", leave=True)
        
        if self.recon_monitor:
            recon_monitor = RECONmonitor("tomo", self.prj_input)
        freeze_disc_steps = 0
        last_recon_np = None   # only updated occasionally to avoid host sync
        step_result = {}

        for step in range(int(self.iter_num)):
            # Apply temporary D freeze by LR=0 (no retracing)
            if hasattr(self, "d_optimizer") and d_lr0 is not None:
                if freeze_disc_steps > 0:
                    set_lr(self.d_optimizer, 0.0)
                elif float(tf.keras.backend.get_value(self.d_optimizer.learning_rate)) == 0.0:
                    set_lr(self.d_optimizer, d_lr0)

            # One step — your function should do forward/backward/updates (prefer @tf.function in it)
            step_result = self.recon_step(prj, ang)
            if "recon" in step_result and (step % log_every == 0):
                y = step_result["recon"]
                if flat_output(y):
                    # 1) roll back to last good snapshot (you already have DeviceSnapshot.restore())
                    snap.restore()
                    # 2) reduce GAN pressure and boost pixel losses
                    if hasattr(self, "lambda_gan"): self.lambda_gan *= 0.5
                    # 3) small LR backoff for D; keep or raise G LR slightly
                    if hasattr(self, "d_optimizer"):
                        cur = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))
                        set_lr(self.d_optimizer, max(1e-6, 0.5 * cur))

            g_loss_t = tf.cast(step_result["g_loss"], tf.float32)
            d_loss_t = tf.cast(step_result["d_loss"], tf.float32)

            # Guard: NaN/Inf
            finite = tf.math.is_finite(g_loss_t) & tf.math.is_finite(d_loss_t)
            if not bool(finite.numpy()):                # scalar sync (cheap)
                snap.restore()                          # instant on-device rollback
                if hasattr(self, "g_optimizer") and g_lr0 is not None:
                    cur = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
                    set_lr(self.g_optimizer, max(lr_floor, cur * lr_backoff))
                if hasattr(self, "d_optimizer") and d_lr0 is not None:
                    cur = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))
                    set_lr(self.d_optimizer, max(lr_floor, cur * lr_backoff))
                freeze_disc_steps = freeze_disc_max
                if step % log_every == 0:
                    pbar.set_postfix_str("NaN/Inf→rollback")
                pbar.update(1)
                continue

            # Update scalar EMAs (device)
            g_ema_v = g_ema.update(g_loss_t)
            d_ema_v = d_ema.update(d_loss_t)

            # Spike detection (occasional host fetch for decision)
            do_check = (step > warmup_steps) and (step % log_every == 0)
            if do_check:
                g_loss = float(g_loss_t.numpy())
                d_loss = float(d_loss_t.numpy())
                g_bar  = float(g_ema_v.numpy())
                d_bar  = float(d_ema_v.numpy())
                if g_loss > spike_factor * max(1e-8, g_bar) or d_loss > spike_factor * max(1e-8, d_bar):
                    snap.restore()
                    if hasattr(self, "g_optimizer") and g_lr0 is not None:
                        cur = float(tf.keras.backend.get_value(self.g_optimizer.learning_rate))
                        set_lr(self.g_optimizer, max(lr_floor, cur * lr_backoff))
                    if hasattr(self, "d_optimizer") and d_lr0 is not None:
                        cur = float(tf.keras.backend.get_value(self.d_optimizer.learning_rate))
                        set_lr(self.d_optimizer, max(lr_floor, cur * lr_backoff))
                    freeze_disc_steps = freeze_disc_max
                    pbar.set_postfix_str("spike→rollback")
                    pbar.update(1)
                    continue

            # Good step: update EMA weights and snapshot occasionally (device only)
            ema.update()                 # on-device
            if step % snapshot_every == 0:
                snap.snapshot()          # on-device, cheap

            # Rare disk save (weights-only)
            if weights_dir and (step % disk_save_every == 0 or step == int(self.iter_num) - 1):
                try:
                    os.makedirs(weights_dir, exist_ok=True)
                    self.generator.save_weights(os.path.join(weights_dir, "generator.weights.h5"))
                    self.discriminator.save_weights(os.path.join(weights_dir, "discriminator.weights.h5"))
                except Exception as e:
                    if step % log_every == 0:
                        pbar.set_postfix_str(f"save err: {e}")

            # Progress/logging (host sync every log_every steps only)
            if step % log_every == 0:
                pbar.set_postfix(G_loss=f"{float(g_loss_t.numpy()):.4f}",
                                D_loss=f"{float(d_loss_t.numpy()):.4f}")

            if self.recon_monitor:
                recon_monitor.update_plot(step_result)
            # countdown
            if freeze_disc_steps > 0:
                freeze_disc_steps -= 1

            pbar.update(1)

        pbar.close()
        if getattr(self, "recon_monitor", False) and 'recon_monitor' in locals() and recon_monitor is not None:
            try: recon_monitor.close_plot()
            except Exception: pass

        # Final output: evaluate with EMA weights (swap in/out on device, no host copy)
        ema.swap_in()
        try:
            # If your generator supports a direct forward to produce recon from (prj, ang),
            # compute it here. Otherwise fall back to the last monitored recon.
            final = last_recon_np
            if final is None and "recon" in step_result:
                final = step_result["recon"].numpy()
            if final is None:
                final = np.zeros(self.shape_output, dtype=np.float32)
        finally:
            ema.swap_out()

        return np.reshape(final.astype(np.float32), self.shape_output)


class GANtensor:
    def __init__(self, prj_input, angle, psi, **kwargs):
        tomo_args = config["GANtensor"]
        tomo_args.update(**kwargs)
        super(GANtensor, self).__init__()
        tf_configures()
        self.prj_input = prj_input
        self.img_h, self.img_w = self.prj_input.shape
        self.angle = angle
        self.psi = psi
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
            self.prj_input.shape[0], self.prj_input.shape[1], self.conv_num, self.conv_size, self.dropout, 3
        )
        self.discriminator = make_discriminator(self.prj_input.shape[0], self.prj_input.shape[1])
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    @tf.function
    # def tfnor_tomo(self, data):
    #     # Calculate the mean and standard deviation of the data
    #     mean = tf.reduce_mean(data)
    #     std = tf.math.reduce_std(data)
    #     # Standardize the data (z-score normalization)
    #     standardized_data = (data - mean) / std
    #     # Find the minimum value in the standardized data
    #     standardized_min = tf.reduce_min(standardized_data)
    #     # Shift the data to start from 0
    #     shifted_data = standardized_data - standardized_min
    #     return shifted_data

    def tfnor_tomo(self, img):
        img = tf.image.per_image_standardization(img)
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    @tf.function
    def recon_step(self, prj, ang, psi):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(prj)
            recon = self.tfnor_tomo(recon)
            tomo_radon_obj = TensorRadon(recon, ang, psi)
            prj_rec = tomo_radon_obj.compute()
            prj_rec = self.tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"recon": recon, "prj_rec": prj_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = self.tfnor_tomo(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        psi = self.psi
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath + "generator.h5")
            print("generator is initilized")
            self.discriminator.load_weights(self.init_wpath + "discriminator.h5")
        # recon = np.zeros((self.iter_num, px, px, 6))
        # gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        if self.recon_monitor:
            recon_monitor = RECONmonitor("tensor", self.prj_input)
            # plot_x, plot_loss = [], []
            # recon_monitor = RECONmonitor("tomo")
            # recon_monitor.initial_plot(self.prj_input)
            # pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):
            step_result = self.recon_step(prj, ang, psi)
            # recon[epoch, :, :, :] = step_result["recon"]
            pbar.set_postfix(G_loss=step_result["g_loss"].numpy(), D_loss=step_result["d_loss"].numpy())
            pbar.update(1) 
            # gen_loss[epoch] = step_result["g_loss"]

            ###########################################################################
            if self.recon_monitor:
                recon_monitor.update_plot(step_result)         
            # if (epoch + 1) % 100 == 0:
            #     if recon_monitor:
            #         prj_rec = np.reshape(step_result["prj_rec"], (nang, px))
            #         prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
            #         rec_plt = np.reshape(recon[epoch, :, :, 0], (px, px))
            #         recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
        if self.save_wpath != None:
            self.generator.save(self.save_wpath + "generator.h5")
            self.discriminator.save(self.save_wpath + "discriminator.h5")
        recon_monitor.close_plot()
        
        recon_out = np.transpose(step_result['recon'][0].numpy(), axes=(2, 0, 1))
        return recon_out.astype(np.float32)


class GANtomo3D:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config["GANphase"]
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
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
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
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, self.l1_ratio)
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
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("tomo")
            recon_monitor.initial_plot(self.prj_input)
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

            plot_x.append(epoch)
            plot_loss = gen_loss[: epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result["prj_rec"], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print(
                    "Iteration {}: G_loss is {} and D_loss is {}".format(
                        epoch + 1, gen_loss[epoch], step_result["d_loss"].numpy()
                    )
                )
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath + "generator.h5")
            self.discriminator.save(self.save_wpath + "discriminator.h5")
        return recon[epoch]
        # return avg_results(recon, gen_loss)


class GANphase:
    def __init__(self, i_input, energy, z, pv, **kwargs):
        phase_args = config["GANphase"]
        phase_args.update(**kwargs)
        super(GANphase, self).__init__()
        tf_configures()
        self.i_input = i_input
        self.img_h, self.img_w = self.i_input.shape
        self.energy = energy
        self.z = z
        self.pv = pv
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
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.generator = make_generator(
            self.i_input.shape[0], self.i_input.shape[1], self.conv_num, self.conv_size, self.dropout, 2
        )
        self.discriminator = make_discriminator(self.i_input.shape[0], self.i_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-3)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)

    @tf.function
    def rec_step(self, i_input, ff):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
            phase = tfnor_phase(recon[:, :, :, 0])
            phase = tf.reshape(phase, [self.img_h, self.img_w])
            absorption = (1 - tfnor_phase(recon[:, :, :, 1])) * self.abs_ratio
            absorption = tf.reshape(absorption, [self.img_h, self.img_w])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            phase_obj = PhaseFresnel(phase, absorption, ff, self.img_w)
            i_rec = phase_obj.compute()
            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
            g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"phase": phase, "absorption": absorption, "i_rec": i_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        ff = ffactor(self.img_w * 2, self.energy, self.z, self.pv)
        i_input = np.reshape(self.i_input, (1, self.img_h, self.img_w, 1))
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()
        # phase = np.zeros((self.iter_num, self.img_h, self.img_w))
        # absorption = np.zeros((self.iter_num, self.img_h, self.img_w))
        # gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        if self.recon_monitor:
            recon_monitor = RECONmonitor("phase", self.i_input)
            # plot_x, plot_loss = [], []
            # recon_monitor = RECONmonitor("phase")
            # recon_monitor.initial_plot(self.i_input)
            # pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_result = self.rec_step(i_input, ff)
            pbar.set_postfix(G_loss=step_result["g_loss"].numpy(), D_loss=step_result["d_loss"].numpy())
            pbar.update(1) 
            # phase[epoch, :, :] = step_results["phase"]
            # absorption[epoch, :, :] = step_results["absorption"]
            # i_rec = step_results["i_rec"]
            # gen_loss[epoch] = step_results["g_loss"]
            # d_loss = step_results["d_loss"]
            ###########################################################################
            if self.recon_monitor:
                recon_monitor.update_plot(step_result)   
                # plot_x.append(epoch)
                # plot_loss = gen_loss[: epoch + 1]
                # pbar.set_postfix(G_loss=gen_loss[epoch], D_loss=d_loss.numpy())
                # pbar.update(1)
            # if (epoch + 1) % 100 == 0:
            #     if recon_monitor:
            #         i_rec = np.reshape(i_rec, (self.px, self.px))
            #         i_diff = np.abs(i_rec - self.i_input.reshape((self.px, self.px)))
            #         phase_plt = np.reshape(phase[epoch], (self.px, self.px))
            #         recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
        if self.recon_monitor:
            recon_monitor.close_plot()
        absorption = np.reshape(step_result['absorption'].numpy().astype(np.float32), (self.img_w, self.img_w))
        phase = np.reshape(step_result['phase'].numpy().astype(np.float32), (self.img_w, self.img_w))
        return absorption, phase


class GANdiffraction:
    def __init__(self, i_input, mask, **kwargs):
        diffraction_args = config["GANdiffraction"]
        diffraction_args.update(**kwargs)
        super(GANdiffraction, self).__init__()
        self.i_input = i_input
        self.mask = mask
        self.px, _ = i_input.shape
        self.iter_num = diffraction_args["iter_num"]
        self.conv_num = diffraction_args["conv_num"]
        self.conv_size = diffraction_args["conv_size"]
        self.dropout = diffraction_args["dropout"]
        self.l1_ratio = diffraction_args["l1_ratio"]
        self.abs_ratio = diffraction_args["abs_ratio"]
        self.g_learning_rate = diffraction_args["g_learning_rate"]
        self.d_learning_rate = diffraction_args["d_learning_rate"]
        self.phase_only = diffraction_args["phase_only"]
        self.save_wpath = diffraction_args["save_wpath"]
        self.init_wpath = diffraction_args["init_wpath"]
        self.init_model = diffraction_args["init_model"]
        self.recon_monitor = diffraction_args["recon_monitor"]
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    @tf.function
    def tfnor_diff(img):
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    def make_model(self):
        self.generator = make_generator(
            self.i_input.shape[0], self.i_input.shape[1], self.conv_num, self.conv_size, self.dropout, 2
        )
        self.discriminator = make_discriminator(self.i_input.shape[0], self.i_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-3)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)

    @tf.function
    def rec_step(self, i_input):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
            phase = self.tfnor_diff(recon[:, :, :, 0])
            phase = tf.reshape(phase, [self.px // 2, self.px // 2])
            phase = tf.pad(phase, [[64, 64], [64, 64]])
            absorption = (1 - self.tfnor_diff(recon[:, :, :, 1])) * self.abs_ratio
            absorption = tf.reshape(absorption, [self.px // 2, self.px // 2])
            absorption = tf.pad(absorption, [[64, 64], [64, 64]])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            phase_obj = PhaseFraunhofer(phase, absorption)
            i_rec = phase_obj.compute()
            mask = tf.reshape(self.mask, [1, self.mask.shape[0], self.mask.shape[1], 1])
            if self.mask:
                i_rec = tf.multiply(i_rec, mask)
            i_rec = self.tfnor_diff(i_rec)
            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
            g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return {"phase": phase, "absorption": absorption, "i_rec": i_rec, "g_loss": g_loss, "d_loss": d_loss}

    @property
    def recon(self):
        i_input = np.reshape(self.i_input, (1, self.px, self.px, 1))
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()
        phase = np.zeros((self.iter_num, self.px, self.px))
        absorption = np.zeros((self.iter_num, self.px, self.px))
        gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor("phase")
            recon_monitor.initial_plot(self.i_input)
            pbar = tqdm(total=self.iter_num, desc="Reconstruction Progress", position=0, leave=True)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_results = self.rec_step(i_input)
            phase[epoch, :, :] = step_results["phase"]
            absorption[epoch, :, :] = step_results["absorption"]
            i_rec = step_results["i_rec"]
            gen_loss[epoch] = step_results["g_loss"]
            d_loss = step_results["d_loss"]
            ###########################################################################

            if self.recon_monitor:
                plot_x.append(epoch)
                plot_loss = gen_loss[: epoch + 1]
                pbar.set_postfix(G_loss=gen_loss[epoch], D_loss=d_loss.numpy())
                pbar.update(1)
            if (epoch + 1) % 100 == 0:
                if self.recon_monitor:
                    i_rec = np.reshape(i_rec, (self.px, self.px))
                    i_diff = np.abs(i_rec - self.i_input.reshape((self.px, self.px)))
                    phase_plt = np.reshape(phase[epoch], (self.px, self.px))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
        if self.recon_monitor:
            recon_monitor.close_plot()
        return absorption[epoch], phase[epoch]
