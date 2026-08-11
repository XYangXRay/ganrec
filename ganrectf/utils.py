import os
import datetime
import numpy as np
from numpy.fft import fftfreq
import tifffile
import tensorflow as tf


def nor_tomo(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img


def angles(nang, ang1=0.0, ang2=180.0):
    return np.linspace(ang1 * np.pi / 180.0, ang2 * np.pi / 180.0, nang)


def nor_prj(img):
    # nang, px = img.shape
    mean_sum = np.mean(np.sum(img, axis=(1, 2)))
    data_corr = np.zeros_like(img)
    for i in range(len(img)):
        data_corr[i, :, :] = img[i, :, :] * mean_sum / np.sum(img[i, :, :])
    return data_corr


def center(prj, cen):
    _, _, px = prj.shape
    cen_diff = px // 2 - cen
    if cen_diff > 0:
        prj = prj[:, :, : -cen_diff * 2]
    if cen_diff < 0:
        prj = prj[:, :, -cen_diff * 2 :]
    prj = np.pad(
        prj,
        (
            (
                0,
                0,
            ),
            (0, 0),
            (np.abs(cen_diff), np.abs(cen_diff)),
        ),
        "constant",
    )
    return prj


def cal_intensity(prj, recon):
    cal_coeff = np.mean(np.sum(prj, axis=(0, 2)))
    recon_corr = np.zeros_like(recon)
    for i in range(len(recon)):
        recon_corr[i, :, :] = recon[i, :, :] * cal_coeff / np.sum(recon[i, :, :])
    return recon_corr


def nor_phase(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    return img


def ffactor(px, energy, z, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv**2
    freq = fftfreq(px)
    xi, eta = np.meshgrid(freq, freq)
    xi = xi.astype("float32")
    eta = eta.astype("float32")
    h = np.exp(-1j * frequ_prefactor * (xi**2 + eta**2) / 2)
    return h


def in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other types
    except NameError:
        return False  # Probably standard Python interpreter


class RECONmonitor:
    """TensorFlow-native reconstruction monitor.

    Logs scalar losses, gradient norms, SSIM scores, and image summaries
    to a TensorBoard logdir.  When running inside a Jupyter notebook, also
    renders inline image panels using ``tf.io.encode_png`` and
    ``IPython.display`` (no matplotlib dependency).

    Parameters
    ----------
    recon_target : str
        ``"tomo"``, ``"phase"``, or ``"tensor"``.
    img_input : ndarray (2-D)
        The measured input image (sinogram / intensity / ...).
    logdir : str or None
        TensorBoard log directory.  Defaults to ``logs/ganrec/<timestamp>``.
    update_rate : int
        Write image / inline display every *update_rate* steps.
        Scalar summaries are written every step.
    """

    def __init__(self, recon_target, img_input, logdir=None, update_rate=100):
        self.recon_target = recon_target
        self.img_input = img_input.astype(np.float32)
        self.img_h, self.img_w = img_input.shape
        self.update_rate = int(update_rate)
        self.step = 0
        self.total_steps = 0  # set by caller for progress display
        self._in_notebook = in_notebook()

        # Accumulated loss history for inline display
        self._g_losses = []
        self._d_losses = []

        if logdir is None:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join("logs", "ganrec", ts)
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(logdir)

        # Map recon_target to the dict keys produced by forward_fn
        if recon_target in ("tomo", "tensor"):
            self._recon_key = "recon"
            self._pred_key  = "prj_rec"
        elif recon_target == "phase":
            self._recon_key = "phase"
            self._pred_key  = "i_rec"
        else:
            self._recon_key = "recon"
            self._pred_key  = "predicted"

        # Also accept "predicted" as a fallback for _pred_key
        self._pred_fallback = "predicted"

        # Write the input image once
        with self.writer.as_default():
            inp_img = self._to_image_tensor(
                tf.constant(self.img_input))
            tf.summary.image("input", inp_img, step=0)
        self.writer.flush()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_image_tensor(t):
        """Normalise a 2-D+ tensor to [0, 1] and reshape to [1, H, W, 1]."""
        t = tf.cast(t, tf.float32)
        t_flat = tf.reshape(t, [-1])
        t_min = tf.reduce_min(t_flat)
        t_max = tf.reduce_max(t_flat)
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)
        if t_norm.shape.ndims is None or t_norm.shape.ndims == 2:
            return tf.reshape(t_norm, [1, tf.shape(t_norm)[0], tf.shape(t_norm)[1], 1])
        elif t_norm.shape.ndims == 3:
            return tf.reshape(t_norm, [1, tf.shape(t_norm)[0], tf.shape(t_norm)[1], tf.shape(t_norm)[2]])
        elif t_norm.shape.ndims == 4:
            return t_norm[:1]
        return tf.reshape(t_norm, [1, tf.shape(t_norm)[0], tf.shape(t_norm)[1], 1])

    @staticmethod
    def _to_png_bytes(img_4d):
        """Convert a [1, H, W, 1] float32 tensor to PNG bytes."""
        img_uint8 = tf.cast(tf.clip_by_value(img_4d[0] * 255.0, 0, 255), tf.uint8)
        return tf.io.encode_png(img_uint8)

    @tf.function(reduce_retracing=True)
    def _tf_ssim(self, pred, target):
        """Compute SSIM between two 2-D tensors (on device)."""
        pred_4d = tf.reshape(pred, [1, tf.shape(pred)[0], tf.shape(pred)[1], 1])
        tgt_4d  = tf.reshape(target, [1, tf.shape(target)[0], tf.shape(target)[1], 1])
        max_val = tf.reduce_max(tgt_4d) - tf.reduce_min(tgt_4d)
        return tf.image.ssim(pred_4d, tgt_4d, max_val=max_val + 1e-8)

    def _get_pred_tensor(self, step_result):
        """Resolve the predicted-measurement tensor from step_result."""
        if self._pred_key in step_result:
            return step_result[self._pred_key]
        return step_result.get(self._pred_fallback)

    # ------------------------------------------------------------------ #
    #  Inline Jupyter display (no matplotlib)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _colorbar_html(tensor, height=256):
        """Generate a vertical HTML/CSS colorbar with min/max value labels.

        Uses a grayscale gradient matching the image normalization
        (white = max, black = min) and annotates with true data values.
        """
        if hasattr(tensor, 'numpy'):
            t_np = tensor.numpy()
        else:
            t_np = np.asarray(tensor)
        vmin, vmax = float(t_np.min()), float(t_np.max())
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'justify-content:space-between;height:{height}px;margin-left:2px;'
            f'font-family:monospace;font-size:10px;color:#333;">'
            f'<div>{vmax:.3g}</div>'
            f'<div style="width:14px;flex:1;margin:2px 0;'
            f'background:linear-gradient(to bottom,#fff,#000);'
            f'border:1px solid #999;"></div>'
            f'<div>{vmin:.3g}</div>'
            f'</div>'
        )

    def _image_panel(self, title, tensor, b64_png, img_width=256):
        """Build an HTML snippet for one image panel with a colorbar."""
        # Compute displayed image height from tensor aspect ratio
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            # Handle [B,H,W,C], [H,W,C], [H,W] tensors
            if shape.ndims == 4:
                th, tw = int(shape[1]), int(shape[2])
            elif shape.ndims == 3:
                th, tw = int(shape[0]), int(shape[1])
            elif shape.ndims == 2:
                th, tw = int(shape[0]), int(shape[1])
            else:
                th, tw = 1, 1
            img_height = int(img_width * th / max(tw, 1))
        else:
            img_height = img_width
        cbar = self._colorbar_html(tensor, height=img_height)
        return (
            f'<div style="display:flex;align-items:flex-start;">'
            f'<div>'
            f'<div style="font-size:12px;font-weight:bold;">{title}</div>'
            f'<img src="data:image/png;base64,{b64_png}" '
            f'style="image-rendering:pixelated;width:{img_width}px;height:{img_height}px;"/>'
            f'</div>'
            f'{cbar}'
            f'</div>'
        )

    def _display_inline(self, step_result, ssim_val):
        """Render an inline HTML panel in the notebook using IPython.display.

        Layout:
          Row 1: Reconstruction (centered)
          Row 2: Input + Predicted (side by side)
          Row 3: Loss convergence curve
        """
        from IPython.display import display, clear_output, HTML
        import base64

        parts = []

        # Progress bar
        if self.total_steps > 0:
            pct = min(100, self.step * 100 / self.total_steps)
            parts.append(
                f'<div style="background:#e0e0e0;border-radius:4px;height:18px;'
                f'width:100%;max-width:600px;margin-bottom:4px;">'
                f'<div style="background:#1976d2;height:100%;border-radius:4px;'
                f'width:{pct:.1f}%;min-width:2px;"></div></div>'
                f'<div style="font-size:12px;margin-bottom:6px;">'
                f'Step {self.step}/{self.total_steps} ({pct:.0f}%)</div>'
            )

        parts.append(
            f"<b>G_loss:</b> {self._g_losses[-1]:.4f} &nbsp; "
            f"<b>D_loss:</b> {self._d_losses[-1]:.4f} &nbsp; "
            f"<b>SSIM:</b> {ssim_val:.4f}"
        )

        # Row 1: Reconstruction
        if self._recon_key in step_result:
            recon_t = step_result[self._recon_key]
            if self.recon_target == "tensor":
                recon_t = recon_t[:, :, :, 0] if recon_t.shape.ndims == 4 else recon_t
            rec_img = self._to_image_tensor(recon_t)
            rec_png = self._to_png_bytes(rec_img).numpy()
            rec_b64 = base64.b64encode(rec_png).decode()
            parts.append(
                '<div style="display:flex;gap:16px;justify-content:center;margin-bottom:8px;">'
            )
            parts.append(self._image_panel("Reconstruction", recon_t, rec_b64))
            parts.append("</div>")

        # Row 2: Input + Predicted
        parts.append(
            '<div style="display:flex;gap:16px;justify-content:center;margin-bottom:8px;">'
        )
        inp_t = tf.constant(self.img_input)
        inp_img = self._to_image_tensor(inp_t)
        inp_png = self._to_png_bytes(inp_img).numpy()
        inp_b64 = base64.b64encode(inp_png).decode()
        parts.append(self._image_panel("Input", inp_t, inp_b64))

        pred_t = self._get_pred_tensor(step_result)
        if pred_t is not None:
            pred_img = self._to_image_tensor(pred_t)
            pred_png = self._to_png_bytes(pred_img).numpy()
            pred_b64 = base64.b64encode(pred_png).decode()
            parts.append(self._image_panel("Predicted", pred_t, pred_b64))
        parts.append("</div>")

        # Row 3: Loss convergence curve
        if len(self._g_losses) > 1:
            parts.append(self._loss_svg())

        clear_output(wait=True)
        display(HTML("\n".join(parts)))

    def _loss_svg(self):
        """Render a compact inline SVG loss chart with axes (no matplotlib)."""
        w, h = 600, 160
        left, right, top, bottom = 60, 20, 15, 40
        plot_w = w - left - right
        plot_h = h - top - bottom
        n = len(self._g_losses)
        if n < 2:
            return ""

        # Combine both loss series to get a shared Y range
        all_vals = self._g_losses + self._d_losses
        ymin, ymax = min(all_vals), max(all_vals)
        if ymax == ymin:
            ymax = ymin + 1

        def _map(vals):
            xs = [left + i / (n - 1) * plot_w for i in range(n)]
            ys = [top + (1 - (v - ymin) / (ymax - ymin)) * plot_h for v in vals]
            return xs, ys

        gx, gy = _map(self._g_losses)
        dx, dy = _map(self._d_losses)

        g_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(gx, gy))
        d_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(dx, dy))

        # X-axis tick labels (step numbers)
        n_xticks = min(5, n)
        xticks = ""
        for i in range(n_xticks):
            idx = int(i / (n_xticks - 1) * (n - 1)) if n_xticks > 1 else 0
            x_pos = left + idx / (n - 1) * plot_w
            step_num = idx  # monitor step index
            xticks += (
                f'<line x1="{x_pos:.1f}" y1="{top + plot_h}" '
                f'x2="{x_pos:.1f}" y2="{top + plot_h + 4}" stroke="#666"/>'
                f'<text x="{x_pos:.1f}" y="{top + plot_h + 16}" '
                f'font-size="10" fill="#666" text-anchor="middle">{step_num}</text>'
            )

        # Y-axis tick labels (loss values)
        n_yticks = 4
        yticks = ""
        for i in range(n_yticks):
            frac = i / (n_yticks - 1)
            y_pos = top + (1 - frac) * plot_h
            val = ymin + frac * (ymax - ymin)
            yticks += (
                f'<line x1="{left - 4}" y1="{y_pos:.1f}" '
                f'x2="{left}" y2="{y_pos:.1f}" stroke="#666"/>'
                f'<text x="{left - 6}" y="{y_pos + 3:.1f}" '
                f'font-size="10" fill="#666" text-anchor="end">{val:.3g}</text>'
            )

        svg = (
            f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#fafafa; border:1px solid #ddd; margin-top:8px;">'
            # Axes
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
            f'stroke="#999" stroke-width="1"/>'
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
            f'y2="{top + plot_h}" stroke="#999" stroke-width="1"/>'
            # Tick marks
            f'{xticks}{yticks}'
            # Axis labels
            f'<text x="{left + plot_w / 2}" y="{h - 2}" font-size="11" '
            f'fill="#333" text-anchor="middle">Step</text>'
            f'<text x="12" y="{top + plot_h / 2}" font-size="11" fill="#333" '
            f'text-anchor="middle" transform="rotate(-90,12,{top + plot_h / 2})">Loss</text>'
            # Data lines
            f'<polyline points="{g_pts}" fill="none" stroke="#d32f2f" stroke-width="1.5"/>'
            f'<polyline points="{d_pts}" fill="none" stroke="#1565c0" stroke-width="1.5"/>'
            # Legend
            f'<line x1="{left + plot_w - 120}" y1="{top + 10}" '
            f'x2="{left + plot_w - 100}" y2="{top + 10}" stroke="#d32f2f" stroke-width="2"/>'
            f'<text x="{left + plot_w - 96}" y="{top + 14}" font-size="10" fill="#d32f2f">'
            f'G: {self._g_losses[-1]:.4f}</text>'
            f'<line x1="{left + plot_w - 120}" y1="{top + 24}" '
            f'x2="{left + plot_w - 100}" y2="{top + 24}" stroke="#1565c0" stroke-width="2"/>'
            f'<text x="{left + plot_w - 96}" y="{top + 28}" font-size="10" fill="#1565c0">'
            f'D: {self._d_losses[-1]:.4f}</text>'
            f'</svg>'
        )
        return svg

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def update_plot(self, step_result):
        """Write summaries and optionally display inline in Jupyter.

        Parameters
        ----------
        step_result : dict
            Dictionary returned by ``recon_step``.
        """
        self.step += 1
        step = self.step

        # Accumulate losses (cheap — just float scalars)
        self._g_losses.append(float(step_result["g_loss"].numpy()))
        self._d_losses.append(float(step_result["d_loss"].numpy()))

        with self.writer.as_default():
            # ---- Scalar summaries (every step) ----
            g_loss = step_result["g_loss"]
            d_loss = step_result["d_loss"]
            tf.summary.scalar("loss/generator", g_loss, step=step)
            tf.summary.scalar("loss/discriminator", d_loss, step=step)

            if "g_gnorm" in step_result:
                tf.summary.scalar("grad_norm/generator",
                                  step_result["g_gnorm"], step=step)
            if "d_gnorm" in step_result:
                tf.summary.scalar("grad_norm/discriminator",
                                  step_result["d_gnorm"], step=step)

            # ---- Image summaries + SSIM (every update_rate steps) ----
            ssim_val = 0.0
            if step % self.update_rate == 0:
                # Reconstruction image
                if self._recon_key in step_result:
                    recon_t = step_result[self._recon_key]
                    if self.recon_target == "tensor":
                        recon_t = recon_t[:, :, :, 0] if recon_t.shape.ndims == 4 else recon_t
                    recon_img = self._to_image_tensor(recon_t)
                    tf.summary.image("reconstruction", recon_img, step=step)

                # Predicted measurement + SSIM
                pred_t = self._get_pred_tensor(step_result)
                if pred_t is not None:
                    pred_img = self._to_image_tensor(pred_t)
                    tf.summary.image("predicted", pred_img, step=step)

                    pred_2d = tf.reshape(
                        tf.cast(pred_t, tf.float32),
                        [self.img_h, self.img_w],
                    )
                    input_tf = tf.constant(self.img_input)
                    ssim_val = float(self._tf_ssim(pred_2d, input_tf)[0].numpy())
                    tf.summary.scalar("metrics/ssim", ssim_val, step=step)

        # Flush + inline display periodically
        if step % self.update_rate == 0:
            self.writer.flush()
            if self._in_notebook:
                self._display_inline(step_result, ssim_val)

    def close_plot(self, last_step_result=None):
        """Flush, do a final inline display, and close the TensorBoard writer."""
        self.writer.flush()
        # Final inline display so progress shows 100%
        if self._in_notebook and last_step_result is not None:
            # Compute final SSIM
            ssim_val = 0.0
            pred_t = self._get_pred_tensor(last_step_result)
            if pred_t is not None:
                pred_2d = tf.reshape(
                    tf.cast(pred_t, tf.float32),
                    [self.img_h, self.img_w],
                )
                input_tf = tf.constant(self.img_input)
                ssim_val = float(self._tf_ssim(pred_2d, input_tf)[0].numpy())
            self._display_inline(last_step_result, ssim_val)
        self.writer.close()


def display_strain_tensor(tensor, profile_index=None):
    """
    Display the components of the strain tensor and a single horizontal profile plot.

    Parameters:
    tensor (numpy.ndarray): A numpy array of shape [3 or 6, h, w] representing the components of the strain tensor.
    profile_index (int): The index of the row for the profile plot. If None, the middle row is used.

    Components are expected to be in the following order if 6 components:
    0: ε_xx
    1: ε_xy
    2: ε_xz
    3: ε_yy
    4: ε_yz
    5: ε_zz

    Components are expected to be in the following order if 3 components:
    0: ε_xx
    1: ε_xy
    2: ε_yy
    """
    if tensor.shape[0] not in [3, 6]:
        raise ValueError("Input tensor must have 3 or 6 components in the first dimension")

    import matplotlib.pyplot as plt

    component_names = (
        [
            r"$\epsilon_{xx}$",
            r"$\epsilon_{xy}$",
            r"$\epsilon_{xz}$",
            r"$\epsilon_{yy}$",
            r"$\epsilon_{yz}$",
            r"$\epsilon_{zz}$",
        ]
        if tensor.shape[0] == 6
        else [r"$\epsilon_{xx}$", r"$\epsilon_{xy}$", r"$\epsilon_{yy}$"]
    )

    rows, cols = (2, 3) if tensor.shape[0] == 6 else (1, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10 if tensor.shape[0] == 6 else 5))
    axes = axes.ravel()

    # Find the global minimum and maximum for the color scale
    vmin = np.min(tensor)
    vmax = np.max(tensor)

    if profile_index is None:
        profile_index = tensor.shape[1] // 2  # Default to the middle row

    profile_colors = ["r", "g", "b", "c", "m", "y"]

    # Plot each component and its profile position marker
    for i in range(tensor.shape[0]):
        row, col = divmod(i, cols)
        ax_image = axes[row * cols + col]
        im = ax_image.imshow(tensor[i], cmap="gray", aspect="equal", vmin=vmin, vmax=vmax)
        ax_image.set_title(component_names[i], fontsize=16)
        ax_image.axis("off")

        # Mark the profile position
        ax_image.axhline(profile_index, color=profile_colors[i], linestyle="--", linewidth=2)
    fig.subplots_adjust(right=0.85)

    # Add a single colorbar for the first two rows of subplots on the right side
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()
    # Plot the profile on the last row

    plt.figure(figsize=(18, 5))
    for i in range(tensor.shape[0]):
        profile = tensor[i, profile_index, :]
        plt.plot(profile, label=component_names[i], color=profile_colors[i], linewidth=2)
        plt.title(f"Profile at Row {profile_index}", fontsize=16)
        plt.legend(fontsize=14)
    plt.show()


# def display_strain_tensor(tensor):
#     """
#     Display the six components of the strain tensor.

#     Parameters:
#     tensor (numpy.ndarray): A numpy array of shape [6, h, w] representing the six components of the strain tensor.

#     Components are expected to be in the following order:
#     0: ε_xx
#     1: ε_xy
#     2: ε_xz
#     3: ε_yx
#     4: ε_yz
#     5: ε_zz
#     """
#     if tensor.shape[0] != 6:
#         raise ValueError("Input tensor must have 6 components in the first dimension")
#     component_names = [r'$\epsilon_{xx}$', r'$\epsilon_{xy}$', r'$\epsilon_{xz}$', 
#                        r'$\epsilon_{yx}$', r'$\epsilon_{yz}$', r'$\epsilon_{zz}$']
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.ravel()
#     vmin = np.min(tensor)
#     vmax = np.max(tensor)
#     for i in range(6):
#         ax = axes[i]
#         im = ax.imshow(tensor[i], cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
#         ax.set_title(component_names[i])
#         ax.axis('off')
#     fig.subplots_adjust(right=0.85)  # Adjust the right space to fit the colorbar
#     cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#     fig.colorbar(im, cax=cbar_ax, orientation='vertical')
#     plt.tight_layout()
#     plt.show()


# Draw a annular shape mask to only inlcude the feature in the annular area
def annular_mask(img, inner_diameter, outer_diameter):
    image_size, _ = img.shape
    x = np.linspace(-image_size // 2, image_size // 2, image_size)
    y = np.linspace(-image_size // 2, image_size // 2, image_size)
    X, Y = np.meshgrid(x, y)

    # Calculate distances from the center
    center = (0, 0)
    distances = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    # Create the mask
    mask = (distances >= inner_diameter / 2) & (distances <= outer_diameter / 2)

    # Apply the mask to an image (white ring on black background)
    img = img * mask

    return img


def save_tiff(image, filename):
    # Extract the directory from the filename
    directory = os.path.dirname(filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    image = nor_tomo(image)
    image = np.array(image, dtype=np.float32)
    # Save the image
    tifffile.imwrite(filename, image)
