import os
import numpy as np
from numpy.fft import fftfreq
import tifffile
import matplotlib.pyplot as plt
import tensorflow as tf


def nor_tomo(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img


def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)


def nor_prj(img):
    # nang, px = img.shape
    mean_sum = np.mean(np.sum(img, axis=(1,2)))
    data_corr = np.zeros_like(img)
    for i in range(len(img)):
        data_corr[i, :, :] = img[i, :, :] * mean_sum / np.sum(img[i, :, :])
    return data_corr


def center(prj, cen):
    _, _, px = prj.shape
    cen_diff = px//2 - cen
    if cen_diff>0:
        prj = prj[:, :, :-cen_diff*2]
    if cen_diff<0:
        prj = prj[:, :, -cen_diff*2:]
    prj = np.pad(prj, ((0, 0,), (0, 0), (np.abs(cen_diff), np.abs(cen_diff))), 'constant')
    return prj


def cal_intensity(prj, recon):
    cal_coeff = np.mean(np.sum(prj, axis=(0, 2)))
    recon_corr = np.zeros_like(recon)
    for i in range(len(recon)):
        recon_corr[i, :, :] = recon[i, :, :]*cal_coeff/np.sum(recon[i, :, :])
    return recon_corr


def nor_phase(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    return img


def ffactor(px, energy, z, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv ** 2
    freq = fftfreq(px)
    xi, eta = np.meshgrid(freq, freq)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    h = np.exp(- 1j * frequ_prefactor * (xi ** 2 + eta ** 2) / 2)
    return h


class RECONmonitor:
    def __init__(self, recon_target):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 8))
        self.recon_target = recon_target
        if self.recon_target == 'tomo':
            self.plot_txt = 'Sinogram'
        elif self.recon_target == 'phase':
            self.plot_txt = 'Intensity'
      

    def initial_plot(self, img_input):
        _, px = img_input.shape
        self.im0 = self.axs[0, 0].imshow(img_input, cmap='gray')
        self.axs[0, 0].set_title(self.plot_txt)
        self.fig.colorbar(self.im0, ax=self.axs[0, 0])
        self.axs[0, 0].set_aspect('equal','box')
        self.im1 = self.axs[1, 0].imshow(img_input, cmap='jet')
        self.tx1 = self.axs[1, 0].set_title('Difference of ' + self.plot_txt + ' for iteration 0')
        self.fig.colorbar(self.im1, ax=self.axs[1, 0])
        self.axs[1, 0].set_aspect('equal')
        self.im2 = self.axs[0, 1].imshow(np.zeros((px, px)), cmap='gray')
        self.fig.colorbar(self.im2, ax=self.axs[0, 1])
        self.axs[0, 1].set_title('Reconstruction')
        self.im3, = self.axs[1, 1].plot([], [], 'r-')
        self.axs[1, 1].set_title('Generator loss')
        plt.tight_layout()

    def update_plot(self, epoch, img_diff, img_rec, plot_x, plot_loss):
        self.tx1.set_text('Difference of ' + self.plot_txt + ' for iteration {0}'.format(epoch))
        vmax = np.max(img_diff)
        vmin = np.min(img_diff)
        self.im1.set_data(img_diff)
        self.im1.set_clim(vmin, vmax)
        self.im2.set_data(img_rec)
        vmax = np.max(img_rec)
        vmin = np.min(img_rec)
        self.im2.set_clim(vmin, vmax)
        self.axs[1, 1].plot(plot_x, plot_loss, 'r-')
        # plt.tight_layout()
        plt.pause(0.1)
    def close_plot(self):
        plt.close()

# Draw a annular shape mask to only inlcude the feature in the annular area
def annular_mask(img, inner_diameter, outer_diameter):
    image_size, _ = img.shape
    x = np.linspace(-image_size // 2, image_size // 2, image_size)
    y = np.linspace(-image_size // 2, image_size // 2, image_size)
    X, Y = np.meshgrid(x, y)

    # Calculate distances from the center
    center = (0, 0)
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    # Create the mask
    mask = (distances >= inner_diameter / 2) & (distances <= outer_diameter / 2)

    # Apply the mask to an image (white ring on black background)
    img = img*mask

    return img

def save_tiff(image, filename):
    # Extract the directory from the filename
    directory = os.path.dirname(filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    image = nor_tomo(image)
    image = np.array(image, dtype = np.float32)
    # Save the image
    tifffile.imwrite(filename, image)
    


def downsample_resize(
    image: tf.Tensor,
    scale: float = 0.5,
    method: str = 'bilinear',
    antialias: bool = True
) -> tf.Tensor:
    """
    Downsample a 2D image (or batch of images) by a given scale factor
    using tf.image.resize.

    Parameters
    ----------
    image : tf.Tensor
        `[H, W]`, `[H, W, C]`, or `[B, H, W, C]` tensor.
    scale : float
        Scaling factor < 1.0 to shrink the image.
    method : str
        One of 'nearest', 'bilinear', 'bicubic', 'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic'.
    antialias : bool
        Whether to apply antialiasing (recommended when downsampling).

    Returns
    -------
    tf.Tensor
        The downsampled image, same rank as input.
    """
    # ensure float32
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    orig_shape = tf.shape(img)

    # handle possible shapes
    rank = img.shape.rank
    if rank == 2:
        img = tf.expand_dims(img, axis=-1)         # [H, W] → [H, W, 1]
        orig_rank = 2
    elif rank == 3:
        orig_rank = 3
    elif rank == 4:
        orig_rank = 4
    else:
        raise ValueError(f"Unsupported tensor rank: {rank}")

    # Batch‐ify
    if img.shape.rank == 3:
        img = tf.expand_dims(img, axis=0)           # [H, W, C] → [1, H, W, C]
        added_batch = True
    else:
        added_batch = False

    # Compute new size
    h = tf.cast(orig_shape[-3], tf.float32)
    w = tf.cast(orig_shape[-2], tf.float32)
    new_h = tf.cast(tf.math.round(h * scale), tf.int32)
    new_w = tf.cast(tf.math.round(w * scale), tf.int32)
    new_size = [new_h, new_w]

    # Resize
    resized = tf.image.resize(
        img,
        size=new_size,
        method=method,
        antialias=antialias
    )

    # remove added dims
    if added_batch:
        resized = tf.squeeze(resized, axis=0)       # [1, H', W', C] → [H', W', C]
    if orig_rank == 2:
        resized = tf.squeeze(resized, axis=-1)      # [H', W', 1] → [H', W']

    return resized


def downsample_avgpool(
    image: tf.Tensor,
    factor: int
) -> tf.Tensor:
    """
    Downsample a 2D image (or batch) by an integer factor using average pooling.

    Parameters
    ----------
    image : tf.Tensor
        `[H, W]`, `[H, W, C]`, or `[B, H, W, C]` tensor.
    factor : int
        Downsampling factor (must evenly divide H and W).

    Returns
    -------
    tf.Tensor
        The downsampled image, same rank as input.
    """
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    rank = img.shape.rank

    # Add batch and channel dims if missing
    if rank == 2:
        img = tf.expand_dims(tf.expand_dims(img, 0), -1)   # [H,W]→[1,H,W,1]
        added_bc = True
    elif rank == 3:
        # Could be [H,W,C] or [B,H,W]
        if img.shape[-1] > 1 and img.shape.rank == 3:
            img = tf.expand_dims(img, axis=0)             # [H,W,C]→[1,H,W,C]
            added_bc = True
        else:
            # ambiguous; assume [B,H,W]
            img = tf.expand_dims(img, axis=-1)            # [B,H,W]→[B,H,W,1]
            added_bc = False
    elif rank == 4:
        added_bc = False
    else:
        raise ValueError(f"Unsupported tensor rank: {rank}")

    # apply average pooling
    # ksize = (factor, factor), strides = (factor, factor)
    pooled = tf.nn.avg_pool2d(
        img,
        ksize=factor,
        strides=factor,
        padding='SAME'
    )

    # remove added dims
    if added_bc:
        pooled = tf.squeeze(pooled, axis=0)              # remove batch
        pooled = tf.squeeze(pooled, axis=-1)             # remove channel
    return pooled