import os
import numpy as np
from numpy.fft import fftfreq
import tifffile
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from IPython.display import display, clear_output


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

def in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other types
    except NameError:
        return False      # Probably standard Python interpreter

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
        plt.tight_layout()
        if in_notebook():
            clear_output(wait=True)
            display(self.fig)
            plt.pause(0.001)  
        else:
            plt.ion()  # Turn on interactive mode
            plt.draw()
            plt.pause(0.001)  
    def close_plot(self):
        plt.close()


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

    component_names = [r'$\epsilon_{xx}$', r'$\epsilon_{xy}$', r'$\epsilon_{xz}$', 
                       r'$\epsilon_{yy}$', r'$\epsilon_{yz}$', r'$\epsilon_{zz}$'] if tensor.shape[0] == 6 else \
                      [r'$\epsilon_{xx}$', r'$\epsilon_{xy}$', r'$\epsilon_{yy}$']
    
    rows, cols = (2, 3) if tensor.shape[0] == 6 else (1, 3)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10 if tensor.shape[0] == 6 else 5))
    axes = axes.ravel()

    # Find the global minimum and maximum for the color scale
    vmin = np.min(tensor)
    vmax = np.max(tensor)

    if profile_index is None:
        profile_index = tensor.shape[1] // 2  # Default to the middle row

    profile_colors = ['r', 'g', 'b', 'c', 'm', 'y']


    # Plot each component and its profile position marker
    for i in range(tensor.shape[0]):
        row, col = divmod(i, cols)
        ax_image = axes[row * cols + col]
        im = ax_image.imshow(tensor[i], cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
        ax_image.set_title(component_names[i], fontsize=16)
        ax_image.axis('off')

        # Mark the profile position
        ax_image.axhline(profile_index, color=profile_colors[i], linestyle='--', linewidth=2)
    fig.subplots_adjust(right=0.85)
    
    # Add a single colorbar for the first two rows of subplots on the right side
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    plt.show()
    # Plot the profile on the last row
    
    plt.figure(figsize=(18, 5))
    for i in range(tensor.shape[0]):
        profile = tensor[i, profile_index, :]
        plt.plot(profile, label=component_names[i], color=profile_colors[i], linewidth=2)
        plt.title(f"Profile at Row {profile_index}", fontsize=16)
        plt.legend(fontsize=14)
    plt.show()

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
    
