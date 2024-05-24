import os
import numpy as np
from numpy.fft import fftfreq
import torch
import tifffile
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from numpy import pi
import skimage.io as io
from skimage.transform import resize
import quantities as pq


def nor_tomo(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img


def angles(nang, ang1=0.0, ang2=180.0):
    return np.linspace(ang1 * np.pi / 180.0, ang2 * np.pi / 180.0, nang, dtype=np.float32)


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
    def __init__(self, recon_target, img_input):
        self.recon_target = recon_target
        self.img_input = tensor_to_np(img_input)
        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 8))

        if self.recon_target == "tomo":
            self.plot_txt = "Sinogram"
            self.dummy_img1 = np.zeros((self.img_input.shape[1], self.img_input.shape[1]))
            self.dummy_img2 = np.zeros_like(self.img_input)
        elif self.recon_target == "phase":
            self.plot_txt = "Intensity"

        self._initialize_plot()

    def _initialize_plot(self):
        # Initialize the subplots
        self.im0 = self.axs[0, 0].imshow(self.img_input, cmap="gray")
        self.axs[0, 0].set_title(self.plot_txt)
        self.fig.colorbar(self.im0, ax=self.axs[0, 0])
        self.axs[0, 0].set_aspect("equal")

        self.im1 = self.axs[0, 1].imshow(self.dummy_img1, cmap="gray")
        self.axs[0, 1].set_title("Reconstruction")
        self.fig.colorbar(self.im1, ax=self.axs[0, 1])
        self.axs[1, 0].set_aspect("equal")

        self.im2 = self.axs[1, 0].imshow(self.dummy_img2, cmap="jet")
        self.tx1 = self.axs[1, 0].set_title("Difference of " + self.plot_txt + " for iteration 0")
        self.fig.colorbar(self.im2, ax=self.axs[1, 0])
        self.axs[1, 0].set_aspect("equal")

        (self.im3,) = self.axs[1, 1].plot([], [], "r-")
        self.axs[1, 1].set_title("Generator loss")
        plt.tight_layout()

    def update_plot(self, epoch, img_diff, img_rec, plot_x, plot_loss):
        self.img_diff = tensor_to_np(img_diff)
        self.img_rec = tensor_to_np(img_rec)
        self.plot_x = plot_x
        self.plot_loss = tensor_to_np(plot_loss)

        self.tx1.set_text("Difference of " + self.plot_txt + " for iteration {0}".format(epoch))

        # Update the difference image
        vmax = np.max(self.img_rec)
        vmin = np.min(self.img_rec)
        self.im1.set_data(self.img_rec)
        self.im1.set_clim(vmin, vmax)

        # Update the reconstruction image
        vmax = np.max(self.img_diff)
        vmin = np.min(self.img_diff)
        self.im2.set_data(self.img_diff)
        self.im2.set_clim(vmin, vmax)

        # Update the loss plot
        self.axs[1, 1].plot(self.plot_x, self.plot_loss, "r-")
        # self.im3.set_data(self.plot_x, self.plot_loss)
        # Check if running in a notebook

        if in_notebook():
            clear_output(wait=True)
            display(self.fig)
            plt.pause(0.001)

        else:
            plt.ion()  # Turn on interactive mode
            plt.draw()
            plt.pause(0.001)  # Pause briefly to ensure the plot is displayed

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


def nor_phase(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    return img


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ffactor(px, py, energy, z, pv, return_both=False):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv**2
    freq_1 = fftfreq(px)
    freq_2 = fftfreq(py)
    xi, eta = np.meshgrid(freq_1, freq_2)
    xi = xi.astype("float32")
    eta = eta.astype("float32")
    if return_both == False:
        return np.exp(-1j * frequ_prefactor * (xi**2 + eta**2) / 2)
    else:

        base = np.exp((xi**2 + eta**2) / 2)
        h = base ** (-1j * frequ_prefactor)
        return h, base


def ffactors(px, py, energy, zs, pv):
    lambda_p = 1.23984122e-09 / energy

    freq_x = fftfreq(px)
    freq_y = fftfreq(py)
    xi, eta = np.meshgrid(freq_x, freq_y)
    xi = xi.astype("float32")
    eta = eta.astype("float32")
    if type(zs) is not list:
        frequ_prefactors = 2 * np.pi * lambda_p * zs / pv**2
        h = np.exp(-1j * frequ_prefactors * (xi**2 + eta**2) / 2)
    else:
        frequ_prefactors = [2 * np.pi * lambda_p * zs[i] / pv**2 for i in range(len(zs))]
        h = [
            ((np.exp(-1j * frequ_prefactors[i] * (xi**2 + eta**2) / 2)).T).astype("complex64")
            for i in range(len(zs))
        ]
    return h


def fresnel_operator(px, py, pv, z, lambda0, upsample_scale):
    # lambda0 = 1.23984122e-09 / energy
    # Scale by which to upsample image
    nx = upsample_scale * px  # Image width in pixels (same as height)
    ny = upsample_scale * py  # Image height in pixels
    grid_size_x = pv * nx
    # Grid size in x-direction
    grid_size_y = pv * ny
    # Grid size in y-direction
    # Inverse space
    fx = np.linspace(-(nx - 1) / 2 * (1 / grid_size_x), (nx - 1) / 2 * (1 / grid_size_x), nx)
    fy = np.linspace(-(ny - 1) / 2 * (1 / grid_size_y), (ny - 1) / 2 * (1 / grid_size_y), ny)
    Fx, Fy = np.meshgrid(fx, fy)
    H = np.exp(1j * (2 * pi / lambda0) * z) * np.exp(1j * pi * lambda0 * z * (Fx**2 + Fy**2))
    return H.T


# class RECONmonitor:
#     def __init__(self, recon_target):
#         self.fig, self.axs = plt.subplots(2, 3, figsize=(23, 8))
#         self.recon_target = recon_target
#         if self.recon_target == 'tomo':
#             self.plot_txt = 'Sinogram'
#         elif self.recon_target == 'phase':
#             self.plot_txt = 'Intensity'

#     def initial_plot(self, img_input):
#         px, py = img_input.shape
#         self.im0 = self.axs[0, 0].imshow(img_input, cmap='gray')
#         self.axs[0, 0].set_title(self.plot_txt)
#         self.fig.colorbar(self.im0, ax=self.axs[0, 0])
#         self.axs[0, 0].set_aspect('equal','box')
#         self.im1 = self.axs[1, 0].imshow(img_input, cmap='jet')
#         self.tx1 = self.axs[1, 0].set_title('Difference of ' + self.plot_txt + ' for iteration 0')
#         self.fig.colorbar(self.im1, ax=self.axs[1, 0])
#         self.axs[0, 0].set_aspect('equal')
#         self.im2 = self.axs[0, 1].imshow(np.zeros((px, py)), cmap='gray')
#         self.fig.colorbar(self.im2, ax=self.axs[0, 1])
#         self.axs[0, 1].set_title('retrieved phase')
#         self.im3, = self.axs[1, 1].plot([], [], 'r-')
#         self.axs[1, 1].set_title('Generator loss')
#         self.axs[0, 2].set_title('plot profile of input')
#         self.axs[0, 2].plot(img_input[int(px/2), :], 'b-')
#         self.axs[0, 2].set_title('plot profile of input')
#         self.im4 = self.axs[1, 2].plot([], 'r-')
#         self.axs[1, 2].set_title('plot profile of recon')

#         plt.tight_layout()

#     def update_plot(self, epoch, img_diff, img_rec, plot_x, plot_loss, save_path = None):
#         self.tx1.set_text('Difference of ' + self.plot_txt + ' for iteration {0}'.format(epoch))
#         vmax = np.max(img_diff)
#         vmin = np.min(img_diff)
#         self.im1.set_data(img_diff)
#         self.im1.set_clim(vmin, vmax)
#         self.im2.set_data(img_rec)
#         vmax = np.max(img_rec)
#         vmin = np.min(img_rec)
#         self.im2.set_clim(vmin, vmax)
#         self.axs[1, 1].plot(plot_x, plot_loss, 'r-')
#         self.axs[1, 2].plot(img_rec[int(img_rec.shape[0]/2), :], 'r-')
#         plt.pause(0.1)

#     def close_plot(self):
#         plt.close()


def tensor_to_np(tensor):
    if type(tensor) is list:
        if len(tensor[0].shape) <= 2:
            try:
                return [t.detach().cpu().numpy() for t in tensor]
            except:
                return [t.numpy()(t) for t in tensor]
        elif len(tensor[0].shape) == 3:
            try:
                return [t.detach().cpu().numpy()[0, :, :] for t in tensor]
            except:
                return [t.numpy()[0, :, :] for t in tensor]
        else:
            try:
                return [t.detach().cpu().numpy()[0, 0, :, :] for t in tensor]
            except:
                return [t.numpy()[0, 0, :, :] for t in tensor]
    else:
        if len(tensor.shape) <= 2:
            try:
                return tensor.detach().cpu().numpy()
            except:
                return tensor.numpy()
        elif len(tensor.shape) == 3:
            try:
                return tensor.detach().cpu().numpy()[0, :, :]
            except:
                return tensor.numpy()[0, :, :]
        else:
            try:
                return tensor.detach().cpu().numpy()[0, 0, :, :]
            except:
                return tensor.numpy()[0, 0, :, :]


# def to_device(data, device=None):
#     """
#     Move tensor(s) to chosen device
#     """
#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if isinstance(data, (list,tuple)):
#         return [to_device(x, device) for x in data]
#     data = data.to(device, non_blocking=True)

#     # # If the data is a model and there are multiple GPUs, use DataParallel
#     # if isinstance(data, torch.nn.Module) and torch.cuda.device_count() > 1:
#     #     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     #     data = torch.nn.DataParallel(data)

#     return data


def to_device(data, device=None):
    """
    Move tensor(s) and model(s) to the chosen device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    data = data.to(device)

    # If the data is a model and there are multiple GPUs, use DataParallel
    # if isinstance(data, torch.nn.Module) and torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     data = torch.nn.DataParallel(data)

    return data


# def to_device(x, device):
# if device is None:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# try:
#     return x.to(device)
# except:
#     return torch_reshape(x).to(device)


def grid_generator(shape_x, shape_y, upscale=1, ps=5.5e-06):
    """
    Parameters: shape_y - shape of the image in y-direction
    #             upscale - scale by which to upsample image
    #             ps - pixel size in microns
    """
    upsample_scale = upscale
    # Scale by which to upsample image
    nx = upsample_scale * shape_x  # Image width in pixels (same as height)
    ny = upsample_scale * shape_y
    grid_size_x = ps * nx
    # Grid size in x-direction
    grid_size_y = ps * ny
    # Grid size in y-direction
    fx = np.linspace(-(nx - 1) / 2 * (1 / grid_size_x), (nx - 1) / 2 * (1 / grid_size_x), nx)
    fy = np.linspace(-(ny - 1) / 2 * (1 / grid_size_y), (ny - 1) / 2 * (1 / grid_size_y), ny)
    Fx, Fy = np.meshgrid(fx, fy)

    return Fx, Fy


def save_path_generator(**kwargs):
    try:
        file_name = os.path.splitext(os.path.basename(kwargs["image_path"]))[0]
        # the folder name of the 'image_path'
        folder = os.path.basename(os.path.dirname(kwargs["image_path"]))
    except:
        file_name = os.path.splitext(os.path.basename(kwargs["image_path"][0]))[0]
        folder = os.path.basename(os.path.dirname(kwargs["image_path"][0]))

    init_path = os.getcwd() + "/data/saved_weights/" + folder + "/"
    save_wpath = os.getcwd() + "/data/retrieved/" + folder + "/"
    if not os.path.exists(init_path):
        os.makedirs(init_path)
    if not os.path.exists(save_wpath):
        os.makedirs(save_wpath)
    kwargs["file_name"] = file_name
    kwargs["save_wpath"] = save_wpath
    kwargs["init_wpath"] = init_path
    return kwargs


def segment(image, type="chan_vese"):
    """
    types: chan_vese, sobel, otsu, local, minimum, mean, triangle, yen, multiotsu, isodata, li
    """
    from skimage.util import img_as_ubyte

    if type == "chan_vese":
        from skimage.segmentation import chan_vese

        return chan_vese(
            image,
            mu=0.25,
            lambda1=1,
            lambda2=1,
            tol=1e-3,
            max_num_iter=200,
            dt=0.5,
            init_level_set="checkerboard",
            extended_output=True,
        )
    elif type == "sobel":
        from skimage.filters import sobel

        return sobel(image)
    elif type == "otsu":
        from skimage.filters import threshold_otsu

        return img_as_ubyte(image > threshold_otsu(image))
    elif type == "local":
        from skimage.filters import threshold_local

        return img_as_ubyte(image > threshold_local(image, block_size=35, offset=10))
    elif type == "minimum":
        from skimage.filters import threshold_minimum

        return img_as_ubyte(image > threshold_minimum(image))
    elif type == "mean":
        from skimage.filters import threshold_mean

        return img_as_ubyte(image > threshold_mean(image))
    elif type == "triangle":
        from skimage.filters import threshold_triangle

        return img_as_ubyte(image > threshold_triangle(image))
    elif type == "yen":
        from skimage.filters import threshold_yen

        return img_as_ubyte(image > threshold_yen(image))
    elif type == "multiotsu":
        from skimage.filters import threshold_multiotsu

        return img_as_ubyte(image > threshold_multiotsu(image))
    elif type == "isodata":
        from skimage.filters import threshold_isodata

        return img_as_ubyte(image > threshold_isodata(image))
    elif type == "li":
        from skimage.filters import threshold_li

        return img_as_ubyte(image > threshold_li(image))
    else:
        print("wrong type")
        return image


def ffactors(px, py, energy, zs, pv):
    lambda_p = 1.23984122e-09 / energy

    freq_x = fftfreq(px)
    freq_y = fftfreq(py)
    xi, eta = np.meshgrid(freq_x, freq_y)
    xi = xi.astype("float32")
    eta = eta.astype("float32")
    if type(zs) is not list:
        frequ_prefactors = 2 * np.pi * lambda_p * zs / pv**2
        h = np.exp(-1j * frequ_prefactors * (xi**2 + eta**2) / 2)
    else:
        frequ_prefactors = [2 * np.pi * lambda_p * zs[i] / pv**2 for i in range(len(zs))]
        h = [
            ((np.exp(-1j * frequ_prefactors[i] * (xi**2 + eta**2) / 2)).T).astype("complex64")
            for i in range(len(zs))
        ]
    return h


def resize_an_image_and_view(image_list, view_factor, ratio=None, title=None, rotate=True):
    if type(image_list) is not list:
        image_list = [image_list]
    new_shape = image_list[-1].shape
    new_shape = (new_shape[0] * view_factor, new_shape[1] * view_factor)
    resized_images = [resize(image, new_shape, anti_aliasing=True) for image in image_list]
    if rotate:
        rotated_images = [np.rot90(image) for image in resized_images]
        f = visualize(rotated_images, title=title, cmap="Greens_r", show_or_plot="show", dict=ratio)
    else:
        f = visualize(resized_images, title=title, cmap="Greens_r", show_or_plot="show", dict=ratio)


def get_all_info(
    path=None,
    images=None,
    idx=None,
    energy_kev=None,
    detector_pixel_size=None,
    distance_sample_detector=None,
    alpha=1e-8,
    delta_beta=1e1,
    pad=2,
    method="TIE",
    file_type="tif",
    image=None,
    **kwargs,
):
    """
    make sure that the unit of energy is in keV, the unit of detector_pixel_size is in meter, and the unit of distance_sample_detector is in meter
    """
    if idx is not None and type(idx) is not list:
        idx = [idx]
    else:
        idx = [0]

    if images is not None:
        image = [images[i] for i in idx]

    if "image_path" in kwargs.keys():
        image_path = kwargs["image_path"]
    else:
        image_path = None

    if "phase" in kwargs.keys():
        phase = kwargs["phase"]
        attenuation = kwargs["attenuation"]
    elif "phase_image" in kwargs.keys():
        phase = kwargs["phase_image"]
        attenuation = kwargs["attenuation_image"]
    else:
        phase = None
        attenuation = None

    if "z" in kwargs.keys():
        distance_sample_detector = kwargs["z"]
    if "pv" in kwargs.keys():
        detector_pixel_size = kwargs["pv"]
    if "energy" in kwargs.keys():
        energy_kev = kwargs["energy"]

    if "mode" in kwargs.keys():
        mode = kwargs["mode"]
    else:
        mode = "reflect"
    if "value" in kwargs.keys():
        value = kwargs["value"]
    else:
        value = "mean"

    if type(distance_sample_detector) is list:
        distance_sample_detector = (
            [distance_sample_detector[i] for i in idx] if idx is not None else distance_sample_detector
        )
    if type(detector_pixel_size) is list:
        detector_pixel_size = [detector_pixel_size[i] for i in idx] if idx is not None else detector_pixel_size
        assert len(detector_pixel_size) == len(
            distance_sample_detector
        ), "detector_pixel_size and distance_sample_detector must have the same length"
    if type(energy_kev) is list:
        energy_kev = [energy_kev[i] for i in idx] if idx is not None else energy_kev
    if type(pad) is list:
        pad = [pad[i] for i in idx] if idx is not None else pad
    lam = wavelength_from_energy(energy_kev).magnitude
    if "fresnel_number" in kwargs.keys():
        fresnel_number = kwargs["fresnel_number"]
    else:
        fresnel_number = (
            [
                fresnel_calculator(energy_kev, lam, detector_pixel_size, distance)
                for distance in distance_sample_detector
            ]
            if type(distance_sample_detector) is list
            else fresnel_calculator(energy_kev, lam, detector_pixel_size, distance_sample_detector)
        )
    if "fresnel_factor" in kwargs.keys():
        fresnel_factor = kwargs["fresnel_factor"]
    else:
        fresnel_factor = None

    if path is not None:
        # if path is a folder
        if type(path) is str:
            if os.path.isdir(path):
                images = list(io.imread_collection(path + "/*." + file_type).files)
                image_path = [images[i] for i in idx]
                image = load_images_parallel(image_path)
            elif os.path.isfile(path):
                images = io.imread(path)
                image_path = path
                if len(images.shape) == 2:
                    image = images
                elif len(images.shape) == 3:
                    image = images[idx, :, :]
                else:
                    image = images[idx, :, :, :]
                image = [image]

        elif type(path) is list:
            if os.path.isfile(path[0]):
                image_path = [path[i] for i in idx]
                image = load_images_parallel(image_path)
            if os.path.isdir(path[0]):
                folders = path
                images = []
                for folder in folders:
                    images += list(io.imread_collection(folder + "/*." + file_type).files)
                image_path = [images[i] for i in idx]
                image = load_images_parallel(image_path)

        elif type(path) is np.array:
            image_path = os.getcwd()
            images = [path]
            image = [path]

        elif "ImageCollection" in str(type(path)):
            image_path = path.files
            image_path = [image_path[i] for i in idx]
            images = load_images_parallel(image_path)
            image = [images[i] for i in idx]

        else:
            if type(path) is not list:
                images = [path]
            image = [images[i] for i in idx]
            try:
                image = load_images_parallel(image)
            except:
                pass
    else:
        if image is None:
            assert phase is not None and attenuation is not None, "phase and attenuation must be given"
            attenuation = attenuation
            from ganrec_dataloader import forward_propagate, tensor_to_np

            shape_x, shape_y = phase.shape
            fresnel_factor = ffactors(
                shape_x * pad, shape_y * pad, energy_kev, distance_sample_detector, detector_pixel_size
            )
            if type(distance_sample_detector) == list:
                image = [
                    tensor_to_np(
                        forward_propagate(
                            shape_x=shape_x,
                            shape_y=shape_y,
                            pad=pad,
                            energy_kev=energy_kev,
                            detector_pixel_size=detector_pixel_size,
                            distance_sample_detector=distance,
                            phase_image=phase,
                            attenuation_image=attenuation,
                            fresnel_factor=fresnel_factor[i],
                            wavefield=None,
                            distance=distance,
                            mode=mode,
                            value=value,
                        )
                    )
                    for i, distance in enumerate(distance_sample_detector)
                ]
            else:
                image = tensor_to_np(
                    forward_propagate(
                        shape_x=shape_x,
                        shape_y=shape_y,
                        pad=pad,
                        energy_kev=energy_kev,
                        detector_pixel_size=detector_pixel_size,
                        distance_sample_detector=distance_sample_detector,
                        phase_image=phase,
                        attenuation_image=attenuation,
                        fresnel_factor=fresnel_factor,
                        wavefield=None,
                        distance=distance_sample_detector,
                        mode=mode,
                        value=value,
                    )
                )

    if image is not None:
        if type(image) is list:
            ND = len(image)
            if len(image[0].shape) == 2:
                shape_x, shape_y = image[0].shape
                Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps=detector_pixel_size)
                ND = 1
                image_path = os.getcwd()
            else:
                shape_x, shape_y = image[0].shape[1:]
                Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps=detector_pixel_size)
        else:
            ND = 1
            shape_x, shape_y = image.shape
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps=detector_pixel_size)
        if image_path is None:
            image_path = os.getcwd()
        if images is None:
            images = image

        if "correct" in kwargs.keys():
            if kwargs["correct"] == True:
                if "mean_dark_image" and "mean_ref_image" in kwargs.keys():
                    mean_dark_image = kwargs["mean_dark_image"]
                    mean_ref_image = kwargs["mean_ref_image"]
                else:
                    all_images = list(io.imread_collection(path + "/*." + file_type).files)
                    mean_ref_image = np.mean(
                        io.imread_collection([im_name for im_name in all_images if "ref" in im_name]), axis=0
                    )
                    mean_dark_image = np.mean(
                        io.imread_collection([im_name for im_name in all_images if "dar" in im_name]), axis=0
                    )

                if len(image) > 1:
                    image = [
                        (image[i] - mean_dark_image) / (mean_ref_image - mean_dark_image)
                        for i in range(len(image))
                    ]
                else:
                    image = (image - mean_dark_image) / (mean_ref_image - mean_dark_image)

        if "downsampling_factor" not in kwargs.keys() or kwargs["downsampling_factor"] == None:
            kwargs["downsampling_factor"] = 1
        else:
            image = [
                resize(
                    image[i],
                    (
                        image[i].shape[0] // kwargs["downsampling_factor"],
                        image[i].shape[1] // kwargs["downsampling_factor"],
                    ),
                    anti_aliasing=True,
                )
                for i in range(len(image))
            ]
            shape_x, shape_y = image[0].shape
            detector_pixel_size = detector_pixel_size * kwargs["downsampling_factor"]
            fresnel_factor = ffactors(
                shape_x * pad, shape_y * pad, energy_kev, distance_sample_detector, detector_pixel_size
            )
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps=detector_pixel_size)

        if fresnel_factor is None:
            fresnel_factor = ffactors(
                shape_x * pad, shape_y * pad, energy_kev, distance_sample_detector, detector_pixel_size
            )

        print(ND, image[0].shape)
        kwargs = {
            "path": path,
            "output_path": os.getcwd(),
            "idx": idx,
            "column_name": "path",
            "energy_J": energy_kev_to_joule(energy_kev),
            "energy_kev": energy_kev,
            "lam": lam,
            "detector_pixel_size": detector_pixel_size,
            "distance_sample_detector": distance_sample_detector,
            "fresnel_number": fresnel_number,
            "wave_number": wave_number(energy_kev),
            "downsampling_factor": kwargs["downsampling_factor"],
            "shape_x": shape_x,
            "px": shape_x,
            "shape_y": shape_y,
            "py": shape_y,
            "pad_mode": mode,
            "shape": [int(shape_x), int(shape_y)],
            "distance": [distance_sample_detector],
            "z": distance_sample_detector,
            "energy": energy_kev,
            "alpha": alpha,
            "pad": pad,
            "nfx": int(shape_x) * pad,
            "nfy": int(shape_y) * pad,
            "pv": detector_pixel_size,
            "pixel_size": [detector_pixel_size, detector_pixel_size],
            "sample_frequency": [1.0 / detector_pixel_size, 1.0 / detector_pixel_size],
            "fx": Fx,
            "fy": Fy,
            "method": method,
            "delta_beta": delta_beta,
            "fresnel_factor": fresnel_factor,
            "i_input": image[0],
            "image_path": image_path,
            "image": image,
            "all_images": images,
            "ND": ND,
            "fresnel_factor": fresnel_factor,
            "phase": phase,
            "attenuation": attenuation,
        }
        kwargs.update(save_path_generator(**kwargs))
        return kwargs

    else:

        assert path is not None
        if type(path) is str:
            assert os.path.exists(path), "path does not exist"
            if os.path.isdir(path):
                images = list(io.imread_collection(path + "/*." + file_type).files)
                image_path = [images[i] for i in idx]
                image = load_images_parallel(image_path)
            else:
                images = list(path)
                image = load_image(path)
            get_all_info(image=image, **kwargs)
        elif type(path) is list:
            path = [path[i] for i in idx]
            if type(path[0]) is str:
                # if the path[0] is a folder
                if os.path.isdir(path[0]):
                    images = list(io.imread_collection(path + "/*." + file_type).files)
                    image_path = [images[i] for i in idx]
                else:
                    image_path = path
                image = load_images_parallel(image_path)
            else:
                image = [path[i] for i in idx]
            get_all_info(image=image, **kwargs)
        else:
            images = path
            image = [images[i] for i in idx]
            get_all_info(image=image, **kwargs)


def load_image(url):

    img = io.imread(url)
    return img


def load_images_parallel(urls=[]):
    if urls == []:
        return None
    """using concurrent.futures"""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_image, urls)
        images = list(results)
    return images


def resize_images_parallel(images=[], shape=(512, 512)):
    if images == []:
        return None
    """using concurrent.futures"""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(resize, images, [shape] * len(images))
    return list(results)


def fun_images_parallel(images, fun, attribute):
    if images == []:
        return None

    if type(images) is not list:
        images = [images]

    if type(attribute) is not list:
        attribute = [attribute] * len(images)

    # change fun from string to function
    if type(fun) is str:
        fun = eval(fun)

    """using concurrent.futures"""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fun, images, attribute)
    return list(results)


def load_dxchange(paths=[]):
    """using dxchange"""
    import dxchange

    if type(paths) == str:
        paths = [paths]
    n = len(paths)
    if n == 1:
        image = dxchange.read_tiff(paths[0])
    else:
        image = dxchange.read_tiff_stack(paths, ind=range(n))
    return image


def wave_transform(phase, attenuation, wave_number):
    return np.exp(1j * phase * wave_number) * np.exp(-attenuation * wave_number)


def propagation_of_image_j_at_distance_i(df, i, j):
    return io.imread(df[df.columns[i]][j])


def detector_image(x):
    # this is what a detector sees (only intensities). float32 is needed for the fft
    return np.real(x * np.conj(x)).astype(np.float32)


def consquative_detections(df, distance_numbers=5, idx=0, interactive_display=False):
    ground_wave = wave_transform(io.imread(df.phase[idx]), io.imread(df.attenuation[idx]), df.wave_number[idx])
    propagations_at_distance = []
    propagations_at_distance.append(ground_wave)
    for i in range(3, distance_numbers + 3):
        propagations_at_distance.append(propagation_of_image_j_at_distance_i(df, i, idx))
    if interactive_display == False:
        return propagations_at_distance
    if interactive_display == True:
        visualize_interact(propagations_at_distance)
        return propagations_at_distance


def energy_kev_to_joule(energy_kev):
    """converts energy in kev to joules"""
    return energy_kev * 1.60217662e-16


def fresnel_calculator(energy_kev=None, lam=None, detector_pixel_size=None, distance_sample_detector=None):
    """calculates the fresnel number, the unit of energy must be in kev, and the unit of the other parameters must be in meters"""
    if energy_kev is not None:
        lam = 6.626 * 10 ** (-34) * 299792458 / energy_kev_to_joule(energy_kev)
    assert detector_pixel_size is not None, "detector_pixel_size must be given"
    assert distance_sample_detector is not None, "distance_sample_detector must be given"
    return detector_pixel_size**2 / (lam * distance_sample_detector)


def eneryg_J(energy):
    if type(energy) == pq.quantity.Quantity:
        if energy.dimensionality == pq.Quantity(1, "keV").dimensionality:
            energy = energy.rescale("J")
        elif energy.dimensionality == pq.Quantity(1, "J").dimensionality:
            energy = energy
        elif energy.dimensionality == pq.Quantity(1, "eV").dimensionality:
            energy = energy.rescale("J")
        elif energy.dimensionality == pq.Quantity(1, "m").dimensionality:
            wavelength = energy
            energy = energy_from_wavelength(wavelength)
    else:
        if type(energy) == str:
            energy = float(energy)
            energy = pq.Quantity(energy, "keV").rescale("J")
        elif type(energy) == int:
            energy = float(energy)
            energy = pq.Quantity(energy, "keV").rescale("J")
        elif type(energy) == float:
            energy = energy
            energy = pq.Quantity(energy, "J")
        else:
            energy = pq.Quantity(energy, "keV").rescale("J")
    return energy


def wavelength_m(lam):
    if type(lam) == pq.quantity.Quantity:
        if lam.dimensionality == pq.Quantity(1, "m").dimensionality:
            lam = lam.rescale("m")
        elif lam.dimensionality == pq.Quantity(1, "nm").dimensionality:
            lam = lam.rescale("m")
        elif lam.dimensionality == pq.Quantity(1, "A").dimensionality:
            lam = lam.rescale("m")
        elif lam.dimensionality == pq.Quantity(1, "keV").dimensionality:
            lam = lam.rescale("m")
    else:
        lam = pq.Quantity(lam, "m")
    return lam


def wavelength_from_energy(energy):
    h = pq.Quantity(6.62607015 * 10**-34, "J*s")
    c = pq.Quantity(299792458, "m/s")
    energy = eneryg_J(energy)
    return h * c / energy


def energy_from_wavelength(lam):
    h = pq.Quantity(6.62607015 * 10**-34, "J*s")
    c = pq.Quantity(299792458, "m/s")
    lam = wavelength_m(lam)
    energy = h * c / lam
    return energy.rescale("keV")


def wave_number(energy):
    lam = wavelength_from_energy(energy)
    wavenumber = 2 * np.pi / lam
    return wavenumber


def energy_from_wave_number(wave_number):
    lam = 2 * np.pi / wave_number
    return energy_from_wavelength(lam)


def shorten(string):
    if "e" in string:
        left = string.split("e")[0][:7]
        right = string.split("e")[1][:7]
        return left + "e" + right
    else:
        if "." in string:
            count = 0
            for i in range(len(string.split(".")[1])):
                if string[i] == "0":
                    count += 1
            return string[: count + 5]
        else:
            return string[:7]


def give_title(image, title="", idx="", min_max=True):
    if min_max:
        min_val_orig = np.min(image)
        max_val_orig = np.max(image)
        txt_min_val = shorten(str(min_val_orig))
        txt_max_val = shorten(str(max_val_orig))
    else:
        txt_min_val = ""
        txt_max_val = ""
    title = "im=" + str(idx + 1) if title == "" else title
    return title + " (" + txt_min_val + ", " + txt_max_val + ")"


def give_titles(images, titles=[], min_max=True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title=titles[i], idx=i, min_max=min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max=min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title=titles[i], idx=i, min_max=min_max) for i in range(len(images))]
    return titles


def get_row_col(images, show_all=False):
    if show_all:
        rows = int(np.sqrt(len(images)))
        cols = int(np.sqrt(len(images)))
        return rows, cols + (len(images) - rows * cols) // rows

    if len(images) == 1:
        rows = 1
        cols = 1
    elif len(images) <= 5:
        rows = 1
        cols = len(images)
    else:
        rows = 2
        cols = len(images) // 2
    if rows * cols > len(images):
        images = images[: rows * cols - int(rows * cols / len(images))]
        rows, cols = get_row_col(images)
    print("rows: ", rows, "cols: ", cols)
    return rows, cols


def chose_fig(images, idx=None, rows=None, cols=None, show_all=False, add_length=None):
    (rows, cols) = get_row_col(images) if rows is None or cols is None else (rows, cols)
    shape = images[0].shape
    if shape[0] > 260:
        fig_size = (shape[1] * cols / 100 + 1, shape[0] * rows / 100)
    elif shape[0] > 100 and shape[0] <= 260:
        fig_size = (shape[1] * cols / 50 + 1, shape[0] * rows / 50)
    else:
        fig_size = (shape[1] * cols / 25 + 1, shape[0] * rows / 25)
    if add_length is not None:
        fig_size = (fig_size[0] + add_length, fig_size[1])
    fig, ax = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    ax.reshape(rows, cols)
    if rows == 1 and cols == 1:
        return fig, ax, rows, cols, fig_size
    elif rows == 1:
        ax = ax.reshape(1, cols)
        return fig, ax, rows, cols, fig_size
    elif cols == 1:
        ax = ax.reshape(rows, 1)
        return fig, ax, rows, cols, fig_size
    else:
        return fig, ax, rows, cols, fig_size


def get_setup_info(dict={}):
    # rearrange them in a descending order based on length
    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: len(item[0]) + len(str(item[1])), reverse=True)}
    len_line = 0
    for key, value in dict.items():
        if type(value) == str or type(value) == int or type(value) == float or type(value) == bool:
            if len(key) > len_line:
                len_line = len(key)
        elif type(value) == np.ndarray:
            if len(value.shape) == 0:
                if len(key) > len_line:
                    len_line = len(key)
        else:
            try:
                from ganrec_dataloader import tensor_to_np

                if type(tensor_to_np(value)) == np.ndarray and len(tensor_to_np(value).shape) == 0:
                    if len(key) > len_line:
                        len_line = len(key)
            except:
                pass
    len_line += 10
    line = "_" * len_line
    information = line + "\n"
    for key, value in dict.items():
        if type(value) == str or type(value) == int or type(value) == float or type(value) == bool:
            information += "| " + key + ": " + str(value) + " \n"
        elif type(value) == np.ndarray and len(value.shape) == 0:
            information += "| " + key + ": " + str(value) + " \n"
        else:
            try:
                from ganrec_dataloader import tensor_to_np

                if type(tensor_to_np(value)) == np.ndarray and len(tensor_to_np(value).shape) == 0:
                    information += "| " + key + ": " + str(tensor_to_np(value)) + " \n"
            except:
                pass
    information += line + " \n"
    print(information)
    return information, len_line


def get_file_nem(dict):
    name = ""
    important_keys = [
        "experiment_name",
        "abs_ratio",
        "iter_num",
        "downsampling_factor",
        "l1_ratio",
        "contrast_ratio",
        "normalized_ratio",
        "brightness_ratio",
        "contrast_normalize_ratio",
        "brightness_normalize_ratio",
        "l2_ratio",
        "fourier_ratio",
    ]
    for key in important_keys:
        if key in dict.keys():
            name += key + "_" + str(dict[key]) + "__"
    return name


def create_table_info(dict={}):
    import pandas as pd

    df = pd.DataFrame()
    for key, value in dict.items():
        if type(value) != np.ndarray:
            df[key] = [value]
        elif type(value) == np.ndarray and len(value.shape) == 0:
            df[key] = [value]
    df = df.T
    # create a plot with the information
    fig, ax = plt.subplots(figsize=(20, 10))
    # make the rows and columns look like a table
    ax.axis("tight")
    ax.axis("off")
    # create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", rowLabels=df.index, cellLoc="center")
    # change the font size
    table.set_fontsize(14)
    # change the cell height
    table.scale(1, 2)

    return df, ax, table


def give_titles(images, titles=[], min_max=True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title=titles[i], idx=i, min_max=min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max=min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title=titles[i], idx=i, min_max=min_max) for i in range(len(images))]
    return titles


def val_from_images(image, type_of_image="nd.array"):
    if "ndarray" in str(type_of_image):
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j, :, :] for j in range(len(image))]
        else:
            val = [image[j, 0, :, :] for j in range(len(image))]
    elif "Tensor" in str(type_of_image):
        from ganrec_dataloader import tensor_to_np

        image = tensor_to_np(image)
        if type(image) is not list:
            if len(image.shape) == 2:
                val = image
            elif len(image.shape) == 3:
                val = [image[j, :, :] for j in range(len(image))]
            elif len(image.shape) == 4:
                val = [image[j, 0, :, :] for j in range(len(image))]
            elif len(image.shape) == 1:
                val = image
        else:
            val = image
    elif type_of_image == "str":
        val = io.imread_collection(image)
    elif "collection" in str(type_of_image):
        val = image
    elif "list" in str(type_of_image):
        val = [val_from_images(image, type_of_image=type(image)) for image in image]
    else:
        print(type_of_image)
        assert False, "type_of_image is not nd.array, list or torch.Tensor"
    return val


def convert_images(images, idx=None):
    if idx is not None:
        images = [images[i] for i in idx]
    if type(images) is list:
        vals = [val_from_images(image, type_of_image=type(image)) for image in images]

        for i, val in enumerate(vals):
            if type(val) is list:
                [vals.append(val[j]) for j in range(len(val))]
                vals.pop(i)
        images = vals
    else:
        images = val_from_images(images, type_of_image=type(images))
    for i, val in enumerate(images):
        if type(val) is list:
            [images.append(val[j]) for j in range(len(val))]
            images.pop(i)
    return images


def visualize(
    images,
    idx=None,
    rows=None,
    cols=None,
    show_or_plot="show",
    cmap="coolwarm",
    title="",
    axis="on",
    plot_axis="half",
    min_max=True,
    dict=None,
    save_path=None,
):
    """
    Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    """
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    images = convert_images(images, idx)
    titles = give_titles(images, title, min_max)
    shape = images[0].shape

    if dict is not None:
        description_title, add_length = get_setup_info(dict)
    else:
        add_length = None
    fig, ax, rows, cols, fig_size = chose_fig(images, idx, rows, cols, add_length)

    if show_or_plot == "plot":
        if plot_axis == "half":
            [ax[i, j].plot(images[i * cols + j][shape[0] // 2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].plot(images[i * cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
    elif show_or_plot == "both":
        [ax[i, j].imshow(images[i * cols + j], cmap=cmap) for i in range(rows) for j in range(cols)]
        if plot_axis == "half":
            [
                ax[i, j].twinx().plot(images[i * cols + j][shape[0] // 2, :])
                for i in range(rows)
                for j in range(cols)
            ]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].twinx().plot(images[i * cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]

    [ax[i, j].axis(axis) for i in range(rows) for j in range(cols)]
    [ax[i, j].set_title(titles[i * cols + j], fontsize=12) for i in range(rows) for j in range(cols)]
    plt.tight_layout()
    if show_or_plot != "plot":
        [
            fig.colorbar(ax[i, j].imshow(images[i * cols + j], cmap=cmap), ax=ax[i, j])
            for i in range(rows)
            for j in range(cols)
        ]
    fig.patch.set_facecolor("xkcd:light grey")

    if dict is not None:
        fig.subplots_adjust(left=add_length / 150)
        fig.suptitle(
            description_title, fontsize=10, y=0.95, x=0.05, ha="left", va="center", wrap=True, color="blue"
        )
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return fig


def sns_visualize(
    images,
    idx=None,
    rows=None,
    cols=None,
    show_or_plot="show",
    cmap="coolwarm",
    title="",
    axis="off",
    plot_axis="half",
):
    """
    Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    """

    import seaborn as sns

    images = convert_images(images, idx)
    titles = give_titles(images, title)
    shape = images[0].shape
    fig, ax, rows, cols = chose_fig(images, idx, rows, cols)

    if rows == 1 and cols == 1:
        if show_or_plot == "plot":
            if plot_axis == "half":
                ax.plot(images[0][shape[0] // 2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.plot(images[0][plot_axis, :])
            ax.set_title("y:" + str(plot_axis) + " " + titles[0], fontsize=12)
        elif show_or_plot == "both":
            if plot_axis == "half":
                ax.twinx().plot(images[0][shape[0] // 2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.twinx().plot(images[0][plot_axis, :])
            ax.set_title("y:" + str(plot_axis) + " " + titles[0], fontsize=12)
        else:
            ax.set_title(titles[0], fontsize=12)
            ax.imshow(images[0], cmap=cmap)

        ax.axis(axis)
        fig.colorbar(ax.imshow(images[0]), ax=ax)
        plt.show()
        return fig
    if show_or_plot == "show":
        [
            sns.heatmap(images[i * cols + j], cmap=cmap, ax=ax[i, j], robust=True)
            for i in range(rows)
            for j in range(cols)
        ]
        [ax[i, j].set_title(titles[i * cols + j], fontsize=12) for i in range(rows) for j in range(cols)]
    elif show_or_plot == "plot":
        if plot_axis == "half":
            [ax[i, j].plot(images[i * cols + j][shape[0] // 2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].plot(images[i * cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
        [
            ax[i, j].set_title("y:" + str(plot_axis) + " " + titles[i * cols + j], fontsize=12)
            for i in range(rows)
            for j in range(cols)
        ]
    elif show_or_plot == "both":
        [sns.heatmap(images[i * cols + j], cmap=cmap, ax=ax[i, j]) for i in range(rows) for j in range(cols)]
        if plot_axis == "half":
            [
                ax[i, j].twinx().plot(images[i * cols + j][shape[0] // 2, :])
                for i in range(rows)
                for j in range(cols)
            ]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].twinx().plot(images[i * cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
        [
            ax[i, j].set_title("y:" + str(plot_axis) + " " + titles[i * cols + j], fontsize=12)
            for i in range(rows)
            for j in range(cols)
        ]
    else:
        assert False, "show_or_plot is not 'show', 'plot' or 'both'"
    [ax[i, j].axis(axis) for i in range(rows) for j in range(cols)]
    plt.tight_layout()
    fig.patch.set_facecolor("xkcd:light blue")
    plt.show()
    return fig


def visualize_interact(pure=[]):
    import ipywidgets as widgets
    from ipywidgets import interact
    from IPython.display import display

    interact(
        visualize,
        pure=widgets.fixed(pure),
        show_or_plot=widgets.Dropdown(options=["show", "plot"], value="show", description="Show or plot:"),
        rows=widgets.IntSlider(min=1, max=10, step=1, value=1, description="Rows:"),
        cols=widgets.IntSlider(min=1, max=10, step=1, value=3, description="Columns:"),
    )


def plot_pandas(df, column_range=None, x_column="abs_ratio", titles=None):
    """
    this function plots the metadata dataframe
    """
    if column_range is None:
        column_range = df.columns[2:-1]
    elif column_range == "all":
        column_range = df.columns
    elif type(column_range) is str:
        column_range = [column_range]
    elif type(column_range) is int:
        column_range = df.columns[column_range:-1]
    fig = plt.figure(figsize=(20, 20))
    min_vals = [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]
    max_vals = [df[column].max() for column in column_range], [df[column].idxmax() for column in column_range]
    if titles is None:
        # titles = [column + '\nmin = ' + str(min_per_column[i])+' at ' + str(df[column].idxmin()) +'\n max = ' + str(df[column].max())+' at ' + str(df[column].idxmax()) for i, column in enumerate(column_range)]
        titles = [
            column
            + "\nmin = "
            + str(min_vals[0][i])
            + " at "
            + str(min_vals[1][i])
            + "\n max = "
            + str(max_vals[0][i])
            + " at "
            + str(max_vals[1][i])
            for i, column in enumerate(column_range)
        ]
    for i, column in enumerate(column_range):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.plot(df[x_column], df[column])
        ax.set_xlabel(x_column)
        ax.set_ylabel(column)
        ax.set_title(titles[i])

    # return the minimum of each column and the corresponding value of x_column and the index in 0, .. format
    return min_vals, max_vals

    # #return the minimum of each column and the corresponding value of x_column
    # return [df[column].min() for column in column_range], [df[column].idxmin() for column in column_range]


def time_to_string(time):
    if time > 60:
        if time > 3600:
            if time > 3600 * 24:
                return (
                    str(int(time // (3600 * 24)))
                    + " days "
                    + str(int((time % (3600 * 24)) // 3600))
                    + " hours "
                    + str(int((time % 3600) // 60))
                    + " minutes "
                    + str(int(time % 60))
                    + " seconds"
                )
            else:
                return (
                    str(int(time // 3600))
                    + " hours "
                    + str(int((time % 3600) // 60))
                    + " minutes "
                    + str(int(time % 60))
                    + " seconds"
                )
        else:
            return str(int(time // 60)) + " minutes " + str(int(time % 60)) + " seconds"
    else:
        return str(int(time % 60)) + " seconds"


def get_list_of_possibilities(value, gap=None, number_of_elements=None):
    if gap is None:
        gap = value * 0.1
    if number_of_elements is None:
        number_of_elements = 6
    values = [value - gap * (i + 1) for i in range(number_of_elements // 2)]
    values2 = [value + gap * (i + 1) for i in range(number_of_elements // 2)]
    values.extend([value])
    values.extend(values2)
    values.sort()
    return values
