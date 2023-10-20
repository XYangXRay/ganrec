import numpy as np
from numpy.fft import fftfreq
from numpy import pi 
import matplotlib.pyplot as plt
import os 
import skimage.io as io
import seaborn as sns

def nor_phase(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    return img


def ffactor(px, py, energy, z, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv ** 2
    freq_1 = fftfreq(px)
    freq_2 = fftfreq(py)
    xi, eta = np.meshgrid(freq_1, freq_2)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    h = np.exp(- 1j * frequ_prefactor * (xi ** 2 + eta ** 2) / 2)
    return h

def ffactors(px, py, energy, zs, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactors = [2 * np.pi  * lambda_p * zs[i] / pv ** 2 for i in range(len(zs))]
    freq_x = fftfreq(px)
    freq_y = fftfreq(py)
    xi, eta = np.meshgrid(freq_x, freq_y)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    h = [((np.exp(- 1j * frequ_prefactors[i] * (xi ** 2 + eta ** 2) / 2)).T).astype('complex64') for i in range(len(zs))]
    return h


def fresnel_operator(px, py, pv, z, lambda0, upsample_scale):
    # lambda0 = 1.23984122e-09 / energy
    # Scale by which to upsample image
    nx = upsample_scale * px # Image width in pixels (same as height)
    ny = upsample_scale * py # Image height in pixels
    grid_size_x = pv * nx;                 # Grid size in x-direction
    grid_size_y = pv * ny;                 # Grid size in y-direction
    # Inverse space
    fx = np.linspace(-(nx-1)/2*(1/grid_size_x), (nx-1)/2*(1/grid_size_x), nx)
    fy = np.linspace(-(ny-1)/2*(1/grid_size_y), (ny-1)/2*(1/grid_size_y), ny)
    Fx, Fy = np.meshgrid(fx, fy)
    H = np.exp(1j*(2 * pi / lambda0) * z) * np.exp(1j * pi * lambda0 * z * (Fx**2 + Fy**2))
    return H.T

class RECONmonitor:
    def __init__(self, recon_target):
        self.fig, self.axs = plt.subplots(2, 3, figsize=(23, 8))
        self.recon_target = recon_target
        if self.recon_target == 'tomo':
            self.plot_txt = 'Sinogram'
        elif self.recon_target == 'phase':
            self.plot_txt = 'Intensity'

    def initial_plot(self, img_input):
        px, py = img_input.shape
        self.im0 = self.axs[0, 0].imshow(img_input, cmap='gray')
        self.axs[0, 0].set_title(self.plot_txt)
        self.fig.colorbar(self.im0, ax=self.axs[0, 0])
        self.axs[0, 0].set_aspect('equal','box')
        self.im1 = self.axs[1, 0].imshow(img_input, cmap='jet')
        self.tx1 = self.axs[1, 0].set_title('Difference of ' + self.plot_txt + ' for iteration 0')
        self.fig.colorbar(self.im1, ax=self.axs[1, 0])
        self.axs[0, 0].set_aspect('equal')
        self.im2 = self.axs[0, 1].imshow(np.zeros((px, py)), cmap='gray')
        self.fig.colorbar(self.im2, ax=self.axs[0, 1])
        self.axs[0, 1].set_title('retrieved phase')
        self.im3, = self.axs[1, 1].plot([], [], 'r-')
        self.axs[1, 1].set_title('Generator loss')
        self.axs[0, 2].set_title('plot profile of input')
        self.axs[0, 2].plot(img_input[int(px/2), :], 'b-')
        self.axs[0, 2].set_title('plot profile of input')
        self.im4 = self.axs[1, 2].plot([], 'r-')
        self.axs[1, 2].set_title('plot profile of recon')
        
        
        plt.tight_layout()

    def update_plot(self, epoch, img_diff, img_rec, plot_x, plot_loss, save_path = None):
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
        self.axs[1, 2].plot(img_rec[int(img_rec.shape[0]/2), :], 'r-')
        plt.pause(0.1)

    def close_plot(self):
        plt.close()


def grid_generator(shape_x, shape_y, upscale = 1, ps = 5.5e-06):
    """ 
    Parameters: shape_y - shape of the image in y-direction
    #             upscale - scale by which to upsample image
    #             ps - pixel size in microns
    """             
    upsample_scale = upscale;                 # Scale by which to upsample image
    nx = upsample_scale * shape_x # Image width in pixels (same as height)
    ny = upsample_scale * shape_y
    grid_size_x = ps * nx;                 # Grid size in x-direction
    grid_size_y = ps * ny;                 # Grid size in y-direction
    fx = np.linspace(-(nx-1)/2*(1/grid_size_x), (nx-1)/2*(1/grid_size_x), nx)
    fy = np.linspace(-(ny-1)/2*(1/grid_size_y), (ny-1)/2*(1/grid_size_y), ny)
    Fx, Fy = np.meshgrid(fx, fy)
    
    return Fx, Fy


def save_path_generator(**kwargs):
    try:
        file_name = os.path.splitext(os.path.basename(kwargs['image_path']))[0]
        #the folder name of the 'image_path'
        folder = os.path.basename(os.path.dirname(kwargs['image_path']))
    except:
        file_name = os.path.splitext(os.path.basename(kwargs['image_path'][0]))[0]
        folder = os.path.basename(os.path.dirname(kwargs['image_path'][0]))
        
    init_path = os.getcwd() + '/data/saved_weights/' + folder + '/'
    save_wpath = os.getcwd() + '/data/retrieved/' + folder + '/'
    if not os.path.exists(init_path):
        os.makedirs(init_path)
    if not os.path.exists(save_wpath):
        os.makedirs(save_wpath)
    kwargs['file_name'] = file_name
    kwargs['save_wpath'] = save_wpath
    kwargs['init_wpath'] = init_path
    return kwargs 


def segment(image, type = 'chan_vese'):
    """
    types: chan_vese, sobel, otsu, local, minimum, mean, triangle, yen, multiotsu, isodata, li
    """
    from skimage.util import img_as_ubyte
    if type == 'chan_vese':        
        from skimage.segmentation import chan_vese
        return chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200,
                         dt=0.5, init_level_set="checkerboard", extended_output=True)
    elif type == 'sobel':
        from skimage.filters import sobel
        return sobel(image)
    elif type == 'otsu':
        from skimage.filters import threshold_otsu
        return img_as_ubyte(image > threshold_otsu(image))
    elif type == 'local':
        from skimage.filters import threshold_local
        return img_as_ubyte(image > threshold_local(image, block_size=35, offset=10))
    elif type == 'minimum':
        from skimage.filters import threshold_minimum
        return img_as_ubyte(image > threshold_minimum(image))
    elif type == 'mean':
        from skimage.filters import threshold_mean
        return img_as_ubyte(image > threshold_mean(image))
    elif type == 'triangle':
        from skimage.filters import threshold_triangle
        return img_as_ubyte(image > threshold_triangle(image))
    elif type == 'yen':
        from skimage.filters import threshold_yen
        return img_as_ubyte(image > threshold_yen(image))
    elif type == 'multiotsu':
        from skimage.filters import threshold_multiotsu
        return img_as_ubyte(image > threshold_multiotsu(image))
    elif type == 'isodata':
        from skimage.filters import threshold_isodata
        return img_as_ubyte(image > threshold_isodata(image))
    elif type == 'li':
        from skimage.filters import threshold_li
        return img_as_ubyte(image > threshold_li(image))
    else:
        print('wrong type')
        return image


def get_all_info(path = None, images = None, idx = 1000, energy_kev = 18.0, detector_pixel_size = 2.57 * 1e-6, distance_sample_detector = 0.15, alpha = 1e-8, delta_beta = 1e1, pad = 1, method = 'TIE', file_type = 'tif', image = None, **kwargs):
    """
    make sure that the unit of energy is in keV, the unit of detector_pixel_size is in meter, and the unit of distance_sample_detector is in meter
    """
    if idx is not None and type(idx) is not list:
        idx = [idx]
    
    if images is not None:
        image = [images[i] for i in idx]
    
    if 'image_path' in kwargs.keys():
        image_path = kwargs['image_path']
    else:
        image_path = None

    phase = None if 'phase' not in kwargs.keys() else kwargs['phase']
    attenuation = None if 'attenuation' not in kwargs.keys() else kwargs['attenuation']    
        
    if path is not None:
        
        #if path is a folder
        if type(path) is str and os.path.isdir(path):
            images = list(io.imread_collection(path + '/*.' + file_type).files)
            image_path = [images[i] for i in idx]
            image = load_images_parallel(image_path)
        #if path is a list of files
        elif type(path) is list and not os.path.isfile(path[0]):
            image_path = [path[i] for i in idx]
            image = load_images_parallel(image_path)
        elif type(path) is list and os.path.isdir(path[0]):
            folders = path
            images = []
            for folder in folders:
                images += list(io.imread_collection(folder + '/*.' + file_type).files)
            image_path = [images[i] for i in idx]
            image = load_images_parallel(image_path)
        elif type(path) is np.array:
            image_path = os.getcwd()
            images = path
            image = [images[i] for i in idx]
        else:
            image = [path[i] for i in idx]
            try:
                image = load_images_parallel(image)
            except:
                pass
                
    else:
        phase = None if 'phase' not in kwargs.keys() else kwargs['phase']
        attenuation = None if 'attenuation' not in kwargs.keys() else kwargs['attenuation']
        if phase is not None and attenuation is not None:
            phase = np.zeros((attenuation.shape[0], attenuation.shape[1])) if phase is None and attenuation is not None else phase
            attenuation = np.zeros((phase.shape[0], phase.shape[1])) if attenuation is None and phase is not None else attenuation
            print("phase shape", phase.shape)
            shape_x, shape_y = phase.shape
            fresnel_factor = ffactor(shape_x*pad, shape_y*pad, energy_kev, distance_sample_detector, detector_pixel_size)
            from ganrec_dataloader import forward_propagate
            image = forward_propagate(shape_x, shape_y, pad, energy_kev, detector_pixel_size, distance_sample_detector, phase_image = phase, attenuation_image = attenuation, fresnel_factor = fresnel_factor)[0,0,:,:].numpy()
    
    if image is not None: 
        if type(image) is list:
            ND = len(image)
            if len(image[0].shape) == 2:
                shape_x, shape_y = image[0].shape
                Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
                ND = 1
                image_path = os.getcwd()
            else:
                shape_x, shape_y = image[0].shape[1:]
                Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
        else:
            ND = 1
            shape_x, shape_y = image.shape
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
        if image_path is None:
            image_path = os.getcwd()
        if images is None:
            images = image

        if 'correct' in kwargs.keys():
            if kwargs['correct'] == True:
                if 'mean_dark_image' and 'mean_ref_image' in kwargs.keys():
                    mean_dark_image = kwargs['mean_dark_image']
                    mean_ref_image = kwargs['mean_ref_image']
                else: 
                    all_images = list(io.imread_collection(path + '/*.' + file_type).files)
                    mean_ref_image = np.mean(io.imread_collection([im_name for im_name in all_images if 'ref' in im_name]), axis = 0)
                    mean_dark_image = np.mean(io.imread_collection([im_name for im_name in all_images if 'dar' in im_name]), axis = 0)
                    
                if len(image) > 1:
                    image = [(image[i] - mean_dark_image) / (mean_ref_image - mean_dark_image) for i in range(len(image))]
                else:
                    image = (image - mean_dark_image) / (mean_ref_image - mean_dark_image)
        
        ff = ffactor(shape_x*pad, shape_y*pad, energy_kev, distance_sample_detector, detector_pixel_size)
        kwargs = {
            "path": path,
            "output_path" : os.getcwd(),
            "idx": idx,
            "column_name": 'path',
            "energy_J": energy_kev_to_joule(energy_kev),
            "energy_kev": energy_kev,
            "lam": wavelength_from_energy(energy_kev),
            "detector_pixel_size": detector_pixel_size,
            "distance_sample_detector": distance_sample_detector,
            "fresnel_number": fresnel_calculator(energy_kev = energy_kev, detector_pixel_size = detector_pixel_size, distance_sample_detector = distance_sample_detector),
            "wave_number": wave_number(energy_kev),

            "shape_x": shape_x,
            "px": shape_x,
            "shape_y": shape_y,
            "py": shape_y,
            "pad_mode": 'symmetric',
            'shape': [int(shape_x), int(shape_y)],
            'distance': [distance_sample_detector],
            'z': distance_sample_detector,
            
            'energy': energy_kev, 
            'alpha': alpha, 
            'pad': pad,
            'nfx': int(shape_x) * pad, 
            'nfy': int(shape_y) * pad,
            'pv': detector_pixel_size,
            'pixel_size': [detector_pixel_size, detector_pixel_size],
            'sample_frequency': [1.0/detector_pixel_size, 1.0/detector_pixel_size],
            'fx': Fx, 'fy': Fy,
            'method': method, 
            'delta_beta': delta_beta,
            "fresnel_factor":  ffactor(shape_x*pad, shape_y*pad, energy_kev, distance_sample_detector, detector_pixel_size),
            "i_input": image[0],
            "image_path": image_path,
            "image": image,
            "all_images": images,
            "ND": ND,
            "fresnel_factor": ff,
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
                images = list(io.imread_collection(path + '/*.' + file_type).files)
                image_path = [images[i] for i in idx]
                image = load_images_parallel(image_path)
            else:
                images = list(path)
                image = load_image(path)
            get_all_info(image=image, **kwargs)
        elif type(path) is list:
            path = [path[i] for i in idx]
            if type(path[0]) is str:
                #if the path[0] is a folder
                if os.path.isdir(path[0]):
                    images = list(io.imread_collection(path + '/*.' + file_type).files)
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

def load_images_parallel(urls = []):
    if urls == []:
        return None
    """using concurrent.futures"""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_image, urls)
        images = list(results)
    return images

def load_dxchange(paths = []):
    """using dxchange"""
    import dxchange
    if type(paths) == str:
        paths = [paths]
    n = len(paths)
    if n == 1:
        image = dxchange.read_tiff(paths[0])
    else:
        image = dxchange.read_tiff_stack(paths, ind = range(n))
    return image


def wave_transform(phase, attenuation, wave_number):
    return np.exp(1j*phase*wave_number)*np.exp(-attenuation*wave_number)

def propagation_of_image_j_at_distance_i(df, i, j):
    return io.imread(df[df.columns[i]][j])
   

def detector_image(x):
    # this is what a detector sees (only intensities). float32 is needed for the fft
    return np.real(x*np.conj(x)).astype(np.float32)

def consquative_detections(df, distance_numbers=5, idx=0, interactive_display = False):
    ground_wave = wave_transform(io.imread(df.phase[idx]), io.imread(df.attenuation[idx]), df.wave_number[idx])
    propagations_at_distance = []
    propagations_at_distance.append(ground_wave)
    for i in range(3, distance_numbers+3):
        propagations_at_distance.append(propagation_of_image_j_at_distance_i(df, i, idx))
    if interactive_display == False:
        return propagations_at_distance
    if interactive_display == True:
        visualize_interact(propagations_at_distance)
        return propagations_at_distance

def energy_kev_to_joule(energy_kev):
    """converts energy in kev to joules"""
    return energy_kev * 1.60217662e-16

def fresnel_calculator(energy_kev = None, lam = None, detector_pixel_size = None, distance_sample_detector = None):
    """calculates the fresnel number, the unit of energy must be in kev, and the unit of the other parameters must be in meters"""
    if energy_kev is not None:
        lam = 6.626 * 10**(-34) * 299792458 / energy_kev_to_joule(energy_kev)
    assert detector_pixel_size is not None, "detector_pixel_size must be given"
    assert distance_sample_detector is not None, "distance_sample_detector must be given"
    return detector_pixel_size**2/(lam*distance_sample_detector)

def wavelength_from_energy(energy):
    import quantities as pq
    h = pq.Quantity(6.626 * 10**(-34), 'J*s')
    c  = pq.Quantity(299792458, 'm/s')
    if type(energy) == pq.quantity.Quantity:
        if energy.dimensionality == pq.Quantity(1, 'keV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'J').dimensionality:
            energy = energy
        elif energy.dimensionality == pq.Quantity(1, 'eV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'm').dimensionality:
            energy = energy.rescale('J')
    else:
        energy = pq.Quantity(energy, 'J')
    return h*c/energy


def wave_number(energy):
    """ if the energy is not a quantity, it assumes that the energy is:
              * in keV if it is a string or int, or 
              * in joules if it is a float. 
              The final wave number is in 1/m"""
    import quantities as pq
    h = pq.Quantity(6.626 * 10**(-34), 'J*s')
    c  = pq.Quantity(299792458, 'm/s')
    if type(energy) == pq.quantity.Quantity:
        if energy.dimensionality == pq.Quantity(1, 'keV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'J').dimensionality:
            energy = energy
        elif energy.dimensionality == pq.Quantity(1, 'eV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'm').dimensionality:
            wave_length = energy
            energy = (h*c/wave_length).rescale('J')
    else:
        if type(energy) == str:
            energy = float(energy)
            energy = pq.Quantity(energy, 'keV').rescale('J')
        elif type(energy) == int:
            energy = float(energy)
            energy = pq.Quantity(energy, 'keV').rescale('J')
        elif type(energy) == float:
            energy = energy
            energy = pq.Quantity(energy, 'J')
        else:
            #joules
            energy = pq.Quantity(energy, 'keV').rescale('J')

    #calculate the wave number
    wave_number = 2*np.pi*(energy/h/c).rescale('1/m')
    wave_number = wave_number.magnitude
    return wave_number

#energy from wave number
def energy_from_wave_number(wave_number):
    import quantities as pq
    h = pq.Quantity(6.626 * 10**(-34), 'J*s')
    c  = pq.Quantity(299792458, 'm/s')
    if type(wave_number) == pq.quantity.Quantity:
        if wave_number.dimensionality == pq.Quantity(1, '1/m').dimensionality:
            wave_number = wave_number
        elif wave_number.dimensionality == pq.Quantity(1, '1/cm').dimensionality:
            wave_number = wave_number.rescale('1/m')
        elif wave_number.dimensionality == pq.Quantity(1, '1/nm').dimensionality:
            wave_number = wave_number.rescale('1/m')
        elif wave_number.dimensionality == pq.Quantity(1, '1/A').dimensionality:
            wave_number = wave_number.rescale('1/m')
    else:
        if type(wave_number) == str:
            wave_number = float(wave_number)
            wave_number = pq.Quantity(wave_number, '1/m')
        elif type(wave_number) == int:
            wave_number = float(wave_number)
            wave_number = pq.Quantity(wave_number, '1/m')
        elif type(wave_number) == float:
            wave_number = wave_number
            wave_number = pq.Quantity(wave_number, '1/m')
        else:
            #joules
            wave_number = pq.Quantity(wave_number, '1/m')

    #calculate the wave number
    energy = (wave_number*h*c/2/np.pi).rescale('J')
    energy = energy.magnitude
    return energy

def shorten(string):
    if 'e' in string:
        left = string.split('e')[0][:5]
        right = string.split('e')[1][:3]
        return left + 'e' + right
    else:
        if '.' in string:
            count = 0
            for i in range(len(string.split('.')[1])):
                if string[i] == '0':
                    count += 1
            return string[:count+2]
        else:
            return string[:5]

def give_title(image, title = '', idx = '', min_max = True):    
    if min_max:
        min_val_orig = np.min(image)
        max_val_orig = np.max(image)
        txt_min_val = shorten(str(min_val_orig))
        txt_max_val = shorten(str(max_val_orig))
    else:
        txt_min_val = ''
        txt_max_val = ''
    title = 'im='+ str(idx+1) if title == '' else title
    return title + ' (' + txt_min_val + ', ' + txt_max_val + ')'

def give_titles(images, titles = []):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title = titles[i], idx=i) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title = titles[i], idx=i) for i in range(len(images))]
    return titles

def get_row_col(images, show_all = False):
    if show_all:
        rows = int(np.sqrt(len(images)))
        cols = int(np.sqrt(len(images)))
        return rows, cols + (len(images) - rows*cols)//rows
    
    if len(images) == 1:
        rows = 1
        cols = 1
    elif len(images) <= 5:
        rows = 1
        cols = len(images)
    else:
        rows = 2
        cols = len(images)//2
    if rows*cols > len(images):
        images = images[:rows*cols - int(rows*cols/len(images))]
        rows, cols = get_row_col(images)
    print('rows: ', rows, 'cols: ', cols)
    return rows, cols

def convert_images(images, idx = None):
    if idx is None:
        if type(images) == np.ndarray:
            return [images]
        elif type(images) == list:
            if type(images[0]) == np.ndarray:
                return images
            
            elif type(images[0]) == str:
                return io.imread_collection(images)
            elif np.iscomplexobj(images[0]):
                return [np.abs(image)**2 for image in images]
            else:
                return images
        elif type(images) == str:
            try:
                return io.imread_collection(images)
            except:
                return [io.imread_collection(images+'/*.tif')]
        else:
            import torch
            from ganrec_dataloader import tensor_to_np
            if type(images) == torch.Tensor:
                return [tensor_to_np(images)]
            elif type(images) == list:
                if type(images[0]) == np.ndarray:
                    return images
                elif type(images[0]) == torch.Tensor:
                    np_im = [tensor_to_np(image) for image in images]
                    if np.iscomplexobj(np_im[0]):
                        return [np.abs(image)**2 for image in np_im]
                    elif len(np_im[0].shape) == 4:
                        np_im = [images[0,0,:,:] for image in images]
                    return np_im
            else:
                assert False, "images are not of type np.ndarray, torch.Tensor, list, str or io.imread_collection"
    else:
        try:
            return convert_images(images[idx])
        except:
            return [convert_images(image) for image in images]

def chose_fig(images, idx = None, rows = None, cols = None, show_all = False):
    (rows, cols) = get_row_col(images) if rows is None or cols is None else (rows, cols)
    shape = images[0].shape
    fig_size = (shape[1]*cols/100+1, shape[0]*rows/100)
    fig, ax = plt.subplots(rows, cols, figsize=fig_size)
    if rows == 1 and cols == 1:
        return fig, ax, rows, cols
    elif rows == 1:
        ax = ax.reshape(1, cols)
        return fig, ax, rows, cols
    elif cols == 1:
        ax = ax.reshape(rows, 1)
        return fig, ax, rows, cols
    else:
        return fig, ax, rows, cols

def visualize(images, idx = None, rows = None, cols = None, show_or_plot = 'show', cmap = 'Blues_r', title = '', axis = 'on', plot_axis = 'half'):
    """
    Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    """
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    
    images = convert_images(images, idx)
    titles = give_titles(images, title)
    shape = images[0].shape
    fig, ax, rows, cols = chose_fig(images, idx, rows, cols)
    if rows == 1 and cols == 1:
        ax.imshow(images[0], cmap = cmap)
        if show_or_plot == 'plot':
            if plot_axis == 'half':
                ax.plot(images[0][shape[0]//2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.plot(images[0][plot_axis, :])
        elif show_or_plot == 'both':
            ax.twinx().plot(images[0][shape[0]//2, :])
        ax.axis(axis)
        ax.set_title(titles[0], fontsize = 12)
        fig.colorbar(ax.imshow(images[0]), ax=ax)
        plt.show()
        return fig
    
    if show_or_plot == 'show':    
        [ax[i, j].imshow(images[i*cols + j], cmap = cmap) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'plot':
        if plot_axis == 'half':
            [ax[i, j].plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'both':
        [ax[i, j].imshow(images[i*cols + j], cmap = cmap) for i in range(rows) for j in range(cols)]
        if plot_axis == 'half':
            [ax[i, j].twinx().plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].twinx().plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
    else:
        assert False, "show_or_plot is not 'show', 'plot' or 'both'"
    [ax[i, j].axis(axis) for i in range(rows) for j in range(cols)]
    [ax[i, j].set_title(titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols)]
    plt.tight_layout()
    [fig.colorbar(ax[i, j].imshow(images[i*cols + j]), ax=ax[i, j]) for i in range(rows) for j in range(cols)]
    fig.patch.set_facecolor('xkcd:purple')
    plt.show()
    return fig

def sns_visualize(images, idx = None, rows = None, cols = None, show_or_plot = 'show', cmap = 'BuGn', title = '', axis = 'off', plot_axis = 'half'):
    """
    Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    """

    images = convert_images(images, idx)
    titles = give_titles(images, title)
    shape = images[0].shape
    fig, ax, rows, cols = chose_fig(images, idx, rows, cols)

    if rows == 1 and cols == 1:
        if show_or_plot == 'plot':
            if plot_axis == 'half':
                ax.plot(images[0][shape[0]//2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.plot(images[0][plot_axis, :])
            ax.set_title('y:'+str(plot_axis)+' '+titles[0], fontsize = 12)
        elif show_or_plot == 'both':
            if plot_axis == 'half':
                ax.twinx().plot(images[0][shape[0]//2, :])
            else:
                assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
                ax.twinx().plot(images[0][plot_axis, :])
            ax.set_title('y:'+str(plot_axis)+' '+titles[0], fontsize = 12)
        else:
            ax.set_title(titles[0], fontsize = 12)
            ax.imshow(images[0], cmap = cmap)
        
        ax.axis(axis)
        fig.colorbar(ax.imshow(images[0]), ax=ax)
        plt.show()
        return fig
    if show_or_plot == 'show':    
        [sns.heatmap(images[i*cols + j], cmap = cmap, ax = ax[i, j], robust=True) for i in range(rows) for j in range(cols)]
        [ax[i, j].set_title(titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'plot':
        if plot_axis == 'half':
            [ax[i, j].plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
        [ax[i, j].set_title('y:'+str(plot_axis)+' '+titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols)]
    elif show_or_plot == 'both':
        [sns.heatmap(images[i*cols + j], cmap = cmap, ax = ax[i, j]) for i in range(rows) for j in range(cols)]
        if plot_axis == 'half':
            [ax[i, j].twinx().plot(images[i*cols + j][shape[0]//2, :]) for i in range(rows) for j in range(cols)]
        else:
            assert type(plot_axis) == int, "plot_axis is not 'half' or an integer"
            [ax[i, j].twinx().plot(images[i*cols + j][plot_axis, :]) for i in range(rows) for j in range(cols)]
        [ax[i,j].set_title('y:'+str(plot_axis)+' '+titles[i*cols + j], fontsize = 12) for i in range(rows) for j in range(cols) ]
    else:
        assert False, "show_or_plot is not 'show', 'plot' or 'both'"
    [ax[i, j].axis(axis) for i in range(rows) for j in range(cols)]
    plt.tight_layout()
    fig.patch.set_facecolor('xkcd:light blue')
    plt.show()
    return fig

def visualize_interact(pure = []):
    import ipywidgets as widgets
    from ipywidgets import interact
    from IPython.display import display
    interact(visualize, pure = widgets.fixed(pure), show_or_plot = widgets.Dropdown(options=['show', 'plot'], value='show', description='Show or plot:'), rows = widgets.IntSlider(min=1, max=10, step=1, value=1, description='Rows:'), cols = widgets.IntSlider(min=1, max=10, step=1, value=3, description='Columns:'))
     
 