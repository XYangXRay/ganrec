import numpy as np
from numpy.fft import fftfreq
import matplotlib.pyplot as plt
import os 
import skimage.io as io

def nor_phase(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    return img


def ffactor(px, py, energy, z, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv ** 2
    freq_x = fftfreq(px)
    freq_y = fftfreq(py)
    xi, eta = np.meshgrid(freq_x, freq_y)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    h = np.exp(- 1j * frequ_prefactor * (xi ** 2 + eta ** 2) / 2)
    h = h.T
    return h.astype('complex64')

def fresnel_operator(px, py, pv, z, energy):
    from numpy.fft import fftfreq
    from numpy import pi 
    lambda0 = 1.23984122e-09 / energy
    upsample_scale = 1;                 # Scale by which to upsample image
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
        self.axs[0, 1].set_title('Reconstruction')
        self.im3, = self.axs[1, 1].plot([], [], 'r-')
        self.axs[1, 1].set_title('Generator loss')
        self.axs[0, 2].set_title('plot profile of input')
        self.axs[0, 2].plot(img_input[int(px/2), :], 'b-')
        self.axs[0, 2].set_title('plot profile of input')
        self.im4 = self.axs[1, 2].plot([], 'r-')
        self.axs[1, 2].set_title('plot profile of recon')

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
        #axs 1,2 is the plot profile of recon
        self.axs[1, 2].plot(img_rec[int(img_rec.shape[0]/2), :], 'r-')
        #save the plot with the epoch number
        if epoch%100==0:
            #5 significant digits
            plt.savefig('data/ganrec/recon/recon_{0:05d}.png'.format(epoch))
        # plt.tight_layout()
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
    file_name = os.path.splitext(os.path.basename(kwargs['image_path']))[0]
    #the folder name of the 'image_path'
    folder = os.path.basename(os.path.dirname(kwargs['image_path']))
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
    
def get_all_info(path = None, images = None, idx = 1000, energy_kev = 18.0, detector_pixel_size = 2.57 * 1e-6, distance_sample_detector = 0.15, alpha = 1e-8, delta_beta = 1e1, pad = 1, method = 'TIE', image = None, phase_path= None, attenuation_path = None, phase_image = None, attenuation_image = None):
    """
    make sure that the unit of energy is in keV, the unit of detector_pixel_size is in meter, and the unit of distance_sample_detector is in meter
    """

    if path is not None:
        images = list(io.imread_collection(path + '/*.tif').files)
        # images = sorted(glob(path + '/*.tif'))
        if type(idx) is list:
            image_path = [images[i] for i in idx]
            image = load_images_parallel(image_path)
            shape_x = image[0].shape[0]
            shape_y = image[0].shape[1]
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
            ND = len(idx)
            print("images are of leng", len(image))
        else:
            image_path = images[idx]
            image = load_image(image_path)
            shape_x = image.shape[0]
            shape_y = image.shape[1]
            Fx, Fy = grid_generator(shape_x, shape_y, upscale=pad, ps = detector_pixel_size)
            ND = 1
        assert images is not None, "either image or path should be provided"
        
    else:
        assert images is not None, "either image or path should be provided"
        image_path = images[idx]
        image = load_image(images[idx])
        Fx, Fy = grid_generator(image.shape[0], image.shape[1], upscale=1, ps = detector_pixel_size)
        shape_x = image.shape[0]
        shape_y = image.shape[1]
        ND = 1
    
    
    if phase_path is not None and attenuation_path is not None:
        phase_images = io.imread_collection(phase_path + '/*.tif').files
        attenuation_images = io.imread_collection(attenuation_path + '/*.tif').files
        if type(idx) is list:
            phase_image_path = []
            attenuation_image_path = []
            for i in idx:
                phase_image_path.append(phase_images[i])
                attenuation_image_path.append(attenuation_images[i])
            phase_image = load_images_parallel(phase_image_path)
            attenuation_image = load_images_parallel(attenuation_image_path)
        else:
            phase_image_path = phase_images[idx]
            attenuation_image_path = attenuation_images[idx]
            phase_image = load_image(phase_image_path)
            attenuation_image = load_image(attenuation_image_path)

    kwargs = {
        "path": path,
        "output_path" : os.getcwd(),
        "idx": idx,
        "column_name": 'path',
        "energy_J": energy_kev_to_joule(energy_kev),
        "energy_kev": energy_kev,
        "lam": wavelength_from_energy(energy_kev),
        "detector_pixel_size": detector_pixel_size,
        "distance_sample_detector": detector_pixel_size,
        "fresnel_number": fresnel_calculator(energy_kev = energy_kev, detector_pixel_size = detector_pixel_size, distance_sample_detector = distance_sample_detector),
        "wave_number": wave_number(energy_kev),

        "shape_x": shape_x,
        "shape_y": shape_y,
        "pad_mode": 'symmetric',
        'shape': [int(shape_x), int(shape_y)],
        'nx': int(shape_x), 'ny': int(shape_y),
        'distance': [distance_sample_detector], 
        'energy': energy_kev, 
        'alpha': alpha, 
        'pad': pad,
        'nfx': int(shape_x) * pad, 
        'nfy': int(shape_y) * pad,
        'pixel_size': [detector_pixel_size, detector_pixel_size],
        'sample_frequency': [1.0/detector_pixel_size, 1.0/detector_pixel_size],
        'fx': Fx, 'fy': Fy,
        'method': method, 
        'delta_beta': delta_beta,

        "image_path": image_path,
        "image": image,
        "ND": ND,
        "phase_path": phase_path,
        "attenuation_path": attenuation_path,
        'phase_image': phase_image,
        'attenuation_image': attenuation_image,
    } 
    kwargs.update(save_path_generator(**kwargs))
    return kwargs



def load_image(url):

    img = io.imread(url)
    return img

def load_images_parallel(urls = []):
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


#plot the images
def plot_or_show_images(images, rows = 1, cols = 5, show_or_plot = 'plot', random = True, cmap = 'None', figsize = (20, 20), title = ''):
    #if images is a pandas dataframe, convert to numpy array
        
    if len(images) == 0:
        print("no images to plot")
        return None
    elif len(images) == 1:
        rows = 1
        cols = 1
    elif len(images) < rows*cols:
        rows = 1
        cols = len(images)
    else:
        pass
    #generate random numbers rows*cols times and take the images with the random numbers
    random_numbers = np.random.randint(0, len(images), rows*cols)
    if random:
        images = [images[i] for i in random_numbers]
    else:
        images = images[:rows*cols]

    #if images are complex, take the absolute value
    if np.iscomplexobj(images[0]):
        #exponent the images
        
        # images = [np.exp(np.imag(image) - np.real(image)) for image in images]
        images = [np.abs(image) for image in images]
        # print("images are complex, taking the np.exp(imag - real) value")

    #if the images is 4D, 
    if len(images[0].shape) == 4:
        shape = images[0].shape
        if shape[0] > shape[1] and shape[0] > shape[2]:
            images = [image[0, :, :, 0] for image in images]
        elif shape[2] > shape[0] and shape[2] > shape[1]:
            images = [image[:, :, 0, 0] for image in images]
        else:
            images = [images[0,:,:,0] for image in images]
        images = [image[0,:, :, 0] for image in images]

    shape = images[0].shape
    if rows == 1 and cols == 1:
        figsize = (10,10)
        plt.imshow(images[0]) if show_or_plot == 'show' else plt.plot(images[0][shape[0]//2, :])
        plt.axis('on')
        if random:
            plt.title('min: ' + str(np.min(images)) + ' max: ' + str(np.max(images)) + 'im_' + str(random_numbers[0]), fontsize = 12)
        else:
            plt.title('min: ' + str(np.min(images)) + ' max: ' + str(np.max(images)), fontsize = 12)
        if title != '':
            plt.title(title)
        #if cmap is 
        plt.gray()
        plt.show()
        return None
    
    figsize = (shape[1]*cols/100, shape[0]*rows/100)
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        counter = 0
        for j in range(cols):
            if show_or_plot == 'show':
                if np.iscomplexobj(images[j]):
                    ax[j].imshow(np.exp(np.imag(images[j]) - np.real(images[j])))
                else:
                    ax[j].imshow(images[j])
                ax[j].axis('on')
                if random:
                    if title == '':
                        title = 'min: ' + str(np.min(images[j])) + ' max: ' + str(np.max(images[j])) + 'im_' + str(random_numbers[counter])

                    if type(title) == list:
                        title = title[counter]
                    if type(title) == str and title != '':
                        title = title + '_im_' + str(random_numbers[counter])
                else:
                    title = 'min: ' + str(np.min(images[j])) + ' max: ' + str(np.max(images[j]))
                ax[j].set_title(title, fontsize = 12)

                
            elif show_or_plot == 'plot':    
                if np.iscomplexobj(images[j]):
                    ax[j].plot((np.exp(np.imag(images[j]) - np.real(images[j])))[shape[0]//2, :])
                else:
                    ax[j].plot(images[j][shape[0]//2, :])
                ax[j].axis('on')
                if random:
                    if title == '':
                        title = 'min: ' + str(np.min(images[j])) + ' max: ' + str(np.max(images[j])) + 'im_' + str(random_numbers[counter])

                    if type(title) == list:
                        title = title[counter]
                    if type(title) == str and title != '':
                        title = title + '_im_' + str(random_numbers[counter])
                else:
                    title = 'min: ' + str(np.min(images[j])) + ' max: ' + str(np.max(images[j]))
                ax[j].set_title(title, fontsize = 12)      
            counter += 1
    else:
        counter = 0
        for i in range(rows):
            for j in range(cols):
                if show_or_plot == 'show':
                    ax[i, j].imshow(images[counter])
                    ax[i, j].axis('on')
                    if random:
                        title = 'min: ' + str(np.min(images[counter])) + ' max: ' + str(np.max(images[counter])) + 'im_' + str(random_numbers[counter])
                    else:
                        title = 'min: ' + str(np.min(images[counter])) + ' max: ' + str(np.max(images[counter]))
                    ax[i, j].set_title(title, fontsize = 12)

                elif show_or_plot == 'plot':    
                    ax[i, j].plot(images[counter][shape[0]//2, :])
                    ax[i, j].axis('on')
                    if random:
                        title = 'min: ' + str(np.min(images[counter])) + ' max: ' + str(np.max(images[counter])) + 'im_' + str(random_numbers[counter])
                    else:
                        title = 'min: ' + str(np.min(images[counter])) + ' max: ' + str(np.max(images[counter]))

                    ax[i, j].set_title(title, fontsize = 12)

                elif show_or_plot == 'both':
                    ax[i, j].imshow(images[counter])
                    ax[i, j].axis('on')
                    if random:
                        title = 'min: ' + str(np.min(images[counter])) + ' max: ' + str(np.max(images[counter])) + 'im_' + str(random_numbers[counter])
                    else:
                        title = 'min: ' + str(np.min(images[counter])) + ' max: ' + str(np.max(images[counter]))
                    ax[i, j].set_title(title, fontsize = 12)
                    
                    ax2 = ax[i, j].twinx()
                    ax2.plot(images[counter][shape[0]//2, :])
                    ax2.axis('on')
                    ax2.set_title(title, fontsize = 12)
                counter += 1
    #cmap gray
    plt.gray()
    plt.show()
    return None

def visualize(pure = [] , show_or_plot = 'show', rows = 1, cols = 5, random = False, in_parallel = False):
    if show_or_plot == 'show':
        plot_or_show_images(pure, show_or_plot='show', rows = rows, cols = cols, random = random)
    elif show_or_plot == 'plot':
        plot_or_show_images(pure, show_or_plot='plot', rows = rows, cols = cols, random = random)

def visualize_interact(pure = []):
    import ipywidgets as widgets
    from ipywidgets import interact
    from IPython.display import display
    interact(visualize, pure = widgets.fixed(pure), show_or_plot = widgets.Dropdown(options=['show', 'plot'], value='show', description='Show or plot:'), rows = widgets.IntSlider(min=1, max=10, step=1, value=1, description='Rows:'), cols = widgets.IntSlider(min=1, max=10, step=1, value=3, description='Columns:'))
     
 
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

