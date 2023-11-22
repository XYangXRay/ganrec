import numpy as np
import skimage.io as io
import quantities as pq
from utils import nor_phase

from numpy.fft import fftfreq
from utils import load_images_parallel
import os
from skimage.transform import resize


def wave_transform(phase, attenuation, wave_number):
    return np.exp(1j*phase*wave_number)*np.exp(-attenuation*wave_number)

def propagation_of_image_j_at_distance_i(df, i, j):
    return io.imread(df[df.columns[i]][j]) 

def detector_image(x):
    # this is what a detector sees (only intensities). float32 is needed for the fft
    return np.real(x*np.conj(x)).astype(np.float32)

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

def eneryg_J(energy):
    if type(energy) == pq.quantity.Quantity:
        if energy.dimensionality == pq.Quantity(1, 'keV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'J').dimensionality:
            energy = energy
        elif energy.dimensionality == pq.Quantity(1, 'eV').dimensionality:
            energy = energy.rescale('J')
        elif energy.dimensionality == pq.Quantity(1, 'm').dimensionality:
            wavelength = energy
            energy = energy_from_wavelength(wavelength)
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
            energy = pq.Quantity(energy, 'keV').rescale('J')
    return energy

def wavelength_m(lam):
    if type(lam) == pq.quantity.Quantity:
        if lam.dimensionality == pq.Quantity(1, 'm').dimensionality:
            lam = lam.rescale('m')
        elif lam.dimensionality == pq.Quantity(1, 'nm').dimensionality:
            lam = lam.rescale('m')
        elif lam.dimensionality == pq.Quantity(1, 'A').dimensionality:
            lam = lam.rescale('m')
        elif lam.dimensionality == pq.Quantity(1, 'keV').dimensionality:
            lam = lam.rescale('m')
    else:
        lam = pq.Quantity(lam, 'm')
    return lam

def wavelength_from_energy(energy):
    h = pq.Quantity(6.62607015* 10**-34, 'J*s')
    c  = pq.Quantity(299792458, 'm/s')
    energy = eneryg_J(energy)
    return h*c/energy

def energy_from_wavelength(lam):
    h = pq.Quantity(6.62607015* 10**-34, 'J*s')
    c  = pq.Quantity(299792458, 'm/s')
    lam = wavelength_m(lam)
    energy = h*c/lam
    return energy.rescale('keV')

def wave_number(energy):
    lam = wavelength_from_energy(energy)
    wavenumber = 2*np.pi/lam
    return wavenumber

def energy_from_wave_number(wave_number):
    lam = 2*np.pi/wave_number
    return energy_from_wavelength(lam)


def get_from_sim(phase, attenuation, energy=None, z=None, fresnel_number=None, pv=None, abs_ratio=0.5):
    import quantities as pq
    energy = pq.Quantity(energy, 'keV') if energy is not None else None
    z = pq.Quantity(z, 'm') if z is not None else None
    pv = pq.Quantity(pv, 'm') if pv is not None else None
    fresnel_number = pq.Quantity(fresnel_number, 'dimensionless') if fresnel_number is not None else None

    if energy is None or z is None or pv is None:
        assert fresnel_number is not None        
        if energy == None and [z, pv] != None:
            lam = pv**2 / (fresnel_number * z)
            energy = energy_from_wavelength(lam)
            energy = energy.rescale('keV')
        elif z == None and [energy, pv] != None:
            lam = wavelength_from_energy(energy)
            z = pv**2 / (fresnel_number * lam)
            print('z: ', z, 'pv: ', pv, 'lam: ', lam)
        elif pv == None and [energy, z] != None:
            lam = wavelength_from_energy(energy)
            pv = np.sqrt(fresnel_number * z * lam)
            pv = pq.Quantity(pv, 'm')
        else:
            raise ValueError('Not enough information provided')
    else:
        lam = wavelength_from_energy(energy)
        fresnel = pv**2 / (lam * z) * pq.Quantity(1, 'dimensionless')

        if fresnel_number is not None:
            if fresnel_number == fresnel:
                pass
            else:
                print('Warning: Fresnel number does not match with the other parameters')
                fresnel_number = fresnel
        else:
            fresnel_number = fresnel
    phase_image_orig = phase * 2 * np.pi / lam.magnitude
    attenuation_image_orig = attenuation * 2*np.pi / lam.magnitude
    phase_image = nor_phase(phase_image_orig)/np.max(nor_phase(phase_image_orig))
    attenuation_image = (nor_phase(attenuation_image_orig)/np.max(nor_phase(attenuation_image_orig))) * abs_ratio

    print('energy: ', energy, 'z: ', z, 'fresnel_number: ', fresnel_number, 'pv: ', pv, 'lam: ', lam)
    return phase_image_orig, attenuation_image_orig, phase_image, attenuation_image, energy.magnitude, z.magnitude, fresnel_number.magnitude, pv.magnitude, lam.magnitude

def get_shape(image):
    if image is None:
        return None, None, None, None
    if type(image) is not list:
        image = [image]
    n_of_images= len(image)
    if len(image[0].shape) == 2:
        px, py = image[0].shape
        ND = n_of_images
        return ND, 1, px, py
    elif len(image[0].shape) == 3:
        shape = image[0].shape
        ND = min(shape)
        shape = [shape[i] for i in range(len(shape)) if shape[i] != ND]
        px, py = shape
        return n_of_images, ND, px, py
    elif len(image[0].shape) == 4:
        shape = image[0].shape
        if 1 in shape:
            shape = [shape[i] for i in range(len(shape)) if shape[i] != 1]
        ND = min(shape)
        shape = [shape[i] for i in range(len(shape)) if shape[i] != ND]
        channels = min(shape)
        shape = [shape[i] for i in range(len(shape)) if shape[i] != channels]
        px, py = shape
        return ND, channels, px, py
    else:
        raise ValueError("The input image has to be 2D, 3D or 4D")

     
def base_coeff(px = None, py=None, image=None):
    if [px, py] == [None, None]:
        ND, channels, px, py = get_shape(image)
    freq_1 = fftfreq(px)
    freq_2 = fftfreq(py)
    xi, eta = np.meshgrid(freq_1, freq_2)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    return np.exp((xi ** 2 + eta ** 2) / 2)

def fresnel_calc(energy, z, pv):
    """z and pv have to in meters"""
    if energy is None or z is None or pv is None:
        return None
    if type(energy) is not list or type(z) is not list or type(pv) is not list:
        wavelength = wavelength_from_energy(eneryg_J(energy)).magnitude
        fresnel_number = pv**2/(wavelength*z) 
    else:
        energy = [energy] if type(energy) is not list else energy
        wavelength = [wavelength_from_energy(eneryg_J(ener)).magnitude for ener in energy]
        z = [z] if type(z) is not list else z
        pv = [pv] if type(pv) is not list else pv
        fresnel_number = []
        for i in range(len(energy)):
            for j in range(len(z)):
                for k in range(len(pv)):
                    fresnel_number.append(pv[k]**2/(wavelength[i]*z[j]))
    return  fresnel_number

def ffactor(energy=None, z=None, pv=None, px = None, py = None, image = None, fresnel_number = None):
    if px== None and py==None and image is not None:
        ND, channel, px, py = get_shape(image)
    if fresnel_number == None:
        fresnel_number = fresnel_calc(energy, z, pv)
    else:
        fresnel_number = fresnel_number
        if [energy, z, pv] != [None, None, None]:
            if fresnel_number != fresnel_calc(energy, z, pv): 
                print("fresnel_number is not consistent with energy, z, and pv")
            
        
    basecoeff = base_coeff(px, py, image)
    if type(fresnel_number) is not list:
        ffs = basecoeff**(-1j*2*np.pi/fresnel_number)
    else:
        ffs = [basecoeff**(-1j*2*np.pi/fres) for fres in fresnel_number]
    return ffs

def propagate(energy=None, z=None, pv=None, px = None, py = None, fresnel_number = None, image = None, return_complex = False):
    if energy is not None and z is not None and pv is not None:
        ffs = ffactor(energy, z, pv, px, py, image)

    elif [energy, z ] == [None, None] and pv != None or [energy, pv] == [None, None] and z != None or [z, pv] == [None, None] and energy != None and fresnel_number != None:
        ffs = ffactor(energy, z, pv, px, py, image, fresnel_number)
    elif [energy, z, pv] == [None, None, None] and fresnel_number != None:
        basecoeff = base_coeff(px, py, image)
        if type(fresnel_number) is not list:
            ffs = basecoeff**(-1j*2*np.pi/fresnel_number)
        else:
            ffs = [basecoeff**(-1j*2*np.pi/fres) for fres in fresnel_number]
    else:
        ffs = None

    fft_image = np.fft.fft2(image)
    if type(ffs) is not list:
        propagated_images = [np.fft.ifft2(ffs * fft_image)]
    else:
        propagated_images = [np.fft.ifft2(ff * fft_image) for ff in ffs]
    if return_complex:
        return propagated_images
    else:
        return np.abs(propagated_images)**2


def prepare_dict(**kwargs):
    """
    make sure that the unit of energy is in keV, the unit of detector_pixel_size is in meter, and the unit of distance_sample_detector is in meter
    """
    similar_terms = [
        ['energy', 'eneryg_J', 'energy_kev'], 
        ['lam', 'lamda', 'wavelength', 'wave_length'],
        ['phase', 'phase_image'],
        ['attenuation', 'attenuation_image'],
        ['path', 'paths','images', 'i_inputs', 'path'],
        ['image_path', 'image_paths'],
        ['detector_pixel_size', 'pv'],
        ['distance_sample_detector', 'z'],
        ['fresnel_number', 'fresnel_number', 'fresnelnumbers', 'fresnelnumbers'],
        ['fresnel_factors', 'ffs', 'frensel_factor', 'fresnelfactor'],
        ['pad', 'pad_value', 'magnification_factor', 'upscale'],
        ['mode', 'pad_mode'],
        ['method', 'propagation_method'],
        ['alpha', 'alpha_value'],
        ['delta_beta', 'delta_beta_value'],
        ['idx', 'indices', 'index'],
        ['file_type', 'file_types', 'filetype', 'filetypes'],
        ['image', 'i_input', 'hologram', 'intensity'],
        ['shape_x', 'px'],
        ['shape_y', 'py'],
        ]
    for terms in similar_terms:
        for term in terms:
            if term in kwargs.keys():
                kwargs[terms[0]] = kwargs[term]
                break
        kwargs[terms[0]] = None if terms[0] not in kwargs.keys() else kwargs[terms[0]]

    kwargs['idx'] = [0] if kwargs['idx'] is None else kwargs['idx']
    kwargs['idx'] = [kwargs['idx']] if type(kwargs['idx']) is not list else kwargs['idx']
    if kwargs['image'] is None:
        if type(kwargs['path']) is list:
            kwargs['path'] = [kwargs['path'][i] for i in kwargs['idx']]
        if type(kwargs['path']) is not list:
            kwargs['path'] = [kwargs['path']]
        kwargs['image_path'] = []
        for path in kwargs['path']:
            if type(path) is str:
                if os.path.isdir(path):
                    kwargs['image_path'] += list(io.imread_collection(path + '/*.' + kwargs['file_type']).files)
                elif os.path.isfile(path):
                    kwargs['image_path'] += [path]
            elif 'collection' in str(type(path)):
                kwargs['image_path'] += path.files
            else:
                kwargs['image_path'] = kwargs['path']
        kwargs['image_path'] = [kwargs['image_path'][i] for i in kwargs['idx']] 
        kwargs['image'] = load_images_parallel(kwargs['image_path']) if kwargs['image_path'][0] is not None else None
    else:
        if type(kwargs['image']) is not list:
            kwargs['image'] = [kwargs['image']]

    
    kwargs['energy'] = None if 'energy' not in kwargs.keys() else kwargs['energy']
    kwargs['lam'] = wavelength_from_energy(eneryg_J(kwargs['energy'])).magnitude if kwargs['energy'] is not None else None
    kwargs['detector_pixel_size'] = None if 'detector_pixel_size' not in kwargs.keys() else kwargs['detector_pixel_size']
    kwargs['distance_sample_detector'] = None if 'distance_sample_detector' not in kwargs.keys() else kwargs['distance_sample_detector']
    kwargs['fresnel_number'] = None if 'fresnel_number' not in kwargs.keys() else kwargs['fresnel_number']
    kwargs['ffs'] = None if 'ffs' not in kwargs.keys() else kwargs['ffs']
    kwargs['base_coeff'] = None if 'base_coeff' not in kwargs.keys() else kwargs['base_coeff']
    kwargs['phase'] = None if 'phase' not in kwargs.keys() else kwargs['phase']
    kwargs['attenuation'] = None if 'attenuation' not in kwargs.keys() else kwargs['attenuation']
    kwargs['propagate'] = True if kwargs['phase'] is not None or kwargs['attenuation'] is not None else False

    kwargs['ND'], kwargs['channels'], kwargs['px'], kwargs['py'] = get_shape(kwargs['image'])
    kwargs['shape'] = [kwargs['px'], kwargs['py']]
    kwargs['shape_x'] = kwargs['px']
    kwargs['shape_y'] = kwargs['py']

    kwargs['fresnel_number'] = fresnel_calc(kwargs['energy'], kwargs['distance_sample_detector'], kwargs['detector_pixel_size']) if kwargs['fresnel_number'] is None else kwargs['fresnel_number']
    kwargs['base_coeff'] = [base_coeff(image = image) for image in kwargs['image']] if kwargs['image'] is not None else kwargs['base_coeff']
    if kwargs['fresnel_number'] is not None:
        kwargs['ffs'] = [ffactor(energy = kwargs['energy'], z = kwargs['distance_sample_detector'], pv = kwargs['detector_pixel_size'], image = image, fresnel_number = kwargs['fresnel_number']) for image in kwargs['image']] if 'ffs' not in kwargs.keys() else kwargs['ffs']
    else:
        kwargs['ffs'] = [ffactor(energy = kwargs['energy'], z = kwargs['distance_sample_detector'], pv = kwargs['detector_pixel_size'], image = image) for image in kwargs['image']] if 'ffs' not in kwargs.keys() and kwargs['image'] is not None else kwargs['ffs']
    
    if 'propagate' in kwargs.keys() and kwargs['propagate'] == True:
        if 'phase' not in kwargs.keys(): 
            kwargs['phase'] = None
        if 'attenuation' not in kwargs.keys():
            kwargs['attenuation'] = None
        if kwargs['phase'] is not None:
            if type(kwargs['phase']) is not list:
                kwargs['phase'] = [kwargs['phase']]
            if type(kwargs['phase'][0]) is str:
                kwargs['phase'] = load_images_parallel(kwargs['phase'])
        
        if kwargs['attenuation'] is not None:
            if type(kwargs['attenuation']) is not list:
                kwargs['attenuation'] = [kwargs['attenuation']]
            if type(kwargs['attenuation'][0]) is str:
                kwargs['attenuation'] = load_images_parallel(kwargs['attenuation'])

        kwargs['phase'] = [nor_phase(phase)/np.max(nor_phase(phase)) for phase in kwargs['phase']] if kwargs['phase'] is not None else None
        kwargs['attenuation'] = [nor_phase(attenuation)/np.max(nor_phase(attenuation)) for attenuation in kwargs['attenuation']] if kwargs['attenuation'] is not None else None

        if kwargs['phase'] is not None and kwargs['attenuation'] is not None:
            kwargs['wavefunction'] = [np.exp(1j*kwargs['phase'][i] - kwargs['attenuation'][i]) for i in range(len(kwargs['phase']))]
        elif kwargs['phase'] is not None and kwargs['attenuation'] is None:
            kwargs['wavefunction'] = [np.exp(1j*kwargs['phase'][i] + np.zeros(kwargs['phase'][0].shape)) for i in range(len(kwargs['phase']))]
        elif kwargs['phase'] is None and kwargs['attenuation'] is not None:
            kwargs['wavefunction'] = [np.exp(1j * np.zeros(kwargs['attenuation'][0].shape) - kwargs['attenuation'][i]) for i in range(len(kwargs['attenuation']))]
        else:
            kwargs['wavefunction'] = kwargs['image']
        kwargs['propagated_images'] = [propagate(energy=kwargs['energy'], z=kwargs['distance_sample_detector'], pv=kwargs['detector_pixel_size'], fresnel_number=kwargs['fresnel_number'], image=kwargs['wavefunction'][i]) for i in range(len(kwargs['wavefunction']))] if 'propagated_images' not in kwargs.keys() else kwargs['propagated_images']

        if kwargs['image'] is None:
            kwargs['image'] = kwargs['propagated_images']
        if kwargs['path'] is None or kwargs['path'][0] is None:
            kwargs['path'] = kwargs['propagated_images']
        if kwargs['image_path'] is None:
            kwargs['image_path'] = os.getcwd()
    else:
        kwargs['propagated_images'] = kwargs['image']

    if 'pad' not in kwargs.keys():
        kwargs['pad'] = 1
    if 'pad_mode' not in kwargs.keys():
        kwargs['pad_mode'] = 'reflect'
    if 'alpha' not in kwargs.keys():
        kwargs['alpha'] = 1e-8
    if 'delta_beta' not in kwargs.keys():
        kwargs['delta_beta'] = 1e1
    if 'method' not in kwargs.keys():
        kwargs['method'] = 'GANREC'
    if 'downsampling_factor' not in kwargs.keys():
        kwargs['downsampling_factor'] = 1
    if 'save_path' not in kwargs.keys():
        kwargs['save_path'] = os.getcwd()
    if 'save_name' not in kwargs.keys():
        kwargs['save_name'] = 'reconstructed'
    if 'save_format' not in kwargs.keys():
        kwargs['save_format'] = 'tif'
    if 'save' not in kwargs.keys():
        kwargs['save'] = False
    if 'save_all' not in kwargs.keys():
        kwargs['save_all'] = False

    if kwargs['downsampling_factor'] != 1:
        kwargs['image'] = [resize(image, (image.shape[0]//kwargs['downsampling_factor'], image.shape[1]//kwargs['downsampling_factor']), anti_aliasing=True) for image in kwargs['image']] if kwargs['image'] is not None else None
        kwargs['detector_pixel_size'] = kwargs['detector_pixel_size'] * kwargs['downsampling_factor'] if kwargs['detector_pixel_size'] is not None else None
        if kwargs['fresnel_number'] is None:
            kwargs['fresnel_number'] = fresnel_calc(kwargs['energy'], kwargs['distance_sample_detector'], kwargs['detector_pixel_size'])  
        else:
            if type(kwargs['fresnel_number']) is not list:
                kwargs['fresnel_number'] = kwargs['fresnel_number']*kwargs['downsampling_factor']**2
            else:
                kwargs['fresnel_number'] = [kwargs['fresnel_number'][i]*kwargs['downsampling_factor']**2 for i in range(len(kwargs['fresnel_number']))]
        
        kwargs['base_coeff'] = [base_coeff(image = image) for image in kwargs['image']] if kwargs['image'] is not None else None
        kwargs['ffs'] = [ffactor(energy = kwargs['energy'], z = kwargs['distance_sample_detector'], pv = kwargs['detector_pixel_size'], image = image) for image in kwargs['image']] if 'ffs' not in kwargs.keys() and kwargs['image'] is not None else kwargs['ffs']
    return kwargs
