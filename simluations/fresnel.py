import numpy as np
from numpy.fft import ifft2, fft2, fftshift, fftfreq

# def padding(img, size, pvalue):
#     h, w = img.shape
#     dh = (size - h)//2
#     dw = (size - w)//2
#     img = np.pad(img, ((dh, dh),(dw, dw)), 'constant', constant_values = ((pvalue, pvalue),(pvalue, pvalue)))
#     return img

def padding(img, target_size, pvalue):
    """Pad the array to the specified target size with a constant value."""
    # Unpack the target size into height and width
    target_height, target_width = target_size
    h, w = img.shape

    # Calculate the padding for height and width separately
    dh = (target_height - h) // 2
    dw = (target_width - w) // 2

    # Pad the image symmetrically with the given constant value
    img = np.pad(img, ((dh, target_height - h - dh), (dw, target_width - w - dw)),
                 'constant', constant_values=pvalue)
    return img

def unpadding(img, target_size):
    """Crop the array to the specified target size."""
    target_height, target_width = target_size
    h, w = img.shape

    # Calculate the starting indices for cropping
    start_y = (h - target_height) // 2
    start_x = (w - target_width) // 2

    # Return the cropped image
    return img[start_y:start_y + target_height, start_x:start_x + target_width]

# def unpadding(img, size):
#     h, w = img.shape
#     dh = (h-size)//2
#     dw = (w-size)//2
#     return img[dh:-dh, dw:-dw]

def ffactor(px, energy, z, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv ** 2
    freq = fftfreq(px)
    xi, eta = np.meshgrid(freq, freq)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    h = np.exp(- 1j* frequ_prefactor * ( xi**2 + eta**2) / 2 )
    return h

def fresnel_propagation(phase, absorption, h, abs_ratio):
    phase= phase/phase.max()*0.5
    absorption = absorption/absorption.max()*0.5*abs_ratio
    aps = np.exp( - absorption + 1j*phase)
    ifp = (abs(ifft2(h*fft2(aps)))**2).astype('float32')
    return ifp

def propagate(img_f, img_a, energy, z, pv, abs_ratio):
    # Automatically detect original image size
    original_size = img_f.shape[-2:]  # Get the height and width of the image
    padded_size = (2 * original_size[0], 2 * original_size[1])  # Pad to twice the original size
    cropped_size = (int(1.5*original_size[0]), int(1.5*original_size[1]))  # Crop to 1.5 times the original size

    h = ffactor(padded_size[0], energy, z, pv)

    # Automatically detect if the input is 2D or 3D
    is_3d = img_f.ndim == 3

    ifp_tests = []  # Store results

    if is_3d:
        for i in range(img_f.shape[0]):
            phase = img_f[i]
            phase = padding(phase, padded_size, pvalue=phase.max())
            absorption = img_a[i]
            absorption = padding(absorption, padded_size, pvalue=absorption.min())          
            ifp_test = fresnel_propagation(phase, absorption, h, abs_ratio)
            ifp_test = unpadding(ifp_test, cropped_size)
            ifp_tests.append(ifp_test)
    else:
        phase = padding(img_f, padded_size, pvalue=img_f.max())
        absorption = padding(img_a, padded_size, pvalue=img_a.min())  
        ifp_test = fresnel_propagation(phase, absorption, h, abs_ratio)
        ifp_test = unpadding(ifp_test, cropped_size)
        ifp_tests.append(ifp_test)

    return ifp_tests if is_3d else ifp_tests[0]
