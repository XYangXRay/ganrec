import tensorflow as tf
from ganrectf.tfutils import tfrotate, apply_gaussian_blur_4d


class TomoRadon:

    def __init__(self, rec, ang):
        self.rec = rec
        self.ang = ang

    def compute(self):
        nang = self.ang.shape[0]
        img = tf.transpose(self.rec, [3, 1, 2, 0])
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfrotate(img, -self.ang, interpolation="bilinear")
        sino = tf.reduce_mean(img, 1, name=None)
        sino = tf.transpose(sino, [2, 0, 1])
        sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
        return sino


class TensorRadon:

    def __init__(self, rec, ang, psi):
        self.strain_tensor = rec
        self.ang = ang
        self.psi = psi

    def tfnor_data(self, img):
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    def compute(self):
        detector_rows = self.strain_tensor.shape[0]
        detector_columns = self.strain_tensor.shape[1]
        strain_tensor = tf.cast(self.strain_tensor, dtype=tf.float32)
        vol_mask = tf.zeros((detector_rows, detector_columns, detector_columns))
        vol_mask = tf.reduce_sum(tf.abs(strain_tensor), axis=3) > 0.0
        vol_mask = tf.reshape(vol_mask, (-1, detector_columns, detector_columns, 1))
        vol_mask = tf.cast(vol_mask, dtype=tf.float32)
        angles = tf.cast(self.ang, dtype=tf.float32)
        thickness = TomoRadon(vol_mask, angles).compute()
        thickness = tf.squeeze(thickness)
        strain_tensor = tf.transpose(strain_tensor, [3, 1, 2, 0])
        proj_strain_comp = TomoRadon(strain_tensor, angles).compute()
        proj_strain_comp = tf.squeeze(proj_strain_comp)
        cos_squared = tf.expand_dims(tf.pow(tf.cos(angles), 2), 1)
        sin_squared = tf.expand_dims(tf.pow(tf.sin(angles), 2), 1)
        cos_psi_squared = tf.pow(tf.cos(self.psi), 2)
        sin_psi_squared = tf.pow(tf.sin(self.psi), 2)
        sin_2angles = tf.expand_dims(tf.sin(2 * angles), 1)
        sin_angles_sin_2psi = tf.expand_dims(tf.sin(angles) * tf.sin(2 * self.psi), 1)
        cos_angles_sin_2psi = tf.expand_dims(tf.cos(angles) * tf.sin(2 * self.psi), 1)
        if proj_strain_comp.shape[0] == 6:
            proj_strain_ws = (
                tf.multiply(proj_strain_comp[0], cos_squared * sin_psi_squared)
                + tf.multiply(proj_strain_comp[1], sin_squared * sin_psi_squared)
                + tf.multiply(proj_strain_comp[2], cos_psi_squared)
                + tf.multiply(proj_strain_comp[3], sin_2angles * sin_psi_squared)
                + tf.multiply(proj_strain_comp[4], sin_angles_sin_2psi)
                + tf.multiply(proj_strain_comp[5], cos_angles_sin_2psi)
                )
        elif proj_strain_comp.shape[0] == 3:
            proj_strain_ws = (
                tf.multiply(proj_strain_comp[0], cos_squared * sin_psi_squared)
                + tf.multiply(proj_strain_comp[1], sin_squared * sin_psi_squared)
                # + tf.multiply(proj_strain_comp[2], cos_psi_squared)
                + tf.multiply(proj_strain_comp[2], sin_2angles * sin_psi_squared)
                # + tf.multiply(proj_strain_comp[4], sin_angles_sin_2psi)
                # + tf.multiply(proj_strain_comp[5], cos_angles_sin_2psi)
                )
            
        # print(f'thickness shape is {thickness.shape}')
        # print(f'proj_strain_ws shape is {proj_strain_ws.shape}')
        # tensor_sino = tf.where(thickness > 0.05, tf.math.divide_no_nan(self.tfnor_data(proj_strain_ws), thickness), 0)
        tensor_sino = proj_strain_ws
        # tensor_sino = tf.math.divide_no_nan(proj_strain_ws, thickness)
        tensor_sino = tf.reshape(tensor_sino, [1, tensor_sino.shape[0], tensor_sino.shape[1], 1])
        return tensor_sino


class PhaseFresnel:

    def __init__(self, phase, absorption, ff, px):
        self.phase = phase
        self.absorption = absorption
        self.ff = ff
        self.px = px

    def compute(self):
        paddings = tf.constant([[self.px // 2, self.px // 2], [self.px // 2, self.px // 2]])
        pvalue = tf.reduce_mean(self.phase[:100, :])
        self.phase = tf.pad(self.phase, paddings, "SYMMETRIC")
        self.absorption = tf.pad(self.absorption, paddings, "SYMMETRIC")
        abfs = tf.complex(-self.absorption, self.phase)
        abfs = tf.exp(abfs)
        ifp = tf.abs(tf.signal.ifft2d(self.ff * tf.signal.fft2d(abfs))) ** 2
        ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
        ifp = tf.image.central_crop(ifp, 0.5)
        ifp = tf.image.per_image_standardization(ifp)
        ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
        ifp = apply_gaussian_blur_4d(ifp, kernel_size=3, sigma=1.5)
        return ifp
    
class PhaseFresnelG:

    def __init__(self, phase, absorption, ff, px):
        self.phase = phase
        self.absorption = absorption
        self.ff = ff
        self.px = px
    
    def gaussian_filter_in_fourier(self, sigma, shape):

        rows = tf.cast(shape[0], tf.float32)
        cols = tf.cast(shape[1], tf.float32)
    
    # Create a meshgrid in frequency space
        x = tf.linspace(-cols / 2, cols / 2, int(cols))
        y = tf.linspace(-rows / 2, rows / 2, int(rows))
        X, Y = tf.meshgrid(x, y)
    
    # Calculate Gaussian in frequency domain
        gaussian_filter = tf.exp(-0.5 * (X**2 + Y**2) / (sigma ** 2))
    
    # Shift to match the FFT frequency ordering
        gaussian_filter = tf.signal.fftshift(gaussian_filter)
        gaussian_filter = tf.cast(gaussian_filter, dtype=tf.complex64)
    
        return gaussian_filter

    def compute(self):
        paddings = tf.constant([[self.px // 2, self.px // 2], [self.px // 2, self.px // 2]])
        pvalue = tf.reduce_mean(self.phase[:100, :])
        self.phase = tf.pad(self.phase, paddings, "SYMMETRIC")
        self.absorption = tf.pad(self.absorption, paddings, "SYMMETRIC")
        abfs = tf.complex(-self.absorption, self.phase)
        abfs = tf.exp(abfs)
        fft_image = self.ff * tf.signal.fft2d(abfs)
    
    # Get the shape of the image for the Gaussian filter
        shape = tf.shape(abfs)[-2:]  # Height and width
    
    # Create Gaussian filter in Fourier domain
        gaussian_filter = self.gaussian_filter_in_fourier(30, shape)
    
    # Apply the Gaussian filter in Fourier domain
        blurred_fft_image = fft_image * gaussian_filter
    
    # Inverse FFT to obtain the blurred image in the spatial domain
        ifp = tf.abs(tf.signal.ifft2d(blurred_fft_image)) ** 2
        # ifp = tf.abs(tf.signal.ifft2d(self.ff * tf.signal.fft2d(abfs))) ** 2
        ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
        ifp = tf.image.central_crop(ifp, 0.5)
        ifp = tf.image.per_image_standardization(ifp)
        ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
        return ifp


class PhaseFraunhofer:

    def __init__(self, phase, absorption, shift_factor=100000):
        self.phase = phase
        self.absorption = absorption
        self.shift_factor = shift_factor

    def compute(self):
        wf = tf.complex(self.absorption, self.phase)
        ifp = tf.square(tf.abs(tf.signal.fft2d(wf)))
        ifp = tf.math.log(ifp + self.shift_factor)
        ifp = tf.signal.fftshift(ifp)
        ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
        ifp = tf.image.per_image_standardization(ifp)
        # ifp = self.tfnor_diff(ifp)
        return ifp
