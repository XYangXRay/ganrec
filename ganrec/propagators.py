import tensorflow as tf
import tensorflow_addons as tfa

class TomoRadon:

    def __init__(self, rec, ang):
        self.rec = rec
        self.ang = ang

    def compute(self):
        nang = self.ang.shape[0]
        img = tf.transpose(self.rec, [3, 1, 2, 0])
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfa.image.rotate(img, -self.ang, interpolation='bilinear')
        sino = tf.reduce_mean(img, 1, name=None)
        sino = tf.image.per_image_standardization(sino)
        sino = tf.transpose(sino, [2, 0, 1])
        sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
        return sino
    
    
class PhaseFresnel:
    
    def __init__(self, phase, absorption, ff, px):
        self.phase = phase
        self.absorption = absorption
        self.ff = ff
        self.px = px

    def compute(self):
        paddings = tf.constant([[self.px // 2, self.px // 2], [self.px // 2, self.px // 2]])
        pvalue = tf.reduce_mean(self.phase[:100, :])
        self.phase = tf.pad(self.phase, paddings, 'SYMMETRIC')
        self.absorption = tf.pad(self.absorption, paddings, 'SYMMETRIC')
        abfs = tf.complex(-self.absorption, self.phase)
        abfs = tf.exp(abfs)
        ifp = tf.abs(tf.signal.ifft2d(self.ff * tf.signal.fft2d(abfs))) ** 2
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