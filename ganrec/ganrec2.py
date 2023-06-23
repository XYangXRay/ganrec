from models import make_generator, make_discriminator, make_filter
from utils import RECONmonitor, ffactor, fresnel_operator, visualize, get_all_info
from skimage.exposure import match_histograms

import skimage.io as io

import tensorflow as tf
import os
import numpy as np
import time


# @tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,
                                                                       labels=tf.ones_like(real_output)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                       labels=tf.zeros_like(fake_output)))
    total_loss = real_loss + fake_loss
    return total_loss


def l1_loss(img1, img2):
    return tf.reduce_mean(tf.abs(img1 - img2))
def l2_loss(img1, img2):
    return tf.square(tf.reduce_mean(tf.abs(img1-img2)))



# @tf.function
def generator_loss(fake_output, img_output, pred, l1_ratio):
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                      labels=tf.ones_like(fake_output))) \
               + l1_loss(img_output, pred) * l1_ratio
    return gen_loss


# @tf.function
def filer_loss(fake_output, img_output, img_filter):
    f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                    labels=tf.ones_like(fake_output))) + \
              l1_loss(img_output, img_filter) *10
              # l1_loss(img_output, img_filter) * 10
    return f_loss



def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img

import tensorflow as tf
def tf_reshape(img):
    if type(img) is np.ndarray or type(img) is tf.Tensor:
        if len(img.shape) == 2:
            img = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    elif type(img) is list:
        if len(img[0].shape) == 2:
            img = tf.stack([tf_reshape(i) for i in img])
    else:
        try:
            img = img.numpy()
            img = tf_reshape(img)
        except:
            raise TypeError("img must be a list, np.ndarray or tf.Tensor")
    img = tf.cast(img, dtype=tf.float32)
    return img


def tfnor_phase(img):
    img = tf.image.per_image_standardization(tf_reshape(img))
    img = img / tf.reduce_max(img)
    img = tf.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])
    return img

#match the 
def tfback_phase(img, input):
    img = tf.numpy_function(match_histograms, [tf_reshape(img)[0,:,:,0], tf_reshape(input)[0,:,:,0]], tf.float32)
    img = tf_reshape(img)
    return img

#contrast limited adaptive histogram equalization
def tf_equalize(img):
    img = tf.image.equalize_adapthist(tf_reshape(img)[0,:,:,0])
    img = tf_reshape(img)
    return img




def FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None):
    """  Parameters: 
            E0 - initial complex field in x-y source plane
            detector_pixel_size - pixel size in microns
            lambda0 - wavelength in nm
            distance_sample_detector - distance_sample_detector-value (distance from sensor to object)
            background - optional background image to divide out from
        
        Returns: E1 - propagated complex field in x-y sensor plane"""  
    import os
    dtype = tf.complex64
    H = tf.cast(ff, dtype)
    detector_wavefield = tf.exp(tf.complex(-absorption, phase))
    detector_wavefield = tf.cast(detector_wavefield, dtype)
    # Compute FFT centered about 0
    E0fft = (tf.signal.fft2d(detector_wavefield))
    E0fft = tf.cast(E0fft, dtype)

    # Multiply spectrum with fresnel phase-factor
    print("E0fft shape: ", E0fft.shape, "H shape: ", H.shape)
    G = H * E0fft
    # Ef = tf.signal.ifft2d(tf.signal.ifftshift(G)) # Output after deshifting Fourier transform
    I = tf.abs(tf.signal.ifft2d(G))**2
    I = tf.cast(I, tf.float32)
    if dark_image is not None and ref_image is not None:
        I = I * (ref_image - dark_image) + dark_image
    I = tf_reshape(I) #without normalizing
    # I = tfnor_phase(tf.reshape(I, [1, I.shape[0], I.shape[1], 1]))
    return I

def phase_fresnel(phase, absorption, ff, shape_x, shape_y):
    paddings = tf.constant([[shape_x // 2, shape_y // 2], [shape_x //2, shape_y // 2]])
    pvalue = tf.reduce_mean(phase[:100, :])
    phase = tf.pad(phase, paddings, 'SYMMETRIC')
    absorption = tf.pad(absorption, paddings, 'SYMMETRIC')
    abfs = tf.complex(-absorption, phase)
    abfs = tf.exp(abfs)
    ifp = tf.abs(tf.signal.ifft2d(ff * tf.signal.fft2d(abfs))) ** 2
    ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.central_crop(ifp, 0.5)
    ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
    ifp = tfnor_phase(ifp)
    return ifp


def ssim_check(image, rec, ff, distance_sample_detector):
    shape_x, shape_y = image.shape
    propagated = FresnelPropagator(rec[1], rec[0], ff, distance_sample_detector)
    data_im = tfnor_phase(tf.reshape(image, [1, shape_x, shape_y, 1]))
    ssim = tf.image.ssim(data_im, propagated, max_val = 1.0)
    print("SSIM between the input image and the reconstructed image is {}".format(ssim))
    return ssim

def peak_signal_to_noise(image, rec_phantom, ff, distance_sample_detector):
    shape_x, shape_y = image.shape
    propagated = FresnelPropagator(rec_phantom[1], rec_phantom[0], ff, distance_sample_detector)
    data_im = tfnor_phase(tf.reshape(image, [1, shape_x, shape_y, 1]))
    # visualize_interact([propagated[0,:,:,0], data_im[0,:,:,0]])
    psnr = tf.image.psnr(data_im, propagated, max_val = 1.0)
    print("Noise to signal ratio is {}".format(psnr))
    return psnr

class GANphase():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        #transform type can be reshape or normalize
        if 'transform_type' not in kwargs:
            self.transform_type = 'normalize'

        if self.transform_type  == 'reshape':
            self.transform_func = tf_reshape 
        elif self.transform_type  == 'normalize':
            self.transform_func = tfnor_phase
        elif self.transform_type  == 'standardize':
            self.transform_func = tf.image.per_image_standardization
        elif self.transform_type == 'contrast':
            factor = 0.7
            self.transform_func = lambda x: tf.image.adjust_contrast(x, factor)
        elif self.transform_type == 'equalize':
            self.transform_func = tf_equalize

        self.transformed_image = self.transform_func(self.image)[0,:,:,0]
        super(GANphase, self).__init__()
        
    def make_model(self):
        if self.filter is True:
            self.filter = make_filter(self.transformed_image.shape[0],
                                  self.transformed_image.shape[1])
        self.generator = make_generator(self.transformed_image.shape[0],
                                        self.transformed_image.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        2)
        self.discriminator = make_discriminator(self.transformed_image.shape[0],
                                                self.transformed_image.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-3)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)

    def make_chechpoints(self):
        checkpoint_dir = '/data/ganrec/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

    @tf.function
    def rec_step(self, transformed_image, ff):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(transformed_image)
            # recon = tfa.image.median_filter2d(recon)
            phase = tfnor_phase(recon[:, :, :, 0]) 
            phase = tf.reshape(phase, [self.shape_x, self.shape_y])
            absorption = (tfnor_phase(recon[:, :, :, 1]))* self.abs_ratio 
            absorption = tf.reshape(absorption, [self.shape_x, self.shape_y])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            # i_rec = tfnor_phase(tf.reshape(FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None), [1, self.shape_x, self.shape_y, 1]))
            propagated = FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None)
            if self.transform_type == 'reshape':
                i_rec = tfback_phase(propagated, transformed_image)[0, :, :, 0]
            else:
                i_rec = self.transform_func(FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None))

            real_output = self.discriminator(transformed_image, training=True)
            fake_output = self.discriminator(i_rec, training=True)
            
            g_loss = generator_loss(fake_output, transformed_image, i_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))


        return {'phase': phase,
                'absorption': absorption,
                'i_rec': i_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}


    @property
    def recon(self):
        # ff = fresnel_operator(self.shape_x, self.shape_y, self.detector_pixel_size, self.distance_sample_detector, self.energy)
        ff = self.fresnel_factor
        transformed_image = tf_reshape(self.transformed_image)
        self.make_model()

        if self.init_model:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')

        phase = np.zeros((self.iter_num, self.shape_x, self.shape_y))
        absorption = np.zeros((self.iter_num, self.shape_x, self.shape_y))
        gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('phase')
            recon_monitor.initial_plot(self.transformed_image)
            # hdisplay = display.display("", display_id=True)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_results = self.rec_step(transformed_image, ff)
            phase[epoch, :, :] = step_results['phase']
            absorption[epoch, :, :] = step_results['absorption']
            i_rec = step_results['i_rec']
            gen_loss[epoch] = step_results['g_loss']
            d_loss = step_results['d_loss']
            # phase[epoch, :, :], absorption[epoch, :, :], i_rec, gen_loss[epoch], d_loss = self.rec_step(transformed_image,
            #                                                                                                   ff)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    i_rec = np.reshape(i_rec, (self.shape_x, self.shape_y))
                    i_diff = np.abs(i_rec - self.transformed_image)
                    phase_plt = np.reshape(phase[epoch], (self.shape_x, self.shape_y))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss, None)
                    # hdisplay.update(fig)
                    if self.save_wpath != None:
                        import skimage.io as io
                        io.imsave(self.save_wpath+ 'iter_' +str(epoch)+'.tif', phase_plt, check_contrast=False)
                        io.imsave(self.save_wpath+ 'iter_' +str(epoch)+'_diff.tif', i_diff, check_contrast=False)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1, gen_loss[epoch], d_loss.numpy()))
        recon_monitor.close_plot()
        
        if self.save_wpath != None:
            import skimage.io as io
            self.generator.save(self.init_wpath+'generator.h5')
            self.discriminator.save(self.init_wpath+'discriminator.h5')
            io.imsave(self.save_wpath+ 'final_phase_iter_' +str(self.iter_num)+'.tif', phase[epoch][1], check_contrast=False)
            io.imsave(self.save_wpath+ 'final_absorption_iter_' +str(self.iter_num)+'.tif', absorption[epoch][0], check_contrast=False)
        return absorption[epoch], phase[epoch], i_rec, gen_loss, d_loss

class Ganrec_Dataloader():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.kwargs.update(get_all_info(**kwargs))
        keys = self.kwargs.keys()
        [self.__setattr__(key, self.kwargs[key]) for key in keys]
        self.dims = (self.ND, self.shape_x, self.shape_y)
        self.transformed_images = None

        super(Ganrec_Dataloader, self).__init__()

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx = None):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            kwargs.update(get_all_info(**kwargs))
            keys = kwargs.keys()
            [self.__setattr__(key, kwargs[key]) for key in keys]
            
            if type(idx) is not list:    
                self.transformed_images = tfnor_phase(self.image)
            else:
                images = [tfnor_phase(self.image[i]) for i in range(len(self.idx))]
                self.transformed_images= tf.stack(images)
        else:
            if type(self.idx) is not list:
                self.transformed_images = tfnor_phase(self.image)
            else:
                self.transformed_images = tf.stack([tfnor_phase(self.image[i]) for i in range(len(self.idx))])
        return self.transformed_images
    def normalize(self, idx = None):
        image = self.__getitem__(idx)
        image = tfnor_phase(image)
        return image
    
    def get_kwargs(self):
        return self.__dict__
    
    def visualize(self, idx = None, random = False):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            kwargs.update(get_all_info(**kwargs))
            keys = kwargs.keys()
            [self.__setattr__(key, kwargs[key]) for key in keys]
            images = self.image
        else:
            images = self.image
        if type(images) is not list:
            images = [images]

        rows = int(np.sqrt(len(images)))
        if rows ==1:
            cols = len(images)
        else:
            cols = rows + 1
        print("rows: {}, cols: {}".format(rows, cols))
        if random == False:
            visualize(images, rows = rows, cols = cols)
        else:
            visualize(images, rows = rows, cols = cols, random=True)

    def normal_visualize(self, idx = None, random = False):
        if self.transformed_images is None:
            self.__getitem__(idx)
        print(self.transformed_images.shape)
        images = [self.transformed_images[i, :, :, 0].numpy() for i in range(self.transformed_images.shape[0])]
        rows = int(np.sqrt(len(images)))
        cols = rows + 1
        if random == False:    
            visualize(images, rows = rows, cols = cols, random = False )
        else:
            visualize(images, random=True)
    
    def create_ganphase_class(self, id = None, **kwargs):
        try:
            self.transformed_images
            if self.transformed_images is None:
                self.transformed_images = self.__getitem__(id)
        except:
            raise ValueError("transformed_images is not defined")

        self.kwargs['transformed_image'] = self.transformed_images[0, :, :, 0]
        self.ganphase = GANphase(**self.kwargs)
        return self.ganphase

    def train_model(self, id = None, **kwargs):
        if kwargs is not None:
            self.kwargs.update(kwargs)
        self.ganphase = self.create_ganphase_class(id, **self.kwargs)
        print("tranformed_images shape: {}".format(self.transformed_images.shape))
        self.retrieved = self.ganphase.recon
        return self.retrieved
    
    def forward_propagate(self):
        self.phase = self.retrieved[1]
        self.attenuation = self.retrieved[0]
        return FresnelPropagator(self.phase, self.attenuation, self.fresnel_factor, self.distance_sample_detector)

    def ssim_check(self):
        self.ssim_value = tf.image.ssim(self.transformed_images[0,:,:,0], self.propagated_forward, max_val = 1.0)
        print("SSIM value is {}".format(self.ssim_value))
        return self.ssim_value
