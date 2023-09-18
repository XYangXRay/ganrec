from models import make_generator, make_discriminator, make_filter
from utils import *
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




def tf_reshape(img):
    """
    output: [1, img.shape[0], img.shape[1], 1] or [img.shape[0], img.shape[1], img.shape[2], 1]
    """
    try:
        img = tf.convert_to_tensor(img)
        return tf.cast(tf.reshape(img, [1, img.shape[0], img.shape[1], 1]), dtype=tf.float32)
    except:
        if type(img) is np.ndarray or type(img) is tf.Tensor:
            if len(img.shape) == 2:
                img = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
            elif len(img.shape) == 3:
                img = tf.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])
            elif len(img.shape) == 4:
                img = img
        elif type(img) is list:
            if len(img[0].shape) == 2:
                img = tf.stack([tf_reshape(i) for i in img])
                img = tf.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])
            elif len(img[0].shape) == 3:
                img = tf.stack([tf_reshape(i) for i in img])
            elif len(img[0].shape) == 4:
                img = tf.stack(img)
            try:
                img = img.numpy()
                img = tf_reshape(img)
            except:
                raise TypeError("img must be a list, np.ndarray or tf.Tensor")
        img = tf.cast(img, dtype=tf.float32)
        return img

def tfnor_phase(img):
    img = tf_reshape(img)
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img

def tfback_phase(img, input):
    img_final = tf.numpy_function(match_histograms, [img, input], tf.float32)
    return tf_reshape(img_final)

def tfbright_adjust(img, factor):
    img = tf_reshape(img)
    tf.image.adjust_brightness(img, factor)
    return img

def tfcontrast_adjust(img, factor):
    img = tf_reshape(img)
    tf.image.adjust_contrast(img, factor)
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
    G = H * E0fft
    # Ef = tf.signal.ifft2d(tf.signal.ifftshift(G)) # Output after deshifting Fourier transform
    I = tf.abs(tf.signal.ifft2d(G))**2
    I = tf.cast(I, tf.float32)
    if dark_image is not None and ref_image is not None:
        I = I * (ref_image - dark_image) + dark_image
    I = tf_reshape(I) #without normalizing
    # I = tfnor_phase(tf.reshape(I, [1, I.shape[0], I.shape[1], 1]))
    return I

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

def measure_reconstruction_quality(img1, img2, experiment_name, csv_file = None, iteration = 0, save = True, epoch_time = 0, total_time = 0):
    assert img1.shape == img2.shape, "img1 and img2 must have the same shape"
    px = img1.shape[0]
    py = img1.shape[1]
    sum_diff = tf.reduce_sum(img1 - img2)/(px*py)
    sum_abs_diff = tf.reduce_sum(tf.abs(img1 - img2))/(px*py)
    sum_squared_diff = tf.reduce_sum(tf.square(img1 - img2))/(px*py)
    psnr = tf.image.psnr(img1, img2, max_val = 1.0)
    ssim = tf.image.ssim(img1, img2, max_val = 1.0)
    nrmse = tf.math.sqrt(tf.reduce_sum(tf.square(img1 - img2))/(px*py))
    mssim = tf.image.ssim_multiscale(img1, img2, max_val = 1.0)
    values = [sum_diff, sum_abs_diff, sum_squared_diff, psnr, ssim, nrmse, mssim]
    values = [v.numpy() for v in values]
    values.insert(0, iteration)
    values.insert(0, experiment_name)
    values.append(epoch_time)
    values.append(total_time)

    import pandas as pd
    df = pd.DataFrame(values, index = ['experiment_name', 'iter', 'sum of pointwise difference', 'sum of absolute value of pointwise difference', 'sum of squared difference', 'PSNR', 'SSIM', 'NRMSE', 'MSSIM', 'epoch_time', 'total_time'])
    print(df)
    if save:
        if csv_file is not None:
            try:
                df_old = pd.read_csv(csv_file, index_col = 0)
                df = pd.concat([df_old, df], axis = 1)
            except:
                pass
            df.to_csv(csv_file)
        else:
            df.to_csv('metrics.csv')

    return df
     
def transform_func(transform_type, transform_factor = 0.5):
    if transform_type   == 'reshape':
        transform_func = tf_reshape
        tf.config.run_functions_eagerly(True)  #important to set this to True due to the match function
        tf.config.experimental_run_functions_eagerly(True)
    elif transform_type   == 'normalize':
        transform_func = tfnor_phase
        tf.config.run_functions_eagerly(False)
    elif transform_type   == 'standardize':
        transform_func = tf.image.per_image_standardization
        tf.config.run_functions_eagerly(False)
    elif transform_type  == 'contrast':
        transform_func = lambda x: tf.image.adjust_contrast(tf_reshape(x), transform_factor)
        tf.config.run_functions_eagerly(False)
    elif transform_type  == 'brightness':
        transform_func = lambda x: tf.image.adjust_brightness(tf_reshape(x), transform_factor)
        tf.config.run_functions_eagerly(False)
    elif transform_type  == 'gamma':
        transform_func = lambda x: tf.image.adjust_gamma(tf_reshape(x), transform_factor)
        tf.config.run_functions_eagerly(False)
    elif transform_type  == 'hue':
        transform_func = lambda x: tf.image.adjust_hue(tf_reshape(x), transform_factor)
        tf.config.run_functions_eagerly(False)
    else:
        transform_func = tfnor_phase
    return transform_func

class GANphase():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if 'transform_type' not in kwargs:
            self.kwargs['transform_type'] = 'normalize'
        if 'transform_factor' not in kwargs:
            self.kwargs['transform_factor'] = 0.5
        self.transform_func = transform_func(self.transform_type, self.transform_factor)

        if self.transformed_images is None:
            transformed_images = self.transform_func(self.image)
            if len(transformed_images.shape) == 3:
                transformed_images = tf.reshape(transformed_images, [transformed_images.shape[0], transformed_images.shape[1],transformed_images.shape[2], 1])
            self.transformed_images = transformed_images
            self.transformed_image = transformed_images[0,:,:,0]
            if self.transform_type == 'crop':
                if 'transform_factor' not in kwargs:
                    self.transform_factor = 0.9
                self.image = tf_reshape(self.image), self.transform_factor
                self.image = tf.image.central_crop(self.image, self.transform_factor)
                self.shape_x = self.image.shape[0]
                self.shape_y = self.image.shape[1]
                self.fresnel_factor = fresnel_operator( self.shape_x,self.shape_y, self.detector_pixel_size, self.distance_sample_detector, self.energy_kev)
                self.kwargs['fresnel_factor'] = self.fresnel_factor
                self.kwargs['shape_x'] = self.shape_x
                self.kwargs['shape_y'] = self.shape_y    
        super(GANphase, self).__init__()
        
    def make_model(self):
        
        if self.filter is True:
            self.filter = make_filter(self.transformed_image.shape[0],
                                  self.transformed_image.shape[1])
        self.generator = make_generator(self.shape_x,
                                        self.shape_y,
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        2)
        # self.generator = unet(self.transformed_image.shape[0],self.transformed_image.shape[1], 2, 6)
        self.discriminator = make_discriminator(self.shape_x,
                                        self.shape_y)
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
            absorption = ( 1 - tfnor_phase(recon[:, :, :, 1] ))* self.abs_ratio
            absorption = tf.reshape(absorption, [self.shape_x, self.shape_y])
            # absorption = tf.nn.silu(absorption)
            #set a constraint that the phase and absorption should be positive
            
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            # i_rec = tfnor_phase(tf.reshape(FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None), [1, self.shape_x, self.shape_y, 1]))
                
            propagated = FresnelPropagator(phase, absorption, ff, ref_image = None, dark_image = None)
            
            if self.transform_type == 'histogram':
                i_rec = tfnor_phase(propagated)
                i_rec = self.transform_func(propagated, transformed_image)
            else:
                i_rec = self.transform_func(propagated)
            
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
        
        
        # ff = self.fresnel_factor
        self.shape_x = self.transformed_image.shape[0]
        self.shape_y = self.transformed_image.shape[1]
        ff = fresnel_operator(self.shape_x, self.shape_y, self.detector_pixel_size, self.distance_sample_detector, self.energy)
        transformed_image = tf.reshape(self.transformed_image, [1, self.shape_x, self.shape_y, 1])
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
        
        epoch_time = []*self.iter_num
        for epoch in range(self.iter_num):
            start_time = time.time()
            ###########################################################################
            ## Call the rconstruction step
            print("shape of the self.transformed image is {}".format(self.transformed_image.shape))
            print("shape of the transformed images is {}".format(transformed_image.shape))
            print("ff shape is {}".format(ff.shape))
                
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
            epoch_time.append(time.time() - start_time)

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
        self.phase = phase[epoch]
        self.absorption = absorption[epoch]
        self.i_rec = i_rec
        self.gen_loss = gen_loss
        self.d_loss = d_loss
        self.epoch_time = sum(epoch_time)/len(epoch_time)
        self.total_time = sum(epoch_time)
        return absorption[epoch], phase[epoch], i_rec, gen_loss, d_loss
    
    def measure_quality(self, interactive = False):
        propagated = FresnelPropagator(self.phase, self.absorption, self.fresnel_factor)
        matched = tfback_phase(self.transformed_image, propagated[0,:,:,0])
        df = measure_reconstruction_quality(matched, propagated, self.experiment_name, 'metrics.csv', iteration = self.iter_num, save = True, epoch_time = self.epoch_time, total_time = self.total_time)
        df = measure_reconstruction_quality(self.transformed_images, propagated, self.experiment_name + '_unmatched', 'metrics.csv', iteration = self.iter_num, save = True, epoch_time = self.epoch_time, total_time = self.total_time)
        
        if interactive:
            visualize_interact([self.phase, self.absorption, propagated[0,:,:,0], matched])
        else:
            plot_or_show_images([self.phase, self.absorption], show_or_plot = 'show', random = False, cmap = 'None', figsize = (20, 20), title = "Figure1: phase and attenuation")
            plot_or_show_images([self.transformed_image, propagated[0,:,:,0]], show_or_plot = 'show', random = False, cmap = 'None', figsize = (20, 20), title = "Figure2: input and propagated image")
            plot_or_show_images([(matched - propagated)[0, :, :, 0]], show_or_plot = 'show', random = False, cmap = 'None', figsize = (20, 20), title = "Figure3: matched - input image")
        
        return df, propagated, matched

class Ganrec_Dataloader():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.kwargs.update(get_all_info(**kwargs))
        keys = self.kwargs.keys()
        [self.__setattr__(key, self.kwargs[key]) for key in keys]
        self.dims = (self.ND, self.shape_x, self.shape_y)

        if 'transform_type' not in kwargs:
            self.kwargs['transform_type'] = 'normalize'
        if 'transform_factor' not in kwargs:
            self.kwargs['transform_factor'] = 0.5

        if self.transform_type  == 'reshape':
            self.transform_func = tf_reshape 
        elif self.transform_type  == 'normalize':
            self.transform_func = tfnor_phase
        elif self.transform_type  == 'standardize':
            self.transform_func = tf.image.per_image_standardization
        elif self.transform_type == 'contrast':
            if 'transform_factor' not in kwargs:
                self.transform_factor = 0.5
            self.transform_func = lambda x: tf.image.adjust_contrast(tf_reshape(x), self.transform_factor)
        elif self.transform_type == 'brightness':
            if 'transform_factor' not in kwargs:
                self.transform_factor = 0.5
            self.transform_func = lambda x: tf.image.adjust_brightness(tf_reshape(x), self.transform_factor)
        elif self.transform_type == 'gamma':
            if 'transform_factor' not in kwargs:
                self.transform_factor = 0.5
            self.transform_func = lambda x: tf.image.adjust_gamma(tf_reshape(x), self.transform_factor)
        elif self.transform_type == 'hue':
            if 'transform_factor' not in kwargs:
                self.transform_factor = 0.5
            self.transform_func = lambda x: tf.image.adjust_hue(tf_reshape(x), self.transform_factor)
        else:
            self.transform_func = tfnor_phase       
        
        if self.transform_type == 'crop':
            self.image = tf_reshape(self.image)
            self.image = tf.image.central_crop(self.image, self.transform_factor)
            self.shape_x = self.image.shape[0]
            self.shape_y = self.image.shape[1]
            self.fresnel_factor = fresnel_operator( self.shape_x,self.shape_y, self.detector_pixel_size, self.distance_sample_detector, self.energy_kev)
            self.kwargs['fresnel_factor'] = self.fresnel_factor
            self.kwargs['shape_x'] = self.shape_x
            self.kwargs['shape_y'] = self.shape_y

        self.transformed_images = self.transform_func(self.image)
        if len(self.transformed_images.shape) == 3:
            self.transformed_images = tf.reshape(self.transformed_images, [self.transformed_images.shape[0], self.transformed_images.shape[1], self.transformed_images.shape[2], 1])
        self.transformed_image = self.transformed_images[0,:,:,0]
        self.kwargs['transformed_image'] = self.transformed_image
        self.kwargs['transformed_images'] = self.transformed_images
        super(Ganrec_Dataloader, self).__init__()

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx = None):
        if idx is not None:
            kwargs = self.kwargs
            kwargs["idx"] = idx
            new = Ganrec_Dataloader(**kwargs)
            return new.transformed_imagess
        else:
            return self.transformed_images
    
    def normalize(self, idx = None):
        image = self.__getitem__(idx)
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
        if idx is not None:
            transformed_images = self.__getitem__(idx)
        else:
            transformed_images = self.transformed_images
        images = [transformed_images[i, :, :, 0].numpy() for i in range(transformed_images.shape[0])]
        rows = int(np.sqrt(len(images)))
        cols = rows + 1
        if random == False:    
            visualize(images, rows = rows, cols = cols, random = False )
        else:
            visualize(images, random=True)
    
    def create_ganphase_class(self, idx = None, **kwargs):
        if idx is not None:
            transformed_images = self.__getitem__(idx)
        else:
            transformed_images = self.transformed_images
        kwargs = self.kwargs
        kwargs['transformed_images'] = transformed_images
        kwargs['idx'] = idx
        ganphase = GANphase(**kwargs)
        return ganphase

    def train_model(self, idx = None, **kwargs):
        self.ganphase = self.create_ganphase_class(idx, **kwargs)
        self.retrieved = self.ganphase.recon
        self.phase = self.retrieved[1]
        self.attenuation = self.retrieved[0]
        return self.retrieved
    
    def forward_propagate(self, distance = None):
        assert self.ganphase is not None, "ganphase is not defined"
        if distance is None:
            distance = self.distance_sample_detector
        self.propagated_forward = FresnelPropagator(self.phase, self.attenuation, self.fresnel_factor, distance)
        return self.propagated_forward
    def ssim_check(self):
        intensity = FresnelPropagator(self.phase, self.attenuation, self.fresnel_factor, self.distance_sample_detector)
        #match
        intensity = tfback_phase(intensity, self.transformed_images[0,:,:,0])
        self.ssim_value = tf.image.ssim(self.transformed_images[0,:,:,0], intensity, max_val = 1.0)
        print("SSIM value is {}".format(self.ssim_value))
        return self.ssim_value
    
    def quality_check(self, x, y):
        self.quality = measure_reconstruction_quality(x, y, self.kwargs['experiment_name'], csv_file = self.kwargs['csv_file'])
        return self.quality
