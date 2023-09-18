import tensorflow as tf
from tensorflow_addons.image import median_filter2d

import numpy as np

from models import *
from utils import *

from joblib import Parallel, delayed


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


def tfnor_tomo(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    img = img - tf.reduce_min(img)
    # img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    return img


def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img

def tfnor_diff(img):
    # img = tf.image.per_image_standardization(img)
    # img = img / tf.reduce_max(img)
    # img = img - tf.reduce_min(img)
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    return img

def avg_results(recon, loss):
    sort_index = np.argsort(loss)
    recon_tmp = recon[sort_index[:10], :, :, :]
    return np.mean(recon_tmp, axis=0)

def phase_fresnel(phase, absorption, ff, px):
    paddings = tf.constant([[px // 2, px // 2], [px // 2, px // 2]])
    # padding1 = tf.constant([[px // 2, px // 2], [0, 0]])
    # padding2 = tf.constant([[0, 0], [px // 2, px // 2]])
    pvalue = tf.reduce_mean(phase[:100, :])
    # phase = tf.pad(phase, paddings, 'CONSTANT',constant_values=1)
    phase = tf.pad(phase, paddings, 'SYMMETRIC')
    # phase = tf.pad(phase, paddings, 'REFLECT')
    absorption = tf.pad(absorption, paddings, 'SYMMETRIC')
    # phase = phase
    # absorption = absorption
    abfs = tf.complex(-absorption, phase)
    abfs = tf.exp(abfs)
    ifp = tf.abs(tf.signal.ifft2d(ff * tf.signal.fft2d(abfs))) ** 2
    ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.central_crop(ifp, 0.5)
    ifp = tf.image.per_image_standardization(ifp)
    ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
    # ifp = tfnor_phase(ifp)
    return ifp


def phase_fresnel_2(cosine, sine, absorption, ff, px):
    paddings = tf.constant([[px // 2, px // 2], [px // 2, px // 2]])
    # padding1 = tf.constant([[px // 2, px // 2], [0, 0]])
    # padding2 = tf.constant([[0, 0], [px // 2, px // 2]])
    pvalue = tf.reduce_mean(cosine[:100, :])
    # phase = tf.pad(phase, paddings, 'CONSTANT',constant_values=1)
    cosine = tf.pad(cosine, paddings, 'SYMMETRIC')
    sine = tf.pad(sine, paddings, 'SYMMETRIC')
    # phase = tf.pad(phase, paddings, 'REFLECT')
    absorption = tf.pad(absorption, paddings, 'SYMMETRIC')
    # abfs = #amplitude * (cosine - i * sine)
    abfs = tf.complex(tf.exp(-1*absorption) * cosine, tf.exp(-1*absorption)  * sine)
    # abfs = tf.complex(-absorption, phase)
    # abfs = tf.exp(abfs)
    ifp = tf.abs(tf.signal.ifft2d(ff * tf.signal.fft2d(abfs))) ** 2
    ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.central_crop(ifp, 0.5)
    ifp = tf.image.per_image_standardization(ifp)
    ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
    # ifp = tfnor_phase(ifp)
    return ifp

def phase_fraunhofer(phase, absorption):
    wf = tf.complex(absorption, phase)
    # wf = tf.complex(phase, absorption)

    # wf = mask_img(wf)
    # wf = tf.multiply(ampl, tf.exp(phshift))
    # wf = tf.manip.roll(wf, [160, 160], [0, 1])
    ifp = tf.square(tf.abs(tf.signal.fft2d(wf)))
 
    
    # # adding log to the fft
    # ifp = tf.math.log(ifp+8000)
    ifp = tf.math.log(ifp+10000)
    # ifp = tf.math.log(tf.abs(tf.signal.fft2d(wf))+1)
    # ifp = tf.math.log(tf.square(tf.abs(tf.signal.fft2d(wf)))+1)
    ifp = tf.signal.fftshift(ifp)
  
    # ifp = tf.roll(ifp, [256, 256], [0, 1])
    ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.per_image_standardization(ifp)
    ifp = tfnor_diff(ifp)
    return ifp


def diffusion_layer(input_tensor, sigma=1.0):
    size = int(sigma*4)
    x = tf.linspace(-3.0, 3.0, size)
    z = (1.0/(sigma*tf.sqrt(2.0*3.1415)))*tf.exp(-tf.square(x)/(2.0*sigma*sigma))
    z_2d = tf.matmul(tf.reshape(z, [size, 1]), tf.reshape(z, [1, size]))
    z_4d = tf.reshape(z_2d, [size, size, 1, 1])
    return tf.nn.depthwise_conv2d(input_tensor, z_4d, [1, 1, 1, 1], 'SAME')

class GANphase:
    def __init__(self, i_input, energy, z, pv, **kwargs):
        phase_args = _get_GANphase_kwargs()
        phase_args.update(**kwargs)
        super(GANphase, self).__init__()
        self.i_input = i_input
        self.px, self.py = i_input.shape
        self.energy = energy
        self.z = z
        self.pv = pv
        self.internal_iter = kwargs['internal_iter']
        self.last_retrieval = kwargs['last_retrieval']
        self.iter_num = phase_args['iter_num']
        self.conv_num = phase_args['conv_num']
        self.conv_size = phase_args['conv_size']
        self.dropout = phase_args['dropout']
        self.l1_ratio = phase_args['l1_ratio']
        self.abs_ratio = phase_args['abs_ratio']
        self.g_learning_rate = phase_args['g_learning_rate']
        self.d_learning_rate = phase_args['d_learning_rate']
        self.phase_only = phase_args['phase_only']
        self.save_wpath = phase_args['save_wpath']
        self.init_wpath = phase_args['init_wpath']
        self.init_model = phase_args['init_model']
        self.save_model = phase_args['save_model']
        self.recon_monitor = phase_args['recon_monitor']
        self.filter_type = phase_args['filter_type']
        self.output_num = phase_args['output_num']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.i_input.shape[0],
                                  self.i_input.shape[1])
        self.generator = make_generator_atan(self.i_input.shape[0],
                                        self.i_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        self.output_num)
        self.discriminator = make_discriminator(self.i_input.shape[0],
                                                self.i_input.shape[1])
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
    def rec_step(self, i_input, ff, phase_input = None, absorption_input = None):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
        
            #change from list to tensor
            if type(recon) == list:
                print(len(recon), recon[0].shape, recon[1].shape)
                sin_phase = recon[0][0, :, :, 0]
                cos_phase = recon[0][0, :, :, 1]
                
                phase = tf.reshape(tf.atan2(sin_phase, cos_phase), [1, self.px, self.py,1])
                absorption = (1 - tfnor_phase(tf.reshape(recon[1][0, :, :, 0], [1, self.px, self.py, 1])) )* self.abs_ratio
            
                phase = tf.reshape((phase), [self.px, self.py])
                absorption = tf.reshape((absorption), [self.px, self.py])
                # i_rec = phase_fresnel(phase, absorption, ff, self.px)
                i_rec = phase_fresnel_2(cos_phase, sin_phase, absorption, ff, self.px)
            else:   
                phase = tfnor_phase(tf.reshape(recon[:, :, :, 0], [1, self.px, self.py, 1]))
                absorption = (1 - tfnor_phase(tf.reshape(recon[:, :, :, 1], [1, self.px, self.py, 1])) )* self.abs_ratio
                phase = tf.reshape(phase, [self.px, self.py])
                absorption = tf.reshape((1- absorption) * self.abs_ratio, [self.px, self.py])
            
                if self.phase_only:
                    absorption = tf.zeros_like(phase)
                i_rec = phase_fresnel(phase, absorption, ff, self.px)
            # i_rec = phase_fresnel_2(tf.reshape((recon[:, :, :, 0]), [self.px, self.py]), tf.reshape((recon[:, :, :, 1]), [self.px, self.py]), tf.reshape(tfnor_phase(recon[:, :, :, 2]), [self.px, self.py]), ff, self.px)
        

            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
            if phase_input is not None and absorption_input is not None:
                #change their shape
                phase_input = tf.reshape(phase_input, [1, self.px, self.py, 1])
                absorption_input = tf.reshape(absorption_input, [1, self.px, self.py, 1])
                phase = tf.reshape(phase, [1, self.px, self.py, 1])
                absorption = tf.reshape(absorption, [1, self.px, self.py, 1])
                #discremenate between the real phase and the generated phase, and the real absorption and the generated absorption and the real intensity and the generated intensity
                real_output_phase = self.discriminator(phase_input, training=True)
                fake_output_phase = self.discriminator(phase, training=True)
                real_output_abs = self.discriminator(absorption_input, training=True)
                fake_output_abs = self.discriminator(absorption, training=True)
                real_output_i = self.discriminator(i_input, training=True)
                fake_output_i = self.discriminator(i_rec, training=True)
                # fake_output_i_2 = self.discriminator(i_rec_2, training=True)

                g_loss = generator_loss(fake_output_i, i_input, i_rec, self.l1_ratio) + \
                            generator_loss(fake_output_phase, phase_input, phase, self.l1_ratio) + \
                            generator_loss(fake_output_abs, absorption_input, absorption, self.l1_ratio)
                            # generator_loss(fake_output_i_2, i_input, i_rec_2, self.l1_ratio)
                            
                
                d_loss = discriminator_loss(real_output, fake_output) + \
                        discriminator_loss(real_output_phase, fake_output_phase) + \
                            discriminator_loss(real_output_abs, fake_output_abs) + \
                            discriminator_loss(real_output_i, fake_output_i)
                                # discriminator_loss(real_output_i, fake_output_i_2)
                    
                
            else:
                g_loss = generator_loss(fake_output, i_input, i_rec, self.l1_ratio)
                d_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        


        return {'phase':  tf.reshape(phase, [self.px, self.py]),
                'absorption': tf.reshape(absorption, [self.px, self.py]),
                'i_rec': i_rec,
                'g_loss': g_loss,
                'd_loss': d_loss,
                'recon':recon}


    def multi_propagation(self, i_input, ffs, retrieved_phase = None, retrieved_absorption = None, filter_type = None):
        if retrieved_phase is not None and retrieved_absorption is not None:
            if filter_type is None:
                filter_type = self.filter_type

            if filter_type == 'median':
                # filter the phase and absorption
                filtered_phase = median_filter2d( tf.reshape(retrieved_phase, [1, self.px, self.py, 1]))[0, :, :, 0]
                filtered_absorption = median_filter2d( tf.reshape(retrieved_absorption, [1, self.px, self.py, 1]))[0, :, :, 0]
                step_results = [self.rec_step(phase_fresnel(filtered_phase, filtered_absorption, ffs[i], self.px), ffs[i], phase_input = filtered_phase, absorption_input = filtered_absorption) for i in range(1, len(ffs))]
            if filter_type == 'contrast':
                # add contrast to the phase and absorption
                filtered_phase = (retrieved_phase - tf.reduce_min(retrieved_phase))/ (tf.reduce_max(retrieved_phase) - tf.reduce_min(retrieved_phase))
                filtered_absorption = (retrieved_absorption - tf.reduce_min(retrieved_absorption))/ (tf.reduce_max(retrieved_absorption) - tf.reduce_min(retrieved_absorption))
                step_results = [self.rec_step(phase_fresnel(filtered_phase, filtered_absorption, ffs[i], self.px), ffs[i], phase_input = filtered_phase, absorption_input = filtered_absorption) for i in range(1, len(ffs))]
            
            if filter_type == 'noise':
                # add noise to the phase and absorption
                filtered_phase = retrieved_phase + tf.random.normal(retrieved_phase.shape, mean=0.0, stddev=0.1, dtype=tf.float32)
                filtered_absorption = retrieved_absorption + tf.random.normal(retrieved_absorption.shape, mean=0.0, stddev=0.1, dtype=tf.float32)
                step_results = [self.rec_step(phase_fresnel(filtered_phase, filtered_absorption, ffs[i], self.px), ffs[i], phase_input = filtered_phase, absorption_input = filtered_absorption) for i in range(1, len(ffs))]
            if filter_type == 'diffuse':
                # diffuse
                # ffs = [ffs[0]]*5
                sigma_coeff = list(np.arange(0.9, 1.1, 0.2))
                ffs = [ffs[0]]*len(sigma_coeff)
                filtered_phases = [diffusion_layer(tf.reshape(retrieved_phase, [1, self.px, self.py, 1]), sigma=sigma_coeff[i])[0, :, :, 0] for i in range(len(sigma_coeff))]
                filtered_absorptions = [diffusion_layer(tf.reshape(retrieved_absorption, [1, self.px, self.py, 1]), sigma=sigma_coeff[i])[0, :, :, 0] for i in range(len(sigma_coeff))]
                step_results = [self.rec_step(phase_fresnel(filtered_phases[i], filtered_absorptions[i], ffs[0], self.px), ffs[0], phase_input = filtered_phases[i], absorption_input = filtered_absorptions[i]) for i in range(len(sigma_coeff))]
            if filter_type == 'phase_only':
                # diffuse
                sigma_coeff = list(np.arange(0.5, 0.5*len(ffs), 0.5))
                filtered_phases = [diffusion_layer(tf.reshape(retrieved_phase, [1, self.px, self.py, 1]), sigma=sigma_coeff[i])[0, :, :, 0] for i in range(len(sigma_coeff))]
                filtered_absorption = np.zeros_like(filtered_phases[0])
                step_results = [self.rec_step(phase_fresnel(filtered_phases[i], filtered_absorption, ffs[i], self.px), ffs[i], phase_input = filtered_phases[i], absorption_input = filtered_absorption) for i in range(len(sigma_coeff))]
            if filter_type == 'alternate':
                step_results = [self.rec_step(phase_fresnel(retrieved_phase, retrieved_absorption, ffs[i], self.px), ffs[i], phase_input = retrieved_phase, absorption_input = retrieved_absorption) for i in range(1, len(ffs))]                
        else:
            step_results = self.rec_step(i_input, ffs[0])
        return step_results
    
    @property
    def recon(self):
        ff = ffactor(self.px * 2, self.py *2, self.energy, self.z, self.pv)
        ffs = []
        
        ffs.append(ff)
        # for i in range(1, self.internal_iter):
            # z_i = self.z * (1 + 1/i)
            # ffs.append(ffactor(self.px * 2, self.py *2, self.energy, z_i, self.pv))
        for i in range(1, self.internal_iter):
            z_i = self.z / (1 + 1/i)
            ffs.append(ffactor(self.px * 2, self.py *2, self.energy, z_i, self.pv))
        ffs.append(ff)
        # print(ff.shape, ff.max(), ff.min())

        i_input = np.reshape(self.i_input, (1, self.px, self.py, 1))
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()

        if self.init_model:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')
            print('discriminator is initilized')

        phase = np.zeros((self.iter_num, self.px, self.py))
        absorption = np.zeros((self.iter_num, self.px, self.py))
        gen_loss = np.zeros(self.iter_num)
        i_rec_all = np.zeros((self.iter_num, self.px, self.py))

        ###########################################################################
        # Reconstruction process monitor
        
        plot_x, plot_loss = [], []
        if self.recon_monitor:
            recon_monitor = RECONmonitor('phase')
            recon_monitor.initial_plot(self.i_input)

        side_propagation = []
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            
            step_results = self.rec_step(i_input, ffs[0])
            side_propagation.append(step_results['recon'])
            # side_propagation = list(tfnor_phase(tf.stack(side_propagation, axis = 0)))
            # print('epoch: {}, g_loss: {}, d_loss: {}'.format(epoch, step_results['g_loss'], step_results['d_loss']))

            if self.last_retrieval:
                if epoch % 100 == 0:
                    retrieved_phase = step_results['phase']
                    retrieved_absorption = step_results['absorption']
                    step_result_i = self.multi_propagation(i_input, ffs, retrieved_phase, retrieved_absorption, filter_type = self.filter_type)
                    # step_result_i = self.rec_step(median_filter2d(i_input), ff)    
                
            phase[epoch, :, :] = step_results['phase']
            absorption[epoch, :, :] = step_results['absorption']
            i_rec = step_results['i_rec']
            i_rec_all[epoch, :, :] = np.reshape(i_rec, (self.px, self.py))
            gen_loss[epoch] = step_results['g_loss']
            d_loss = step_results['d_loss']
            ###########################################################################
                
            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]

            if (epoch + 1) % self.iter_num == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if self.recon_monitor:  
                    i_rec = np.reshape(i_rec, (self.px, self.py))
                    i_diff = np.abs(i_rec - np.reshape(self.i_input, (self.px, self.py)))
                    phase_plt = np.reshape(phase[epoch], (self.px, self.py))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
        if self.recon_monitor:
            recon_monitor.close_plot()

        if self.save_model:
            self.generator.save(self.save_wpath+'generator.h5')
            self.discriminator.save(self.save_wpath+'discriminator.h5')

        return absorption[epoch], phase[epoch], i_rec_all[epoch], gen_loss, side_propagation

def train_one_image(data, **kwargs):
    import time
    import os
    import dxchange
    from ganrec_dataloader import measure_reconstruction_quality, tf_reshape, tfback_phase
    abs_ratio = kwargs['abs_ratio']
    idx = kwargs['idx']

    assert type(data) is not list, 'data should be a list'
    gan_phase_object = GANphase(data, **kwargs)
    start = time.time()
    absorption, phase, i_rec_all, gen_loss, side_propagation = gan_phase_object.recon
    end = time.time()
    
    if 'save_wpath' not in kwargs.keys():
        save_wpath = None

    else:
        save_wpath = kwargs['save_wpath']
        
    if save_wpath is not None:
        #if path does not exist, create it
        if not os.path.exists(save_wpath):
            os.makedirs(save_wpath)
            os.makedirs(save_wpath+'/abs/')
            os.makedirs(save_wpath+'/phase/')
            os.makedirs(save_wpath+'/propagation/')
            os.makedirs(save_wpath+'/gen_loss/')

        dxchange.write_tiff(absorption, save_wpath+'abs/abs{}'.format(idx))
        dxchange.write_tiff(phase, save_wpath+'phase/phase_{}'.format(idx))
        dxchange.write_tiff(i_rec_all, save_wpath+'propagation/propagation{}'.format(idx))
        np.save(save_wpath+'gen_loss/gen_loss_{}'.format(idx), gen_loss)
        # np.save(save_wpath+'side_propagation/side_propagation_{}'.format(idx), side_propagation)
    matched = tfback_phase(tf_reshape(i_rec_all), tf_reshape(data))
    time_count = np.array([end-start])
    df = measure_reconstruction_quality(img1=tf_reshape(i_rec_all), img2 = tf_reshape(data), experiment_name = 'reconstructed'+str(idx)+'_'+str(abs_ratio), csv_file = 'Vojtech_fixed.csv', iteration = kwargs['iter_num'], save = True, epoch_time = time_count[0], total_time = np.sum(time_count))
    return absorption, phase, i_rec_all, gen_loss, side_propagation


def train(data, **kwargs):
    import time
    import os
    import dxchange
    from ganrec_dataloader import measure_reconstruction_quality, tf_reshape, tfback_phase

    if type(data) is not list:
        data = [data]
    n_items = len(data)
    px, py = data[0].shape
    kwargs['px'] = px
    kwargs['py'] = py

    iter_num = kwargs['iter_num']
    absorption = np.zeros((n_items, px, py))
    phase = np.zeros((n_items, px, py))
    i_rec_all = np.zeros((n_items, px, py))
    gen_loss = np.zeros((n_items, iter_num))
    side_propagations = []


    for i in range(n_items):
        if type(kwargs['z']) is list:
            args = kwargs.copy()
            args['z'] = args['z'][i]
            print(args['z'])
        else:
            args = kwargs.copy()
        absorption[i], phase[i], i_rec_all[i], gen_loss[i], side_propagation = train_one_image(data[i], **args)
        side_propagations.append(side_propagation)
    return absorption, phase, i_rec_all, gen_loss, side_propagations


def save_as_video(side_propagation, px, py, file_name = '10000_phases.mp4'):
    import time
    import cv2
    output_file = 'output_file.mp4'
    # Stack the images into a 4D tensor
    print(len(side_propagation), side_propagation[0].shape, side_propagation[0].dtype)
    try:
        stacked_images = tf.reshape(tf.concat(side_propagation, axis=0), [len(side_propagation), 1, px, py, 1])
    except:
        try:
            stacked_images = tf.reshape(tf.stack(side_propagation), [len(side_propagation), 1, px, py, 1])
        except:
            stacked_images = tf.stack(side_propagation)
    #make the image's bounded between 0 and 255
    stacked_images = (stacked_images - tf.reduce_min(stacked_images))/(tf.reduce_max(stacked_images) - tf.reduce_min(stacked_images)) * 255
    print(stacked_images.shape, stacked_images.dtype)
    # Set the frame rate to 30 FPS
    fps = 30
    frame_interval = 1 / fps
    # Create the video writer
    writer = tf.io.TFRecordWriter(output_file)

    # Write the images to the video file
    for i in range(stacked_images.shape[0]):
        # Encode the image as a JPEG
        image_jpeg = tf.image.encode_jpeg(tf.cast(tf.reshape(stacked_images[i][0,:,:,0], [px, py, 1]), tf.uint8))
        mean = tf.reduce_mean(stacked_images[i])
        stddev = tf.math.reduce_std(stacked_images[i])
        # Create a video frame, while adding the mean and stddev to the frame

        #add the mean and the stddev to the image
        video_frame = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_jpeg.numpy()])),
                    'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
                }
            ),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'image/encoded': tf.train.FeatureList(feature=[
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_jpeg.numpy()]))
                    ]),
                }
            )
        )
        writer.write(video_frame.SerializeToString())

        
        # Wait for the specified frame interval
        if i < stacked_images.shape[0] - 1:
            time.sleep(frame_interval)

    #converting to mp4
    import subprocess 
    subprocess.run(['ffmpeg', '-i', 'output_file.mp4', file_name])

    # Close the video writer
    writer.close()

def _get_GANphase_kwargs():
    return{
        'iter_num': 1000,
        'conv_num': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'l1_ratio': 10,
        'abs_ratio': 0.05,
        'g_learning_rate': 1e-3,
        'd_learning_rate': 1e-5,
        'phase_only': False,
        'save_wpath': None,
        'init_wpath': None,
        'init_model': True,
        'save_model': False,
        'recon_monitor': True,
        'output_num': 2,
        'filter_type': 'median',
        'internal_iter': 3,
        'last_retrieval': False,
    }