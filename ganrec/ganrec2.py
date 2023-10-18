import os
import numpy as np
import json
import tensorflow as tf
import tensorflow_addons as tfa

from ganrec.propagators import TomoRadon, PhaseFresnel, PhaseFraunhofer
from ganrec.models import make_generator, make_generator_fno, make_discriminator, make_filter
from ganrec.utils import RECONmonitor, ffactor


# Load the configuration from the JSON file
def load_config(filename):
        # Get the directory of the script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the full path to the config file
    config_path = os.path.join(dir_path, filename)
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Use the configuration
config = load_config('config.json')

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


def tomo_bp(sinoi, ang):
    # prj = tfnor_data(sinoi)
    d_tmp = sinoi.shape
    # print d_tmp
    prj = tf.reshape(sinoi, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tfa.image.rotate(prj, ang)
    bp = tf.reduce_mean(prj, 0)
    bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], bp.shape[2]])
    return bp


# @tf.function
# def tomo_radon(rec, ang):
#     nang = ang.shape[0]
#     img = tf.transpose(rec, [3, 1, 2, 0])
#     img = tf.tile(img, [nang, 1, 1, 1])
#     img = tfa.image.rotate(img, -ang, interpolation='bilinear')
#     sino = tf.reduce_mean(img, 1, name=None)
#     sino = tf.image.per_image_standardization(sino)
#     sino = tf.transpose(sino, [2, 0, 1])
#     sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
#     return sino


# def phase_fresnel(phase, absorption, ff, px):
#     paddings = tf.constant([[px // 2, px // 2], [px // 2, px // 2]])
#     # padding1 = tf.constant([[px // 2, px // 2], [0, 0]])
#     # padding2 = tf.constant([[0, 0], [px // 2, px // 2]])
#     pvalue = tf.reduce_mean(phase[:100, :])
#     # phase = tf.pad(phase, paddings, 'CONSTANT',constant_values=1)
#     phase = tf.pad(phase, paddings, 'SYMMETRIC')
#     # phase = tf.pad(phase, paddings, 'REFLECT')
#     absorption = tf.pad(absorption, paddings, 'SYMMETRIC')
#     # phase = phase
#     # absorption = absorption
#     abfs = tf.complex(-absorption, phase)
#     abfs = tf.exp(abfs)
#     ifp = tf.abs(tf.signal.ifft2d(ff * tf.signal.fft2d(abfs))) ** 2
#     ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
#     ifp = tf.image.central_crop(ifp, 0.5)
#     ifp = tf.image.per_image_standardization(ifp)
#     ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
#     # ifp = tfnor_phase(ifp)
#     return ifp


# def phase_fraunhofer(phase, absorption):
#     wf = tf.complex(absorption, phase)
#     # wf = tf.complex(phase, absorption)

    # wf = mask_img(wf)
    # wf = tf.multiply(ampl, tf.exp(phshift))
    # wf = tf.manip.roll(wf, [160, 160], [0, 1])

    ## records from linux machine
#    ifp = tf.math.multiply(tf.square(tf.abs(tf.signal.fft2d(wf))), tf.square(tf.abs(tf.signal.fft2d(wf))))
 
    
    # # adding log to the fft
    # ifp = tf.math.log(ifp+8000)

    ## records from linux machine
#   ifp = tf.math.log(ifp+50)


    # ifp = tf.math.log(tf.abs(tf.signal.fft2d(wf))+1)
    # ifp = tf.math.log(tf.square(tf.abs(tf.signal.fft2d(wf)))+1)
    ## records from linux machine
#    ifp = tf.signal.fftshift(ifp)
#     # wf = mask_img(wf)
#     # wf = tf.multiply(ampl, tf.exp(phshift))
#     # wf = tf.manip.roll(wf, [160, 160], [0, 1])
#     ifp = tf.square(tf.abs(tf.signal.fft2d(wf)))
 
    
#     # # adding log to the fft
#     # ifp = tf.math.log(ifp+8000)
#     ifp = tf.math.log(ifp+10000)
#     # ifp = tf.math.log(tf.abs(tf.signal.fft2d(wf))+1)
#     # ifp = tf.math.log(tf.square(tf.abs(tf.signal.fft2d(wf)))+1)
#     ifp = tf.signal.fftshift(ifp)
  
#     # ifp = tf.roll(ifp, [256, 256], [0, 1])
#     ifp = tf.reshape(ifp, [1, ifp.shape[0], ifp.shape[1], 1])
#     ifp = tf.image.per_image_standardization(ifp)
#     ifp = tfnor_diff(ifp)
#     return ifp


class GANrec:
    def __init__(self, prj_input, angle, **kwargs):
        rec_args = _get_GANtomo_kwargs()
        rec_args.update(**kwargs)
        super(GANtomo, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = rec_args['iter_num']
        self.conv_num = rec_args['conv_num']
        self.conv_size = rec_args['conv_size']
        self.dropout = rec_args['dropout']
        self.l1_ratio = rec_args['l1_ratio']
        self.g_learning_rate = rec_args['g_learning_rate']
        self.d_learning_rate = rec_args['d_learning_rate']
        self.save_wpath = rec_args['save_wpath']
        self.init_wpath = rec_args['init_wpath']
        self.init_model = rec_args['init_model']
        self.recon_monitor = rec_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.prj_input.shape[0],
                                  self.prj_input.shape[1])
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        1)
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    def make_chechpoints(self):
        checkpoint_dir = '/data/ganrec/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

    @tf.function
    def recon_step(self, img_input):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(img_input)
            recon = tfnor_data(recon)
            img_forward = forward_model(recon)
            img_forward = tfnor_data(img_forward)
            real_output = self.discriminator(img_input, training=True)
            fake_output = self.discriminator(img_forward, training=True)
            g_loss = generator_loss(fake_output, img_input, img_forward, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_rec': img_forward,
                'g_loss': g_loss,
                'd_loss': d_loss}

    def recon_step_filter(self, prj, ang):
        with tf.GradientTape() as filter_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            prj_filter = self.filter(prj)
            prj_filter = tfnor_data(prj_filter)
            recon = self.generator(prj_filter)
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            filter_output = self.discriminator(prj_filter, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            f_loss = filer_loss(filter_output, prj, prj_filter)
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_filter = filter_tape.gradient(f_loss,
                                                   self.filter.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.filter_optimizer.apply_gradients(zip(gradients_of_filter,
                                                  self.filter.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_filter': prj_filter,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo')
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.recon_step(prj, ang)
            step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step_filter(prj, ang)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.train_step_filter(prj, ang)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result['prj_rec'], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].numpy()))
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath+'generator.h5')
            self.discriminator.save(self.save_wpath+'discriminator.h5')
        return recon[epoch]
        # return avg_results(recon, gen_loss)


class GANtomo:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config['GANtomo']
        tomo_args.update(**kwargs)
        super(GANtomo, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = tomo_args['iter_num']
        self.conv_num = tomo_args['conv_num']
        self.conv_size = tomo_args['conv_size']
        self.dropout = tomo_args['dropout']
        self.l1_ratio = tomo_args['l1_ratio']
        self.g_learning_rate = tomo_args['g_learning_rate']
        self.d_learning_rate = tomo_args['d_learning_rate']
        self.save_wpath = tomo_args['save_wpath']
        self.init_wpath = tomo_args['init_wpath']
        self.init_model = tomo_args['init_model']
        self.recon_monitor = tomo_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.prj_input.shape[0],
                                  self.prj_input.shape[1])
  
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        1)       
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    def make_chechpoints(self):
        checkpoint_dir = '/data/ganrec/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
    
                                         discriminator=self.discriminator)
        
    @tf.function    
    def tfnor_tomo(self, img):
        img = tf.image.per_image_standardization(img)
        # img = img / tf.reduce_max(img)
        # img = img - tf.reduce_min(img)
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
        return img

    @tf.function
    def recon_step(self, prj, ang):      
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(prj)
            recon = self.tfnor_tomo(recon)
            tomo_radon_obj = TomoRadon(recon, ang)
            prj_rec = tomo_radon_obj.compute()
            prj_rec = self.tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1)) 
      
        prj = tf.cast(prj, dtype=tf.float32)
        prj = self.tfnor_tomo(prj)
        
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo')
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.recon_step(prj, ang)
            step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step_filter(prj, ang)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.train_step_filter(prj, ang)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result['prj_rec'], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].numpy()))
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath+'generator.h5')
            self.discriminator.save(self.save_wpath+'discriminator.h5')
        recon_monitor.close_plot()
        return recon[epoch]
        # return avg_results(recon, gen_loss)

class GANtomo3D:
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = config['GANphase']
        tomo_args.update(**kwargs)
        super(GANtomo, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = tomo_args['iter_num']
        self.conv_num = tomo_args['conv_num']
        self.conv_size = tomo_args['conv_size']
        self.dropout = tomo_args['dropout']
        self.l1_ratio = tomo_args['l1_ratio']
        self.g_learning_rate = tomo_args['g_learning_rate']
        self.d_learning_rate = tomo_args['d_learning_rate']
        self.save_wpath = tomo_args['save_wpath']
        self.init_wpath = tomo_args['init_wpath']
        self.init_model = tomo_args['init_model']
        self.recon_monitor = tomo_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.prj_input.shape[0],
                                  self.prj_input.shape[1])
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        1)
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    def make_chechpoints(self):
        checkpoint_dir = '/data/ganrec/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

    @tf.function
    def recon_step(self, prj, ang):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(prj)
            recon = tfnor_tomo(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_tomo(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    def recon_step_filter(self, prj, ang):
        with tf.GradientTape() as filter_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            prj_filter = self.filter(prj)
            prj_filter = tfnor_data(prj_filter)
            recon = self.generator(prj_filter)
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            filter_output = self.discriminator(prj_filter, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            f_loss = filer_loss(filter_output, prj, prj_filter)
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_filter = filter_tape.gradient(f_loss,
                                                   self.filter.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.filter_optimizer.apply_gradients(zip(gradients_of_filter,
                                                  self.filter.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_filter': prj_filter,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        # prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo')
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.recon_step(prj, ang)
            step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step_filter(prj, ang)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.train_step_filter(prj, ang)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result['prj_rec'], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].numpy()))
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath+'generator.h5')
            self.discriminator.save(self.save_wpath+'discriminator.h5')
        return recon[epoch]
        # return avg_results(recon, gen_loss)

class GANphase:
    def __init__(self, i_input, energy, z, pv, **kwargs):
        phase_args = config['GANphase']
        phase_args.update(**kwargs)
        super(GANphase, self).__init__()
        self.i_input = i_input
        self.px, _ = i_input.shape
        self.energy = energy
        self.z = z
        self.pv = pv
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
        self.recon_monitor = phase_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.i_input.shape[0],
                                  self.i_input.shape[1])
        self.generator = make_generator(self.i_input.shape[0],
                                        self.i_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        2)
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
    def rec_step(self, i_input, ff):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
            phase = tfnor_phase(recon[:, :, :, 0])
            phase = tf.reshape(phase, [self.px, self.px])
            absorption = (1 - tfnor_phase(recon[:, :, :, 1]))* self.abs_ratio
            absorption = tf.reshape(absorption, [self.px, self.px])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            phase_obj = PhaseFresnel(phase, absorption, ff, self.px)
            i_rec = phase_obj.compute()
            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
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
        return {'phase': phase,
                'absorption': absorption,
                'i_rec': i_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        ff = ffactor(self.px * 2, self.energy, self.z, self.pv)
        # print(ff.shape, ff.max(), ff.min())

        i_input = np.reshape(self.i_input, (1, self.px, self.px, 1))
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()
        phase = np.zeros((self.iter_num, self.px, self.px))
        absorption = np.zeros((self.iter_num, self.px, self.px))
        gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('phase')
            recon_monitor.initial_plot(self.i_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_results = self.rec_step(i_input, ff)
            phase[epoch, :, :] = step_results['phase']
            absorption[epoch, :, :] = step_results['absorption']
            i_rec = step_results['i_rec']
            gen_loss[epoch] = step_results['g_loss']
            d_loss = step_results['d_loss']
            # phase[epoch, :, :], absorption[epoch, :, :], i_rec, gen_loss[epoch], d_loss = self.rec_step(i_input,
            #                                                                                                   ff)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    i_rec = np.reshape(i_rec, (self.px, self.px))
                    i_diff = np.abs(i_rec - self.i_input.reshape((self.px, self.px)))
                    phase_plt = np.reshape(phase[epoch], (self.px, self.px))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
                # print(phase.max(), phase.min())
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1, gen_loss[epoch], d_loss.numpy()))
        recon_monitor.close_plot()
        return absorption[epoch], phase[epoch]
        # return avg_results(recon, gen_loss)

class GANdiffraction:
    def __init__(self, i_input, mask, **kwargs):
        phase_args = config['GANdiffraction']
        phase_args.update(**kwargs)
        super(GANdiffraction, self).__init__()
        self.i_input = i_input
        self.mask = mask
        self.px, _ = i_input.shape
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
        self.recon_monitor = phase_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.i_input.shape[0],
                                  self.i_input.shape[1])
        self.generator = make_generator(self.i_input.shape[0],
                                        self.i_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        2)
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
    def rec_step(self, i_input):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            recon = self.generator(i_input)
            # recon = tfa.image.median_filter2d(recon)
            phase = tfnor_diff(recon[:, :, :, 0])
            phase = tf.reshape(phase, [self.px, self.px])
            # add median filter to the result
            # phase = tfa.image.median_filter2d(phase)
            
            
            absorption = (1 - tfnor_diff(recon[:, :, :, 1]))* self.abs_ratio
            absorption = tf.reshape(absorption, [self.px, self.px])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            
            phase_obj = PhaseFraunhofer(phase, absorption)
            i_rec = phase_obj.compute()
            # i_rec = phase_fraunhofer(phase, absorption)
            mask = tf.reshape(self.mask, [1, self.mask.shape[0], self.mask.shape[1], 1])
            # i_rec = tf.multiply(i_rec, mask)
            i_rec = tfnor_diff(i_rec)
            real_output = self.discriminator(i_input, training=True)
            fake_output = self.discriminator(i_rec, training=True)
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
        return {'phase': phase,
                'absorption': absorption,
                'i_rec': i_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        # ff = ffactor(self.px * 2, self.energy, self.z, self.pv)
        # print(ff.shape, ff.max(), ff.min())

        i_input = np.reshape(self.i_input, (1, self.px, self.px, 1))
    
        i_input = tf.cast(i_input, dtype=tf.float32)
        self.make_model()
        phase = np.zeros((self.iter_num, self.px, self.px))
        absorption = np.zeros((self.iter_num, self.px, self.px))
        gen_loss = np.zeros(self.iter_num)

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('phase')
            recon_monitor.initial_plot(self.i_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_results = self.rec_step(i_input)
            phase[epoch, :, :] = step_results['phase']
            absorption[epoch, :, :] = step_results['absorption']
            i_rec = step_results['i_rec']
            gen_loss[epoch] = step_results['g_loss']
            d_loss = step_results['d_loss']
            # phase[epoch, :, :], absorption[epoch, :, :], i_rec, gen_loss[epoch], d_loss = self.rec_step(i_input,
            #                                                                                                   ff)
            ###########################################################################
            # print(i_rec.shape)
            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    i_rec = np.reshape(i_rec, (self.px, self.px))
                    i_diff = np.abs(i_rec - self.i_input.reshape((self.px, self.px)))
                    phase_plt = np.reshape(phase[epoch], (self.px, self.px))
                    recon_monitor.update_plot(epoch, i_diff, phase_plt, plot_x, plot_loss)
                # print(phase.max(), phase.min())
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1, gen_loss[epoch], d_loss.numpy()))
        recon_monitor.close_plot()
        return absorption[epoch], phase[epoch]
        # return avg_results(recon, gen_loss)


def _get_GANtomo_kwargs():
    return{
        'iter_num': 1000,
        'conv_num': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'l1_ratio': 10,
        'g_learning_rate': 1e-3,
        'd_learning_rate': 1e-5,
        'save_wpath': None,
        'init_wpath': None,
        'init_model': False,
        'recon_monitor': True,
    }


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
        'phase_only': True,
        'save_wpath': None,
        'init_wpath': None,
        'init_model': False,
        'recon_monitor': True,
    }
    

def _get_GANdiffraction_kwargs():
    return{
        'iter_num': 1000,
        'conv_num': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'l1_ratio': 100,
        'abs_ratio': 0.0,
        'g_learning_rate': 1e-3,
        'd_learning_rate': 1e-6,
        'phase_only': True,
        'save_wpath': None,
        'init_wpath': None,
        'init_model': False,
        'recon_monitor': True,
    }
