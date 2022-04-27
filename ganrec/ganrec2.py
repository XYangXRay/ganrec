import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from ganrec.models import make_generator, make_discriminator, make_filter
from ganrec.utils import RECONmonitor, ffactor


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
             l1_loss(img_output, img_filter) * 100
    return f_loss


def tfnor_data(img):
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    return img


def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img


def avg_results(recon, loss):
    sort_index = np.argsort(loss)
    recon_tmp = recon[sort_index[:10], :, :, :]
    return np.mean(recon_tmp, axis=0)


def tomo_bp(sinoi, ang):
    prj = tfnor_data(sinoi)
    d_tmp = sinoi.shape
    # print d_tmp
    prj = tf.reshape(prj, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tfa.image.rotate(prj, ang)
    bp = tf.reduce_mean(prj, 0)
    bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], bp.shape[2]])
    return bp


@tf.function
def tomo_radon(rec, ang):
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    img = tf.tile(img, [nang, 1, 1, 1])
    img = tfa.image.rotate(img, -ang, interpolation='bilinear')
    sino = tf.reduce_mean(img, 1, name=None)
    sino = tf.image.per_image_standardization(sino)
    sino = tf.transpose(sino, [2, 0, 1])
    sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
    return sino


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


def phase_fraunhofer(phase, absorption):
    wf = tf.complex(absorption, phase)
    # wf = tf.complex(phase, absorption)

    # wf = mask_img(wf)
    # wf = tf.multiply(ampl, tf.exp(phshift))
    # wf = tf.manip.roll(wf, [160, 160], [0, 1])
    ifp = tf.square(tf.abs(tf.signal.fft2d(wf)))
    ifp = tf.roll(ifp, [256, 256], [0, 1])
    ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.per_image_standardization(ifp)
    ifp = tfnor_phase(ifp)
    return ifp


class GANtomo:
    def __init__(self, prj_input, angle, iter_num):
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = iter_num
        self.conv_num = 32
        self.conv_size = 3
        self.dropout = 0.25
        self.l1_ratio = 10
        self.g_learning_rate = 5e-4
        self.d_learning_rate = 1e-4
        self.recon_monitor = True
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
    def recon_step(self, prj, ang):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(prj)
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_data(prj_rec)
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

    def train_step_filter(self, prj, ang):
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
            g_loss = generator_loss(fake_output, prj_filter, prj_rec)
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
        return recon, prj_rec, g_loss, d_loss

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
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
        return recon[epoch]
        # return avg_results(recon, gen_loss)


class GANphase:
    def __init__(self, i_input, energy, z, pv, iter_num):
        self.i_input = i_input
        self.px, _ = i_input.shape
        self.energy = energy
        self.z = z
        self.pv = pv
        self.iter_num = iter_num
        self.conv_num = 32
        self.conv_size = 3
        self.dropout = 0.25
        self.l1_ratio = 100
        self.g_learning_rate = 5e-4
        self.d_learning_rate = 1e-4
        self.phase_only = True
        self.recon_monitor = True
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
            phase = tfnor_phase(recon[:, :, :, 0]) - 1
            phase = tf.reshape(phase, [self.px, self.px])
            absorption = (1 - tfnor_phase(recon[:, :, :, 1])) * 0.05
            absorption = tf.reshape(absorption, [self.px, self.px])
            if self.phase_only:
                absorption = tf.zeros_like(phase)
            i_rec = phase_fresnel(phase, absorption, ff, self.px)
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

    def rec_step_filter(self, prj, ang):
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
            g_loss = generator_loss(fake_output, prj_filter, prj_rec)
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
        return recon, prj_rec, g_loss, d_loss

    @property
    def recon(self):
        ff = ffactor(self.px * 2, self.energy, self.z, self.pv)
        print(ff.shape, ff.max(), ff.min())

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
                print(phase.max(), phase.min())
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1, gen_loss[epoch], d_loss.numpy()))
        return phase[epoch]
        # return avg_results(recon, gen_loss)
