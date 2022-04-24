import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np

from ganrec.models import make_generator, make_discriminator

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,
                                                                       labels=tf.ones_like(real_output)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                       labels=tf.zeros_like(fake_output)))
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, img_output, pred):
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                      labels=tf.ones_like(fake_output))) \
               + tf.reduce_mean(tf.abs(img_output - pred)) * 10
    return gen_loss


def tfnor_data(img):
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    return img


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


# @tf.function
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


class RECONmonitor:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 8))

    def initial_plot(self, img_input):
        _, px = img_input.shape
        self.im0 = self.axs[0, 0].imshow(img_input, cmap='gray')
        self.axs[0, 0].set_title('Sinogram')
        self.fig.colorbar(self.im0, ax=self.axs[0, 0])
        self.im1 = self.axs[1, 0].imshow(img_input, cmap='jet')
        self.tx1 = self.axs[1, 0].set_title('Difference of sinogram for iteration 0')
        self.fig.colorbar(self.im1, ax=self.axs[1, 0])
        self.im2 = self.axs[0, 1].imshow(np.zeros((px, px)), cmap='gray')
        self.fig.colorbar(self.im2, ax=self.axs[0, 1])
        self.axs[0, 1].set_title('Reconstruction')
        self.im3, = self.axs[1, 1].plot([0, 1], [1, 1], '-')
        self.axs[1, 1].set_title('Generator loss')
        plt.tight_layout()

    def update_plot(self, epoch, img_diff, img_rec, plot_x, plot_loss):
        self.tx1.set_text('Difference of sinogram for iteration {0}'.format(epoch))
        vmax = np.max(img_diff)
        vmin = np.min(img_diff)
        self.im1.set_data(img_diff)
        self.im1.set_clim(vmin, vmax)
        self.im2.set_data(img_rec)
        vmax = np.max(img_rec)
        vmin = np.min(img_rec)
        self.im2.set_clim(vmin, vmax)
        self.im3.set_xdata(plot_x)
        self.im3.set_ydata(plot_loss)
        plt.pause(0.1)


class GANtomo:
    def __init__(self, prj_input, angle, iter_num):
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = iter_num
        self.conv_num = 32
        self.conv_size = 3
        self.dropout = 0.25
        self.l_ratio = 10
        self.g_learning_rate = 5e-4
        self.d_learning_rate = 1e-4
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.discriminator = None
        self.generator = None

    def make_model(self):
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout)
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)

    @tf.function
    def train_step(self, prj, ang):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(prj)
            # recon = generator(sino, training=True)
            # tf.print(tf.reduce_min(recon), tf.reduce_max(recon))
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            # tf.print(tf.reduce_min(sino_rec), tf.reduce_max(sino_rec))
            prj_rec = tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec)
            d_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return recon, prj_rec, g_loss, d_loss

    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()

        # checkpoint_dir = '/data/ganrec/training_checkpoints'
        # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
        #                                  discriminator_optimizer=self.discriminator_optimizer,
        #                                  generator=self.generator,
        #                                  discriminator=self.discriminator)
        #
        # ############################################################

        plot_x, plot_loss = [], []
        recon_monitor = RECONmonitor()
        recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            recon, prj_rec, g_loss, d_loss = self.train_step(prj, ang)
            plot_x.append(epoch)
            plot_loss.append(g_loss.numpy())
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)

                prj_rec = np.reshape(prj_rec, (nang, px))
                prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                rec_plt = np.reshape(recon, (px, px))
                recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1, g_loss.numpy(), d_loss.numpy()))

        return np.reshape(recon.numpy(), (px, px))


