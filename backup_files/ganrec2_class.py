import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as xp
from plotly.subplots import make_subplots
import numpy as np
from tensorflow.keras import layers, Input
import time
import dxchange


def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())
    return img


def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)


def dense_norm(units, dropout, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        layers.Dense(units, activation=tf.nn.tanh, use_bias=True, kernel_initializer=initializer))
    result.add(layers.Dropout(dropout))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    #     result.add(layers.LeakyReLU())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=strides, padding='same',
                      kernel_initializer=initializer, activation=tf.nn.elu))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    # result.add(layers.LeakyReLU())

    return result


def dconv2d_norm(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=strides,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def make_generator(nang, img_size, conv_num, conv_size, dropout):
    units = 128
    fc_size = img_size ** 2
    inputs = Input(shape=(nang, img_size, 1))
    x = tf.keras.layers.Flatten()(inputs)
    fc_stack = [
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(fc_size, 0),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size+2, 1),
        conv2d_norm(conv_num, conv_size+2, 1),
        conv2d_norm(conv_num, conv_size, 1),

    ]

    dconv_stack = [
        dconv2d_norm(conv_num, conv_size+2, 1),
        dconv2d_norm(conv_num, conv_size+2, 1),
        dconv2d_norm(conv_num, conv_size, 1),
    ]

    last = conv2d_norm(1, 3, 1)

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_size, img_size, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def filter_net():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        conv2d_norm(64, 4, 2, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        conv2d_norm(128, 4, 2),  # (batch_size, 64, 64, 128)
        conv2d_norm(256, 4, 2),  # (batch_size, 32, 32, 256)
        conv2d_norm(512, 4, 2),  # (batch_size, 16, 16, 512)
        conv2d_norm(512, 4, 2),  # (batch_size, 8, 8, 512)
        conv2d_norm(512, 4, 2),  # (batch_size, 4, 4, 512)
        conv2d_norm(512, 4, 2),  # (batch_size, 2, 2, 512)
        conv2d_norm(512, 4, 2),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        dconv2d_norm(512, 4, 2),  # (batch_size, 16, 16, 1024)
        dconv2d_norm(256, 4, 2),  # (batch_size, 32, 32, 512)
        dconv2d_norm(128, 4, 2),  # (batch_size, 64, 64, 256)
        dconv2d_norm(64, 4, 2),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_discriminator(nang, px):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[nang, px, 1]))
    model.add(layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


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


def main():
    fname_data = '/data/ganrec/prj_shale.tiff'
    data = dxchange.read_tiff(fname_data)
    nang, nslice, px = data.shape
    theta = angles(nang, ang1=0, ang2=180)
    slice = 100
    iter_num = 2000
    prj = data[:, slice, :]
    prj = nor_data(prj)
    gan_tomo_object = GANtomo(prj, theta, iter_num)
    start = time.time()
    rec = gan_tomo_object.recon()
    end = time.time()
    print('Running time is {}'.format(end - start))
    dxchange.write_tiff(rec, '/data/ganrec/test', overwrite=True)


if __name__ == "__main__":
    main()