import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers, activations, Input
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

#########################################

# def make_generator(nang, img_size, conv_num, conv_size, dropout):
#     fc_size = img_size ** 2
#     inputs = Input(shape=(nang, img_size, 1))
#     # inputs = tf.keras.layers.Input(shape=[img_size, img_size, 1])
#     # initializer = tf.random_normal_initializer(0., 0.02)
#     x = tf.keras.layers.Flatten()(inputs)
#     x = dense_norm(conv_num * 4, dropout)(x)
#     x = dense_norm(conv_num * 4, dropout)(x)
#     x = dense_norm(conv_num * 4, dropout)(x)
#     x = dense_norm(fc_size, dropout)(x)
#     x = tf.reshape(x, shape=[-1, img_size, img_size, 1])
#     x = conv2d_norm(conv_num, conv_size, 1)(x)
#     x = conv2d_norm(conv_num, conv_size, 1)(x)
#     x = conv2d_norm(conv_num * 2, conv_size, 1)(x)
#     x = conv2d_norm(conv_num * 2, conv_size, 1)(x)
#     x = conv2d_norm(1, conv_size, 1)(x)
#
#     return tf.keras.Model(inputs=inputs, outputs=x)


def make_generator(nang, img_size, conv_num, conv_size, dropout):

    fc_size = img_size ** 2
    inputs = Input(shape=(nang, img_size, 1))
    # inputs = tf.keras.layers.Input(shape=[img_size, img_size, 1])
    x = tf.keras.layers.Flatten()(inputs)
#     print(inputs.shape)
    fc_stack = [
        dense_norm(conv_num * 4, dropout),
        dense_norm(conv_num * 4, dropout),
        dense_norm(conv_num * 4, dropout),
        dense_norm(fc_size, dropout),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size, 1),
        conv2d_norm(conv_num, conv_size, 1),
        conv2d_norm(conv_num, conv_size, 1),

    ]

    dconv_stack = [
        dconv2d_norm(conv_num, conv_size, 1),
        dconv2d_norm(conv_num, conv_size, 1),
        dconv2d_norm(conv_num , conv_size, 1),
    ]


    last = conv2d_norm(1, conv_size, 1)

    # Fully connected layers through the model

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


def make_discriminator_model(nang, px):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[nang, px, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

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
               + tf.reduce_mean(tf.abs(img_output - pred))*10
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

# @tf.function
def tomo_learn(sinoi, ang, px, reuse, conv_nb, conv_size, dropout, method):
    if method == 'backproj':
        bp = tomo_bp(sinoi, ang)
        bp = tfnor_data(bp)
        bp = tf.reshape(bp, [bp.shape[0], bp.shape[1], bp.shape[2], 1])
        recon = filter_net(bp, conv_nb, conv_size, dropout, px, reuse=reuse)
    elif method == 'fc':
        inputs = tf.convert_to_tensor(sinoi)
        recon = mdnn_net(inputs)
    else:
        os.exit('Please provide a correct method. Options: backproj, conv1d, fc')

    recon = tfnor_data(recon)
    sinop = tomo_radon(recon, ang)
    sinop = tfnor_data(sinop)
    return sinop, recon

@tf.function
def recon_step(sino, ang, generator, discriminator, generator_optimizer, discriminator_optimizer):
    # noise = tf.random.normal([1, 181, 366, 1])
    # noise = tf.cast(noise, dtype=tf.float32)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
        recon = generator(sino)
        # recon = generator(sino, training=True)
        # tf.print(tf.reduce_min(recon), tf.reduce_max(recon))
        recon = tfnor_data(recon)
        sino_rec = tomo_radon(recon, ang)
        # tf.print(tf.reduce_min(sino_rec), tf.reduce_max(sino_rec))
        sino_rec = tfnor_data(sino_rec)
        real_output = discriminator(sino, training=True)
        fake_output = discriminator(sino_rec, training=True)
        gen_loss = generator_loss(fake_output, sino, sino_rec)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return recon, sino_rec, gen_loss, disc_loss


def init_recon_mointor(img, px):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    im0 = axs[0, 0].imshow(img, cmap='jet')
    tx0 = axs[0, 0].set_title('Sinogram')
    fig.colorbar(im0, ax=axs[0, 0])
    tx1 = axs[1, 0].set_title('Difference of sinogram for iteration 0')
    im1 = axs[1, 0].imshow(img, cmap='jet')
    fig.colorbar(im1, ax=axs[1, 0])
    im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='jet')
    fig.colorbar(im2, ax=axs[0, 1])
    tx2 = axs[0, 1].set_title('Reconstruction')
    xdata, plot_loss = [], []
    im3, = axs[1, 1].plot(xdata, plot_loss)
    tx3 = axs[1, 1].set_title('Generator loss')
    plt.tight_layout()
def update_recon_monitor(epoch, sino_rec, sino_input, recon, xdata, plot_loss):
    xdata.append(epoch)
    plot_loss.append(g_loss)
    sino_plt = np.reshape(sino_rec, (nang, px))
    sino_plt = np.abs(sino_plt - sino_input)
    rec_plt = np.reshape(recon, (px, px))
    tx1.set_text('Difference of sinogram for iteration {0}'.format(epoch))
    vmax = np.max(sino_plt)
    vmin = np.min(sino_plt)
    im1.set_data(sino_plt)
    im1.set_clim(vmin, vmax)
    im2.set_data(rec_plt)
    vmax = np.max(rec_plt)
    vmin = np.min(rec_plt)
    im2.set_clim(vmin, vmax)
    im3.set_xdata(xdata)
    im3.set_ydata(plot_loss)
    plt.pause(0.1)



def ganrec(sino_input, ang, num_iter):
    nang, px = sino_input.shape
    # init_recon_mointor(sino_input, px)
    # plt.imshow(sino_input)
    # plt.show()
    sino = np.reshape(sino_input, (1, nang, px, 1))
    sino = tf.cast(sino, dtype=tf.float32)
    sino = tfnor_data(sino)
    ang = tf.cast(ang, dtype=tf.float32)
    generator = make_generator(nang, px, 64, 3, 0.25)
    discriminator = make_discriminator_model(nang, px)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
    checkpoint_dir = '/data/ganrec/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    ############################################################
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    im0 = axs[0, 0].imshow(sino_input, cmap='jet')
    tx0 = axs[0, 0].set_title('Sinogram')
    fig.colorbar(im0, ax=axs[0, 0])
    tx1 = axs[1, 0].set_title('Difference of sinogram for iteration 0')
    im1 = axs[1, 0].imshow(sino_input, cmap='jet')
    fig.colorbar(im1, ax=axs[1, 0])
    im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='jet')
    fig.colorbar(im2, ax=axs[0, 1])
    tx2 = axs[0, 1].set_title('Reconstruction')
    xdata, plot_loss = [], []
    im3, = axs[1, 1].plot(xdata, plot_loss)
    tx3 = axs[1, 1].set_title('Generator loss')
    plt.tight_layout()

    ###########################################################################
    for epoch in range(num_iter):

        recon, sino_rec, g_loss, d_loss = recon_step(sino, ang, generator, discriminator,
                                                     generator_optimizer,
                                                     discriminator_optimizer)

        ##########################################################################

        # update_recon_monitor(epoch, sino_rec, sino_input, recon, xdata, plot_loss)
        #############################################################################
        # Produce images for the GIF as you go
        #         display.clear_output(wait=True)
        #         generate_and_save_images(generator,
        #                              num_iter + 1,
        #                              sino)

        # Save the model every 15 epochs
        xdata.append(epoch)
        plot_loss.append(g_loss.numpy())
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

            sino_plt = np.reshape(sino_rec, (nang, px))
            sino_plt = np.abs(sino_plt - sino_input.reshape((nang, px)))
            rec_plt = np.reshape(recon, (px, px))
            tx1.set_text('Difference of sinogram for iteration {0}'.format(epoch))
            vmax = np.max(sino_plt)
            vmin = np.min(sino_plt)
            im1.set_data(sino_plt)
            im1.set_clim(vmin, vmax)
            im2.set_data(rec_plt)
            vmax = np.max(rec_plt)
            vmin = np.min(rec_plt)
            im2.set_clim(vmin, vmax)
            im3.set_xdata(xdata)
            im3.set_ydata(plot_loss)
            plt.pause(0.1)
            print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1, g_loss.numpy(), d_loss.numpy()))

    return np.reshape(recon.numpy(), (px, px))

def main():
    fname_data = '/data/ganrec/prj_shale.tiff'
    data = dxchange.read_tiff(fname_data)
    nang, nslice, px = data.shape
    theta = angles(nang, ang1=0, ang2=180)
    slice = 100
    prj = data[:, slice, :]
    prj = nor_data(prj)
    recon = ganrec(prj, theta, 2000)
    dxchange.write_tiff(recon, '/data/ganrec/test', overwrite=True)


if __name__ == "__main__":
    main()