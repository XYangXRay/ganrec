from __future__ import absolute_import, division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.python.framework import ops


def dense_norm(units, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Dense(units, activation=None, use_bias=True, kernel_initializer=initializer))
    result.add(layers.Dropout(0.25))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.tanh())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=strides, padding='same',
                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result


def dconv2d_norm(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

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


def mdnn_net(conv_num, conv_size):
    img_size = input.shape[2]
    fc_size = img_size ** 2

    # inputs = tf.keras.layers.Input(shape=[img_size, img_size, 1])
    x = tf.keras.layers.Flattern(input)

    fc_stack = [
        dense_norm(conv_num * 4),
        dense_norm(conv_num * 4),
        dense_norm(conv_num * 4),
        dense_norm(fc_size),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size, 1),
        conv2d_norm(conv_num, conv_size, 1),
        conv2d_norm(conv_num * 2, conv_size, 1),
        conv2d_norm(conv_num * 2, conv_size, 1),
        conv2d_norm(conv_num, conv_size, 1),
        conv2d_norm(conv_num, conv_size, 1),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    # Fully connected layers through the model

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_size, img_size, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)
    x = last(x)

    return tf.keras.Model(inputs=input, outputs=x)


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


def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = conv2d_norm(64, 4, 2)(x)  # (batch_size, 128, 128, 64)
    down2 = conv2d_norm(128, 4, 2)(down1)  # (batch_size, 64, 64, 128)
    down3 = conv2d_norm(256, 4, 2)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


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


def cost_mse(ytrue,
             ypred):
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    return mse


def cost_ssim(ytrue,
              ypred):
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    ssim = tf.reduce_mean(tfa.image.ssim(ytrue, ypred, max_val=1))
    return tf.divide(mse, ssim)
    # return 1-tf.reduce_mean(tfa.image.ssim(ytrue, ypred, max_val=1.0))


def cost_ssimms(ytrue,
                ypred):
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    ssim = tf.reduce_mean(tfa.image.ssim_multiscale(ytrue, ypred, max_val=1))
    return tf.divide(mse, ssim ** 0.5)

gen_G = mdnn_net(name="generator_G")
gen_F = tomo_learn(name="generator_F")

# Get the discriminators
disc_X = discriminator(name="discriminator_X")
disc_Y = discriminator(name="discriminator_Y")

class GANrec(keras.Model):
    def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
            lambda_l1=10.0,
            lambda_identity=0.5,
    ):
        super(GANrec, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_l1 = lambda_l1
        self.lambda_identity = lambda_identity

    def compile(
            self,
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer,
            gen_loss_fn,
            disc_loss_fn,
    ):
        super(GANrec, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def recon_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
            )
            id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }


def rec_gan(prj, ang, save_wpath, init_wpath=None, **kwargs):
    gan_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method', 'cost_rate', 'gl_tol', 'iter_plot']
    kwargs_defaults = _get_ganrec_kwargs()
    for kw in gan_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True
    _, nang, px, _ = prj.shape
    img_input = tf.keras.Input(shape=prj.shape)
    img_output = tf.keras.Input(shape=prj.shape)

    pred, recon = tomo_learn(img_input, ang, px, reuse=False, conv_nb=kwargs['conv_nb'],
                             conv_size=kwargs['conv_size'],
                             dropout=kwargs['dropout'],
                             method=kwargs['method']
                             )
    disc_real = discriminator(img_output)
    disc_fake = discriminator(pred, reuse=True)

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                      labels=tf.ones_like(disc_fake))) \
               + tf.reduce_mean(tf.abs(img_output - pred)) * kwargs['cost_rate']
    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                            labels=tf.ones_like(disc_real)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                            labels=tf.zeros_like(disc_fake)))
    disc_loss = disc_loss_real + disc_loss_fake

    gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_gen = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
    # optimizer_disc = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
    optimizer_disc = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)

    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
    ######################################################################
    # # plots for debug
    if kwargs['iter_plot']:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        im0 = axs[0, 0].imshow(prj.reshape(nang, px), cmap='jet')
        tx0 = axs[0, 0].set_title('Sinogram')
        fig.colorbar(im0, ax=axs[0, 0])
        tx1 = axs[1, 0].set_title('Difference of sinogram for iteration 0')
        im1 = axs[1, 0].imshow(prj.reshape(nang, px), cmap='jet')
        fig.colorbar(im1, ax=axs[1, 0])
        im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='jet')
        fig.colorbar(im2, ax=axs[0, 1])
        tx2 = axs[0, 1].set_title('Reconstruction')
        xdata, g_loss = [], []
        # im3, = axs[1, 1].plot(xdata, g_loss, 'r-')
        im3, = axs[1, 1].semilogy(xdata, g_loss, 'r-')
        tx3 = axs[1, 1].set_title('Generator loss')
        plt.tight_layout()
    #########################################################################
    # ani_init()
    rec_tmp = tf.zeros([1, px, px, 1])

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        # Run the initializer
        sess.run(init)
        if kwargs['weights_init']:
            if init_wpath == None:
                print('Please provide the file name of initial weights.')
            saver.restore(sess, init_wpath)
        for step in range(1, kwargs['num_steps'] + 1):
            with tf.device('/device:GPU:1'):
                dl, _ = sess.run([disc_loss, train_disc],
                                 feed_dict={img_input: prj, img_output: prj})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: prj, img_output: prj})

            if np.isnan(gl):
                # gl = np.mean(g_loss)
                sess.run(init)
            if kwargs['iter_plot']:
                xdata.append(step)
                g_loss.append(gl)
                # print(g_loss)
                if step % kwargs['display_step'] == 0 or step == 1:
                    pred, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                                      conv_size=kwargs['conv_size'],
                                                      dropout=kwargs['dropout'],
                                                      method=kwargs['method']))
                    ###########################################################
                    sino_plt = np.reshape(pred, (nang, px))
                    sino_plt = np.abs(sino_plt - prj.reshape((nang, px)))
                    rec_plt = np.reshape(recon, (px, px))
                    tx1.set_text('Difference of sinogram for iteration {0}'.format(step))
                    vmax = np.max(sino_plt)
                    vmin = np.min(sino_plt)
                    im1.set_data(sino_plt)
                    im1.set_clim(vmin, vmax)
                    im2.set_data(rec_plt)
                    vmax = np.max(rec_plt)
                    vmin = np.min(rec_plt)
                    im2.set_clim(vmin, vmax)
                    # axs[1, 1].plot(xdata, g_loss, 'r-')
                    axs[1, 1].semilogy(xdata, g_loss, 'r-')
                    # figname = '/gpfs/petra3/scratch/yangx/data_tomo/test_pattern_exp/presentation/ganrec_log_steps_s100_2000/'\
                    #           +'iter_'+str(step)+'.png'
                    # plt.savefig(figname, dpi=200)
                    plt.pause(0.1)
                    ######################################################################
                    print("Step " + str(step) + ", Generator Loss= " + "{:.7f}".format(gl) +
                          ', Discriminator loss = ' + "{:.7f}".format(dl))
            if gl < kwargs['gl_tol']:
                _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                               conv_size=kwargs['conv_size'],
                                               dropout=kwargs['dropout'],
                                               method=kwargs['method']))
                break
            if step > (kwargs['num_steps'] - 10):
                _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                               conv_size=kwargs['conv_size'],
                                               dropout=kwargs['dropout'],
                                               method=kwargs['method']))
                rec_tmp = tf.concat([rec_tmp, recon], axis=0)
                # print(rec_tmp.shape)
                # _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb = kwargs['conv_nb'],
                #                      conv_size = kwargs['conv_size'],
                #                      dropout = kwargs['dropout'],
                #                      method = kwargs['method']))
        plt.close(fig)
        saver.save(sess, save_wpath)
        if rec_tmp.shape[0] > 1:
            rec_tmp = tf.slice(rec_tmp, [1, 0, 0, 0], [10, px, px, 1])
            recon = tf.reduce_mean(rec_tmp, axis=0).eval()
            # recon = recon.eval()
        # print(recon.shape)
    return recon


def _get_ganrec_kwargs():
    return {
        'learning_rate': 5e-3,
        'num_steps': 10000,
        'display_step': 100,
        'conv_nb': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'weights_init': False,
        'method': 'backproj',
        'cost_rate': 10,
        'gl_tol': 1e-6,
        'iter_plot': True
    }


def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)
