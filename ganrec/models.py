import tensorflow as tf
from tensorflow.keras import layers, Input


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


def make_generator(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w ** 2
    inputs = Input(shape=(img_h, img_w, 1))
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

    last = conv2d_norm(output_num, 3, 1)

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_w, img_w, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_filter(img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    down_stack = [
        conv2d_norm(32, 3, 1, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        # conv2d_norm(128, 4, 2),  # (batch_size, 64, 64, 128)
        # conv2d_norm(256, 4, 2),  # (batch_size, 32, 32, 256)
        # conv2d_norm(512, 4, 2),  # (batch_size, 16, 16, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 8, 8, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 4, 4, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 2, 2, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        # dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        # dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        # dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        # dconv2d_norm(512, 4, 2),  # (batch_size, 16, 16, 1024)
        # dconv2d_norm(256, 4, 2),  # (batch_size, 32, 32, 512)
        # dconv2d_norm(128, 4, 2),  # (batch_size, 64, 64, 256)
        dconv2d_norm(32, 3, 1),  # (batch_size, 128, 128, 128)
    ]
    last = conv2d_norm(1, 3, 1)
    # initializer = tf.random_normal_initializer(0., 0.02)
    # last = tf.keras.layers.Conv2DTranspose(1, 3,
    #                                        strides=1,
    #                                        padding='same',
    #                                        kernel_initializer=initializer,
    #                                        activation='tanh')  # (batch_size, 256, 256, 3)

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
        # x = tf.keras.layers.Concatenate()([x, skip])

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
