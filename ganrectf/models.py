from pickle import TRUE

import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Reshape,
    LayerNormalization,
    LeakyReLU,
    MaxPool2D,
    Reshape,
    UpSampling2D,
    Concatenate,
)


def dense_norm(units, dropout, apply_batchnorm=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(Dense(units, use_bias=True, kernel_initializer=initializer))
    result.add(Dropout(dropout))
    if apply_batchnorm:
        result.add(LayerNormalization())
    result.add(LeakyReLU())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Conv2D(filters, size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False)
    )

    if apply_batchnorm:
        result.add(LayerNormalization())

    result.add(LeakyReLU())

    return result


def dconv2d_norm(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Conv2DTranspose(
            filters, size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False
        )
    )
    if apply_dropout:
        result.add(Dropout(0.25))
    result.add(LeakyReLU())
    return result


def dense_res(x, filters, size):
    x = Conv2D(filters, size, padding="same")(x)
    fx = Conv2DTranspose(filters, size, activation="relu", padding="same")(x)
    fx = BatchNormalization()(fx)
    fx = Conv2DTranspose(filters, size, padding="same")(fx)
    out = Concatenate([x, fx], axis=3)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)


def conv_res(x, filters, size):
    x = Conv2D(filters, size, padding="same")(x)
    fx = Conv2DTranspose(filters, size, activation="relu", padding="same")(x)
    fx = BatchNormalization()(fx)
    fx = Conv2DTranspose(filters, size, padding="same")(fx)
    out = Concatenate([x, fx], axis=3)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)
    return out


def make_generator(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w**2
    inputs = Input(shape=(img_h, img_w, 1))
    x = Flatten()(inputs)
    fc_stack = [
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(fc_size, 0),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size + 2, 1),
        conv2d_norm(conv_num, conv_size + 2, 1),
        conv2d_norm(conv_num, conv_size, 1),
    ]

    dconv_stack = [
        dconv2d_norm(conv_num, conv_size + 2, 1),
        dconv2d_norm(conv_num, conv_size + 2, 1),
        dconv2d_norm(conv_num, conv_size, 1),
    ]

    last = conv2d_norm(output_num, 3, 1)

    for fc in fc_stack:
        x = fc(x)

    x = Reshape((img_w, img_w, 1))(x)
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)
    return Model(inputs=inputs, outputs=x)


def make_generator_3d(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w**2
    inputs = Input(shape=(img_h, img_w, 1))
    x = tf.keras.layers.Flatten()(inputs)
    fc_stack = [
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(fc_size, 0),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size + 2, 1),
        conv2d_norm(conv_num, conv_size + 2, 1),
        conv2d_norm(conv_num, conv_size, 1),
    ]

    dconv_stack = [
        dconv2d_norm(conv_num, conv_size + 2, 1),
        dconv2d_norm(conv_num, conv_size + 2, 1),
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


class FourierNeuralOperator:
    def __init__(self, img_h, img_w, conv_num, conv_size, strides, dropout, output_num):
        self.img_h = img_h
        self.img_w = img_w
        self.conv_num = conv_num
        self.conv_size = conv_size
        self.strides = strides
        self.dropout = dropout
        self.output_num = output_num

        # Initialize the model
        self.model = self.make_generator_fno()

    def P(self, X, conv_num, conv_size, strides, stage):
        # Name definition
        P_Name = "P-layer" + str(stage)
        P_BN_Name = "P-layer-BN" + str(stage)

        X = Conv2D(
            filters=conv_num,
            kernel_size=(conv_size, conv_size),
            strides=(strides, strides),
            padding="valid",
            name=P_Name,
            kernel_initializer=glorot_uniform(seed=0),
        )(X)
        X = BatchNormalization(axis=3, name=P_BN_Name)(X)
        X = Activation("relu")(X)

        return X

    def FourierLayer(self, X, stage):
        # Name definition
        FFT_name = "fft-layer" + str(stage)
        RFFT_name = "ifft-layer" + str(stage)

        Res_X = X

        # Perform Fourier Transform
        X = tf.signal.fft2d(tf.cast(X, tf.complex64), name=FFT_name)

        # Initialize an empty tensor for output in Fourier space
        out_ft = tf.zeros_like(X, dtype=tf.complex64)

        # Define the regions for multiplication
        out_ft_first = X[:, :, :5, :5] * tf.cast(Res_X[:, :, :5, :5], tf.complex64)
        out_ft_end = X[:, :, -5:, :5] * tf.cast(Res_X[:, :, -5:, :5], tf.complex64)

        # Combine the results
        o1 = tf.concat([out_ft_first, out_ft[:, :, :5, 5:]], axis=3)
        o2 = tf.concat([out_ft[:, :, -5:, 5:], out_ft_end], axis=3)
        out_ft = tf.concat([o1, o2], axis=2)

        # Perform Inverse Fourier Transform
        X = tf.signal.ifft2d(X, name=RFFT_name)

        # Add the residual connection
        X = Add()([tf.cast(X, tf.float32), Res_X])
        X = Activation("relu")(X)

        return X

    def Q(self, X, conv_num, conv_size, strides, stage):
        # Name definition
        Q_Name = "Q-layer" + str(stage)
        Q_BN_Name = "Q-layer-BN" + str(stage)

        X = Conv2D(
            filters=conv_num,
            kernel_size=(conv_size, conv_size),
            strides=(strides, strides),
            padding="same",
            name=Q_Name,
            kernel_initializer=glorot_uniform(seed=0),
        )(X)
        X = BatchNormalization(axis=3, name=Q_BN_Name)(X)
        X = Activation("relu")(X)

        return X

    def make_generator_fno(self):
        X_input = Input(shape=(self.img_h, self.img_w, 1))

        # Example usage of the P layer
        # X = self.P(X_input, conv_num=self.conv_num, conv_size=self.conv_size, strides=self.strides, stage=1)

        # Middle Part with FourierLayer
        X = self.FourierLayer(X_input, stage=1)
        X = self.FourierLayer(X, stage=2)
        X = self.FourierLayer(X, stage=3)
        X = self.FourierLayer(X, stage=4)

        # Example usage of the Q layer
        # X = self.Q(X, conv_num=512, conv_size=1, strides=1, stage=1)

        # Create model
        model = Model(inputs=X_input, outputs=X, name="Fourier-Neural-Operator")

        return model


def make_discriminator(nang, px):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same", input_shape=[nang, px, 1]))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding="same"))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Conv2D(32, (5, 5), strides=(1, 1), padding="same"))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same"))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    # model.add(LayerNormalization())
    model.add(Dense(128))

    return model


def block(x_img, x_ts):
    x_parameter = Conv2D(128, kernel_size=3, padding="same")(x_img)
    x_parameter = Activation("relu")(x_parameter)

    time_parameter = Dense(128)(x_ts)
    time_parameter = Activation("relu")(time_parameter)
    time_parameter = Reshape((1, 1, 128))(time_parameter)
    x_parameter = x_parameter * time_parameter

    # -----
    x_out = Conv2D(128, kernel_size=3, padding="same")(x_img)
    x_out = x_out + x_parameter
    x_out = LayerNormalization()(x_out)
    x_out = Activation("relu")(x_out)

    return x_out


def diffusion_model(img_h, img_w, conv_num, conv_size, dropout, output_num):
    x = x_input = Input(shape=(img_h, img_w, 1), name="x_input")

    x_ts = x_ts_input = Input(shape=(1,), name="x_ts_input")
    x_ts = Dense(192)(x_ts)
    x_ts = LayerNormalization()(x_ts)
    x_ts = Activation("relu")(x_ts)

    # ----- left ( down ) -----
    print(f"Input shape: {x.shape}")
    x = x32 = block(x, x_ts)
    x = MaxPool2D(2, padding="same")(x)
    print(f"x32 shape: {x.shape}")
    x = x16 = block(x, x_ts)
    print(f"x16 shape: {x.shape}")
    x = MaxPool2D(2, padding="same")(x)
    print(f"x8 shape: {x.shape}")
    x = x8 = block(x, x_ts)
    print(f"x8 shape: {x.shape}")
    x = MaxPool2D(2, padding="same")(x)

    x = x4 = block(x, x_ts)

    # ----- MLP -----
    x = Flatten()(x)
    x = Concatenate()([x, x_ts])
    x = Dense(128)(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    x = Dense((img_h // 8) * (img_w // 8) * 32)(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)
    x = Reshape((img_h // 8, img_w // 8, 32))(x)

    # ----- right ( up ) -----
    x = Concatenate()([x, x4])

    x = block(x, x_ts)

    x = UpSampling2D(2)(x)

    x = Concatenate()([x, x8])
    x = block(x, x_ts)
    x = UpSampling2D(2)(x)

    x = Concatenate()([x, x16])
    x = block(x, x_ts)
    x = UpSampling2D(2)(x)

    x = Concatenate()([x, x32])
    x = block(x, x_ts)

    # ----- output -----
    x = Conv2D(3, kernel_size=1, padding="same")(x)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model
