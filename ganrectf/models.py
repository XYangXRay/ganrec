from pickle import TRUE

import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.initializers import glorot_uniform, glorot_normal, HeNormal, HeUniform
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    InputLayer,
    Flatten,
    GlobalAveragePooling2D,
    Reshape,
    Layer,
    LayerNormalization,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    Concatenate,
)


def model_initializer():
    return glorot_normal()

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


def make_discriminator(img_h, img_w):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", input_shape=[img_h, img_w, 1]))
    model.add(Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer=model_initializer()))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())


    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=model_initializer()))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding="same", kernel_initializer=model_initializer()))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())


    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=model_initializer()))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same", kernel_initializer=model_initializer()))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())


    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer=model_initializer()))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same", kernel_initializer=model_initializer()))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())


    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=model_initializer()))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    # model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer=model_initializer()))
    # model.add(LayerNormalization())
    model.add(Dense(128, kernel_initializer=model_initializer()))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    return model

# def make_discriminator(img_h, img_w):
#     model = Sequential()
#     def add_conv_block(model, filters, kernel_size, strides, dropout_rate=0.2):
#         model.add(Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_initializer=model_initializer()))
#         model.add(LayerNormalization())
#         model.add(LeakyReLU())
#         if dropout_rate > 0:
#             model.add(Dropout(dropout_rate))
#     # Input layer (no need to specify input shape here)
#     model.add(InputLayer(input_shape=(img_h, img_w, 1)))   
#     # Convolutional blocks
#     conv_params = [
#         (32, (5, 5), (2, 2)),
#         (32, (5, 5), (1, 1)),
#         (64, (5, 5), (2, 2)),
#         (64, (5, 5), (1, 1)),
#         (128, (3, 3), (2, 2)),
#         (128, (3, 3), (1, 1)),
#         (256, (3, 3), (2, 2)),
#         (256, (3, 3), (1, 1)),
#     ]  
#     for filters, kernel_size, strides in conv_params:
#         add_conv_block(model, filters, kernel_size, strides)
#     # model.add(GlobalAveragePooling2D())
#     # Flatten and Dense layers
#     model.add(Flatten())
#     dense_params = [
#         (256, 0.2),
#         (128, 0.2),
#         (128, 0.2)
#     ]
#     for units, dropout_rate in dense_params:
#         model.add(Dense(units, kernel_initializer=model_initializer()))
#         model.add(LayerNormalization())
#         model.add(LeakyReLU())
#         if dropout_rate > 0:
#             model.add(Dropout(dropout_rate))
#     return model


class DownBlock(Layer):
    def __init__(self, filters):
        super(DownBlock, self).__init__()
        self.conv1 = Sequential([
            Conv2D(filters, kernel_size=3, padding="same", kernel_initializer=model_initializer()),
            LeakyReLU()
        ])
        self.dense = Sequential([
            Dense(filters, kernel_initializer=model_initializer()),
            LeakyReLU(),
            Reshape((1, 1, filters))
        ])
        self.conv2 = Sequential([
            Conv2D(filters, kernel_size=3, padding="same", kernel_initializer=model_initializer()),
            # LayerNormalization(),
            LeakyReLU()
        ])

    def call(self, x_img, x_ts):
        x_parameter = self.conv1(x_img)
        time_parameter = self.dense(x_ts)
        x_parameter = x_parameter * time_parameter
        x_out = self.conv2(x_img) + x_parameter
        return x_out

class UpBlock(Layer):
    def __init__(self, filters):
        super(UpBlock, self).__init__()
        self.conv_transpose1 = Sequential([
            Conv2DTranspose(filters, kernel_size=3, padding="same", kernel_initializer=model_initializer()),
            LeakyReLU()
        ])
        self.dense = Sequential([
            Dense(filters, kernel_initializer=model_initializer()),
            LeakyReLU(),
            Reshape((1, 1, filters))
        ])
        self.conv_transpose2 = Sequential([
            Conv2DTranspose(filters, kernel_size=3, padding="same", kernel_initializer=model_initializer()),
            # LayerNormalization(),
            LeakyReLU()
        ])

    def call(self, x_img, x_ts):
        x_parameter = self.conv_transpose1(x_img)
        time_parameter = self.dense(x_ts)
        x_parameter = x_parameter * time_parameter
        x_out = self.conv_transpose2(x_img) + x_parameter
        return x_out

class DiffusionUNet(tf.keras.Model):
    def __init__(self, img_h, img_w):
        super(DiffusionUNet, self).__init__()

        self.img_h = img_h
        self.img_w = img_w
        
        self.ts_input = Sequential([
            Dense(192, kernel_initializer=model_initializer()),
            # LayerNormalization(),
            LeakyReLU(),
        ])

        # Down blocks
        self.down1 = DownBlock(32)
        self.down2 = DownBlock(32)
        self.down3 = DownBlock(16)
        self.down4 = DownBlock(16)

        # MLP part
        self.flatten = Flatten()
        self.concat1 = Concatenate()
        self.mlp1 = Sequential([
            Dense(128, kernel_initializer=model_initializer()),
            # LayerNormalization(),
            LeakyReLU(),
            Dense(128, kernel_initializer=model_initializer()),
            # LayerNormalization(),
            LeakyReLU(),
            Dense((img_h // 8) * (img_w // 8) * 32, kernel_initializer=model_initializer()),
            LayerNormalization(),
            LeakyReLU(),
            Reshape((img_h // 8, img_w // 8, 32)),
        ])
       
        # Up blocks
        self.up1 = UpBlock(128)
        self.up2 = UpBlock(64)
        self.up3 = UpBlock(32)
        self.up4 = UpBlock(16)

        # Output layer
        self.output_layer = Sequential([
            Conv2DTranspose(8, kernel_size=3, padding="same", kernel_initializer=model_initializer()),
            Conv2D(1, kernel_size=1, padding="same", kernel_initializer=model_initializer()),
        ])

    def call(self, inputs):
        x_img, x_t = inputs
        
        x_ts = self.ts_input(x_t)
        # Down blocks
        x = self.down1(x_img, x_ts)
        x32 = x
        x = MaxPooling2D(2, padding="same")(x)

        x = self.down2(x, x_ts)
        x16 = x
        x = MaxPooling2D(2, padding="same")(x)

        x = self.down3(x, x_ts)
        x8 = x
        x = MaxPooling2D(2, padding="same")(x)

        x = self.down4(x, x_ts)
        x4 = x

        # MLP
        x = self.flatten(x)
        x = self.concat1([x, x_ts])
        x = self.mlp1(x)

        # Up blocks
        x = Concatenate()([x, x4])
        x = self.up1(x, x_ts)
        x = UpSampling2D(2)(x)

        x = Concatenate()([x, x8])
        x = self.up2(x, x_ts)
        x = UpSampling2D(2)(x)

        x = Concatenate()([x, x16])
        x = self.up3(x, x_ts)
        x = UpSampling2D(2)(x)

        x = Concatenate()([x, x32])
        x = self.up4(x, x_ts)

        # Output layer
        x = self.output_layer(x)

        return x

def diffusion_model(img_h, img_w):
    model = DiffusionUNet(img_h, img_w)
    x_img = tf.keras.Input(shape=(img_h, img_w, 32), name="x_input")
    x_t = tf.keras.Input(shape=(1,), name="x_ts_input")
    outputs = model([x_img, x_t])
    return Model(inputs=[x_img, x_t], outputs=outputs)




