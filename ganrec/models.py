from pickle import TRUE

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Layer, Dense, Conv2D, Conv2DTranspose, \
    Flatten, concatenate, LayerNormalization, MaxPool2D, Concatenate, \
        BatchNormalization, Dropout, UpSampling2D, Reshape, \
            ReLU,LeakyReLU, Activation, Add
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten,
                                     Layer, LeakyReLU, ReLU, concatenate)
from tensorflow.signal import (fft, fft2d, ifft, ifft2d, irfft, irfft2d, rfft,
                               rfft2d)

def dense_norm(units, dropout, apply_batchnorm=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Dense(units, 
            #   activation=tf.nn.tanh, 
              use_bias=True, 
              kernel_initializer=initializer))
    result.add(Dropout(dropout))

    if apply_batchnorm:
        # result.add(BatchNormalization())
        result.add(LayerNormalization())

    result.add(LeakyReLU())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Conv2D(filters, 
               size, 
               strides=strides, 
               padding='same',
               kernel_initializer=initializer, 
               use_bias=False))

    if apply_batchnorm:
        # result.add(BatchNormalization())
        result.add(LayerNormalization())

    result.add(LeakyReLU())

    return result


def dconv2d_norm(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Conv2DTranspose(filters, 
                        size, 
                        strides=strides,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False))

    # result.add(LayerNormalization())

    if apply_dropout:
        result.add(Dropout(0.25))

    result.add(LeakyReLU())

    return result


# Define the residual block as a new layer
class Res_dense(Layer):
    def __init__(self, units, dropout,**kwargs):
        super(Res_dense, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout

    def call(self, x):
        # the residual block using Keras functional API
        first_layer = Activation("linear", trainable=False)(x)
        x = Dense(self.units, 
                  activation=tf.nn.tanh, 
                  use_bias=True)(first_layer)
        x = Dense(self.units, 
                  activation=tf.nn.tanh, 
                  use_bias=True)(x)
        residual = Add()([x, first_layer])
        x = Activation(tf.nn.tanh)(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class Res_conv(Layer):
    def __init__(self, filters, size,**kwargs):
        super(Res_conv, self).__init__(**kwargs)
        self.filters = filters
        self.size = size

    def call(self, x):
        # the residual block using Keras functional API
        first_layer = Activation("linear", trainable=False)(x)
        x = Conv2D(self.filters,
                   self.size,
                   padding="same")(first_layer)
        x = Activation("relu")(x)
        x = Conv2D(self.filters,
                   self.size,
                   padding="same")(x)
        residual = Add()([x, first_layer])
        x = Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    
class Res_dconv(Layer):
    def __init__(self, filters, size,**kwargs):
        super(Res_dconv, self).__init__(**kwargs)
        self.filters = filters
        self.size = size

    def call(self, x):
        # the residual block using Keras functional API
        first_layer = Activation("linear", trainable=False)(x)
        x = Conv2DTranspose(self.filters,
                            self.size,
                            padding="same")(first_layer)
        x = Activation("relu")(x)
        x = Conv2DTranspose(self.filters,
                            self.size,
                            padding="same")(x)
        residual = Add()([x, first_layer])
        x = Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def dense_res(x, filters, size):
    x = Conv2D(filters, size, padding='same')(x)
    fx = Conv2DTranspose(filters, size, activation='relu', padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv2DTranspose(filters, size, padding='same')(fx)
    out = concatenate([x,fx], axis=3)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)


def conv_res(x, filters, size):
    x = Conv2D(filters, size, padding='same')(x)
    fx = Conv2DTranspose(filters, size, activation='relu', padding='same')(x)
    fx = BatchNormalization()(fx)
    fx = Conv2DTranspose(filters, size, padding='same')(fx)
    out = concatenate([x,fx], axis=3)
    out = LeakyReLU()(out)
    out = BatchNormalization()(out)
    return out



def make_generator_rb(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w ** 2
    inputs = Input(shape=(img_h, img_w, 1))
    x = tf.keras.layers.Flatten()(inputs)
    

def make_generator(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w ** 2
    inputs = Input(shape=(img_h, img_w, 1))
    x = Flatten()(inputs)
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
    # diffraction test temp with down sampled output
    # last = conv2d_norm(output_num, 3, 2)

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_w, img_w, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)
    return Model(inputs=inputs, outputs=x)

def make_generator_diff(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = (img_w//2) ** 2
    inputs = Input(shape=(img_h, img_w, 1))
    x = Flatten()(inputs)
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
    # diffraction test temp with down sampled output
    # last = conv2d_norm(output_num, 3, 2)

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_w//2, img_w//2, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)
    return Model(inputs=inputs, outputs=x)


def make_generator_3d(img_h, img_w, conv_num, conv_size, dropout, output_num):
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



def P(X, F1, k_size, s, stage):
    
    # Name definition
    P_Name = 'P-layer' + str(stage)
    P_BN_Name = 'P-layer-BN' + str(stage)
    
    X = Conv2D(filters = F1, kernel_size = (k_size, k_size), strides = (s, s), padding = 'valid',
              name = P_Name, kernel_initializer = glorot_uniform(seed = 0))(X)
    
    X = BatchNormalization(axis = 3, name = P_BN_Name)(X)
    
    X = Activation('relu')(X)
    
    return X

def FourierLayer(X, stage):
    
    # Name definition
    FFT_name = 'fft-layer' + str(stage)
    RFFT_name = 'ifft-layer' + str(stage)
    
    
    Res_X = X
    
    
    X = rfft2d(X, name = FFT_name)
    
    
    out_ft = tf.zeros((1, X.shape[1], X.shape[2], X.shape[3]),dtype=tf.dtypes.complex64)

    
    out_ft_first = tf.Variable(np.array(out_ft.shape))
    out_ft_first = tf.math.multiply(X[:, :, :5, :5], tf.cast(Res_X[:, :, :5, :5], tf.complex64))

    out_ft_end  = tf.Variable(np.array(out_ft.shape))
    out_ft_end = tf.math.multiply(X[:, :, -5:, :5], tf.cast(Res_X[:, :, -5:, :5], tf.complex64))


    o1 = tf.concat([out_ft_first,out_ft[:,:,:5,5:]],axis=3)
    o2 = tf.concat([out_ft[:,:,-5:,5:],out_ft_end],axis=3)
    
    out_ft = tf.concat([o1,o2],axis=2)

    
    X = irfft2d(X,name = RFFT_name)
    
    
    X = Add()([X, Res_X])
    
    X = Activation('relu')(X)

    
    return X

def Q(X, F1, k_size, s, stage):
    
    # Name definition
    Q_Name = 'Q-layer' + str(stage)
    Q_BN_Name = 'Q-layer-BN' + str(stage)
    
    
    X = Conv2D(filters = F1, kernel_size = (k_size, k_size), strides = (s, s), padding = 'same',
              name = Q_Name, kernel_initializer = glorot_uniform(seed = 0))(X)
    
    X = BatchNormalization(axis = 3, name = Q_BN_Name)(X)
    
    X = Activation('relu')(X)
    
    return X

def make_generator_fno(img_h, img_w, conv_num, conv_size, dropout, output_num):

    
    X_input = Input(shape = (img_h, img_w, 1))
    
    # Zero-Padding
    # X = ZeroPadding2D((2, 2))(X_input)
    
    # First Part
    # X = P(X, F1 = 256, k_size = 1, s = 1, stage = 1)
#    X = P(X, F1 = 256, k_size = 2, s = 2, stage = 2)
#    X = P(X, F1 = 256, k_size = 2, s = 2, stage = 3)
    # X = MaxPooling2D((2, 2), strides = (2, 2))(X)
    # X = tf.keras.layers.Dropout(0.3)(X)
    
    
    # Middle Part
    X = FourierLayer(X_input, stage = 1)
    X = FourierLayer(X, stage = 2)
    X = FourierLayer(X, stage = 3)
    X = FourierLayer(X, stage = 4)
    
    # Final Part

#     X = Q(X, F1 = 512, k_size = 1, s = 1, stage = 1)
# #    X = Q(X, F1 = 1024, k_size = 1, s = 2, stage = 2)
# #    X = Q(X, F1 = 256, k_size = 1, s = 2, stage = 3)
# #    X = UpSampling2D((2, 2))(X)
# #    X = tf.keras.layers.Dropout(0.4)(X)
#     X = Flatten()(X)
#     X = Dense(1024, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = Dense(len(classess), activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name = 'Fourier-Neural-Operator')
    
    return model

def make_filter(img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    down_stack = [
        conv2d_norm(16, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(16, 3, 1),
        # conv2d_norm(16, 3, 1),
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
        dconv2d_norm(16, 3, 1),  # (batch_size, 128, 128, 128)
        dconv2d_norm(16, 3, 1)
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

    return Model(inputs=inputs, outputs=x)


def make_discriminator(nang, px):
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[nang, px, 1]))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    # model.add(LayerNormalization())
    model.add(Dense(128))

    return model


def block(x_img, x_ts):
    x_parameter = Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_parameter = Activation('relu')(x_parameter)

    time_parameter = Dense(128)(x_ts)
    time_parameter = Activation('relu')(time_parameter)
    time_parameter = Reshape((1, 1, 128))(time_parameter)
    x_parameter = x_parameter * time_parameter
    
    # -----
    x_out = Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_out = x_out + x_parameter
    x_out = LayerNormalization()(x_out)
    x_out = Activation('relu')(x_out)
    
    return x_out

def diffusion_model(img_h, img_w, conv_num, conv_size, dropout, output_num):
    x = x_input = Input(shape=(img_h, img_w, 1), name='x_input')
    
    x_ts = x_ts_input = Input(shape=(1,), name='x_ts_input')
    x_ts = Dense(192)(x_ts)
    x_ts = LayerNormalization()(x_ts)
    x_ts = Activation('relu')(x_ts)
    
    # ----- left ( down ) -----
    print(f'Input shape: {x.shape}')
    x = x32 = block(x, x_ts)
    x = MaxPool2D(2, padding = 'same')(x)
    print(f'x32 shape: {x.shape}')
    x = x16 = block(x, x_ts)
    print(f'x16 shape: {x.shape}')
    x = MaxPool2D(2, padding = 'same')(x)
    print(f'x8 shape: {x.shape}')
    x = x8 = block(x, x_ts)
    print(f'x8 shape: {x.shape}')
    x = MaxPool2D(2, padding = 'same')(x)
    
    x = x4 = block(x, x_ts)
    
    # ----- MLP -----
    x = Flatten()(x)
    x = Concatenate()([x, x_ts])
    x = Dense(128)(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)

    x = Dense((img_h//8) * (img_w//8) * 32)(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((img_h//8, img_w//8, 32))(x)
    print(f'x4 shape: {x.shape}')
    
    # ----- right ( up ) -----
    x = Concatenate()([x, x4])
    print(f'x4 shape: {x.shape}')
    x = block(x, x_ts)
    
    x = UpSampling2D(2)(x)
    print(f'x8 shape: {x.shape}')
    x = Concatenate()([x, x8])
    x = block(x, x_ts)
    x = UpSampling2D(2)(x)
    
    x = Concatenate()([x, x16])
    x = block(x, x_ts)
    x = UpSampling2D(2)(x)
    
    x = Concatenate()([x, x32])
    x = block(x, x_ts)
    
    # ----- output -----
    x = Conv2D(3, kernel_size=1, padding='same')(x)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model
