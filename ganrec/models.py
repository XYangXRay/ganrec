
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, MaxPooling2D, UpSampling2D, concatenate, Dense, Conv2D, Conv2DTranspose, \
    Flatten, concatenate, \
        BatchNormalization, Dropout, \
            ReLU,LeakyReLU, Add, Activation
from tensorflow.keras.models import Model, Sequential, load_model

def dense_norm(units, dropout, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = Sequential()
    result.add(
        Dense(units, 
            #   activation=tf.nn.tanh, 
              use_bias=True, 
              kernel_initializer=initializer))
    result.add(Dropout(dropout))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=True):
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
        result.add(BatchNormalization())

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

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.25))

    result.add(LeakyReLU())

    return result



def make_generator(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_h * img_w
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

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_h, img_w, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model


def make_generator_rev(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_h * img_w
    inputs = Input(shape=(img_h, img_w, 1))

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

    
    last_1 = conv2d_norm(1, 3, 1)
    last = conv2d_norm(output_num, 3, 1)
    
    #model structure
    x = conv2d_norm(conv_num, conv_size+2, 1)(inputs)
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = dconv2d_norm(conv_num, conv_size+2, 1)(x)
    
    x = last_1(x)
    x = Flatten()(x)

    for fc in fc_stack:
        x = fc(x)
    x = tf.reshape(x, shape=[-1, img_h, img_w, 1])

    x = last(x)    
    model = Model(inputs=inputs, outputs=x) 
    return model


def make_filter(img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    down_stack = [
        conv2d_norm(16, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(16, 3, 1)
    ]

    up_stack = [

        dconv2d_norm(16, 3, 1),  # (batch_size, 128, 128, 128)
        dconv2d_norm(16, 3, 1)
    ]
    last = conv2d_norm(1, 3, 1)
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
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1))

    return model

def unet(img_h, img_w, output_num = 2, depth=4, conv_num=64, conv_size=3):
    inputs = Input(shape=(img_h, img_w, 1))
    conv_stack = [
        conv2d_norm(conv_num, conv_size, 1),
        conv2d_norm(conv_num*2, conv_size, 1),
        conv2d_norm(conv_num*3, conv_size, 1),
        conv2d_norm(conv_num*4, conv_size, 1),
    ]

    dconv_stack = [
        dconv2d_norm(conv_num*3, conv_size, 1),
        dconv2d_norm(conv_num*2, conv_size, 1),
        dconv2d_norm(conv_num, conv_size, 1),
    ]

    up = UpSampling2D(size=(2, 2))
     
    down = MaxPooling2D(pool_size=(2, 2))
    last = conv2d_norm(output_num, conv_size, 1)

    #model structure
    x = inputs
    for conv in conv_stack:
        x = conv(x)
        x = down(x)

    for dconv in dconv_stack:
        x = up(x)
        x = dconv(x)

    x = last(x)

    #to original size
    x = UpSampling2D(size=(2, 2))(x)
    model = Model(inputs=inputs, outputs=x)
    print(model.summary())
    return model
