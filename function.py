import numpy as np
from keras.layers.merge import _Merge
from keras.layers import Reshape, multiply, Concatenate, Conv2D, Conv3D, Activation
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Lambda
from keras import backend as K


def simple_slice(x, index):
    return x[:, index, :, :]

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((10, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)



def test_predictImage(image, model, BATCH_SIZE, FRAME, img_rows, img_cols):
    preimage = image[:, 0:5]
    nt = 6
    for i in range(FRAME):
        input_img = np.reshape(image[:, 0 + i:5 + i], newshape=(image.shape[0], nt - 1, img_rows, img_cols, 1))
        input_img = np.concatenate([input_img, np.zeros((input_img.shape[0], 1, img_rows, img_cols, 1))], axis=1)
        pre_img = model.predict(input_img, BATCH_SIZE)
        preimage = np.concatenate([preimage, np.reshape(pre_img[:, 5, :, :, 0], newshape=(input_img.shape[0], 1, img_rows, img_cols))], axis=1)
        image = preimage
    return preimage*255

def predictMulImage (image, model, BATCH_SIZE, FRAME, img_rows, img_cols):
    preimage = image[:, 0:5]
    for i in range(FRAME):
        result = model.predict(np.reshape(preimage[:, i: i + 5], newshape=(image.shape[0], 5, img_rows, img_cols, 1)), BATCH_SIZE)
        result = result[:, 4, :, :, 0]
        preimage = np.concatenate((preimage, np.reshape(result, newshape=(image.shape[0], 1, img_rows, img_cols))), axis=1)
    return preimage * 255.

def channel_attention(input_feature, ratio=8):
    inputs_shape = K.int_shape(input_feature)
    if len(inputs_shape) == 3:
        batchsize, dim1, channels = inputs_shape

    elif len(inputs_shape) == 4:
        batchsize, dim1, dim2, channels = inputs_shape

    elif len(inputs_shape) == 5:
        batchsize, dim1, dim2, dim3, channels = inputs_shape
    else:
        raise ValueError('Input dimension is wrong)')

    channels_ratio = channels // ratio
    if channels_ratio == 0:
        channels_ratio = 1
    else:
        channels_ratio = channels // ratio
    layer_one = Dense(channels_ratio, kernel_initializer='he_normal', use_bias=True, activation='relu', bias_initializer='zeros')
    layer_two = Dense(channels, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    layer_one_2 = Dense(channels_ratio, kernel_initializer='he_uniform', use_bias=True, activation='relu', bias_initializer='zeros')
    layer_two_2 = Dense(channels, kernel_initializer='he_uniform', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channels))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channels)
    avg_pool = layer_one_2(avg_pool)
    assert avg_pool.shape[1:] == (1, 1,  channels_ratio)
    avg_pool = layer_two_2(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channels)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channels))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channels)
    max_pool = layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1,  channels_ratio)
    max_pool = layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channels)


    c_feature = Add()([avg_pool, max_pool])
    c_feature = Activation('hard_sigmoid')(c_feature)

    return multiply([input_feature, c_feature])


def STIC_attention(input_feature):
    stic_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=4, keepdims=True))(stic_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=4, keepdims=True))(stic_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=4)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    stic_feature = Conv3D(filters=1,
                          kernel_size=(7, 7, 5),
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert stic_feature.shape[-1] == 1


    return multiply([input_feature, stic_feature])


def CS_attention(feature, ratio=8):
    cs_feature = channel_attention(feature, ratio)
    cs_feature = spatial_attention(cs_feature, )
    return cs_feature


def spatial_attention(input_feature):
    s_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(s_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(s_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    s_feature = Conv2D(filters=1,
                          kernel_size=(7, 7),
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert s_feature.shape[-1] == 1

    return multiply([input_feature, s_feature])
