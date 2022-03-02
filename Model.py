from keras.layers import Input, Dense,  Flatten, Lambda, Concatenate, LeakyReLU, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from functools import partial
from function import RandomWeightedAverage, wasserstein_loss, gradient_penalty_loss, extrap_loss
from function import simple_slice, CS_attention, STIC_attention
from ArgcPredNet import Argc_PredNet



def build_GAN_model(latent_dim, img_shape, input_img_num):
    stack_sizes = (1, 128, 128, 256)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    prednet = Argc_PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='prediction', return_sequences=True)

    noise = Input(latent_dim, name='g_model_input')
    prediction = prednet(noise)
    prediction = STIC_attention(prediction)
    result_img = Lambda(simple_slice, output_shape=img_shape, arguments={'index': input_img_num})(prediction)
    real_img = Lambda(simple_slice, output_shape=img_shape, arguments={'index': input_img_num})(noise)
    generated_images = Concatenate(axis=3)([real_img, result_img])
    return Model(noise, generated_images)

def build_STPIN_model(nt, img_rows, img_cols):
    input_shape = (img_rows, img_cols, 1)
    stack_sizes = (1, 128, 128, 256)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    prednet = Argc_PredNet(stack_sizes, R_stack_sizes,
                           A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                           output_mode='prediction', return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)

    prediction = prednet(inputs)
    prediction = STIC_attention(prediction)
    model = Model(input=inputs, output=prediction)
    model.compile(loss=extrap_loss, optimizer='adam')

    return model

def d_model_1(input, d_input_shape):
    mix_con2d_layer1 = Conv2D(256, kernel_size=3, strides=2, input_shape=d_input_shape, padding="same")(input)
    mix_lre_layer1 = LeakyReLU(alpha=0.2)(mix_con2d_layer1)
    mix_lre_layer1 = CS_attention(mix_lre_layer1)
    mix_con2d_layer2 = Conv2D(128, kernel_size=3, strides=2, padding="same")(mix_lre_layer1 )
    mix_lre_layer2 = LeakyReLU(alpha=0.2)(mix_con2d_layer2)
    mix_con2d_layer3 = Conv2D(64, kernel_size=3, strides=2, padding="same")(mix_lre_layer2)
    mix_lre_layer3 = LeakyReLU(alpha=0.2)(mix_con2d_layer3)
    mix_con2d_layer4 = Conv2D(64, kernel_size=3, strides=2, padding="same")(mix_lre_layer3)
    mix_lre_layer4 = LeakyReLU(alpha=0.2)(mix_con2d_layer4)

    mix_flatten_layer = Flatten()(mix_lre_layer4)

    mix_dense_layer1 = Dense(1024)(mix_flatten_layer)
    mix_dense_layer2 = Dense(1)(mix_dense_layer1)

    return Model(input, mix_dense_layer2)


def build_discriminator_model(d_input_shape):
    img = Input(shape=d_input_shape, name='d_model_input')
    model1 = d_model_1(img, d_input_shape)
    out1 = model1.output
    return Model(img, out1)


def build_generator(latent_dim, img_shape, img_rows, img_cols, input_img_num):
    g = build_GAN_model(latent_dim, img_shape, input_img_num)
    return g


def build_gan(BATCH_SIZE, d_input_shape, latent_dim, img_shape, img_rows, img_cols, input_img_num, name):
    g = build_generator(latent_dim, img_shape, img_rows, img_cols, input_img_num)
    d = build_discriminator_model(d_input_shape)
    adam_opt = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9)

    g.trainable = False

    real_img = Input(shape=d_input_shape, name='real_img')

    z_disc = Input(shape=latent_dim, name='z_disc')
    fake_img = g(z_disc)
    fake = d(fake_img)
    valid = d(real_img)

    interpolated_img = RandomWeightedAverage()([real_img, fake_img])
    validity_interpolated = d(interpolated_img)


    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model(inputs=[real_img, z_disc],
                   outputs=[valid, fake, validity_interpolated])
    critic_model.compile(loss=[wasserstein_loss,
                         wasserstein_loss,
                         partial_gp_loss],
                   optimizer=adam_opt,
                   loss_weights=[1, 1, 10])

    d.trainable = False
    g.trainable = True

    z_gen = Input(shape=latent_dim, name='z_gen')
    img = g(z_gen)
    valid = d(img)
    generator_model = Model(z_gen, valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=adam_opt)
    return d, g, critic_model, generator_model



