import numpy as np
import hickle as hkl
from Model import build_gan



def gan_train(g, d, critic_model, generator_model, BATCH_SIZE, img_rows, img_cols, input_img_num):
    X_train = hkl.load('/run/media/root/img128*128/train.hkl') / 255.
    X_test = hkl.load('/run/media/root/img128*128/test.hkl') / 255.
    test_imgs = X_test[45]
    test_imgs = np.reshape(test_imgs, newshape=(1, X_train.shape[1], img_rows, img_cols, 1))
    X_train = np.expand_dims(X_train, axis=4)
    n_critic = 5
    valid = -np.ones((BATCH_SIZE, 1))
    fake = np.ones((BATCH_SIZE, 1))
    dummy = np.zeros((BATCH_SIZE, 1))

    onece_epoch = round(X_train.shape[0] / BATCH_SIZE + 0.5)
    total_epoch = 100 * onece_epoch

    for epoch in range(0, total_epoch):
        for frame in range(7):
            for _ in range(n_critic):
                idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
                imgs = X_train[idx]
                noise = imgs[:, frame: frame + input_img_num + 1]
                image_batch = imgs[:, frame + input_img_num]
                d_loss = critic_model.train_on_batch([image_batch, noise], [valid, fake, dummy])

            g_loss = generator_model.train_on_batch(noise, valid)

        if epoch % 5 == 0:
            g.save_weights('/run/media/root/train_result/weight/g/GAN_' + str(epoch) + '.h5')
            d.save_weights('/run/media/root/train_result/weight/d/GAN_' + str(epoch) + '.h5')

img_rows = 128
img_cols = 128
channels = 1
input_img_num = 5
img_shape = (img_rows, img_cols, channels)
latent_dim = (6,) + (img_rows, img_cols, channels)
d_input_shape = (img_rows, img_cols, 1)
BATCH_SIZE = 10
nt = 6


d, g, critic_model, generator_model = build_gan(BATCH_SIZE, d_input_shape, latent_dim,
                                                img_shape, img_rows, img_cols, input_img_num, 'STPIN')
gan_train(g, d, critic_model, generator_model, BATCH_SIZE, img_rows, img_cols, input_img_num)


