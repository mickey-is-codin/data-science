from networks import DCGAN

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():

    print("Beginning number generation")

    deep_conv_gan = DCGAN()

    img_rows = 28
    img_cols = 28
    channel = 1

    x_train = input_data.read_data_sets(
        'mnist',
        one_hot=True
    ).train.images

    x_train = x_train.reshape(
        -1,
        img_rows,
        img_cols,
        1
    ).astype(np.float32)

    generator = deep_conv_gan.generator()
    discriminator = deep_conv_gan.discriminator_model()
    adversarial = deep_conv_gan.adversarial_model()

    train(
        x_train,
        generator,
        discriminator,
        adversarial,
        img_rows, img_cols
    )

def train(
    x_train,
    generator,
    discriminator,
    adversarial,
    img_rows, img_cols,
    train_steps=2000,
    batch_size=256,
    save_interval=0):

    noise_input = None

    if save_interval > 0:
        noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])

    print("==Training==")
    for i in range(train_steps):

        # The actual data samples
        images_train = x_train[np.random.randint(
            0,
            x_train.shape[0],
            size=batch_size), :, :, :]

        # Our input to the generator NN
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

        # Forward pass of the generator NN to produce fake images
        images_fake = generator.predict(noise)

        # Make an input for the discriminator 0 = fake, 1 = real
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        d_loss = discriminator.train_on_batch(x, y)

        y = np.ones([batch_size, 1])

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])

        a_loss = adversarial.train_on_batch(noise, y)

        print("Epoch %d:\n\tD Network Loss: %.3f Accuracy: %.3f" % (i, d_loss[0], d_loss[1]))
        print("\tOverall A Loss: %.3f Accuracy: %.3f" % (a_loss[0], a_loss[1]))

        if (i % 10 == 0):
            plot_images(
                x_train,
                img_rows, img_cols,
                generator,
                save=True,
                fake=True,
                samples=16,
                noise=None,
                step=i+1
            )

    generator.save('trained_generator.model')
    discriminator.save('trained_discriminator.model')
    adversarial.save('trained_adversarial.model')

def plot_images(
    x_train,
    img_rows,
    img_cols,
    generator,
    save=True,
    fake=True,
    samples=16,
    noise=None,
    step=0):

    filename = 'mnist_{}_train_steps.png'.format(step)

    if fake:
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        else:
            filename = 'mnist_{}.png'.format(step)
        images = generator.predict(noise)
    else:
        i = np.random.randint(0, x_train.shape[0], samples)
        images = x_train[i, :, :, :]

    plt.figure()

    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [img_rows, img_cols])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    main()
