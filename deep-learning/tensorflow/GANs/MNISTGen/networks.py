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

class DCGAN(object):

    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel

        # The discriminator and generator
        self.D = None
        self.G = None

        # Adversarial model and discriminator model
        self.AM = None
        self.DM = None

    def discriminator(self):

        # Input:  28x28x1 image of a number, depth = 1

        if self.D:
            return self.D

        depth = 64
        dropout = 0.4

        self.D = Sequential()

        input_shape = (self.img_rows, self.img_cols, self.channel)

        self.D.add(
            Conv2D(
                depth*1,
                5,
                strides=2,
                input_shape=input_shape,
                padding='same',
                activation=LeakyReLU(alpha=0.2)
            )
        )
        self.D.add(
            Dropout(dropout)
        )

        self.D.add(
            Conv2D(
                depth*2,
                5,
                strides=2,
                input_shape=input_shape,
                padding='same',
                activation=LeakyReLU(alpha=0.2)
            )
        )
        self.D.add(
            Dropout(dropout)
        )

        self.D.add(
            Conv2D(
                depth*4,
                5,
                strides=2,
                input_shape=input_shape,
                padding='same',
                activation=LeakyReLU(alpha=0.2)
            )
        )
        self.D.add(
            Dropout(dropout)
        )

        self.D.add(
            Conv2D(
                depth*8,
                5,
                strides=2,
                input_shape=input_shape,
                padding='same',
                activation=LeakyReLU(alpha=0.2)
            )
        )
        self.D.add(
            Dropout(dropout)
        )
        # Current State: 14x14x1 convolution feature map, depth=64

        self.D.add(
            Flatten()
        )
        self.D.add(
            Dense(1)
        )
        self.D.add(
            Activation('sigmoid')
        )
        # Current State: Scalar prediction (whether or not generated)

        self.D.summary()
        return self.D

    def generator(self):

        # Input Dimensions: 100

        if self.G:
            return self.G

        depth = 64 + 64 + 64 + 64
        dropout = 0.4
        dim = 7

        self.G = Sequential()

        self.G.add(
            Dense(
                dim * dim * depth,
                input_dim = 100
            )
        )
        self.G.add(
            BatchNormalization(momentum=0.9)
        )
        self.G.add(
            Activation('relu')
        )
        self.G.add(
            Reshape((dim, dim, depth))
        )
        self.G.add(
            Dropout(dropout)
        )
        # Current Dimensions: dim x dim x depth

        self.G.add(
            UpSampling2D()
        )
        self.G.add(
            Conv2DTranspose(
                int(depth/2),
                5,
                padding='same'
            )
        )
        self.G.add(
            BatchNormalization(momentum=0.9)
        )
        self.G.add(
            Activation('relu')
        )

        self.G.add(
            UpSampling2D()
        )
        self.G.add(
            Conv2DTranspose(
                int(depth/4),
                5,
                padding='same'
            )
        )
        self.G.add(
            BatchNormalization(momentum=0.9)
        )
        self.G.add(
            Activation('relu')
        )

        self.G.add(
            Conv2DTranspose(
                int(depth/4),
                5,
                padding='same'
            )
        )
        self.G.add(
            BatchNormalization(momentum=0.9)
        )
        self.G.add(
            Activation('relu')
        )
        # Current Dimensions: 2*dim x 2*dim x depth/2

        self.G.add(
            Conv2DTranspose(
                1,
                5,
                padding='same'
            )
        )
        self.G.add(
            Activation('sigmoid')
        )
        # Current Dimensions: 28 x 28 x 1 grayscale image [0.0,1.0] per pix

        self.G.summary()
        return self.G

    def discriminator_model(self):

        if self.DM:
            return self.DM

        loss = 'binary_crossentropy'
        optimizer = RMSprop(lr=0.0002, decay=6e-8)

        self.DM = Sequential()

        self.DM.add(
            self.discriminator()
        )

        self.DM.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        return self.DM

    def adversarial_model(self):

        if self.AM:
            return self.AM

        loss = 'binary_crossentropy'
        optimizer = RMSprop(lr=0.0001, decay=3e-8)

        self.AM = Sequential()

        self.AM.add(
            self.generator(),
        )
        self.AM.add(
            self.discriminator()
        )

        self.AM.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        return self.AM


