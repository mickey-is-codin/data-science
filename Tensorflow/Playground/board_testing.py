import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
import keras.layers as layers
import keras.models as models
import keras.callbacks as callbacks

name = 'cats_v_dogs_cnn_64x2-{}'.format(int(time.time()))
tensorboard = callbacks.TensorBoard(log_dir='logs/{}'.format(name))

X = pickle.load(open('cat_dogs_X.pickle', 'rb'))
y = pickle.load(open('cat_dogs_y.pickle', 'rb'))

X = X / 255.0

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

# Training parameters
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])
