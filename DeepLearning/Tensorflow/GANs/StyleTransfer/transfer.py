import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

import cv2
import PIL

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras import backend as K
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from scipy.optimize import fmin_l_bfgs_b

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# GLOBAL VARIABLES EEP #

target_width  = 512
target_height = 512
target_size = (target_width, target_height)

style_path = 'data/medieval_style_image.jpg'
content_path = 'data/mickey_matt.JPG'
generated_path = 'results/result_image.jpg'

c_img = load_img(path=content_path, target_size=target_size)
c_img_array = np.expand_dims(img_to_array(c_img), axis=0)
c_img_array = K.variable(preprocess_input(c_img_array), dtype='float32')

s_img = load_img(path=style_path, target_size=target_size)
s_img_array = np.expand_dims(img_to_array(s_img), axis=0)
s_img_array = K.variable(preprocess_input(s_img_array), dtype='float32')

g_img0 = np.random.randint(
    256,
    size=(target_width, target_height, 3)
).astype('float64')
g_img0 = preprocess_input(np.expand_dims(g_img0, axis=0))
g_img_placeholder = K.placeholder(shape=(1, target_width, target_height, 3))

def get_feature_reps(x, layer_names, model):

    feature_matrices = []

    for ln in layer_names:
        selected_layer = model.get_layer(ln)

        feature_raw = selected_layer.output
        feature_raw_shape = K.shape(feature_raw).eval(session=tf_session)

        N_l = feature_raw_shape[-1]
        M_l = feature_raw_shape[1] * feature_raw_shape[2]

        feature_matrix = K.reshape(feature_raw, (M_l, N_l))
        feature_matrix = K.transpose(feature_matrix)

        feature_matrices.append(feature_matrix)

    return feature_matrices

def get_content_loss(F, P):
    return 0.5 * K.sum(K.square(F - P))

def get_gram_matrix(F):
    return K.dot(F, K.transpose(F))

def get_style_loss(ws, Gs, As):

    style_loss = K.variable(0.)

    for w, G, A in zip(ws, Gs, As):

        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]

        G_gram = get_gram_matrix(G)
        A_gram = get_gram_matrix(A)

        style_loss += w * 0.25 * K.sum(K.square( G_gram - A_gram) / (N_l**2 * M_l**2) )

    return style_loss

def get_total_loss(g_img_placeholder, alpha=1.0, beta=10000.0):

    F = get_feature_reps(
        g_img_placeholder,
        layer_names=[c_layer_name],
        model=g_model
    )[0]

    Gs = get_feature_reps(
        g_img_placeholder,
        layer_names=s_layer_names,
        model=g_model
    )

    content_loss = get_content_loss(F, P)
    style_loss = get_style_loss(ws, Gs, As)
    total_loss = alpha * content_loss + beta * style_loss

    return total_loss

def calculate_loss(g_img_array):

    if g_img_array.shape != (1, target_width, target_width, 3):
        g_img_array = g_img_array.reshape((1, target_width, target_height, 3))

    loss_fn = K.function([g_model.input], [get_total_loss(g_model.input)])

    return loss_fn([g_img_array])[0].astype('float64')

def get_grad(g_img_array):

    if g_img_array.shape != (1, target_width, target_height, 3):
        g_img_array = g_img_array.reshape((1, target_width, target_height, 3))

    grad_fn = K.function(
        [g_model.input],
        K.gradients(
            get_total_loss(g_model.input),
            [g_model.input]
        )
    )

    grad = grad_fn([g_img_array])[0].flatten().astype('float64')

    return grad

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (target_width, target_height, 3):
        x = x.reshape((target_width, target_height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

tf_session = K.get_session()

c_model = VGG16(include_top=False, weights='imagenet', input_tensor=c_img_array)
s_model = VGG16(include_top=False, weights='imagenet', input_tensor=s_img_array)
g_model = VGG16(include_top=False, weights='imagenet', input_tensor=g_img_placeholder)

c_layer_name = 'block4_conv2'
s_layer_names = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1'
]

P = get_feature_reps(
    x=c_img_array,
    layer_names=[c_layer_name],
    model=c_model
)[0]
As = get_feature_reps(
    x=s_img_array,
    layer_names=s_layer_names,
    model=s_model
)
ws = np.ones(len(s_layer_names)) / float(len(s_layer_names))

iterations = 10

x_val = g_img0.flatten()
xopt, fval, info = fmin_l_bfgs_b(
    calculate_loss,
    x_val,
    fprime=get_grad,
    maxiter=iterations,
    disp=True
)

x_out = postprocess_array(xopt)
x_img = PIL.Image.fromarray(x_out)
x_img.save(generated_path)

