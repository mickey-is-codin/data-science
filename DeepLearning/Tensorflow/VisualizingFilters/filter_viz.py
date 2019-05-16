import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():

    print("\nBeginning filter visualization program\n")

    # Parse command line args
    if len(sys.argv) == 3:
        mode = sys.argv[1]
        image_path = sys.argv[2]
    else:
        mode = '--advanced'
        image_path = 'data/2666.jpg'


    # Get model and create dictionary for its layers
    model = VGG16()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Basic diagnostics to view network architecture
    print("==VGG Network Architecture==")
    for key, val in layer_dict.items():
        print(key)
    print("==End NETWORK Architecture\n")

    # Allow the user to choose which layer they want to visualize
    choice_default = "block1_conv2"
    print("Earlier layers will more closely resemble original image")
    chosen_layer = input('Enter layer: (default = block1_conv2) ')
    if not chosen_layer:
        chosen_layer = choice_default

    basic = False
    advanced = False
    if mode == '--basic' or mode == '-b':
        basic = True
        advanced = False
    elif mode == '--advanced' or mode == '-a':
        basic = False
        advanced = True
    else:
        basic = False
        advanced = True

    if basic:
        # Plot the basic filters
        basic_plot(layer_dict, chosen_layer)

    if advanced:
        # Plot the advanced filters
        advanced_plot(model, layer_dict, chosen_layer, image_path)

def advanced_plot(model, layer_dict, chosen_layer, image_path):

    viz_model = Model(inputs=model.inputs, outputs=layer_dict[chosen_layer].output)

    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature_maps = viz_model.predict(image)

    # Plotting geometry
    row_choice_default = 3
    try:
        n_rows = int(input('Number rows to plot: (default = 3) '))
    except ValueError:
        n_rows = row_choice_default

    col_choice_default = 3
    try:
        n_cols = int(input('Number cols to plot: (default = 3) '))
    except ValueError:
        n_cols = col_choice_default

    print("Plotting {} filters".format(n_rows * n_cols))

    index = 1

    for i in range(n_rows):
        for j in range(n_cols):

            ax = plt.subplot(n_rows, n_cols, index)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
            index += 1

    plt.suptitle('Advanced Vizualization of {}'.format(chosen_layer))
    plt.show()

    predictions = model.predict(image)
    predictions = predictions[0]

    labels_dict = create_labels_dict()
    prediction_list = []

    for ix, certainty in enumerate(predictions):
        if certainty > 0.075:
            guess = list(labels_dict.values())[ix]
            print("Classification made!: {}".format(guess))
            prediction_list.append(guess)

    original_image = load_img(image_path)
    ax = plt.subplot()
    plt.imshow(original_image)
    plt.subplots_adjust(right=0.68)

    for label_ix, label in enumerate(prediction_list):
        plt.figtext(
            0.8,
            0.25*(label_ix+1),
            label,
            horizontalalignment='center'
        )

    plt.suptitle('Predictions')
    plt.axis('off')
    plt.show()

def create_labels_dict(path='data/labels/labels.txt'):

    labels_dict = {}

    label_file = open(path, 'r')
    file_lines = label_file.readlines()

    for line in file_lines:
        (key, value) = line.split(':')
        labels_dict[int(key)] = value

    return labels_dict

def basic_plot(layer_dict, chosen_layer):

    # Retrieve the filters from the model
    filters, biases = get_filters(layer_dict, chosen_layer)

    # Normalize the filters to [0,1]
    filters = normalize_filters(filters)

    # Plotting geometry
    row_choice_default = 3
    try:
        n_rows = int(input('Number rows to plot: (default = 3) '))
    except ValueError:
        n_rows = row_choice_default

    col_choice_default = 3
    try:
        n_cols = int(input('Number cols to plot: (default = 3) '))
    except ValueError:
        n_cols = col_choice_default

    print("Plotting {} filters".format(n_rows * n_cols))

    index = 1

    for i in range(n_rows):
        f = filters[:,:,:,i]

        # Plot each channel separately
        for j in range(n_cols):

            ax = plt.subplot(n_rows, n_cols, index)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(f[:, :, j], cmap='viridis')
            index += 1

    plt.suptitle('Basic Vizualization of {}'.format(chosen_layer))
    plt.show()

def normalize_filters(filters):

    f_min, f_max = np.amin(filters), np.amax(filters)
    norm_filters = (filters - f_min) / (f_max - f_min)

    return norm_filters

def get_filters(layer_dict, chosen_layer):

    filter_index = 0
    filters, biases = layer_dict[chosen_layer].get_weights()

    print("Layer weights shape: {}".format(filters.shape))

    return filters, biases

if __name__ == '__main__':
    main()
