import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense

def main():

    print("Beginning program execution...")

    # Open path and read in text to memory
    data_path = "data/shakespeare.txt"
    text_chars = open(data_path, "rb").read().decode(encoding="utf-8")
    text_words = text_chars.split(" ")
    vocab = sorted(set(text_chars))

    # Explore the basic information of the text
    #explore_text(text_chars, text_words)

    # __Vectorize__
    # Goal: Have every character represented by an integer
    # text_as_int is the entire text file but instead of the correct
    # characters it has the index of each character in our char_lookup dict
    encoded_text_chars, indexer = chars_to_int(text_chars, vocab)
    print("{} ----> mapped to ints ----> {}" \
        .format(str(text_chars[:13]), str(encoded_text_chars[:13])))
    print("Shape of text characters data object: {}".format(encoded_text_chars.shape))

    seq_length = 100
    examples_per_epoch = len(text_chars) / seq_length

    char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text_chars)

    print("\n")
    for i in char_dataset.take(5):
        print(indexer[i.numpy()])

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    print("\n")
    for item in sequences.take(5):
        print(repr("".join(indexer[item.numpy()])))

    dataset = sequences.map(split_input_target)

    print("\n")
    for input_example, target_example in dataset.take(1):
        print("Input data: {}".format(repr("".join(indexer[input_example.numpy()]))))
        print("Target data: {}".format(repr("".join(indexer[target_example.numpy()]))))

        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(indexer[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(indexer[target_idx])))

    batch_size = 64
    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print(dataset)

    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size
    )

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
        print(sampled_indices)

        print("Input: \n", repr("".join(indexer[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(indexer[sampled_indices])))

    print(model.summary())

    model.compile(optimizer="adam", loss=loss)

    epochs=10

    history = model.fit(dataset, epochs=epochs)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    model = Sequential([
        Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),
        Dense(vocab_size)
    ])

    return model

def split_input_target(chunk):

    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def chars_to_int(text_chars, vocab):

    char_lookup = {u:i for i, u in enumerate(vocab)}
    indexer = np.array(vocab)
    encoded_text_chars = np.array([char_lookup[letter] for letter in text_chars])

    return encoded_text_chars, indexer

def explore_text(text, text_words):

    print("Length of text: {} characters".format(len(text)))
    print("First 250 characters: \n{}".format(text[:250]))

    print("Text contains {} words".format(len(text_words)))

if __name__ == "__main__":
    main()
