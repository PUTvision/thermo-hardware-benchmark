from typing import List

import tensorflow as tf


def MetwalyFNN(input_shape: List[int] = (24, 32, 1), squeeze: bool = False) -> tf.keras.Model:

    x = tf.keras.Input(shape=input_shape)

    y = tf.keras.layers.Flatten()(x)

    y = tf.keras.layers.Dense(512)(y)
    y = tf.keras.layers.ReLU()(y)

    if squeeze:
        y = tf.keras.layers.Dense(512)(y)
        y = tf.keras.layers.ReLU()(y)

        y = tf.keras.layers.Dense(512)(y)
        y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.Dense(6)(y)
    y = tf.keras.layers.Softmax()(y)

    model = tf.keras.Model(inputs=x, outputs=y)

    return model
