from typing import List

import tensorflow as tf


def KraftUNet(input_shape: List[int] = (24, 32, 1), in_out_filters: int = 16) -> tf.keras.Model:

    x = tf.keras.Input(shape=input_shape)

    # Encoder
    y1 = tf.keras.layers.Conv2D(filters=in_out_filters, kernel_size=3, strides=1, padding='same')(x)
    y1 = tf.keras.layers.BatchNormalization()(y1)
    y1 = tf.keras.layers.ReLU()(y1)

    y1 = tf.keras.layers.Conv2D(filters=in_out_filters, kernel_size=3, strides=1, padding='same')(y1)
    y1 = tf.keras.layers.BatchNormalization()(y1)
    y1 = tf.keras.layers.ReLU()(y1)


    y2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y1)
    y2 = tf.keras.layers.Conv2D(filters=2*in_out_filters, kernel_size=3, strides=1, padding='same')(y2)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    y2 = tf.keras.layers.ReLU()(y2)

    y2 = tf.keras.layers.Conv2D(filters=2*in_out_filters, kernel_size=3, strides=1, padding='same')(y2)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    y2 = tf.keras.layers.ReLU()(y2)


    y3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(y2)
    y3 = tf.keras.layers.Conv2D(filters=4*in_out_filters, kernel_size=3, strides=1, padding='same')(y3)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.ReLU()(y3)

    y3 = tf.keras.layers.Conv2D(filters=4*in_out_filters, kernel_size=3, strides=1, padding='same')(y3)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.ReLU()(y3)

    # Decoder
    y3 = tf.keras.layers.Conv2DTranspose(filters=4*in_out_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(y3)
    
    
    y23 = tf.keras.layers.Concatenate(axis=-1)([y2, y3])

    y23 = tf.keras.layers.Conv2D(filters=2*in_out_filters, kernel_size=3, strides=1, padding='same')(y23)
    y23 = tf.keras.layers.BatchNormalization()(y23)
    y23 = tf.keras.layers.ReLU()(y23)

    y23 = tf.keras.layers.Conv2D(filters=2*in_out_filters, kernel_size=3, strides=1, padding='same')(y23)
    y23 = tf.keras.layers.BatchNormalization()(y23)
    y23 = tf.keras.layers.ReLU()(y23)

    y23 = tf.keras.layers.Conv2DTranspose(filters=2*in_out_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(y23)
    

    y123 = tf.keras.layers.Concatenate(axis=-1)([y1, y23])

    y123 = tf.keras.layers.Conv2D(filters=in_out_filters, kernel_size=3, strides=1, padding='same')(y123)
    y123 = tf.keras.layers.BatchNormalization()(y123)
    y123 = tf.keras.layers.ReLU()(y123)

    y123 = tf.keras.layers.Conv2D(filters=in_out_filters, kernel_size=3, strides=1, padding='same')(y123)
    y123 = tf.keras.layers.BatchNormalization()(y123)
    y123 = tf.keras.layers.ReLU()(y123)

    y = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(y123)
    y = tf.keras.layers.ReLU()(y)

    model = tf.keras.Model(inputs=x, outputs=y)

    return model
