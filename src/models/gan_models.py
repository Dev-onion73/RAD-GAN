# src/models/gan_models.py
import tensorflow as tf
from tensorflow.keras import layers

def build_generator(input_shape=(64,64,1), base_filters=64):
    """UNet-like generator mapping noise tensor to spectrogram image."""
    inp = layers.Input(shape=input_shape)
    # Encoder
    e1 = layers.Conv2D(base_filters, 4, strides=2, padding='same')(inp)
    e1 = layers.LeakyReLU(0.2)(e1)
    e2 = layers.Conv2D(base_filters*2, 4, strides=2, padding='same')(e1)
    e2 = layers.BatchNormalization()(e2); e2 = layers.LeakyReLU(0.2)(e2)
    e3 = layers.Conv2D(base_filters*4, 4, strides=2, padding='same')(e2)
    e3 = layers.BatchNormalization()(e3); e3 = layers.LeakyReLU(0.2)(e3)
    # Decoder
    d3 = layers.Conv2DTranspose(base_filters*2, 4, strides=2, padding='same')(e3)
    d3 = layers.BatchNormalization()(d3); d3 = layers.ReLU()(d3)
    d2 = layers.Concatenate()([d3, e2])
    d2 = layers.Conv2DTranspose(base_filters, 4, strides=2, padding='same')(d2)
    d2 = layers.BatchNormalization()(d2); d2 = layers.ReLU()(d2)
    d1 = layers.Concatenate()([d2, e1])
    out = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')(d1)
    return tf.keras.Model(inp, out, name='Generator')

def build_patch_discriminator(input_shape=(64,64,1), base_filters=64):
    """PatchGAN discriminator"""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(base_filters, 4, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(base_filters*2, 4, strides=2, padding='same')(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(base_filters*4, 4, strides=2, padding='same')(x); x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)  # output feature map (Patch)
    return tf.keras.Model(inp, x, name='PatchDiscriminator')
