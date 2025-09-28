# src/models/encoders.py
import tensorflow as tf
from tensorflow.keras import layers

def build_encoder(input_shape=(64,64,1), feat_dim=128):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,3,2,'same', activation='relu')(inp)
    x = layers.Conv2D(64,3,2,'same', activation='relu')(x)
    x = layers.Conv2D(128,3,2,'same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feat_dim)(x)
    return tf.keras.Model(inp, x, name='Encoder')
