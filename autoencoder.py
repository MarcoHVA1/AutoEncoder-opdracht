# models/autoencoder.py
"""
Convolutional Autoencoder voor Surface Crack Anomaly Detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(input_shape=(64, 64, 3)):
    """
    Bouw een convolutional autoencoder voor anomaly detection.

    Returns:
        autoencoder: volledig model (encoder + decoder)
        encoder: alleen het encoder-deel (voor K-means clustering)
    """
    # ==========================
    # ENCODER
    # ==========================
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")

    # Block 1: 64x64 -> 32x32
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(encoder_input)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)

    # Block 2: 32x32 -> 16x16
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)

    # Block 3: 16x16 -> 8x8
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same", name="encoded")(x)

    # Maak los encoder-model (voor K-means)
    encoder = models.Model(encoder_input, encoded, name="encoder")

    # ==========================
    # DECODER
    # ==========================
    # 8x8 -> 16x16
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_conv1")(encoded)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up1")(x)

    # 16x16 -> 32x32
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up2")(x)

    # 32x32 -> 64x64
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up3")(x)

    decoder_output = layers.Conv2D(
        input_shape[2],
        (3, 3),
        activation="sigmoid",  # omdat je input tussen 0 en 1 zit
        padding="same",
        name="decoder_output",
    )(x)

    autoencoder = models.Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return autoencoder, encoder


if __name__ == "__main__":
    # Kleine zelftest
    print("🏗️  Testing Autoencoder Architecture")
    model, enc = build_autoencoder(input_shape=(64, 64, 3))
    model.summary()
    import numpy as np

    x = np.random.rand(1, 64, 64, 3).astype("float32")
    y = model.predict(x, verbose=0)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)