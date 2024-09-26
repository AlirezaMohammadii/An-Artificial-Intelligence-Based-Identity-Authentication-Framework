import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import sys

sys.path.append("..")
import guardian.constants as c


def discriminator_model(optimizer, loss, metrics, dropout_rate, l1_l2_value):
    """
    Creates the discriminator model with adjustable dropout rates, batch normalization,
    and L1/L2 regularization.

    Parameters:
    - optimizer: The optimization algorithm.
    - loss: The loss function.
    - metrics: The list of metrics for evaluation.
    - dropout_rate: The dropout rate between 0.2 and 0.5.
    - l1_l2_value: The regularization factor for both L1 and L2 regularization.
        - L1 provides sparsity with setting some weights to zero,
        - L2 tries to keep the weights small but not necessarily set it to zero.
    Returns:
    - model: The compiled keras model.
    - num_layer: The number of layers in the model.
    """
    inputs = keras.Input(shape=(32, 32, 1))
    # Convolutional Layer 1
    x = layers.Conv2D(
        32,
        kernel_size=(4, 4),
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Convolutional Layer 2
    # x = layers.Conv2D(
    #     64,
    #     kernel_size=(3, 3),
    #     activation="relu",
    #     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
    # )(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # x = layers.Dropout(dropout_rate)(x)

    # Dense Layer 1
    x = layers.Flatten()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense Layer 2
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output Layer
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_discriminator_model")

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    num_layer = len(model.layers)
    return model, num_layer


if __name__ == "__main__":
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ["accuracy"]
    dropout_rate = 0.2  # Adjustable dropout rate
    l1_l2_value = 0.01  # Regularization factor for L1/L2

    model, num_layer = discriminator_model(
        optimizer, loss, metrics, dropout_rate, l1_l2_value
    )
    name_model = random.randrange(1000000000, 9999999999)
    print(f"Model ID: {name_model}")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    model.save(c.DISCRIMINATOR_MODEL + str(name_model) + ".h5")
