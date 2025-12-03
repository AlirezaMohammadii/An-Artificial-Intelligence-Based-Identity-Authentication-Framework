import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import sys
import configparser  # For reading the config file

import tensorflow_addons as tfa  # For RectifiedAdam

sys.path.append("..")
import guardian.constants as c

# Function to parse metrics from config file
def get_metrics(metrics_list):
    metrics_map = {
        "accuracy": "accuracy",
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
        "auc": tf.keras.metrics.AUC(name="auc"),
    }
    return [metrics_map[m.strip()] for m in metrics_list if m.strip() in metrics_map]

def get_activation_layer(activation):
    """Returns the appropriate activation layer based on config input."""
    if activation.lower() == "leaky_relu":
        return layers.LeakyReLU(alpha=0.01)
    elif activation.lower() == "relu":
        return layers.ReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def residual_block(x, filters, kernel_size, activation, l1_l2_value):
    shortcut = x
    # Adjust the channels in the shortcut if they do not match
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, (1, 1), padding='same',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(
        filters, kernel_size, padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value)
    )(x)
    x = layers.BatchNormalization()(x)
    x = get_activation_layer(activation)(x)
    x = layers.Conv2D(
        filters, kernel_size, padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value)
    )(x)
    x = layers.BatchNormalization()(x)

    # Add the adjusted shortcut to the output
    x = layers.add([x, shortcut])
    x = get_activation_layer(activation)(x)
    return x

def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = layers.multiply([input_tensor, se])
    return x

def discriminator_model(config):
    """
    Creates the discriminator model with adjustable dropout rates, batch normalization,
    L1/L2 regularization, and the number of convolutional blocks.

    Parameters:
    - config: The configuration object with all model parameters.

    Returns:
    - model: The keras model (not compiled).
    - num_layer: The number of layers in the model.
    """
    # Parse parameters from config
    dropout_rate = config.getfloat('main', 'dropout_rate')
    l1_l2_value = config.getfloat('main', 'l1_l2_value')
    activation = config['main']['activation']
    conv_block_count = config.getint('main', 'conv_block_count')

    inputs = keras.Input(shape=(32, 32, 1))

    # Convolutional Layer 1 with residual block
    x = layers.Conv2D(
        64,  # Increased from 32 to 64
        kernel_size=(4, 4),
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = get_activation_layer(activation)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Adding residual blocks based on `conv_block_count`
    if conv_block_count >= 2:
        x = residual_block(x, 128, (3, 3), activation, l1_l2_value)  # Increased filters
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)

    if conv_block_count >= 3:
        x = residual_block(x, 256, (3, 3), activation, l1_l2_value)  # Increased filters
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)

    # Add Squeeze-and-Excitation (SE) block after convolutional blocks
    x = se_block(x, ratio=16)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense Layer 1
    x = layers.Dense(
        128,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value)
    )(x)
    x = layers.BatchNormalization()(x)
    x = get_activation_layer(activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense Layer 2
    x = layers.Dense(
        32,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value)
    )(x)
    x = layers.BatchNormalization()(x)
    x = get_activation_layer(activation)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output Layer
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_discriminator_model")

    # Removed model.compile()
    num_layer = len(model.layers)
    return model, num_layer

def print_config_parameters(config):
    print("\n--- Model Configuration Parameters ---")
    for section in config.sections():
        print(f"\n[{section}]")
        for key in config[section]:
            print(f"{key}: {config[section][key]}")
    print("\n----------------------------------------")

if __name__ == "__main__":
    # Load config file
    config_file_path = 'config_15_11_24.ini'
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Print the configuration parameters
    print_config_parameters(config)

    # Build the model with parameters from config file
    model, num_layer = discriminator_model(config)

    # Save the model with a random ID
    name_model = random.randrange(1000000000, 9999999999)
    print(f"Model ID: {name_model}")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    model.save(c.DISCRIMINATOR_MODEL + str(name_model) + ".h5")
