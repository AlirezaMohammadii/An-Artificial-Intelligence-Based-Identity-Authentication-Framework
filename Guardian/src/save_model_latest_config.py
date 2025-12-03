import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import sys
import configparser  # For reading the config file

sys.path.append("..")
import guardian.constants as c

# Function to parse metrics from config file
def get_metrics(metrics_list):
    metrics_map = {
        "accuracy": "accuracy",
        "precision": tf.keras.metrics.Precision(),
        "recall": tf.keras.metrics.Recall(),
    }
    return [metrics_map[m] for m in metrics_list if m in metrics_map]


def discriminator_model(config):
    """
    Creates the discriminator model with adjustable dropout rates, batch normalization,
    L1/L2 regularization, and the number of convolutional blocks.

    Parameters:
    - config: The configuration object with all model parameters.

    Returns:
    - model: The compiled keras model.
    - num_layer: The number of layers in the model.
    """
    # Parse parameters from config
    optimizer = getattr(tf.keras.optimizers, config['main']['optimizer'])()
    loss = getattr(tf.keras.losses, config['main']['loss'])()
    metrics = get_metrics(config['main']['metrics'].split(','))
    dropout_rate = config.getfloat('main', 'dropout_rate')
    l1_l2_value = config.getfloat('main', 'l1_l2_value')
    activation = config['main']['activation']
    conv_block_count = config.getint('main', 'conv_block_count')

    inputs = keras.Input(shape=(32, 32, 1))

    # Convolutional Layer 1
    x = layers.Conv2D(
        32,
        kernel_size=(4, 4),
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Conditionally add the second block if conv_block_count is 2
    if conv_block_count == 2:
        x = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)

    # Dense Layer 1
    x = layers.Flatten()(x)
    x = layers.Dense(
        128,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_l2_value, l2=l1_l2_value),
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense Layer 2
    x = layers.Dense(
        32,
        activation=activation,
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


def print_config_parameters(config):
    print("\n--- Model Configuration Parameters ---")
    for section in config.sections():
        print(f"\n[{section}]")
        for key in config[section]:
            print(f"{key}: {config[section][key]}")
    print("\n----------------------------------------")


if __name__ == "__main__":
    # Load config file
    config_file_path = 'config.ini'
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
