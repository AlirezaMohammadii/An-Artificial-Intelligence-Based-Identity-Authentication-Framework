import sys
import tensorflow.keras.backend as K
import math

from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input, GRU, Dropout, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.models import Model

sys.path.append("..")
from authentication_model.constants import *


def clipped_relu(inputs):
    # Custom activation function that clips the output between 0 and 20
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    # Defines a residual block with two convolutional layers
    conv_name_base = "res{}_{}_branch".format(stage, block)

    # First convolutional layer
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.00001),
        name=conv_name_base + "_2a",
    )(input_tensor)
    x = BatchNormalization(name=conv_name_base + "_2a_bn")(x)
    x = Activation("relu")(x)  # Changed from clipped_relu to relu

    # Second convolutional layer
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.00001),
        name=conv_name_base + "_2b",
    )(x)
    x = BatchNormalization(name=conv_name_base + "_2b_bn")(x)

    # Adding the input tensor to the output of the second conv layer
    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)  # Changed from clipped_relu to relu
    return x


def identity_block2(input_tensor, kernel_size, filters, stage, block):
    # Defines a residual block with three convolutional layers, including 1x1 convolutions
    conv_name_base = "res{}_{}_branch".format(stage, block)

    # First convolutional layer (1x1)
    x = Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.00001),
        name=conv_name_base + "_conv1_1",
    )(input_tensor)
    x = BatchNormalization(name=conv_name_base + "_conv1.1_bn")(x)
    x = Activation("relu")(x)  # Changed from clipped_relu to relu

    # Second convolutional layer (main 3x3)
    x = Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.00001),
        name=conv_name_base + "_conv3",
    )(x)
    x = BatchNormalization(name=conv_name_base + "_conv3_bn")(x)
    x = Activation("relu")(x)  # Changed from clipped_relu to relu

    # Third convolutional layer (1x1)
    x = Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        activation=None,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.00001),
        name=conv_name_base + "_conv1_2",
    )(x)
    x = BatchNormalization(name=conv_name_base + "_conv1.2_bn")(x)

    # Adding the input tensor to the output of the third conv layer
    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)  # Changed from clipped_relu to relu
    return x


def conv_and_res_block(inp, filters, stage, use_dropout=False):
    # Defines a block consisting of a convolutional layer followed by residual blocks
    conv_name = "conv{}-s".format(filters)

    # Initial convolutional layer
    o = Conv2D(
        filters,
        kernel_size=5,
        strides=2,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.00001),
        name=conv_name,
    )(inp)
    o = BatchNormalization(name=conv_name + "_bn")(o)
    o = Activation("relu")(o)  # Changed from clipped_relu to relu

    # Adding residual blocks
    for i in range(3):
        o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        if use_dropout:
            o = Dropout(0.3)(o)  # Added dropout for regularization
    return o


def convolutional_model(
    input_shape=(NUM_FRAMES, 64, 1),
    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH,
    num_frames=NUM_FRAMES,
    use_dropout=False,
):
    # Constructs the main convolutional model
    inputs = Input(shape=input_shape)
    x = conv_and_res_block(inputs, 64, stage=1, use_dropout=use_dropout)
    x = conv_and_res_block(x, 128, stage=2, use_dropout=use_dropout)
    x = conv_and_res_block(x, 256, stage=3, use_dropout=use_dropout)
    x = conv_and_res_block(x, 512, stage=4, use_dropout=use_dropout)

    # Reshape and average pooling
    x = Lambda(
        lambda y: K.reshape(y, (-1, math.ceil(num_frames / 16), 2048)), name="reshape"
    )(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name="average")(x)

    # Dense layer and L2 normalization
    x = Dense(512, kernel_regularizer=regularizers.l2(0.00001), name="affine")(x)
    x = Dropout(0.3)(x)  # Added dropout for regularization
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name="ln")(x)

    model = Model(inputs, x, name="convolutional")
    return model


def convolutional_model_simple(
    input_shape=(NUM_FRAMES, 64, 1),
    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH,
    num_frames=NUM_FRAMES,
    use_dropout=False,
):
    # Constructs a simplified version of the convolutional model with fewer stages
    def conv_and_res_block(inp, filters, stage):
        conv_name = "conv{}-s".format(filters)

        # Initial convolutional layer
        o = Conv2D(
            filters,
            kernel_size=5,
            strides=2,
            padding="same",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(0.00001),
            name=conv_name,
        )(inp)
        o = BatchNormalization(name=conv_name + "_bn")(o)
        o = Activation("relu")(o)  # Changed from clipped_relu to relu

        # Adding residual blocks
        for i in range(3):
            o = identity_block2(o, kernel_size=3, filters=filters, stage=stage, block=i)
            if use_dropout:
                o = Dropout(0.3)(o)  # Added dropout for regularization
        return o

    def cnn_component(inp):
        # Defining the simplified CNN component
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        return x_

    inputs = Input(shape=input_shape)
    x = cnn_component(inputs)

    # Reshape and average pooling
    x = Lambda(
        lambda y: K.reshape(y, (-1, math.ceil(num_frames / 8), 2048)), name="reshape"
    )(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name="average")(x)

    # Dense layer and L2 normalization
    x = Dense(512, kernel_regularizer=regularizers.l2(0.00001), name="affine")(x)
    x = Dropout(0.3)(x)  # Added dropout for regularization
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name="ln")(x)

    model = Model(inputs, x, name="convolutional_simple")
    return model


def recurrent_model(
    input_shape=(NUM_FRAMES, 64, 1),
    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH,
    num_frames=NUM_FRAMES,
):
    # Constructs a recurrent model using GRU layers
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(
        64,
        kernel_size=5,
        strides=2,
        padding="same",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=regularizers.l2(0.0001),
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)  # Changed from clipped_relu to relu

    # Reshape to fit the GRU input requirements
    x = Lambda(
        lambda y: K.reshape(y, (-1, math.ceil(num_frames / 2), 2048)), name="reshape"
    )(x)

    # Adding multiple GRU layers
    x = GRU(1024, return_sequences=True)(x)
    x = GRU(1024, return_sequences=True)(x)
    x = GRU(1024, return_sequences=True)(x)

    # Average pooling and dense layer
    x = Lambda(lambda y: K.mean(y, axis=1), name="average")(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = Dropout(0.3)(x)  # Added dropout for regularization
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name="ln")(x)

    model = Model(inputs, x, name="recurrent")
    return model


if __name__ == "__main__":
    # Create and summarize the convolutional model
    model = convolutional_model(use_dropout=True)
    print("Convolutional Model Summary:")
    print(model.summary())

    # Create and summarize the simplified convolutional model
    model_simple = convolutional_model_simple(use_dropout=True)
    print("Simple Convolutional Model Summary:")
    print(model_simple.summary())

    # Create and summarize the recurrent model
    recurrent = recurrent_model()
    print("Recurrent Model Summary:")
    print(recurrent.summary())
