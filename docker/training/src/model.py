from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


def _get_last_layer_units_and_activation(
        num_classes: int
    ) -> Tuple[int, str]:
    """
    Gets the number of units and activation
    function for the last network layer.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def create_mlp_model(
        layers: int,
        units: int,
        dropout_rate: float,
        input_shape: Tuple[int],
        num_classes: int,
        learning_rate: float,
    ) -> tf.keras.models.Sequential:
    """
    Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.
    """

    op_units, op_activation = \
        _get_last_layer_units_and_activation(num_classes)
    
    model = tf.keras.models.Sequential()

    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=op_units, activation=op_activation))
    

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    return model