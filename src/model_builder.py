# Import TensorFlow and Keras libraries
import keras
import tensorflow as tf
# Import Keras layers to building model
from keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D
from keras.layers import Conv2DTranspose, UpSampling2D, MaxPooling2D
# Import Keras layers for data augmentation
from keras.layers import RandomFlip, RandomRotation, RandomHeight, RandomWidth, RandomZoom, Rescaling
# Import Keras class
from keras import Input, Sequential, Model
# Import layers function for adding layers
from keras import layers
# Metric class to creating new custom class DiceScore metric
from keras.metrics import Metric
# Loss class to creating new custom class DiceLoss loss
from keras.losses import Loss
# Keras backend for some operations in DiceScore metric
from keras.backend import backend as K_back

import numpy as np
from typing import Tuple


class DiceLoss(Loss):
    """Class for creating the loss function is named dice loss in compiling the model."""
    def __init__(self,
                 name: str = 'dice_loss',
                 gama: int = 2):
        """
        Initialize attribute for class;
        :param name: name of this loss function;
        :param gama: some constant for squaring y_true and y_pred
        """
        super(DiceLoss, self).__init__(name=name)
        self.gama = gama

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Compute dice loss"""
        y_true, y_pred = np.cast(y_true, dtype=tf.float32), np.cast(y_pred, dtype=tf.float32)

        nominator = 2 * np.reduce_sum(np.abs(np.multiply(y_pred, y_true)))
        denominator = np.reduce_sum(y_pred ** self.gama) + np.reduce_sum(y_true ** self.gama)

        return 1 - np.divide(nominator, denominator)


class DiceScore(Metric):
    """Class for use metric dice score (f1-score) in model compile metrics."""
    def __init__(self,
                 name: str = 'dice_score',
                 dtype: str = 'float32',
                 num_classes: int = 2,
                 **kwargs):
        """
        Initialize attribute for class;
        :param name: refers to the name of the metric;
        :param dtype: it's data type for this metric;
        """
        super().__init__(name=name,
                         dtype=dtype,
                         **kwargs)
        # Setting begin confusion matrix
        self.total_cm = self.add_weight(
                "total_confusion_matrix",
                shape=(num_classes, num_classes),
                initializer="zeros"
        )
        self.num_classes = num_classes

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None):
        """Accumulate the prediction to current confusion matrix."""
        current_cm = tf.math.confusion_matrix(
                y_true,
                y_pred,
                self.num_classes,
                weights=sample_weight,
                dtype=self._dtype
        )

        self.total_cm.asssifn_add(current_cm)

    def result(self):
        """Compute the Dice score via the confusion matrix."""
        sum_over_row = tf.cast(
                tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = tf.cast(
                tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )

        true_positives = tf.cast(
                tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #       2 * true_positives + false_positives + false_negatives
        denominator = sum_over_row + sum_over_col

        return tf.math.divide_no_nan(2 * true_positives, denominator)

    def reset_state(self):
        """Override reset_state method from Metric"""
        K_back.set_value(
                self.total_cm, np.zeros((self.num_classes, self.num_classes))
        )


def get_model(image_size: Tuple[int, int], num_classes: int) -> keras.Model:
    """
    Function for construct the model;
    :param image_size: size of the images which will be used model when getting data;
    :param num_classes: that number of classes that model will be predicted on the image;
    :return: built Keras model.
    """
    # Create data augmentation layer
    data_augmentation = Sequential([
        RandomFlip('horizontal'),  # randomly flip images on horizontal edge
        RandomRotation(0.1),  # randomly rotate images by a specific amount
        RandomHeight(0.003),  # randomly adjust the height of an image by a specific amount
        RandomWidth(0.003),  # randomly adjust the width of an image by a specific amount
        RandomZoom(0.1),  # randomly zoom into an image
    ])

    # Initialize Input layer
    inputs = Input(shape=image_size + (3,))
    # Initialize Data augmentation layer
    x = data_augmentation(inputs)  # augment images (only happens during training)

    # ## [First half of the network: down-sampling inputs] ## #

    # Entry block
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        # Project residual
        residual = Conv2D(filters, 1, strides=2, padding='same')(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # ## [Second half of the network: up-sampling inputs] ## #

    for filters in [256, 128, 64, 32]:
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Project residual
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding='same')(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = Conv2D(num_classes, 3, activation='sigmoid', padding='same')(x)

    # Define the model
    return Model(inputs, outputs)
