# Import TensorFlow and Keras libraries
import tensorflow as tf
# Import Keras layers for building model
from keras.layers import Conv2D, BatchNormalization, Activation, SeparableConv2D
from keras.layers import Conv2DTranspose, UpSampling2D, MaxPooling2D
# Import Keras layers for data augmentation
from keras.layers import RandomFlip, RandomRotation, RandomHeight, RandomWidth, RandomZoom, Rescaling
# Import Keras class
from keras import Input, Sequential, Model
# Import layers function for adding layers
from keras import layers


def get_model(image_size, num_classes):
    """
    Function for construct the model.
    :param image_size:
    :param num_classes:
    :return:
    """
    # Create data augmentation layer
    data_augmentation = Sequential([
        RandomFlip('horizontal'),  # randomly flip images on horizontal edge
        RandomRotation(0.2),  # randomly rotate images by a specific amount
        RandomHeight(0.003),  # randomly adjust the height of an image by a specific amount
        RandomWidth(0.003),  # randomly adjust the width of an image by a specific amount
        RandomZoom(0.2),  # randomly zoom into an image
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
    outputs = Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

    # Define the model
    model = Model(inputs, outputs)

    return model
