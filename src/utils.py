import keras.callbacks
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
# To process config YAML File
import yaml

CONFIG_PATH = "../cfg/"


def load_config(config_name: str) -> dict:
    """
    Load config from YAML type file.
    :param config_name: name of config yaml file;
    :return: dictionary of values from yaml config.
    """
    try:
        with open(os.path.join(CONFIG_PATH, config_name), 'r') as conf_file:
            config = yaml.safe_load(conf_file)
    except Exception as e:
        raise "Error reading the config file"

    return config


def tf_rle_decode(rle_string: str, shape=(768, 768)) -> tf.Tensor:
    """
    Function for decoding run-length encoding mask from string.
    :param rle_string: run-length string of encoded segmentation mask from csv file;
    :param shape: shape of output image;
    :type shape: Tuple[int, int]
    :return: tensor as image mask.
    """
    # Initialize tensor as shape of image
    shape_tensor = tf.convert_to_tensor(shape, tf.int64)
    # Initialize tensor of image size
    size = tf.math.reduce_prod(shape)

    # Split and convert string to tensor of number from string
    rle_tensor = tf.strings.split(rle_string)
    rle_tensor = tf.strings.to_number(rle_tensor, tf.int64)

    # Split start and lengths data from rle string
    starts = rle_tensor[::2] - 1
    lengths = rle_tensor[1::2]

    # Make ones to be scattered
    total_ones = tf.reduce_sum(lengths)
    ones = tf.ones([total_ones], tf.uint8)

    # Make scattering indices
    ones_range = tf.range(total_ones)
    lens_cumsum = tf.math.cumsum(lengths)
    rle_ssorted = tf.searchsorted(lens_cumsum, ones_range, 'right')
    idx = ones_range + tf.gather(starts - tf.pad(lens_cumsum[:-1], [(1, 0)]), rle_ssorted)

    # Scatter ones into flattened mask
    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])

    # Reshape into mask
    mask = tf.reshape(mask_flat, shape_tensor)

    # Cast tensor to the new data type and normalize values
    return tf.cast(mask, tf.float32) / 255.


def train_valid_ids_split(list_ids: list[str], train_split=0.8, shuffle=False, random_seed=17) -> tuple[list, list]:
    """
    Function for split list_ids to train and valid list with ids.
    :param list_ids: list with train list_ids;
    :param train_split: percent value of training split;
    :type train_split: float
    :param shuffle: Shuffle training list of ids of images, if False not shuffle it;
    :param random_seed: the seed for the reproducible splits;
    :type random_seed: int
    :return: two list of train and test ids.
    """
    np.random.seed(random_seed)
    # define size of list with ids
    list_size = np.size(list_ids)

    # define train size of list with ids
    train_size = int(train_split * list_size)

    # Set train & valid list with ids
    train_list_ids = list_ids[:train_size]
    valid_list_ids = list_ids[train_size:]

    # Shuffle training list if needed
    if shuffle:
        np.random.shuffle(train_list_ids)

    return train_list_ids, valid_list_ids


def load_prepare_image(image_path: str, image_size: tuple[int, int]) -> tf.Tensor:
    """
    Loading and processing image from directory.
    :param image_path: path to file where store image;
    :param image_size: size of the images that is expected after loading;
    :return: loaded and resized image.
    """
    # Read image file from path to tensor
    image = tf.io.read_file(image_path)
    # Define size variable for resizing
    tensor_size = tf.constant(image_size)
    # Decode image as jpeg through tf.image.decode_jpeg function
    decoded_image = tf.image.decode_jpeg(image, channels=3)  # colour image
    # Convert uint8 tensor to floats in the [0, 1] range
    decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    # Resize the image into image_size
    decoded_image = tf.image.resize(decoded_image, size=tensor_size)

    return decoded_image


def plot_history_curves(history: keras.callbacks.History) -> None:
    """
    Shows plotted separate curves of dice loss and score for training and validation process.
    :param history: history from process of training model.
    """
    # Get dice loss and validation loss from history
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get dice score and validation score from history
    dice_score = history.history['dice_score']
    val_dice_score = history.history['val_dice_score']

    # Get epochs from history
    epochs = range(len(history.history['loss']))

    # Plotting dice loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Plotting dice score
    plt.figure()
    plt.plot(epochs, dice_score, label='training_dice_score')
    plt.plot(epochs, val_dice_score, label='val_dice_score')
    plt.title('Dice Score')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


