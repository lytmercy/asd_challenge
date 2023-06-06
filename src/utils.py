import tensorflow as tf
import numpy as np

import os
# To process config YAML File
import yaml

CONFIG_PATH = "../cfg/"


def load_config(config_name):
    """"""
    try:
        with open(os.path.join(CONFIG_PATH, config_name), 'r') as conf_file:
            config = yaml.safe_load(conf_file)
    except Exception as e:
        raise "Error reading the config file"

    return config


def tf_rle_decode(rle_string, shape=(768, 768)):
    """
    Function for decoding run-length encoding mask from string;
    :param rle_string: run-length string from csv file;
    :param shape: shape of output image;
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
    return tf.reshape(mask_flat, shape_tensor)


def train_valid_ids_split(list_ids, train_split=0.8):
    """
    Function for split list_ids to train and valid list with ids
    :param list_ids: list with train list_ids;
    :param train_split: float per cent value of training split;
    :return: two list of train and test ids.
    """
    # define size of list with ids
    list_size = np.size(list_ids)

    # define train size of list with ids
    train_size = int(train_split * list_size)

    # Set train & valid list with ids
    train_list_ids = list_ids[:train_size]
    valid_list_ids = list_ids[train_size:]

    return train_list_ids, valid_list_ids


def load_prepare_image(image_path, image_size):
    """ Loading and processing image from directory """
    image = tf.io.read_file(image_path)
    tensor_size = tf.constant(image_size)
    decoded_image = tf.image.decode_jpeg(image, channels=3)  # colour images
    # Convert uint8 tensor to floats in the [0, 1] range
    decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
    # Resize the image into image_size
    decoded_image = tf.image.resize(decoded_image, size=tensor_size)

    return decoded_image



