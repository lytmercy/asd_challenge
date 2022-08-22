import tensorflow as tf
from keras.utils import Sequence
from keras_preprocessing.image import load_img

import numpy as np
import pandas as pd

import os
# importing global variables
from globals import BATCH_SIZE, IMAGE_SIZE


# create function for decoding run-length mask from "train_ship_segmentations_v2.csv"
def tf_rle_decode(rle_string, shape=(768, 768)):
    """
    Function for decoding run-length encoding mask from string.

    :param rle_string: run-length string from csv file
    :param shape: shape of output image
    :return: tensor as image mask
    """
    shape_tensor = tf.convert_to_tensor(shape, tf.int64)
    size = tf.math.reduce_prod(shape)

    rle_tensor = tf.strings.split(rle_string)
    rle_tensor = tf.strings.to_number(rle_tensor, tf.int64)

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


def create_image_paths(image_dir, ground_truth):
    """
    Function for build array with correct & clean image paths
    :param image_dir: string with path of directory with images.
    :param ground_truth: string with path of *.csv file of ground truth for images.
    :return: sorted array with all image paths in image_dir.
    """
    ground_truth_dataframe = pd.read_csv(ground_truth)
    return sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith('.jpg') & ground_truth_dataframe['ImageId'].searchsorted(fname) == fname[0]
        ]
    )


class PreprocessData(Sequence):

    def __init__(self, image_paths, image_ground_truth):
        self.batch_size = BATCH_SIZE
        self.image_size = IMAGE_SIZE

        self.input_image_paths = image_paths
        self.input_ground_truth = image_ground_truth

    def __len__(self):
        return len(self.input_ground_truth) // self.batch_size

    def __getitem__(self, idx):
        """ Method for preprocessing train image and ground truth
        :returns tuple (input, masks) correspond to batch #idx.
        """
        i = idx * self.batch_size
        batch_input_image_paths = self.input_image_paths[i:i + self.batch_size]
        ground_truth_dataframe = pd.read_csv(self.input_ground_truth)
        # Create x variable for train image
        x = np.zeros((self.batch_size,) + self.image_size + (3,), dtype='float32')
        # Create y variable for train ground truth masks
        y = np.zeros((self.batch_size,) + self.image_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_input_image_paths):
            image = load_img(path, target_size=self.image_size)
            x[j] = image

            image_id = path.split('\\')[-1]
            # Make a list with the masks that image_id match
            image_masks = ground_truth_dataframe.loc[ground_truth_dataframe['ImageId'] == image_id,
                                                     'EncodedPixels'].tolist()

            if pd.isnull(image_masks[0]):
                print(image_masks[0])
                continue

            # Take the individual ship masks and create a single mask array for all ships
            all_masks = tf.zeros((768, 768), tf.uint8)
            for mask in image_masks:
                all_masks += tf.transpose(tf_rle_decode(mask, self.image_size))
            y[j] = tf.expand_dims(all_masks, 2)

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        return x, y

