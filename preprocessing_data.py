import tensorflow as tf
from keras.utils import Sequence
from keras_preprocessing.image import load_img

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as dataset_split

import os
# importing global variables
from globals import BATCH_SIZE, IMAGE_SIZE

# set global variable AUTOTUNE (from TensorFlow)
AUTOTUNE = tf.data.experimental.AUTOTUNE


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


class PreprocessData:

    def __init__(self, image_paths, image_ground_truth):
        self.batch_size = BATCH_SIZE
        self.image_size = IMAGE_SIZE

        self.input_image_paths = image_paths
        self.input_ground_truth = pd.read_csv(image_ground_truth)

        self.tensor_of_paths = None
        self.tensor_dataset = None

        self.create_image_paths()

    def create_image_paths(self):
        """
        Method for build tensor with correct image paths that have ground truth in input_ground_truth (*.csv file)
        :param image_dir: string with path of directory with images.
        :param ground_truth: string with path of *.csv file of ground truth for images.
        :return: sorted array with all image paths in image_dir.
        """
        # self.tensor_of_paths = tf.data.Dataset.from_tensors(tf.convert_to_tensor())
        for fname in os.listdir(self.input_image_paths):
            if fname.endswith('.jpg') & self.input_ground_truth[self.input_ground_truth['ImageId'] == fname &
                                                                self.input_ground_truth['EncodedPixels'] is not None]:
                print(fname)
                self.tensor_of_paths.append(os.path.join(self.input_image_paths, fname))
        print(self.tensor_of_paths)

    def decode_image(self, image):
        """"""
        image = tf.image.decode_jpeg(image, channels=3)  # colour images
        # convert uint8 tensor to floats in the [0, 1] range
        image = tf.image.convert_image_dtype(image, tf.float32)
        # resize the image into image_size
        return tf.image.resize(image, [self.image_size])

    def process_dataset(self, image_paths):
        """"""
        image_id = tf.strings.split(image_paths, '\\')[-1]
        ground_truth_for_image = self.input_ground_truth.loc[self.input_ground_truth['ImageId'] == image_id,
                                                             'ImageId'].tolist()
        all_masks = tf.zeros(self.image_size)

        for mask in ground_truth_for_image:
            all_masks = tf.add(all_masks, tf_rle_decode(mask))

        image = tf.io.read_file(image_paths)
        image = self.decode_image(image)
        return image, all_masks

    def get_dataset(self):
        """"""
        self.tensor_dataset = self.tensor_of_paths.map(self.process_dataset, num_parallel_calls=AUTOTUNE)

        train_set, valid_set = dataset_split(self.tensor_dataset, test_size=0.2, train_size=0.8)

        return train_set, valid_set
