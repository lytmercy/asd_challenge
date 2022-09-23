import tensorflow as tf
# import Sequence class for creating own DataGenerator class
from keras.utils import Sequence

# import other libraries
import numpy as np

# importing global variables
from globals import BATCH_SIZE, IMAGE_SIZE


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


class DataGenerator(Sequence):
    """ Class generates data for Keras model """

    def __init__(self, list_ids, ground_truth_dataframe, mode='fit',
                 base_path='dataset\\train_v2', batch_size=BATCH_SIZE, img_size=IMAGE_SIZE,
                 color_channels=3, random_state=17, shuffle=True):
        """
        Initialize attributes for class;
        :param list_ids: list with ids that data will be generated for;
        :param ground_truth_dataframe: dataframe with data that will be generated for ids from list_ids;
        :param mode: string variable that indicate data generated mode;
        :param base_path: string with the path where the images are stored;
        :param batch_size: number of the batch size;
        :param img_size: the shape of the image that will be processed in the model;
        :param color_channels: number of color channels;
        :param random_state: random state for shuffle image ids from list_ids;
        :param shuffle: bool that indicates, shuffling image ids or not;
        """
        self.image_size = img_size
        self.batch_size = batch_size
        self.ground_truth_dataframe = ground_truth_dataframe
        self.mode = mode
        self.base_path = base_path
        self.list_ids = list_ids
        self.color_channels = color_channels
        self.shuffle = shuffle
        self.random_state = random_state
        # init indexes for store all indexes
        self.indexes = None

        # run on_epoch_end method for shuffle and init 'indexes' attribute
        self.on_epoch_end()
        # first set random seed
        np.random.seed(self.random_state)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Find list of IDs
        list_ids_batch = [self.list_ids[k] for k in indexes]

        # Generate X variable that contains images
        X = self.__generate_x(list_ids_batch)

        # Check string attribute mode for 'fit' mode
        if self.mode == 'fit':
            # Generate y variable that contains masks for images
            y = self.__generate_y(list_ids_batch)
            # And return both X and y variables
            return X, y
        # Check string attribute mode for 'predict' mode
        elif self.mode == 'predict':
            # Return only one variable X (images)
            return X
        else:
            # When mode is not 'fit' or 'predict' then rise Attribute Error
            raise AttributeError("The mode parameter should be set to 'fit' or 'predict'.")

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_x(self, list_ids_batch):
        """Generates data (images) in the current batch sample"""
        # Initialization X as an emtpy NumPy array
        X = np.empty((self.batch_size, *self.image_size, self.color_channels))

        # Generate data from ground_truth_dataframe with ids from list_ids_batch
        for i, ID in enumerate(list_ids_batch):
            image_id = self.ground_truth_dataframe['ImageId'].iloc[ID]
            image_path = f"{self.base_path}\\{image_id}"
            image = load_prepare_image(image_path, self.image_size)

            # Store sample image in NumPy array
            X[i, ] = image

        return X

    def __generate_y(self, list_ids_batch):
        """Generates masks for the image in the current batch sample"""
        # Initialization y as an empty NumPy array
        y = np.empty((self.batch_size, *self.image_size, 1))

        # Generate masks from ground_truth_dataframe with ids from list_ids_batch
        for i, ID in enumerate(list_ids_batch):
            image_id = self.ground_truth_dataframe['ImageId'].iloc[ID]
            ground_truth_dataframe = self.ground_truth_dataframe[self.ground_truth_dataframe['ImageId'] == image_id]

            coded_rle_strings = ground_truth_dataframe['EncodedPixels'].values

            all_masks = tf.zeros((768, 768), dtype=tf.uint8)
            for mask in coded_rle_strings:
                all_masks += tf.transpose(tf_rle_decode(mask))

            # Resize masks to image_size for model
            all_masks = tf.image.resize(tf.expand_dims(all_masks, -1), size=self.image_size)

            # Store sample mask in NumPy array
            y[i, ] = all_masks

        return y
