import pandas as pd
import tensorflow as tf
# import Sequence class for creating own DataGenerator class
from keras.utils import Sequence

import numpy as np

# To use help functions
from src.utils import load_prepare_image, tf_rle_decode


class DataGenerator(Sequence):
    """ Class generates data for Keras model """

    def __init__(self,
                 list_ids: list[str],
                 ground_truth_df: pd.DataFrame,
                 mode: str = 'fit',
                 base_path: str = '../input/dataset/train_v2',
                 batch_size: int = 32,
                 img_size: tuple[int, int] = (256, 256),
                 color_channels: int = 3,
                 random_state: int = 17,
                 shuffle: bool = True):
        """
        Initialize attributes for class;
        :param list_ids: list with image ids from base_path directory;
        :param ground_truth_df: contain data about masks indexed by image ids from list_ids;
        :param mode: string variable that indicate data generated mode;
        :param base_path: string with the path where the images are stored;
        :param batch_size: number of the size for batch;
        :param img_size: the shape of the image that will be processed by the model;
        :param color_channels: number of color channels of images;
        :param random_state: random state for reproducible shuffle process of image ids from list_ids;
        :param shuffle: indicates, shuffling image ids or not;
        """
        self.image_size = img_size
        self.batch_size = batch_size
        self.gt_df = ground_truth_df
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

    def __getitem__(self, index: int):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        # Find list of IDs
        list_ids_batch = np.array([self.list_ids[k] for k in batch_indexes])

        # Generate X variable that contains images
        X = self.__generate_x(list_ids_batch)

        # Check string attribute mode for 'fit' mode
        if self.mode == 'fit':
            # Generate y variable that contains masks for images
            y = self.__generate_y(list_ids_batch)
            # And return both X and y (images & masks)
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

    def __generate_x(self, list_ids_batch: np.ndarray) -> np.ndarray:
        """Generates data (images) in the current batch sample"""
        # Initialization X as an emtpy NumPy array
        X = np.empty((self.batch_size, *self.image_size, self.color_channels))

        # Generate data from ground_truth_dataframe with ids from list_ids_batch
        for i, image_id in enumerate(list_ids_batch):
            image_path = f"{self.base_path}\\{image_id}"
            image = load_prepare_image(image_path, self.image_size)

            # Store sample image in NumPy array
            X[i, ] = image

        return X

    def __generate_y(self, list_ids_batch: np.ndarray) -> np.ndarray:
        """Generates masks for the image in the current batch sample"""
        # Initialization y as an empty numpy array with determined shape
        y = np.empty((self.batch_size, *self.image_size, 1))
        # Generate masks from gt_df with ids from list_ids_batch
        for i, image_id in enumerate(list_ids_batch):
            # Extract image id for get masks from gt_df
            # image_id = self.gt_df["ImageId"].iloc[ID]
            # Get masks for concrete image
            gt_df = self.gt_df[self.gt_df["ImageId"] == image_id]
            # Get coded rle string from dataframe of masks for concrete image
            coded_rle_strings = gt_df["EncodedPixels"].values
            # Set tensor with zeros value for all masks
            # (768, 768) - it's shape of ground truth masks
            all_masks = tf.zeros((768, 768), dtype=tf.float32)
            # Iter through rle strings of all masks for concrete image
            if coded_rle_strings[0] is not np.nan:
                for mask in coded_rle_strings:
                    # Add mask to only one tensor for all masks
                    all_masks += tf.transpose(tf_rle_decode(mask))
                
            # Resize masks to image_size for model
            all_masks = tf.image.resize(tf.expand_dims(all_masks, -1), size=self.image_size)
            # Store sample mask in y (numpy.array)
            y[i, ] = all_masks
            
        return y
