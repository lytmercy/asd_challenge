import tensorflow as tf
from keras_preprocessing.image import array_to_img
from keras import Model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# importing class for data preprocessing and function for creating image paths array
from preprocessing_data import PreprocessData

# importing global variable from globals.py
from globals import TEST_DIR, TEST_GROUND_TRUTH


def display_mask(i, val_image_path, val_preds):
    """Quick utility to display a model's prediction"""
    mask = tf.argmax(val_preds[i], axis=-1)
    mask = tf.expand_dims(mask, axis=-1)
    image = array_to_img(mask)
    val_image = mpimg.imread(val_image_path)
    plt.imshow(val_image)
    plt.imshow(image, alpha=0.4)
    plt.axis('False')
    plt.show()


def inference(model: Model, image_number):
    """ This function for take model and make prediction on image
    Display mask predicted by our model
    :param model: trained Keras model.
    :param image_number: image number that will be used for prediction.
    """

    # Set test & ground_truth paths
    test_path = TEST_DIR
    test_ground_truth_path = TEST_GROUND_TRUTH

    # Initializing test dataset class
    test_dataset_gen = PreprocessData(test_path, test_ground_truth_path)

    # Get test dataset
    test_data, _ = test_dataset_gen.get_dataset()

    val_preds = model.predict(test_data)

    # Display results for validation image #number
    i = image_number

    # Display  input image
    for image, ground_truth in test_data.take(1):
        plt.imshow(image)
        plt.axis('False')
        plt.show()


    # Display mask predicted by our model
    display_mask(i, test_data[i], val_preds)
