import tensorflow as tf
from keras_preprocessing.image import array_to_img
from keras import Model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# importing class for data preprocessing and function for creating image paths array
from preprocessing_data import PreprocessData, create_image_paths

# importing global variable from globals.py
from globals import VALIDATION_DIR, VALIDATION_GROUND_TRUTH


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

    validation_paths = create_image_paths(VALIDATION_DIR)

    validation_gen = PreprocessData(validation_paths, VALIDATION_GROUND_TRUTH)

    val_preds = model.predict(validation_gen)

    # Display results for validation image #number
    i = image_number

    # Display  input image
    plt.imshow(validation_paths[i])
    plt.axis('False')
    plt.show()

    # Display mask predicted by our model
    display_mask(i, validation_paths[i], val_preds)
