import tensorflow as tf
# Import Keras function for converting array of numbers to image
from keras_preprocessing.image import array_to_img
from keras import Model

# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt

# importing class for data preprocessing and function for creating image paths array
from preprocessing_data import DataGenerator

# importing global variable from globals.py
from globals import TEST_DIR, TEST_GROUND_TRUTH, TRAIN_DIR, TRAIN_GROUND_TRUTH


def display_mask(true_image, image_number, test_preds):
    """Quick utility to display a model's prediction"""
    # Find index with the largest value in test_preds along -1 axis
    # mask = tf.argmax(test_preds, axis=-1)
    # Expand dims in mask tensor along -1 axis
    mask_image = test_preds[image_number]
    # Convert mask tensor to image with array_to_img function

    # Show true image with predicted masks (with alpha=0.4 value)
    plt.imshow(true_image)
    plt.imshow(mask_image, alpha=0.4)
    plt.axis('off')
    plt.show()


def inference(model: Model, batch_number):
    """ This function for take model and make prediction on image
    Display mask predicted by our model
    :param model: trained Keras model;
    :param batch_number: batch number that will be used for prediction.
    """

    train_ground_truth = pd.read_csv(TRAIN_GROUND_TRUTH)
    train_list_ids = train_ground_truth.index[train_ground_truth['EncodedPixels'].isnull() == False].tolist()
    train_data_gen = DataGenerator(train_list_ids, train_ground_truth, base_path=TRAIN_DIR)
    train_images, train_masks = train_data_gen[6]
    train_image, train_mask = train_images[4], train_masks[4]
    plt.imshow(train_image/255.)
    plt.imshow(train_mask, alpha=0.4)
    plt.axis('off')
    plt.show()

    # Set test & ground_truth paths
    test_path = TEST_DIR
    test_ground_truth_path = TEST_GROUND_TRUTH

    # Set ground truth dataframe
    test_ground_truth = pd.read_csv(test_ground_truth_path)
    # Set list of ids
    test_list_ids = test_ground_truth.index.tolist()

    # Initializing test dataset class
    test_data_gen = DataGenerator(test_list_ids, test_ground_truth, base_path=test_path, mode='predict')

    # Take prediction from model
    test_preds = model.predict(test_data_gen[batch_number])

    # Define image_number for image from batch with number in batch_number
    image_number = 16

    # Display  input image
    true_image_batch = test_data_gen[batch_number]
    true_image = true_image_batch[image_number]/255.
    plt.imshow(true_image)
    plt.axis('off')
    plt.show()

    # Display mask predicted by our model
    display_mask(true_image, image_number, test_preds)
