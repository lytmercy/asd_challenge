import tensorflow as tf
# Importing Keras class Model for force typing in function inference
from keras import Model
# Importing Keras optimizer Adam
from keras.optimizers import Adam

# Importing other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing class for data preprocessing and function for creating image paths array
from preprocessing_data import DataGenerator

# Importing class for loss function and Dice score metric
from model_training import DiceLoss, DiceScore

# Importing global variable from globals.py
from cfg.globals import TEST_DIR, TEST_GROUND_TRUTH, TRAIN_DIR, TRAIN_GROUND_TRUTH


def inference(model: Model, batch_number, image_number):
    """ This function for take model and make prediction on image
    Display mask predicted by our model
    :param model: trained Keras model;
    :param batch_number: batch number that will be used for prediction;
    :param image_number: image number that will be used for get image from true and predicted batch.
    """

    # Preprocess train data for evaluating model
    train_ground_truth = pd.read_csv(TRAIN_GROUND_TRUTH)
    train_list_ids = train_ground_truth.index[train_ground_truth['EncodedPixels'].isnull() == False].tolist()
    train_list_size = np.size(train_list_ids)
    evaluation_list_ids = train_list_ids[:int(train_list_size * 0.20)]
    eval_data_gen = DataGenerator(evaluation_list_ids, train_ground_truth, base_path=TRAIN_DIR)

    # Initializing class DiceScore for metrics in compiling the model
    dice_score = DiceScore()
    dice_loss = DiceLoss()

    # Configure the model for training.
    model.compile(loss=dice_loss,
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[dice_score])

    # Evaluate the model
    model.evaluate(eval_data_gen)

    # Set test & ground_truth paths
    test_path = TEST_DIR
    test_ground_truth_path = TEST_GROUND_TRUTH

    # Set ground truth dataframe
    test_ground_truth = pd.read_csv(test_ground_truth_path)
    # Set list of ids
    test_list_ids = test_ground_truth.index.tolist()

    # Initializing test dataset class
    test_data_gen = DataGenerator(test_list_ids, test_ground_truth, base_path=test_path, mode='predict')

    # Define batch which will be used for prediction
    true_image_batch = test_data_gen[batch_number]

    # Take prediction from model
    test_preds = model.predict(true_image_batch)

    # Define true image and predicted mask
    true_image = true_image_batch[image_number]
    pred_mask = test_preds[image_number]

    # Show true image only
    plt.imshow(true_image)
    plt.axis('off')
    plt.show()

    # Display true image and mask predicted by our model (pred mask with alpha=0.4 value)
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(true_image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(true_image)
    plt.imshow(pred_mask, alpha=0.4)
    plt.axis('off')
    plt.show()

    for i in range(22):
        plt.figure(figsize=(13, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(true_image_batch[i])
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(test_preds[i])
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(true_image_batch[i])
        plt.imshow(test_preds[i], alpha=0.4)
        plt.axis('off')
        plt.show()
