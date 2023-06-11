import tensorflow as tf
# Importing Model class for type the model parameter in the model_train function
from keras import Model
# Importing Adam optimizer for compiling the model
from keras.optimizers import Adam
# Importing ModelCheckpoint callback for checkpointing model weight during training
from keras.callbacks import ModelCheckpoint

# Importing other libraries
import numpy as np
import pandas as pd

# Class for preprocessing data
from src.data_handler import DataGenerator
# Function for split image ids to train and validation set
from src.utils import train_valid_ids_split
# To load config variables
from src.utils import load_config
# To simplify operations with config
from attrdict import AttrDict


def model_train(model: Model):
    """
    Function for getting the model and training that model on the data;
    :param model: built Keras model;
    :return: trained model.
    """

    # Setting train path & train ground truth path
    train_path = TRAIN_DIR
    train_ground_truth_path = TRAIN_GROUND_TRUTH

    # Setting ground truth dataframe
    train_ground_truth = pd.read_csv(train_ground_truth_path)
    # Defining filtered lists of ids
    train_list_ids = train_ground_truth.index[train_ground_truth['EncodedPixels'].isnull() == False].tolist()

    # Splitting dataset on train & validation list with ids
    train_list_ids, valid_list_ids = train_valid_ids_split(train_list_ids)
    # Set 30% of data in train_list_ids and valid_list_ids
    train_list_size = np.size(train_list_ids)
    train_list_ids = train_list_ids[:int(train_list_size * 0.30)]
    valid_list_size = np.size(valid_list_ids)
    valid_list_ids = valid_list_ids[:int(valid_list_size * 0.30)]

    # Initializing data generator
    train_data_gen = DataGenerator(train_list_ids, train_ground_truth, base_path=train_path)
    valid_data_gen = DataGenerator(valid_list_ids, train_ground_truth, base_path=train_path)

    # Initializing class DiceScore for metrics in compiling the model
    dice_score = DiceScore()
    dice_loss = DiceLoss()

    # Configure the model for training.
    model.compile(loss=dice_loss,
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[dice_score])

    # Create callback for saving best weights during training
    callbacks = [
        ModelCheckpoint(WEIGHT_CHECKPOINT_PATH,
                        monitor='val_dice_score',
                        save_weights_only=True,
                        save_best_only=True
                        )
    ]

    # Training the model, doing validation at the end of each epoch.
    epochs = EPOCHS
    model_history = model.fit(train_data_gen,
                              steps_per_epoch=len(train_data_gen),
                              epochs=epochs,
                              validation_data=valid_data_gen,
                              validation_steps=len(valid_data_gen),
                              callbacks=callbacks)

    # Plotting history of model training curves
    plot_history_curves(model_history)

    return model
