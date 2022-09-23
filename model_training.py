import tensorflow as tf
# Importing Model class for type the model parameter in the model_train function
from keras import Model
# Importing Adam optimizer for compiling the model
from keras.optimizers import Adam
# Importing ModelCheckpoint callback for checkpointing model weight during training
from keras.callbacks import ModelCheckpoint
# Importing Metric class for creating new class DiceScore metric for use it in compiling the model
from keras.metrics import Metric, MeanIoU
# Importing Loss class for creating new class DiceLoss loss for use it in compiling the model
from keras.losses import Loss

# Importing other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing class for preprocessing data, and function for create image paths array
from preprocessing_data import DataGenerator, train_valid_ids_split
# Importing global variable from globals.py
from globals import EPOCHS, TRAIN_DIR, TRAIN_GROUND_TRUTH, WEIGHT_CHECKPOINT_PATH
from globals import TEST_DIR, TEST_GROUND_TRUTH


def plot_history_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    :param history: history from process of training model.
    :return: plotted separate curves for training and validation.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    dice_score = history.history['dice_score']
    val_dice_score = history.history['val_dice_score']

    epochs = range(len(history.history['loss']))

    # Plotting loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Plotting dice_score
    plt.figure()
    plt.plot(epochs, dice_score, label='training_dice_score')
    plt.plot(epochs, val_dice_score, label='val_dice_score')
    plt.title('Dice Score')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


class DiceLoss(Loss):
    """Class for creating the loss function is named dice loss in compiling the model."""
    def __init__(self, name='dice_loss', smooth=1e-6, gama=2):
        """
        Initialize attribute for class;
        :param name: name of this loss function;
        :param smooth: number for smoothing gradient;
        :param gama: some constant for squaring y_true and y_pred
        """
        super(DiceLoss, self).__init__(name=name)
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        """Compute dice loss"""
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.abs(tf.multiply(y_pred, y_true))) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        dice_loss = 1 - tf.divide(nominator, denominator)

        return dice_loss


class DiceScore(Metric):
    """Class for use metric dice score (f1-score) in model compile metrics."""
    def __init__(self,
                 name='dice_score',
                 dtype='float32',
                 smooth=1e-6,
                 gama=2,
                 **kwargs):
        """
        Initialize attribute for class;
        :param name: refers to the name of the metric;
        :param dtype: it's data type for this metric;
        :param smooth: using in method result for avoiding division by zero.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        # Setting intersection and union for computing the dice score.
        self.intersection = self.add_weight(
            name='intersection', dtype=dtype, initializer='zeros'
        )
        self.union = self.add_weight(
            name='union', dtype=dtype, initializer='zeros'
        )
        self.smooth = smooth
        self.gama = gama

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Override update_state method from Metric"""
        intersection = tf.reduce_sum(tf.abs(tf.multiply(y_true, y_pred)))
        union = tf.reduce_sum(y_true ** self.gama) + tf.reduce_sum(y_pred ** self.gama)

        self.intersection.assign_add(tf.reduce_sum(intersection))
        self.union.assign_add(tf.reduce_sum(union))

    def result(self):
        """Compute the dice score"""
        dice_score = tf.divide((2. * self.intersection + self.smooth), (self.union + self.smooth))
        return dice_score

    def reset_state(self):
        """Override reset_state method from Metric"""
        self.intersection.assign(0)
        self.union.assign(0)


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
