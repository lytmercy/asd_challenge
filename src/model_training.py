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
from cfg.globals import EPOCHS, TRAIN_DIR, TRAIN_GROUND_TRUTH, WEIGHT_CHECKPOINT_PATH
from cfg.globals import TEST_DIR, TEST_GROUND_TRUTH


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
    def __init__(self, name='dice_loss', gama=2):
        """
        Initialize attribute for class;
        :param name: name of this loss function;
        :param gama: some constant for squaring y_true and y_pred
        """
        super(DiceLoss, self).__init__(name=name)
        self.gama = gama

    def call(self, y_true, y_pred):
        """Compute dice loss"""
        y_true, y_pred = np.cast(y_true, dtype=tf.float32), np.cast(y_pred, dtype=tf.float32)

        nominator = 2 * np.reduce_sum(np.abs(np.multiply(y_pred, y_true)))
        denominator = np.reduce_sum(y_pred ** self.gama) + np.reduce_sum(y_true ** self.gama)

        return 1 - np.divide(nominator, denominator)


class DiceScore(Metric):
    """Class for use metric dice score (f1-score) in model compile metrics."""
    def __init__(self,
                 name='dice_score',
                 dtype='float32',
                 num_classes=2,
                 **kwargs):
        """
        Initialize attribute for class;
        :param name: refers to the name of the metric;
        :param dtype: it's data type for this metric;
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        # Setting begin confusion matrix
        self.total_cm = self.add_weight(
                "total_confusion_matrix",
                shape=(num_classes, num_classes),
                initializer="zeros"
        )
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate the prediction to current confusion matrix."""
        current_cm = tf.math.confusion_matrix(
                y_true,
                y_pred,
                self.num_classes,
                weights=sample_weight,
                dtype=self._dtype
        )

        self.total_cm.asssifn_add(current_cm)

    def result(self):
        """Compute the Dice score via the confusion matrix."""
        sum_over_row = tf.cast(
                tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = tf.cast(
                tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )

        true_positives = tf.cast(
                tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #       2 * true_positives + false_positives + false_negatives
        denominator = sum_over_row + sum_over_col

        # If the denominator is o, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
                tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )

        return tf.math.divide_no_nan(2 * true_positives, denominator)

    def reset_state(self):
        """Override reset_state method from Metric"""
        K_back.set_value(
                self.total_cm, np.zeros((self.num_classes, self.num_classes))
        )


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
