import tensorflow as tf
# Importing Model class for type the model parameter in the model_train function
from keras import Model
# Importing Adam optimizer for compiling the model
from keras.optimizers import Adam
# Importing ModelCheckpoint callback for checkpointing model weight during training
from keras.callbacks import ModelCheckpoint
# Importing Metric class for creating new class DiceScore metric for use it in compiling the model
from keras.metrics import Metric, MeanIoU
from keras import backend

# Importing other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing class for preprocessing data, and function for create image paths array
from preprocessing_data import DataGenerator, train_valid_ids_split
# Importing global variable from globals.py
from globals import IMAGE_SIZE, EPOCHS, TRAIN_DIR, TRAIN_GROUND_TRUTH, WEIGHT_CHECKPOINT_PATH


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


class DiceScore(Metric):
    """Class for use metric dice score (f1-score) in model compile metrics"""
    def __init__(self,
                 name='dice_score',
                 dtype='float32',
                 smooth=1,
                 **kwargs):
        """
        Initialize attribute for class;
        :param name: refers to the name of the metric;
        :param dtype: it's data type for this metric;
        :param smooth: using in method result for avoiding division by zero.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        # self.true_positives = self.add_weight(
        #     name='tp', dtype=dtype, initializer='zeros'
        # )
        # self.false_positives = self.add_weight(
        #     name='fp', dtype=dtype, initializer='zeros'
        # )
        # self.false_negatives = self.add_weight(
        #     name='fn', dtype=dtype, initializer='zeros'
        # )
        self.intersection = self.add_weight(
            name='intersection', dtype=dtype, initializer='zeros'
        )
        self.union = self.add_weight(
            name='union', dtype=dtype, initializer='zeros'
        )
        self.smooth = smooth

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Override update_state method from Metric"""
        # y_pred = tf.math.greater_equal(y_pred, self.threshold)
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)
        #
        # true_positives = tf.cast(y_true & y_pred, self.dtype)
        # false_positives = tf.cast(~y_true & y_pred, self.dtype)
        # false_negatives = tf.cast(y_true & ~y_pred, self.dtype)
        #
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, self.dtype)
        #     true_positives *= sample_weight
        #     false_positives *= sample_weight
        #     false_negatives *= sample_weight
        #
        # self.true_positives.assign_add(tf.reduce_sum(true_positives))
        # self.false_positives.assign_add(tf.reduce_sum(false_positives))
        # self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

        intersection = backend.sum(y_true * y_pred, axis=[1, 2, 3])
        union = backend.sum(y_true, axis=[1, 2, 3]) + backend.sum(y_pred, axis=[1, 2, 3])

        self.intersection.assign_add(tf.reduce_sum(intersection))
        self.union.assign_add(tf.reduce_sum(union))

    def result(self):
        """Compute the intersection-over-union via the confusion matrix."""
        dice = backend.mean((2. * self.intersection + self.smooth) / (self.union + self.smooth))
        return dice

    def reset_state(self):
        """Override reset_state method from Metric"""
        # self.true_positives.assign(0)
        # self.false_positives.assign(0)
        # self.false_negatives.assign(0)

        self.intersection.assign(0)
        self.union.assign(0)

        # backend.set_value(
        #     self.total_cm, tf.zeros((self.num_classes, self.num_classes))
        # )


def model_train(model: Model):
    """
    Function for getting the model and training that model on the data.

    :param model: built Keras model.
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
    # Set 10% of data in train_list_ids and valid_list_ids
    train_list_size = np.size(train_list_ids)
    train_list_ids = train_list_ids[:int(train_list_size * 0.1)]
    valid_list_size = np.size(valid_list_ids)
    valid_list_ids = valid_list_ids[:int(valid_list_size * 0.1)]

    # Initializing data generator
    train_data_gen = DataGenerator(train_list_ids, train_ground_truth, base_path=train_path)
    valid_data_gen = DataGenerator(valid_list_ids, train_ground_truth, base_path=train_path)

    # train_images, train_masks = train_data_gen[6]
    # train_image, train_mask = train_images[4], train_masks[4]
    # plt.imshow(train_image/255.)
    # plt.imshow(train_mask, alpha=0.5)
    # plt.axis('off')
    # plt.show()

    # Initializing class DiceScore for metrics in compiling the model
    dice_score = DiceScore()
    iou_metric = MeanIoU(num_classes=1)

    # Configure the model for training.
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=[dice_score])

    # Create callback for saving best weights during training
    callbacks = [
        ModelCheckpoint(WEIGHT_CHECKPOINT_PATH,
                        save_weights_only=True,
                        )
    ]

    # Training the model, doing validation at the end of each epoch.
    epochs = EPOCHS
    model_history = model.fit(train_data_gen,
                              epochs=epochs,
                              validation_data=valid_data_gen,
                              callbacks=callbacks)

    # Plotting history of model training curves
    plot_history_curves(model_history)

    return model
