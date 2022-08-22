import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.metrics import Metric

import pandas as pd
import matplotlib.pyplot as plt

# importing class for preprocessing data, and function for create image paths array
from preprocessing_data import PreprocessData, create_image_paths
# importing global variable from globals.py
from globals import IMAGE_SIZE, EPOCHS, TRAIN_DIR, TRAIN_GROUND_TRUTH, VALIDATION_DIR, VALIDATION_GROUND_TRUTH


class DiceScore(Metric):
    def __init__(self, name='dice_score', dtype='float32', threshold=0.5, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = 0.5
        self.true_positives = self.add_weight(
            name='tp', dtype=dtype, initializer='zeros'
        )
        self.false_positives = self.add_weight(
            name='fp', dtype=dtype, initializer='zeros'
        )
        self.false_negatives = self.add_weight(
            name='fn', dtype=dtype, initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.greater_equal(y_pred, self.threshold)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.cast(y_true & y_pred, self.dtype)
        false_positives = tf.cast(~y_true & y_pred, self.dtype)
        false_negatives = tf.cast(y_true & ~y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            true_positives *= sample_weight
            false_positives *= sample_weight
            false_negatives *= sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return precision * recall * 2.0 / (precision + recall)

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


def model_train(model: Model):
    """This function for getting data and training model on that data

    :param model: built Keras model.
    :return: trained model.
    """
    # Initializing paths of data
    train_paths = create_image_paths(TRAIN_DIR, TRAIN_GROUND_TRUTH)
    validation_paths = create_image_paths(VALIDATION_DIR, VALIDATION_GROUND_TRUTH)

    train_gen = PreprocessData(train_paths, TRAIN_GROUND_TRUTH)
    validation_gen = PreprocessData(validation_paths, VALIDATION_GROUND_TRUTH)

    print(train_paths)

    # Initializing class DiceScore for metrics in compiling the model
    dice_score = DiceScore()

    # Configure the model for training.
    # We use the "sparse" version of binary_crossentropy
    # because our target data is integers.

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=[dice_score])

    # Create callback for finish training early
    callbacks = [
        ModelCheckpoint('asd_challenge.h5', save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    epochs = EPOCHS
    model_history = model.fit(train_gen, epochs=epochs, validation_data=validation_gen, callbacks=callbacks)

    # Plotting history of model training curves
    pd.DataFrame(model_history.history).plot()
    plt.title('Model training curves')

    return model
