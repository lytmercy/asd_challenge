# Importing Model class for type the model parameter in the model_train function
from keras import Model
# Importing Adam optimizer for compiling the model
from keras.optimizers import Adam
# Importing ModelCheckpoint callback for checkpointing model weight during training
from keras.callbacks import ModelCheckpoint

# Class for preprocessing data
from src.data_handler import DataGenerator
# Dice Loss & Score for model
from src.model_builder import DiceLoss, DiceScore
# To plotting history curves after training
from src.utils import plot_history_curves
# To load config variables
from src.utils import load_config
# To simplify operations with config
from attrdict import AttrDict


def model_train(model: Model,
                train_data_gen: DataGenerator,
                valid_data_gen: DataGenerator):
    """
    Function for getting the model and training that model on the data;
    :param model: built Keras model;
    :param train_data_gen: class with methods to generate batch of data for training;
    :param valid_data_gen: class with methods to generate batch of data for validation;
    :return: trained model.
    """

    # Load config
    cfg = AttrDict(load_config("config.yaml"))

    # Initializing hyperparameters
    lr_rate = cfg.hyper.lr_rate
    epochs = cfg.hyper.epochs

    # Configure the model for training.
    model.compile(loss=DiceLoss(),
                  optimizer=Adam(learning_rate=lr_rate),
                  metrics=[DiceScore()])

    # Create callback for saving best weights during training
    callbacks = [
        ModelCheckpoint(cfg.model.ckpt_paths.weights,
                        monitor='val_dice_score',
                        save_weights_only=True,
                        save_best_only=True
                        )
    ]

    # Training the model, doing validation at the end of each epoch.
    model_history = model.fit(train_data_gen,
                              steps_per_epoch=len(train_data_gen),
                              epochs=epochs,
                              validation_data=valid_data_gen,
                              validation_steps=len(valid_data_gen),
                              callbacks=callbacks)

    # Plotting history of model training curves
    plot_history_curves(model_history)

    return model
