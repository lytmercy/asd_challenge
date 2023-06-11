# Importing Keras class Model for force typing in function inference
from keras import Model
# Importing Keras optimizer Adam
from keras.optimizers import Adam

# Importing other libraries
import matplotlib.pyplot as plt

# Class for preprocessing data
from src.data_handler import DataGenerator
# Dice Loss & Score for model
from src.model_builder import DiceLoss, DiceScore
# To load config variables
from src.utils import load_config
# To simplify operations with config
from attrdict import AttrDict


def inference(model: Model,
              test_data_gen: DataGenerator,
              predict_data_gen: DataGenerator,
              batch_number: int,
              image_number: int):
    """ This function for take model and make prediction on image
    Display mask predicted by our model
    :param model: trained Keras model;
    :param test_data_gen: class with methods to generate batch of data for testing;
    :param predict_data_gen: class with methods to generate batch of data for prediction;
    :param batch_number: batch number that will be used for prediction;
    :param image_number: image number that will be used for get image from true and predicted batch.
    """

    # Load config
    cfg = AttrDict(load_config("config.yaml"))

    # Initializing hyperparameters
    lr_rate = cfg.hyper.lr_rate

    # Configure the model for training.
    model.compile(loss=DiceLoss(),
                  optimizer=Adam(learning_rate=lr_rate),
                  metrics=[DiceScore()])

    # Evaluate the model
    model.evaluate(test_data_gen)

    # Define batch which will be used for prediction
    true_image_batch = predict_data_gen[batch_number]

    # Take prediction from model
    masks_preds = model.predict(true_image_batch)

    # Define true image and predicted mask
    true_image = true_image_batch[image_number]
    pred_mask = masks_preds[image_number]

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
    # Save results and images
    plt.savefig(cfg.output.graph_results)
    plt.imsave(cfg.output.original, true_image)
    plt.imsave(cfg.output.with_mask, pred_mask)

    for i in range(cfg.preprocess.batch_size):
        plt.figure(figsize=(14, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(true_image_batch[i])
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(masks_preds[i])
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(true_image_batch[i])
        plt.imshow(masks_preds[i], alpha=0.4)
        plt.axis('off')
        plt.tight_layout(h_pad=0.1, w_pad=0.1)
        plt.show()
