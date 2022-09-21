import os
# import function for loading Keras model from directory
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend
# importing function for build model
from model_build import get_model
# importing function for training model
from model_training import model_train
# importing function for take inference from trained model
from model_inference import inference

# importing global variable from globals.py
from globals import IMAGE_SIZE, WEIGHT_CHECKPOINT_PATH, MODEL_WEIGHT_SAVING_PATH


def main():
    """ The primary function for calling all other modules with Keras model functionality """
    # Free up RAM in case the model definition cells were run multiple times.
    # backend.clear_session()

    # Building model
    asd_model = get_model(IMAGE_SIZE, 1 )

    # Check if exist model in filesystem
    if os.path.exists('model\\asd_checkpoint_weight\\checkpoint'):
        # Load model from file 'asd_model.h5'
        asd_model.load_weights(WEIGHT_CHECKPOINT_PATH)
    elif os.path.exists(MODEL_WEIGHT_SAVING_PATH):
        # Load model from file 'asd.checkpoint.h5'
        asd_model.load_weights(MODEL_WEIGHT_SAVING_PATH)
    else:
        # Training model
        asd_model = model_train(asd_model)
        # Saving model
        asd_model.save_weights(MODEL_WEIGHT_SAVING_PATH)

    # Making prediction on batch #17
    inference(asd_model, 50)


if __name__ == "__main__":
    main()
