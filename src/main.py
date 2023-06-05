import os
# importing function for build model
from model_build import get_model
# importing function for training model
from model_training import model_train
# importing function for take inference from trained model
from model_inference import inference

# importing global variable from globals.py
from cfg.globals import IMAGE_SIZE, WEIGHT_CHECKPOINT_PATH, MODEL_WEIGHT_SAVING_PATH


def main():
    """ The primary function for calling all other modules with Keras model functionality """

    # Building model
    asd_model = get_model(IMAGE_SIZE, 1)

    # Check if exist model in filesystem
    if os.path.exists(MODEL_WEIGHT_SAVING_PATH):
        # Load model weights from file 'asd_model'
        asd_model.load_weights(MODEL_WEIGHT_SAVING_PATH)
    elif os.path.exists('models\\asd_checkpoint_weight'):
        # Load model weights from file WEIGHT_CHECKPOINT_PATH
        asd_model.load_weights(WEIGHT_CHECKPOINT_PATH)
    else:
        # Training model
        asd_model = model_train(asd_model)
        # Saving model
        asd_model.save_weights(MODEL_WEIGHT_SAVING_PATH)

    # Making prediction on batch #34 and image #5
    inference(asd_model, 68, 6)


if __name__ == "__main__":
    main()
