
# importing function for build model
from model_build import get_model
# importing function for training model
from model_training import model_train
# importing function for take inference from trained model
from model_inference import inference

# importing global variable from globals.py
from globals import IMAGE_SIZE


def main():

    # Building model
    asd_model = get_model(IMAGE_SIZE, 1)

    # Training model
    asd_model = model_train(asd_model)

    # Making prediction on image #17
    inference(asd_model, 17)


if __name__ == "__main__":
    main()
