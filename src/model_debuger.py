import os
# To build model
from src.model_builder import get_model
# To train model
from src.training_process import model_train
# To take inference from trained model
from src.inference_process import inference
# To load config variables
from src.utils import load_config
# To simplify operations with config
from attrdict import AttrDict


def debugger():
    """The primary module to debug any process for the model."""

    # Load config
    cfg = AttrDict(load_config("config.yaml"))

    # Get the model
    asd_model = get_model(cfg.preprocess.image_size, 1)


if __name__ == "__main__":
    debugger()
