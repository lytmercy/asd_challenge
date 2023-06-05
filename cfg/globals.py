
"""This file contain all global variables for this project"""
# Initialize globals
global BATCH_SIZE
global IMAGE_SIZE
global EPOCHS

global TRAIN_DIR
global TRAIN_GROUND_TRUTH
global TEST_DIR
global TEST_GROUND_TRUTH

global WEIGHT_CHECKPOINT_PATH
global MODEL_WEIGHT_SAVING_PATH

# Set globals
BATCH_SIZE = 22
IMAGE_SIZE = (160, 160)
EPOCHS = 4

TRAIN_DIR = 'input\\dataset\\train_v2'
TRAIN_GROUND_TRUTH = 'input\\dataset\\train_ship_segmentations_v2.csv'
TEST_DIR = 'input\\dataset\\test_v2'
TEST_GROUND_TRUTH = 'input\\dataset\\sample_submission_v2.csv'

WEIGHT_CHECKPOINT_PATH = 'output\\asd_weights\\asd_checkpoint_weight\\asd_weight.ckpt'
MODEL_WEIGHT_SAVING_PATH = 'output\\asd_weights\\asd_model_weight\\asd_model'
