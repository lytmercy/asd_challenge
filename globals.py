
"""This file contain all global variables for this project"""
# Initialize globals
global BATCH_SIZE
global BUFFER_SIZE
global IMAGE_SIZE
global EPOCHS
global TRAIN_LENGTH
global STEPS_PER_EPOCH

global TRAIN_DIR
global TRAIN_GROUND_TRUTH
global VALIDATION_DIR
global VALIDATION_GROUND_TRUTH

# Set globals
BATCH_SIZE = 10
BUFFER_SIZE = 1000
IMAGE_SIZE = (768, 768)
EPOCHS = 15

TRAIN_DIR = 'dataset\\train_v2'
TRAIN_GROUND_TRUTH = 'dataset\\train_ship_segmentations_v2.csv'
VALIDATION_DIR = 'dataset\\test_v2'
VALIDATION_GROUND_TRUTH = 'dataset\\sample_submission_v2.csv'

