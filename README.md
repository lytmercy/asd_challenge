# asd_challenge
### Airbus Ship Detection Challenge

---

## About Project
That my implementation for Airbus Ship Detection Challenge.
My model based on TensorFlow/Keras frameworks. 
Implementation containing model architecture such as a simple U-Net for image semantic segmentation.
In project also used the libraries: NumPy, Pandas and Matplotlib.

**_Project have that files_**:
- **dataset_analysis.ipynb** -- containing exploring of data from Challenge.
- **globals.py** -- for storing global variables for project.
- **main.py** -- it's core code file with calling all function and classes for preprocessing data, 
  building, training and using model.
- **model_build.py** -- file with function for building model.
- **model_training.py** -- file with function for training model and class for creating new metric for scoring model.
- **model_inference.py** -- file with function for testing model.
- **preprocessing_data.py** -- file with class for building batch with images and masks.
- **requirements.txt** -- file which store name all libraries that require for this project.

---

## Model Architecture

Model architecture has:
- Input layer with shape=(160, 160)
- data_augmentation layer (for augmented data only in training process)

And standard U-Net architecture:
[Model build code](model_build.py)

---

## Downloading and testing model

First get the repository
```commandline
git clone https://github.com/lytmercy/asd_challenge.git
```
Next run the main.py in console
```commandline
python main.py
```

If you want check training process you can move out files of weight from path `asd_challenge/model/*`
```commandline
mv 'asd_challenge/model/*' 'your/path/'
```

---

## Demonstrating the result

