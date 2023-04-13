
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
np.set_printoptions(suppress=True)

# Load the model
model = load_model("video_train/01converted_keras/keras_model.h5", compile=False)

# Load the labels
class_names = open("video_train/01converted_keras/labels.txt", "r").readlines()