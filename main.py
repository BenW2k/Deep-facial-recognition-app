# Import General Dependencies
import secret
import cv2
import os
import random
import numpy as np
import uuid
from matplotlib import pyplot as plt

# Import TensorFlow Dependencies
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Setting GPU Memory Consumption Growth to avoid 'Out of memory' errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Path setup
POS_PATH = os.path.join(secret.path_prefix, 'data', 'positive')
NEG_PATH = os.path.join(secret.path_prefix, 'data', 'negative')
ANCHOR_PATH = os.path.join(secret.path_prefix, 'data', 'anchor')

# Variables grab all images from each directory
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(300)
anchor = tf.data.Dataset.list_files(ANCHOR_PATH + '\*.jpg').take(300)

# Preprocessing

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) # Loads image from file path
    img = tf.io.decode_jpeg(byte_img) # Decodes jpeg
    img = tf.image.resize(img, (100,100)) # Resizing image to 100x100x3
    img = img / 255.0 # Scaling image to be between 0 and 1
    return img

# Creating labelled dataset

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
data = positives.concatenate(negatives)

# Creating the preprocess twin for test and train split
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Creating the dataloading pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training split
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Test split
test_data = data.skip(round(len(data)*.7)) # To skip the 70% of the data that is used for training.
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def create_embedding():
    inp = Input(shape=(100,100,3), name = 'input_image')
    c1 = Conv2D(64, (10,10), activation='relu')(inp) # Convolutional layer, 64

# return Model(inputs=, outputs=, name=,)