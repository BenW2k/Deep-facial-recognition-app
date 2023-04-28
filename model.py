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
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Creating the preprocess twin for test and train split
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Creating the dataloading pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

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
    inp = Input(shape=(100,100,3), name ='input_image')

    # Block One
    c1 = Conv2D(64, (10,10), activation='relu')(inp) # Convolutional layer, 64
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1) # MaxPooling layer

    # Block Two
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    # Block Three
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    # Block Four
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4) # Flattens the 3 dimensional ouput of the convolutions into a single dimension
    d1 = Dense(4096, activation='sigmoid')(f1) # Condenses output of f1 



    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = create_embedding()

class L1Dist(Layer): # New class for L1 distance layer
    def __init__(self, **kwargs ): # passing in **kwargs to allow us to pass in a variable number of arguments into the function
        super().__init__() # For inheritance from parent class

    #Similarity Calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def siamese_model_creation():

    # Input handling
    
    #Input image in the model
    input_image = Input(name='input_img', shape=(100,100,3))

    # Validation image in the model
    validation_image = Input(name='validation_img', shape=(100,100,3))

    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)


    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(inp_embedding, val_embedding)

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = siamese_model_creation()