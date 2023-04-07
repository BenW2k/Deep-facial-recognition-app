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


# Webcam setup
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #Resizing frame to 250x250 pixels
    frame = frame[120:120+250, 200:200+250, :]

    # Anchor collection
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Creating file path
        imgname = os.path.join(ANCHOR_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Writing out anchor image
        cv2.imwrite(imgname, frame)

    # Positives collection
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Creating file path
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Writing out positive image
        cv2.imwrite(imgname, frame)

    # showing image
    cv2.imshow('Webcam', frame)
    # Breaking while loop
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
# Closing the webcam frame
cap.release()
cv2.destroyAllWindows()

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
