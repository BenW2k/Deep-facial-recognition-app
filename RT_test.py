# Import General Dependencies
import secret
import cv2
import os
import random
import numpy as np
import uuid
import model as md
from matplotlib import pyplot as plt

# Import TensorFlow Dependencies
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

model_import = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': md.L1Dist, 'Binarycrossentropy': tf.losses.BinaryCrossentropy})

def verify(model, detection_threshold, verification_threshold):
    # Building results array
    results = []
    for image in os.listdir(os.path.join(secret.path_prefix, 'application', 'verification_image')):
        input_img = md.preprocess(os.path.join(secret.path_prefix, 'application', 'input_image', 'input_image.jpg'))
        validation_img = md.preprocess(os.path.join(secret.path_prefix, 'application', 'verification_image', image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
        results.append(result)
    
    # Detection Threshold: Metric above which the prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions with regard to the total number of predictions
    verification = detection / len(os.listdir(os.path.join(secret.path_prefix, 'application', 'verification_image')))
    verified = verification > verification_threshold

    return results, verified
    
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #Resizing frame to 250x250 pixels
    frame = frame[120:120+250, 200:200+250, :]
    
    cv2.imshow('Verification', frame)

    # Start verification process
    if cv2.waitKey(10) & 0XFF == ord('v'):
        # Save an input image to input image folder
        cv2.imwrite(os.path.join(secret.path_prefix, 'application', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(model_import, 0.5, 0.7)
        print(np.sum(np.squeeze(results) > 0.9) / 50)
        print(verified)

    if cv2.waitKey(10) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    