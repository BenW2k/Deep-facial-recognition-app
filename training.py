# Import General Dependencies
import secret
import cv2
import os
import random
import numpy as np
import uuid
import model
from matplotlib import pyplot as plt

# Import TensorFlow Dependencies
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

siamese_model = model.Siamese_model_creation()

# Setting GPU Memory Consumption Growth to avoid 'Out of memory' errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Loss setup and optimisation
binary_cross_loss = tf.losses.binary_crossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 1e-4 denotes a learning rate of 0.0001

# Checkpoint creation
checkpoint_dir = os.path.join(secret.path_prefix, 'training_checkpoints') # Creating directory for storing checkpoint data
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Train step creation
@tf.function # Decorator to compile the training function more effectively
def train_step(batch):

    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Loss calculation
        loss = binary_cross_loss(y, yhat)
    
    # Gradient Calculation
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Updated weights calculation and application to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables)) # Calculating the new weights using the Adam optimisation model