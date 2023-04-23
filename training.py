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
from keras.metrics import Precision, Recall

EPOCHS = 20
siamese_model = model.siamese_model_creation()

# Setting GPU Memory Consumption Growth to avoid 'Out of memory' errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Loss setup and optimisation
binary_cross_loss = tf.losses.BinaryCrossentropy()
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

def train(data, EPOCHS):
    
    
    # Looping through epochs
    for epoch in range(1, EPOCHS+1):
        print(f"\n Epoch {epoch}/{EPOCHS}")
        progbar = tf.keras.utils.Progbar(len(data))

    # Creating metric object
        r = Recall()
        p = Precision()
    # Looping through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

train(model.train_data, EPOCHS)

    
