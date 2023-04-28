# Import dependencies

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer

# Custom L1Distance layer - need to import because h5 file needs custom objects to be passed through when loaded.

class L1Dist(Layer): # New class for L1 distance layer
    def __init__(self, **kwargs ): # passing in **kwargs to allow us to pass in a variable number of arguments into the function
        super().__init__() # For inheritance from parent class

    #Similarity Calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)