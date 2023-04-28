# Import Dependencies

import cv2
import numpy as np
import tensorflow as tf
from Layers import L1Dist
import os
import secret

# Import Kivy Dependencies

# For application layout
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
# For UX components
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
# Miscellaneous
from kivy.logger import Logger
from kivy.uix.image import Image
from kivy.uix.button import Button

# App layout

class CameraApp(App):

    def build(self):
        # Main layout
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text='Verify', on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1, 0.1))

        # Add main components to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load Tensorflow model

        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist})

        # Video capture device setup
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        
        return layout
    
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontal and convert image to texture (necessary to see image in app)
        # Essentially converting a raw OpenCV image array into a texture for rendering
        buffer = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buffer, colorfmt = 'bgr', bufferfmt= 'ubyte')
        self.web_cam.texture = img_texture
    
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path) # Loads image from file path
        img = tf.io.decode_jpeg(byte_img) # Decodes jpeg
        img = tf.image.resize(img, (100,100)) # Resizing image to 100x100x3
        img = img / 255.0 # Scaling image to be between 0 and 1
        return img
    
    # Verification function

    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.6

        # Image Capture from webcam
        SAVE_PATH = os.path.join(secret.path_prefix, 'kivy_application', 'application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        # Building results array
        results = []
        for image in os.listdir(os.path.join(secret.path_prefix, 'kivy_application', 'application_data', 'verification_image')):
            input_img = self.preprocess(os.path.join(secret.path_prefix, 'kivy_application', 'application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join(secret.path_prefix, 'kivy_application', 'application_data', 'verification_image', image))

            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)
        
        # Detection Threshold: Metric above which the prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions with regard to the total number of predictions
        verification = detection / len(os.listdir(os.path.join(secret.path_prefix, 'kivy_application', 'application_data', 'verification_image')))
        verified = verification > verification_threshold

        # Set verification
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log details
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(np.sum(np.array(results) > 0.5))

        return results, verified

if __name__ == '__main__':
    CameraApp().run()