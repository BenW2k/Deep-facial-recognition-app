import tensorflow as tf
import numpy as np
import os


''' def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))

        data.append '''
person = 'rihanna'
input_folder = fr'C:\Users\Ben\Documents\Resume projects\Deep facial recognition app\data\positive\{person}'
output = fr'C:\Users\Ben\Documents\Resume projects\Deep facial recognition app\data\positive\{person}_aug'

def augment_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            
            img = tf.io.read_file(input_path)
            img = tf.image.decode_jpeg(img, channels=3)
            
            data = []
            for i in range(9):
                img_aug = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
                img_aug = tf.image.stateless_random_contrast(img_aug, lower=0.6, upper=1, seed=(1,3))
                img_aug = tf.image.stateless_random_flip_left_right(img_aug, seed=(np.random.randint(100), np.random.randint(100)))
                img_aug = tf.image.stateless_random_jpeg_quality(img_aug, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
                img_aug = tf.image.stateless_random_saturation(img_aug, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
                
                data.append(img_aug.numpy())
            
            for i, img_aug in enumerate(data):
                output_file = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
                output_path = os.path.join(output_folder, output_file)
                
                img_aug = tf.image.encode_jpeg(img_aug, quality=100)
                tf.io.write_file(output_path, img_aug)

augment_images(input_folder, output)

def rename_images(folder_path, number):
    counter = number
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            old_path = os.path.join(folder_path, filename)
            new_filename = f'{person}{counter}.jpg'
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            counter += 1

rename_images(input_folder, 0)
rename_images(output, 4)
