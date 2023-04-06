import os
import secret

lfw_path = os.path.join(secret.path_prefix, 'lfw')

# Script to extract the images from the Labeled Faces in the Wild dataset and move them to the data/negative repository
for directory in os.listdir(lfw_path):
    for file in os.listdir(os.path.join(lfw_path, directory)):
        CURRENT_PATH = os.path.join(lfw_path, directory, file)
        NEW_PATH = os.path.join(os.path.join(secret.path_prefix, 'data', 'negative'), file)
        os.replace(CURRENT_PATH, NEW_PATH)

print('done')