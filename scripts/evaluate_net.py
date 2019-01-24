""" Calculating the accuracy of the model """
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import pandas as pd

sat = 'Google'
MODEL = 'google_mobnet.h5'
IMG_SIZE = 256

# load model and predict
model = tf.keras.models.load_model('results/{}'.format(MODEL))
# list of files to be used for training -----------------
data_list = pd.read_csv('data/{}/data_index.csv'.format(sat))  # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

# drop ones without pictures
existing = []
for file in os.listdir('data/{}/test/'.format(sat)):
    if file.endswith('.png'):
        existing.append(file)

data_list = data_list[data_list['filename'].isin(existing)]
print("# of samples in test: ", data_list.shape[0])


def load_images(files, directory):
    """
    given a list of files it returns them from the directory
    - files: list
    - directory: str (path to directory)
    - returns: list containing the images
    """
    images = []
    for f in files:
        image = Image.open(directory + f, 'r')
        image = np.array(image)[-IMG_SIZE:, :IMG_SIZE, :] / 255.
        images.append(image)
    return images


# run
X = np.array(load_images(data_list['filename'], 'data/{}/test/'.format(sat)))
y_pred = model.predict(X)
y_true = data_list['value'].astype(int)

# classification report
from sklearn.metrics import classification_report
y = tf.keras.utils.to_categorical(np.array(y_true)-1, num_classes=3)
print(classification_report(y, np.round(y_pred)))
