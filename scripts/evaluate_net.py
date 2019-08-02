""" Calculating the accuracy of the model """
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.path.append('src')
from utils import load_images
import argparse
from sklearn.metrics import classification_report


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--satellite", required=True, help="google/sentinel")
ap.add_argument("-m", "--model", required=True, help="google_cnn/google_vgg16 etc...")
ap.add_argument("-z", "--size", required=True, help="image size (256/400)")

args = vars(ap.parse_args())

IMG_SIZE = args['size']

# load model and predict
model = tf.keras.models.load_model('results/{}.h5'.format(args['model']))

# list of files used for training
data_list = pd.read_csv('data/{}/data_index.csv'.format(args['satellite']))  # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

# drop ones without pictures
existing = []
for file in os.listdir('data/{}/test/'.format(args['satellite'])):
    if file.endswith('.png'):
        existing.append(file)

data_list = data_list[data_list['filename'].isin(existing)][:3000]
print("# of samples in test: ", data_list.shape[0])

# run
X = np.array(load_images(data_list['filename'], 'data/{}/test/'.format(args['satellite']), img_size=int(IMG_SIZE)))
y_pred = model.predict(X)
y_true = data_list['value'].astype(int)

# classification report
y = tf.keras.utils.to_categorical(np.array(y_true)-1, num_classes=3)
print(classification_report(y, np.round(y_pred)))
