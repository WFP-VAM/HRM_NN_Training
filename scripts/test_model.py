"""
Utility used to check some images, predictions and labels
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

sat = 'Google'
model = 'google_mobnet.h5'
IMG_SIZE = 256

# model
net = tf.keras.models.load_model('results/'+model, compile=False)

# data
data_list = pd.read_csv('data/{}/data_index.csv'.format(sat))
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

for i in range(1,5):
    d = data_list.sample(n=1)
    val = d.to_string(columns=['value'], header=False, index=False)

    if sat == 'Google':
        image = Image.open('data/Google/images/{}'.format(d.to_string(columns=['filename'], header=False, index=False)), 'r')
        image = image.crop((  # crop center
            int(image.size[0] / 2 - IMG_SIZE / 2),
            int(image.size[1] / 2 - IMG_SIZE / 2),
            int(image.size[0] / 2 + IMG_SIZE / 2),
            int(image.size[1] / 2 + IMG_SIZE / 2)
        ))
        image = np.array(image)/ 255.
        pred = net.predict(image.reshape(1, IMG_SIZE, IMG_SIZE, 3))

    if sat == 'Sentinel':
        image = Image.open('data/Sentinel/images/5.3011_-4.082.png', 'r')
        image.show()
        pred = net.predict(image.reshape(1, IMG_SIZE, IMG_SIZE, 3))

    # show
    plt.imshow(image)
    plt.title('prediction: ' + str(np.argmax(pred) + 1) + ' label: ' + str(val))
    plt.pause(0.5)
    time.sleep(5)
    plt.close()

    i += 1
