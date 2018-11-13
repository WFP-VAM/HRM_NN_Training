"""
Random forest check.
"""

import pandas as pd
import os
from PIL import Image
import numpy as np

# parameters --------------------------------
IMAGES_DIR = 'data/images/'
img_size = 256
classes = 3
batch_size = 16
epochs = 35
split = 0.3

# list images -------------------------------
df = pd.read_csv('data_index.csv')
df['filename'] = df.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1)

# drop ones without pictures
existing = []
for file in os.listdir(IMAGES_DIR):
    if file.endswith('.png'):
        existing.append(file)

# load data ---------------------------------
df = df[df['filename'].isin(existing)]
df.value = df.value.astype(int)
data_list = df.sample(frac=1, random_state=666)  # shuffle the data


def load_images(files, directory):

    def get_image(image_path):
        image = Image.open(image_path, 'r')
        image = image.convert('RGB')
        image = image.resize((img_size, img_size))
        image = np.array(image)
        return image

    images = np.zeros((len(files), img_size, img_size, 3))
    c = 0
    for f in files:
        im = get_image(directory + f)
        images[c, :, :, :]
        c += 1
    return images.reshape(len(files),img_size*img_size*3)


img = load_images(df.filename[:int(df.shape[0]*split)], IMAGES_DIR)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=0, verbose=2, n_jobs=-1)
clf.fit(img, data_list[:int(df.shape[0]*split)].value.astype(int))

img_test = load_images(df.filename[int(df.shape[0]*split*2):], IMAGES_DIR)
clf.score(img, data_list[:int(df.shape[0]*split)].value.astype(int))
clf.score(img_test, data_list[int(df.shape[0]*split*2):].value.astype(int))