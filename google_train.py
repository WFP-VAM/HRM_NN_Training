import tensorflow as tf
import pandas as pd
from shutil import rmtree
import os
import numpy as np
import matplotlib.pyplot as plt
from src.models import google_netowrk
from src.data_loader import data_directories
from time import time

# parameters --------------------------------
split = 0.8
IMAGES_DIR = 'data/Google/'
img_size = 400
classes = 3
batch_size = 32
epochs = 30

# list of files to be used for training -----------------
data_list = pd.read_csv('data/Google/data_index.csv') # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

# drop ones without pictures
existing = []
for file in os.listdir(IMAGES_DIR+'images/'):
    if file.endswith('.png'):
        existing.append(file)

data_list = data_list[data_list['filename'].isin(existing)]

data_list.value = data_list.value.astype(int)
data_list = data_list.sample(frac=1, random_state=666)  # shuffle the data

# split in train and validation sets
training_list = pd.DataFrame(columns=data_list.columns)
validation_list = pd.DataFrame(columns=data_list.columns)
for cls in data_list.value.unique():

    training_size_cls = int(len(data_list[data_list.value == cls]) * split)
    validation_size_cls = int(len(data_list[data_list.value == cls]) - training_size_cls)

    training_list_cls = data_list[data_list.value == cls][:training_size_cls]
    validation_list_cls = data_list[data_list.value == cls][training_size_cls:]

    training_list = training_list.append(training_list_cls)
    validation_list = validation_list.append(validation_list_cls)
# -----------------------------------------------

# split files into respective directories -------------------
data_directories(training_list, validation_list, IMAGES_DIR)
# -----------------------------------------------------

# generators ------------------------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        height_shift_range=0.2)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/Google/train',
        target_size=(img_size, img_size),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        'data/Google/test',
        target_size=(img_size, img_size),
        batch_size=batch_size)


# model --------------------------------------------
model = google_netowrk(img_size)

tboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()), write_graph=False)
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=100, callbacks=[tboard])

# remove ad hoc class folders -------
for dir in ['train', 'test']:
    rmtree('data/Google/{}'.format(dir))


# save training history --------------------------------
def save_history_plot(history, path):
    plt.switch_backend('agg')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)


save_history_plot(history, 'Google_history.png')

# save model
model.save('Google.h5')