import pandas as pd
import os
import numpy as np
from src.models import *
from src.utils import save_history_plot, train_test_move
from PIL import Image
import tensorflow as tf
import datetime as dt

# parameters --------------------------------
split = 0.8
IMAGES_DIR = 'data/Sentinel/images/'
img_size = 500
classes = 3
batch_size = 8
epochs = 40

# list of files to be used for training -----------------
data_list = pd.read_csv('data/Sentinel/data_index.csv') # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

# drop ones without pictures
existing = []
for file in os.listdir(IMAGES_DIR):
    if file.endswith('.png'):
        existing.append(file)

data_list = data_list[data_list['filename'].isin(existing)]
print("# of samples: ", data_list.shape[0])
data_list.value = data_list.value.astype(int)

# split in train and validation sets ----------------------
training_list = pd.DataFrame(columns=data_list.columns)
validation_list = pd.DataFrame(columns=data_list.columns)

for cls in data_list.value.unique():

    training_size_cls = int(len(data_list[data_list.value == cls]) * split)
    validation_size_cls = int(len(data_list[data_list.value == cls]) - training_size_cls)

    training_list_cls = data_list[data_list.value == cls][:training_size_cls]
    validation_list_cls = data_list[data_list.value == cls][training_size_cls:]

    training_list = training_list.append(training_list_cls)
    validation_list = validation_list.append(validation_list_cls)

training_list = training_list.sample(frac=1, random_state=901)#333)  # shuffle the data
validation_list = validation_list.sample(frac=1, random_state=901)#333)  # shuffle the data
train_test_move(training_list, validation_list, 'data/Sentinel/')


def load_images(files, directory, flip=False):
    """
    given a list of files it returns them from the directory
    - files: list
    - directory: str (path to directory)
    - flip: data augnemntation, rotate the image by 180, doubles the batch size.
    - returns: list containing the images
    """
    images = []
    for f in files:
        image = Image.open(directory + f, 'r')
        if flip:
            image = image.rotate(180)
            image = np.array(image)[:500, :500, :] / 255.
        else:
            image = np.array(image)[:500, :500, :] / 255.

        images.append(image)
    return images


def data_generator(labels, files, batch_size, flip=False):
    """
    Python generator, returns batches of images in array format
    files: list of images files
    labels:
    batch_size: int
    returns: np arrays of (batch_size,rows,cols,channels)
    """

    size = len(files)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < size:
            limit = min(batch_end, size)

            X = load_images(files[batch_start:limit], IMAGES_DIR)
            Y = labels[batch_start:limit]
            if flip:
                X.extend(load_images(files[batch_start:limit], IMAGES_DIR, flip=True))
                Y = np.concatenate((Y, Y))

            yield (np.array(X), Y)

            batch_start += batch_size
            batch_end += batch_size

train_labels = tf.keras.utils.to_categorical(training_list['value'].values - 1, num_classes=classes)
valid_labels = tf.keras.utils.to_categorical(validation_list['value'].values - 1, num_classes=classes)

model = sentinel_net(img_size)
tboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}-{}".
                                       format(dt.datetime.now().hour, dt.datetime.now().minute), write_graph=False)
history = model.fit_generator(
    data_generator(train_labels, training_list['filename'], batch_size, flip=True),
    validation_data=data_generator(valid_labels, validation_list['filename'], batch_size, flip=True),
    validation_steps=40,
    steps_per_epoch=80,
    epochs=epochs, verbose=1, callbacks=[tboard])

save_history_plot(history, 'results/Sentinel_history.png')

# save model
model.save('results/Sentinel.h5')

# checking layers weights
for layer in model.layers:
    try:
        print(layer)
        print(layer.get_weights()[0].max(), layer.get_weights()[0].min(), layer.get_weights()[0].mean())
    except IndexError:
        pass

