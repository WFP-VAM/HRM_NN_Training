import tensorflow as tf
import pandas as pd
from shutil import copyfile, rmtree
import os
import numpy as np
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt

# parameters --------------------------------
split = 0.8
IMAGES_DIR = 'data/images/'
img_size = 400
classes = 3
batch_size = 16
epochs = 35

# list of files to be used for training -----------------
data_list = pd.read_csv('data_index.csv') # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

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
if os.path.exists('data/train/0'):
    for dir in ['train', 'test']:
        for i in [0,1,2]:
            rmtree('data/{}/{}'.format(dir, str(i)))

os.makedirs('data/train/0', )
os.makedirs('data/train/1')
os.makedirs('data/train/2')
for i, row in training_list.iterrows():

    copyfile('data/images/{}'.format(row['filename']),
             'data/train/{}/{}'.format(row['value'], row['filename']))

os.makedirs('data/test/0')
os.makedirs('data/test/1')
os.makedirs('data/test/2')
for i, row in validation_list.iterrows():

    copyfile('data/images/{}'.format(row['filename']),
             'data/test/{}/{}'.format(row['value'], row['filename']))
# -----------------------------------------------------


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        vertical_flip=True,
        zoom_range=0.2,
        channel_shift_range=0.2,
        horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_size, img_size),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_size, img_size),
        batch_size=batch_size)


# model --------------------------------------------
def netowrk(size):
    inputs = Input((size, size, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)

    dense1 = Dense(256)(pool3)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.5)(dense1)

    out = Flatten(name='flatten')(dense1)
    out = Dense(3)(out)
    out = Activation('softmax')(out)

    model = Model(inputs=[inputs], outputs=[out])

    # compile and train ----------------------------------------------
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
                  metrics=['accuracy'])

    return model


model = netowrk(img_size)
from time import time
#tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()), write_graph=False)
stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=100) #, callbacks=[stopper])

# remove ad hoc class folders -------
for dir in ['train', 'test']:
    rmtree('data/{}'.format(dir))


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


save_history_plot(history, 'training_history.png')

# save model
model.save('model.h5')
