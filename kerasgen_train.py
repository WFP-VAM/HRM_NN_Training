import tensorflow as tf
import pandas as pd
from shutil import copyfile, rmtree
import os

# parameters --------------------------------
split = 0.8
IMAGES_DIR = 'data/images/'
img_rows, img_cols = 256, 256
classes = 3
batch_size = 32
epochs = 25

# list of files to be used for training -----------------
data_list = pd.read_csv('data_index.csv') # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(x['y']) + '_' + str(x['x']) + '.png', axis=1) # filename is lon_lat

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
os.makedirs('data/train/0')
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
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        #target_size=(150, 150),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        #target_size=(150, 150),
        batch_size=batch_size)
# model --------------------------------------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(None, None, 3), name='B1'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(32, (3, 3), name='B2'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), name='B3'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(tf.keras.layers.Dense(256, name='D1'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(classes))
model.add(tf.keras.layers.Activation('softmax'))

from time import time
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()), write_graph=False)
stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')

# compile and train ----------------------------------------------
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=100)

# remove ad hoc class folders -------
for dir in ['train', 'test']:
    rmtree('data/{}'.format(dir))


# save training history --------------------------------
import matplotlib.pyplot as plt
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