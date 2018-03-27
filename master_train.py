# Training a NN that uses Sentinel 2 images to extract features for insecurity.
# Baseline trains on nightlights.

import pandas as pd
from PIL import Image
import numpy as np
import tensorflow as tf

# parameters --------------------------------
split = 0.8
IMAGES_DIR = 'data/images/'
img_rows, img_cols = 400, 400
classes = 3
batch_size = 4
epochs = 10

# list of files to be used for training ------
data_list = pd.read_csv('data_index.csv') # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(x['y']) + '_' + str(x['x']) + '.png', axis=1) # filename is lon_lat

data_list = data_list.sample(frac=1)  # shuffle the data
labels = tf.keras.utils.to_categorical(data_list.value, 3)  # convert output to 1-hot

# split in train and validation sets
training_size = int(len(data_list) * split)
validation_size = int(len(data_list) - training_size)

training_list = data_list[:training_size]
training_labels = labels[:training_size]
validation_list = data_list[training_size:]
validation_labels = labels[training_size:]

print('tot size: ', len(data_list), ' training size: ', len(training_list), ' validation size: ', len(validation_list))


# generators -----------------------
def load_images(files, directory):
    """
    given a list of iamge files it returns them from the directory
    - files: list
    - directory: str (path to directory)
    - returns: list containing the images
    """
    images = []
    for f in files:
        image = Image.open(directory+f, 'r')
        image = image.convert('RGB').resize((400, 400), Image.ANTIALIAS)
        image = np.array(image)
        images.append(image)
    return images


def image_preprocessing(img):
    """ scale images by ..."""
    img = img / np.amax(img)
    return img


def data_generator(files, labels, batch_size):
    """
    Python generator, returns batches of images in array format
    files: list with filenames
    labels: list of targets
    batch_size: int
    returns: np arrays of (batch_size,rows,cols,channels)
    """

    size = len(files)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < size:
            limit = min(batch_end, size)

            X = np.array(load_images(files[batch_start:limit], IMAGES_DIR))
            Y = labels[batch_start:limit]

            X = image_preprocessing(X)

            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


# model --------------------------------------------
# load pretrained VGG16 on ImageNet
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(400,400,3))

# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers:
    print(layer.name, ' : ', layer.trainable)

# build a classifier model to put on top of the convolutional model
top_model = tf.keras.models.Sequential()
top_model.add(tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
top_model.add(tf.keras.layers.Dropout(0.5))
top_model.add(tf.keras.layers.Dense(3, activation='softmax'))

# add the model on top of the convolutional base
model = tf.keras.models.Model(inputs=base_model.input, outputs=top_model(base_model.output))

# compile and train ----------------------------------------------
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

print('INFO: training ...')
history = model.fit_generator(data_generator(training_list['filename'], training_labels, batch_size),
                              validation_data=data_generator(validation_list['filename'], validation_labels, batch_size),
                              validation_steps=validation_size / batch_size, steps_per_epoch=training_size / batch_size,
                              epochs=epochs)