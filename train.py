import pandas as pd
import os
import numpy as np
from src.utils import save_history_plot, train_test_move
import datetime as dt
import argparse
from src.models import *
from src.utils import load_images

CLASSES = 3

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--satellite", required=True, help="google/sentinel")
ap.add_argument("-m", "--model", required=True, help="small cnn from scratch (cnn), finetune VGG16 (vgg16), finetune Mobile Net (mobnet)")
args = vars(ap.parse_args())

# parameters --------------------------------
SPLIT = 0.8

if args['satellite'] == 'google':
    IMG_SIZE = 256
    IMAGES_DIR = 'data/Google/'
    INDEX_DIR = 'data/Google/data_index.csv'

    if args['model'] == 'mobnet':
        print('finetuning mobilenet for google images')
        BATCH_SIZE = 4
        EPOCHS = 50
        EPOCHS = 50
        model = google_mobnet_finetune(classes=3)

    elif args['model'] == 'vgg16':
        print('finetuning VGG16 for google images')
        BATCH_SIZE = 4
        EPOCHS = 100
        model = vgg16_finetune(classes=3)

    elif args['model'] == 'cnn':
        print('training CNN for google images')
        BATCH_SIZE = 8
        EPOCHS = 50
        model = google_net(size=IMG_SIZE)

    elif args['model'] == 'simple':
        print('training simple CNN for google images')
        BATCH_SIZE = 8
        EPOCHS = 100
        model = google_simple(size=IMG_SIZE)

    else:
        raise ValueError("wrong model parameter")

elif args['satellite'] == 'sentinel':
    IMAGES_DIR = 'data/Sentinel/'
    IMG_SIZE = 256
    INDEX_DIR = 'data/Sentinel/data_index.csv'

    if args['model'] == 'cnn':
        print('training CNN for Sentinel images')
        BATCH_SIZE = 8
        EPOCHS = 50
        model = sentinel_net(size=IMG_SIZE)

    else:
        raise ValueError("wrong model parameter")

else:
    raise ValueError("wrong sat parameter")

# list of files to be used for training -----------------
data_list = pd.read_csv(INDEX_DIR)  # this is the list produced from "master_getdata.py"
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

# drop ones without pictures
existing = []
for file in os.listdir(IMAGES_DIR+'images'):
    if file.endswith('.png'):
        existing.append(file)

data_list = data_list[data_list['filename'].isin(existing)]
print("# of samples: ", data_list.shape[0])
data_list.value = data_list.value.astype(int)

# split in train and validation sets ----------------------
training_list = pd.DataFrame(columns=data_list.columns)
validation_list = pd.DataFrame(columns=data_list.columns)

for cls in data_list.value.unique():

    training_size_cls = int(len(data_list[data_list.value == cls]) * SPLIT)
    validation_size_cls = int(len(data_list[data_list.value == cls]) - training_size_cls)

    training_list_cls = data_list[data_list.value == cls][:training_size_cls]
    validation_list_cls = data_list[data_list.value == cls][training_size_cls:]

    training_list = training_list.append(training_list_cls)
    validation_list = validation_list.append(validation_list_cls)

# shuffling
training_list = training_list.sample(frac=1, random_state=7777)  # shuffle the data
validation_list = validation_list.sample(frac=1, random_state=7777)  # shuffle the data
if os.path.exists(IMAGES_DIR+'/train'):
    print('train and test folders already created.')
else:
    train_test_move(training_list, validation_list, IMAGES_DIR)


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

            X = load_images(files[batch_start:limit], IMAGES_DIR+'images/', IMG_SIZE)
            Y = labels[batch_start:limit]
            if flip:  # also add the rotated images
                X.extend(load_images(files[batch_start:limit], IMAGES_DIR+'images/', IMG_SIZE, flip=True))
                Y = np.concatenate((Y, Y))

            yield (np.array(X), Y)

            batch_start += batch_size
            batch_end += batch_size


# 1 hot encode the labels
train_labels = tf.keras.utils.to_categorical(training_list['value'].values - 1, num_classes=CLASSES)
valid_labels = tf.keras.utils.to_categorical(validation_list['value'].values - 1, num_classes=CLASSES)

# callbacks ---------------
tboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}-{}-{}".format(
    args['satellite'],
    args['model'],
    dt.datetime.now().minute), write_graph=False)
weights_filepath="models/{}_{}_weights_best.hdf5".format(args['satellite'], args['model'])
checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# model ----------------
history = model.fit_generator(
    generator=data_generator(train_labels, training_list['filename'], BATCH_SIZE, flip=True),
    validation_data=data_generator(valid_labels, validation_list['filename'], BATCH_SIZE, flip=True),
    validation_steps=40,
    steps_per_epoch=80,
    #class_weight={0:1, 1:1.2, 2:1},
    epochs=EPOCHS, verbose=1, callbacks=[tboard, checkpoint])

save_history_plot(history, 'results/{}_{}_history.png'.format(args['satellite'], args['model']))
print('training history saved: ', 'results/{}_{}_history.png'.format(args['satellite'], args['model']))

# save best model
model.load_weights(weights_filepath)
model.save('results/{}_{}.h5'.format(args['satellite'], args['model']))
print('model saved: ', 'results/{}_{}.h5'.format(args['satellite'], args['model']))

# checking layers weights
for layer in model.layers:
    try:
        print(layer.name)
        print(layer.get_weights()[0].max(), layer.get_weights()[0].min(), layer.get_weights()[0].mean())
    except IndexError:
        pass
