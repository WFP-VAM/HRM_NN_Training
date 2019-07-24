from shutil import copyfile, rmtree
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


def load_images(files, directory, img_size, flip=False):
    """
    given a list of images it returns them from the directory as np arrays
    - files: list
    - directory: str (path to directory)
    - img_size: the image will be resized with PIL
    - raw_size: we crop the pic before resizing to cut logos
    - flip: data augmentation, rotate the image by 180, doubles the batch size.
    - returns: list containing the images
    """
    images = []
    for f in files:
        image = Image.open(directory + f, 'r')
        image = image.crop((  # crop
            int(image.size[0] / 2 - img_size / 2),
            int(image.size[1] / 2 - img_size / 2),
            int(image.size[0] / 2 + img_size / 2),
            int(image.size[1] / 2 + img_size / 2)
        ))
        #image = image.resize((img_size, img_size), Image.ANTIALIAS)
        if flip:
            image = image.rotate(180)
            image = np.array(image) / 255.
        else:
            image = np.array(image) / 255.

        images.append(image)
    return images


def train_test_move(training_list, validation_list, img_dir):
    """ given a list of training and validation files it splits them into train/test directories. """
    os.makedirs(img_dir + 'train')
    os.makedirs(img_dir + 'test')

    for i, row in training_list.iterrows():

        copyfile(img_dir+'images/{}'.format(row['filename']),
                 img_dir+'train/{}'.format(row['filename']))

    for i, row in validation_list.iterrows():

        copyfile(img_dir+'images/{}'.format(row['filename']),
                 img_dir+'test/{}'.format(row['filename']))

    return 'directories for classification ready.'


def data_directories(training_list, validation_list, img_dir):
    """
    because we are using the Keras DataGenerator, we want the images used for training
    to be split in the label's directories, i.e. all images with luminosity = 1
    go into directory /1 etc ...
    :param training_list: DataFrame containing the list of files to be used for training,
    'filename' for image name and 'value' for the label.
    :param validation_list: DataFrame containing the list of files to be used for validation,
    'filename' for image name and 'value' for the label.
    :param img_dir: string, path to the directory containing the images.
    :return: None
    """
    # if dirs there remove them
    if os.path.exists(img_dir+'train/0'):
        for dir in ['train', 'test']:
            for i in [1,2,3]:
                rmtree(img_dir+'{}/{}'.format(dir, str(i)))
    # create new dirs
    for i in [1, 2, 3]:
        os.makedirs(img_dir+'train/{}'.format(i))
    # move files there
    for i, row in training_list.iterrows():

        copyfile(img_dir+'images/{}'.format(row['filename']),
                 img_dir+'train/{}/{}'.format(row['value'], row['filename']))

    # create new dirs
    for i in [1, 2, 3]:
        os.makedirs(img_dir+'test/{}'.format(i))
    for i, row in validation_list.iterrows():

        copyfile(img_dir+'images/{}'.format(row['filename']),
                 img_dir+'test/{}/{}'.format(row['value'], row['filename']))

    return 'directories for classification ready.'


# save training history --------------------------------
def save_history_plot(history, path):
    """
    Save trining history plot to file.
    """
    plt.switch_backend('agg')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('crossentropy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.axhline(y=max(history.history['acc']))
    plt.text(20, max(history.history['acc']), round(max(history.history['acc']), 2), fontdict={'color':'b'})

    plt.axhline(y=max(history.history['val_acc']))
    plt.text(20, max(history.history['val_acc']), round(max(history.history['val_acc']), 2), fontdict={'color':'orange'})

    plt.savefig(path)
    return


def custom_shuffler(in_df):
    """ shuffle across the repeated labels (3,3,3,2,2,2,1,1,1) ..."""
    df = in_df.copy()
    df['cumcount'] = df.groupby('value').cumcount()
    df = df.sort_values('cumcount')
    df = df.sample(frac=1, random_state=18)
    return df.drop('cumcount', axis=1)