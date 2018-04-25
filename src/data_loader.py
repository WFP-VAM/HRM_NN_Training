from shutil import copyfile, rmtree
import os


def data_directories(training_list, validation_list):
    # split files into respective directories -------------------
    if os.path.exists('data/train/0'):
        for dir in ['train', 'test']:
            for i in [0,1,2]:
                rmtree('data/{}/{}'.format(dir, str(i)))

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

    return 'directories for classification ready.'
