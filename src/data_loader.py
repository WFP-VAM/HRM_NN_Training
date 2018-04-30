from shutil import copyfile, rmtree
import os


def data_directories(training_list, validation_list, img_dir):
    # split files into respective directories -------------------
    if os.path.exists(img_dir+'train/0'):
        for dir in ['train', 'test']:
            for i in [0,1,2]:
                rmtree(img_dir+'{}/{}'.format(dir, str(i)))

    os.makedirs(img_dir+'train/0')
    os.makedirs(img_dir+'train/1')
    os.makedirs(img_dir+'train/2')
    for i, row in training_list.iterrows():

        copyfile(img_dir+'images/{}'.format(row['filename']),
                 img_dir+'train/{}/{}'.format(row['value'], row['filename']))

    os.makedirs(img_dir+'test/0')
    os.makedirs(img_dir+'test/1')
    os.makedirs(img_dir+'test/2')
    for i, row in validation_list.iterrows():

        copyfile(img_dir+'images/{}'.format(row['filename']),
                 img_dir+'test/{}/{}'.format(row['value'], row['filename']))

    return 'directories for classification ready.'
