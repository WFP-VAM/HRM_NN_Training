"""
Utility used to check some images
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

sat = 'Sentinel'
IMG_SIZE = 256

# data
data_list = pd.read_csv('data/{}/data_index.csv'.format(sat))
data_list['filename'] = data_list.apply(lambda x: str(np.round(x['y'], 4)) + '_' + str(np.round(x['x'],4)) + '.png', axis=1) # filename is lon_lat

for i in range(1, 5):
    d = data_list.sample(n=1)
    val = d.to_string(columns=['value'], header=False, index=False)

    image = Image.open('data/{}/images/{}'.format(sat, d.to_string(columns=['filename'], header=False, index=False)),
                       'r')
    image = image.crop((  # crop center
        int(image.size[0] / 2 - IMG_SIZE / 2),
        int(image.size[1] / 2 - IMG_SIZE / 2),
        int(image.size[0] / 2 + IMG_SIZE / 2),
        int(image.size[1] / 2 + IMG_SIZE / 2)
    ))

    # show
    plt.imshow(image)
    plt.title('label: ' + str(val))
    plt.pause(0.5)
    time.sleep(5)
    plt.close()

    i += 1