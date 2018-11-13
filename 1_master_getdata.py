"""
Downloads Sentinel 2 and Google images for given coordinate location.
We use a raster that need to be preprocessed:

    _Nightlights_
    luminosity at night
    produce the raster (data/africa_nightlights_bin.tif) with "nightlights_prep.py"

"""
from osgeo import gdal
import numpy as np
import pandas as pd
from src.SentinelExp import sentinel_downlaoder, download_and_unzip, rgbtiffstojpg
import os
import urllib
from io import BytesIO
import scipy.misc

# parameters ------
# dates from and to used for Sentinel imgaes
START_DATE = '2016-01-01'
END_DATE = '2017-12-01'

# loop over the nightlights and landuse rasters for each country.
raster = "data/nightlights_africa_HRM_bin.tif"

print('raster: {} \n'.format(raster))
# Load centroids of raster ------------------------
r = gdal.Open(raster)
band = r.GetRasterBand(1)  # bands start at one
a = band.ReadAsArray().astype(np.float)
(y_index, x_index) = np.nonzero(a >= 0)
(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
x_coords = x_index * x_size + upper_left_x + (x_size / 2)
y_coords = y_index * y_size + upper_left_y + (y_size / 2)

# make dataframe with list of files and targets --------
nl_data = pd.DataFrame({'x': x_coords, 'y': y_coords, 'value': a[y_index, x_index]})
nl_data.x = nl_data.x.round(decimals=4)
nl_data.y = nl_data.y.round(decimals=4)

# take same counts for the 3 classes ----------------
nl_data.value = nl_data.value.astype(int)
nl_data_1 = nl_data[nl_data.value == 1]
nl_data_2 = nl_data[nl_data.value == 2]
nl_data_3 = nl_data[nl_data.value == 3]

# n in this case will be the count of the least representative class (or 1k max per class and raster)
s = min(nl_data_3.shape[0], nl_data_1.shape[0], nl_data_2.shape[0], 2750)
nl_data_0 = nl_data_1.sample(n=s, random_state=4321)
nl_data_1 = nl_data_2.sample(n=s, random_state=4321)
nl_data_2 = nl_data_3.sample(n=s, random_state=4321)

nl_data = pd.concat((nl_data_0, nl_data_1, nl_data_2))
print('images to download: ', nl_data.shape[0])
# Images Download ------------------------------
for source in ['Google']:#, 'Sentinel']:
    print('Source: ', source)
    img_dir = 'data/{}/'.format(source)
    c = 0
    for x, y in zip(nl_data.x, nl_data.y):
        if os.path.exists(img_dir+'images/{}_{}.png'.format(y, x)):
            pass
        else:
            print('downloading: {}_{}'.format(y, x))
            if source == 'Sentinel':
                url = sentinel_downlaoder(y, x, START_DATE, END_DATE)
            else:
                url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(y) + ',' + \
                  str(x) + '&zoom=16&size=400x500&maptype=satellite&key=' + os.environ['Google_key']

            ur = urllib.request.urlopen(url).read()
            buffer = BytesIO(ur)
            if source == 'Sentinel':
                gee_tif = download_and_unzip(buffer, img_dir)
                rgbtiffstojpg(gee_tif, img_dir, '{}_{}.png'.format(y, x))
            else:
                image = scipy.misc.imread(buffer, mode='RGB')
                scipy.misc.imsave(img_dir + 'images/{}_{}.png'.format(y, x), image)

        c += 1

        if c%10 == 0: print('{} images downlaoded ({}%)'.format(c, np.round(c/len(nl_data), 2)*100))

    # write file index to csv -----------------------------
    try:
        data_index_prev = pd.read_csv(img_dir+'data_index.csv')
        data_index = pd.concat((data_index_prev, nl_data))
        data_index.to_csv(img_dir+'data_index.csv', index=False)
        print('added data to the list')

    except FileNotFoundError:
        print('no previous data_index')
        nl_data.to_csv(img_dir+'data_index.csv', index=False)