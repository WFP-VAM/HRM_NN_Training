# Downlaods Sentinel 2 images around a given coordinate location.
from osgeo import gdal
import numpy as np
import pandas as pd
from SentinelExp import *
import os
import urllib
from io import BytesIO

start_date = '2015-01-01'
end_date = '2015-12-01'
raster = "data/nightlights_bin_Senegal.tif"  # nightlights ratser

# Load centroids of raster ------------------------
r = gdal.Open(raster)
band = r.GetRasterBand(1) #bands start at one
a = band.ReadAsArray().astype(np.float)
(y_index, x_index) = np.nonzero(a >= 0)
(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
x_coords = x_index * x_size + upper_left_x + (x_size / 2) #add half the cell size
y_coords = y_index * y_size + upper_left_y + (y_size / 2) #to centre the point

# make dataframe with list of files and targets --------
nl_data = pd.DataFrame({'x': x_coords, 'y': y_coords, 'value': a[y_index, x_index]})
nl_data.x = nl_data.x.round(decimals=4)
nl_data.y = nl_data.y.round(decimals=4)

# take same counts for the 3 classes ----------------
nl_data_0 = nl_data[nl_data.value == 0]
nl_data_1 = nl_data[nl_data.value == 1]
nl_data_2 = nl_data[nl_data.value == 2]
# 1878 is the count for the least represented class
nl_data_0 = nl_data_0.sample(n=1000, random_state=1234)
nl_data_1 = nl_data_1.sample(n=1000, random_state=1234)
nl_data_2 = nl_data_2.sample(n=1000, random_state=1234)

nl_data = pd.concat((nl_data_0, nl_data_1, nl_data_2))

# Sentinel Images Download ------------------------------
c = 0
for x, y in zip(nl_data.x, nl_data.y):
    if os.path.exists('data/images/{}_{}.png'.format(y, x)):
        print('{}_{} already downloaded'.format(y, x))
    else:
        print('downloading: {}_{}'.format(y, x))
        url = sentinelDownlaoder(y, x, start_date, end_date) you d
        ur = urllib.request.urlopen(url).read()
        buffer = BytesIO(ur)
        gee_tif = download_and_unzip(buffer, 'data')
        rgbtiffstojpg(gee_tif, 'data/', '{}_{}.png'.format(y, x))

    c += 1

    if c%100==0: print(c)

# write file index to csv -----------------------------
nl_data.to_csv('data_index.csv', index=False)