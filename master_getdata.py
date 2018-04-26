# Downlaods Sentinel 2 images around a given coordinate location.
from osgeo import gdal
import numpy as np
import pandas as pd
from src.SentinelExp import sentinelDownlaoder, download_and_unzip, rgbtiffstojpg
import os
import urllib
from io import BytesIO

start_date = '2015-01-01'
end_date = '2016-12-01'

for raster, landuse in zip(
        ['data/nightlights_bin_Senegal.tif', 'data/nightlights_bin_Nigeria.tif',
         'data/nightlights_bin_Uganda.tif', 'data/nightlights_bin_Malawi.tif'],
        ['data/esa_landcover_Senegal_b_10.tif', 'data/esa_landcover_Nigeria_b_10.tif',
         'data/esa_landcover_Uganda_b_10.tif', 'data/esa_landcover_Malawi_b_10.tif']):

    # Load centroids of raster ------------------------
    r = gdal.Open(raster)
    band = r.GetRasterBand(1) #bands start at one
    a = band.ReadAsArray().astype(np.float)
    (y_index, x_index) = np.nonzero(a >= 0)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
    x_coords = x_index * x_size + upper_left_x + (x_size / 2)
    y_coords = y_index * y_size + upper_left_y + (y_size / 2)

    # make dataframe with list of files and targets --------
    nl_data = pd.DataFrame({'x': x_coords, 'y': y_coords, 'value': a[y_index, x_index]})
    nl_data.x = nl_data.x.round(decimals=4)
    nl_data.y = nl_data.y.round(decimals=4)

    # get the landuse for each tile -----------------------
    import georasters as gr
    esa = gr.load_tiff(landuse)

    # Find location of point (x,y) on raster, e.g. to extract info at that location
    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(landuse)

    def lu_extract(row):
        c, r = gr.map_pixel(row['x'], row['y'], GeoT[1], GeoT[-1], GeoT[0], GeoT[3])
        lu = esa[c, r]
        return lu

    nl_data['landuse'] = nl_data.apply(lu_extract, axis=1)

    nl_data['landuse'].value_counts()
    nl_data = nl_data[nl_data['landuse'] > 0]  # take only built areas

    # take same counts for the 3 classes ----------------
    nl_data_0 = nl_data[nl_data.value < 1]
    nl_data_1 = nl_data[nl_data.value == 1]
    nl_data_2 = nl_data[nl_data.value == 2]
    # n in this case will be the count of the least representative class (or 1k max per class and raster)
    s = min(nl_data_0.shape[0], nl_data_1.shape[0], nl_data_2.shape[0], 1000)
    nl_data_0 = nl_data_0.sample(n=s, random_state=1234)
    nl_data_1 = nl_data_1.sample(n=s, random_state=1234)
    nl_data_2 = nl_data_2.sample(n=s, random_state=1234)

    nl_data = pd.concat((nl_data_0, nl_data_1, nl_data_2))

    # Sentinel Images Download ------------------------------
    c = 0
    for x, y in zip(nl_data.x, nl_data.y):
        if os.path.exists('data/images/{}_{}.png'.format(y, x)):
            # print('{}_{} already downloaded'.format(y, x))
            pass
        else:
            print('downloading: {}_{}'.format(y, x))
            url = sentinelDownlaoder(y, x, start_date, end_date)
            ur = urllib.request.urlopen(url).read()
            buffer = BytesIO(ur)
            gee_tif = download_and_unzip(buffer, 'data')
            rgbtiffstojpg(gee_tif, 'data/', '{}_{}.png'.format(y, x))

        c += 1

        if c%10 == 0: print('{} images downlaoded ({}%)'.format(c, np.round(c/len(nl_data),2)*100))

    # write file index to csv -----------------------------
    try:
        data_index_prev = pd.read_csv('data_index.csv')
        data_index = pd.concat((data_index_prev, nl_data))
        data_index.to_csv('data_index.csv', index=False)
        print('added data to the list')

    except FileNotFoundError:
        print('no previous data_index')
        nl_data.to_csv('data_index.csv', index=False)