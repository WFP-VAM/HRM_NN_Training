# Downlaods Sentinel 2 images around a given coordinate location.
from osgeo import gdal
import numpy as np
import pandas as pd
from src.SentinelExp import sentinelDownlaoder, download_and_unzip, rgbtiffstojpg
import os
import urllib
from io import BytesIO
import yaml
import scipy as sp

start_date = '2015-01-01'
end_date = '2016-12-01'

with open('private_config.yml', 'r') as cfgfile:
    tokens = yaml.load(cfgfile)

# to get the rasters:
#   - produce the nightlights with "nightlights_prep.py"
#   - produce the landuse from HRM/application/resample_esa_raster.py
for raster, landuse in zip(
        ['data/nightlights_bin_Senegal.tif', 'data/nightlights_bin_Nigeria.tif', 'data/nightlights_bin_Uganda.tif',
         'data/nightlights_bin_Malawi.tif', 'data/nightlights_bin_Zimbawe.tif'],
        ['data/esa_landcover_Senegal_b_10.tif', 'data/esa_landcover_Nigeria_b_10.tif', 'data/esa_landcover_Uganda_b_10.tif',
         'data/esa_landcover_Malawi_b_10.tif', 'data/esa_landcover_Zimbawe_b_10.tif']):

    print('raster: {} \nlanduse: {}'.format(raster, landuse))
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

    # Images Download ------------------------------
    for source in ['Google']:#,'Sentinel']:
        print('Source: ', source)
        img_dir = 'data/{}/'.format(source)
        c = 0
        for x, y in zip(nl_data.x, nl_data.y):
            if os.path.exists(img_dir+'images/{}_{}.png'.format(y, x)):
                pass
            else:
                print('downloading: {}_{}'.format(y, x))
                if source == 'Sentinel':
                    url = sentinelDownlaoder(y, x, start_date, end_date)
                else:
                    url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(y) + ',' + \
                      str(x) + '&zoom=16&size=400x500&maptype=satellite&key=' + tokens['Google']

                ur = urllib.request.urlopen(url).read()
                buffer = BytesIO(ur)
                if source == 'Sentinel':
                    gee_tif = download_and_unzip(buffer, img_dir)
                    rgbtiffstojpg(gee_tif, img_dir, '{}_{}.png'.format(y, x))
                else:
                    image = sp.misc.imread(buffer, mode='RGB')
                    sp.misc.imsave(img_dir + 'images/{}_{}.png'.format(y, x), image)

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