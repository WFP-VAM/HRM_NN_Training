"""
Correlations between survey and nightlights.
"""
import pandas as pd
import numpy as np

for filename, indicator in zip([
    '../HRM/HRM/Data/datasets/WFP_ENSAN_Senegal_2013_cluster.csv',
    '../HRM/HRM/Data/datasets/WB_Uganda_2011_cluster.csv'],
        ['FCS_mean',
         'cons']):
    dataset = pd.read_csv(filename)
    try:
        dataset = dataset[dataset.cons <= 5]
    except AttributeError:
        dataset = dataset[dataset.FCS_mean <= 30]

    import georasters as gr
    nightlights = 'data/nightlights.tif'
    esa = gr.load_tiff(nightlights)

    # Find location of point (x,y) on raster, e.g. to extract info at that location
    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(nightlights)


    def lu_extract(row):
        try:
            c, r = gr.map_pixel(row['gpsLongitude'], row['gpsLatitude'], GeoT[1], GeoT[-1], GeoT[0], GeoT[3])
            lu = esa[c, r]
            return lu
        except IndexError:
            print('coordinates {} {} at sea!'.format(row['gpsLongitude'], row['gpsLatitude']))

    dataset['nightlights'] = dataset.apply(lu_extract, axis=1)

    print('file {} correlates {} with nightlights.'.format(
        filename, np.round(dataset[indicator].corr(dataset['nightlights']), 2)))








