"""
download the nightlights from: https://ngdc.noaa.gov/eog/download.html

this script bins them into classes for training.
"""

from osgeo import gdal
import numpy as np

raster = gdal.Open('data/nightlights.tif')
nightlights = np.array(raster.GetRasterBand(1).ReadAsArray())

np.unique(nightlights)

print('0: ', len(nightlights[nightlights == 0]) / len(nightlights.ravel()))
print('below 10: ', len(nightlights[(nightlights > 0) & (nightlights < 10)]) / len(nightlights.ravel()))
print('above 10: ', len(nightlights[(nightlights >= 10)]) / len(nightlights.ravel()))
#print('above 50: ', len(nightlights[nightlights > 50]) / len(nightlights.ravel()))

# Bin
nightlights[(nightlights > 0) & (nightlights < 10)] = 1
nightlights[(nightlights >= 10)] = 2

# if need to aggregate georasters.aggregate(data,NDV,(10,10))

# write out
[cols, rows] = nightlights.shape

output_raster = gdal.GetDriverByName('GTiff').Create('data/nightlights_crop_bin.tif', rows, cols, 1, gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(raster.GetGeoTransform())  # Specify its coordinates
output_raster.SetProjection(raster.GetProjection())
output_raster.GetRasterBand(1).SetNoDataValue(-99)

output_raster.GetRasterBand(1).WriteArray(nightlights)  # Writes my array to the raster
output_raster.FlushCache()  # saves to disk!!

# then for each country:
# gdalwarp -crop_to_cutline -cutline C:\Users\lorenzo.riches\Downloads\UGA_adm_shp\UGA_adm0.shp nightlights_crop_bin.tif nightlights_bin_Uganda.tif