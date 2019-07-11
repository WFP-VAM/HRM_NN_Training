"""
pull nightlights using GEE with the src/nightlights_gee.js.
This script bins them into classes for training.
"""

from osgeo import gdal
import numpy as np

# read the ni
raster = gdal.Open('data/nightlights_africa_settlements.tif')
band = raster.GetRasterBand(1)
print('minimum and maximum: ', band.ComputeRasterMinMax(True))

nightlights = band.ReadAsArray()

# nightlights = np.round(np.float64(nightlights), 2)
intervals = [0.5, 10, 25, 40]
print('0: ', len(nightlights[(nightlights <= intervals[0]) | (np.isnan(nightlights))]) / len(nightlights.ravel()))
print('poor: ', len(nightlights[(nightlights > intervals[0]) & (nightlights <= intervals[1])]) )
print('medium: ', len(nightlights[(nightlights > intervals[1]) & (nightlights <= intervals[2])]))
print('rich: ', len(nightlights[(nightlights > intervals[2]) & (nightlights <= intervals[3])]))

# if need to aggregate georasters.aggregate(data,NDV,(10,10))

# Bin
nightlights[(nightlights <= intervals[0]) | (np.isnan(nightlights)) | (nightlights > intervals[3])] = np.nan
nightlights[(nightlights > intervals[0]) & (nightlights <= intervals[1])] = 1
nightlights[(nightlights > intervals[1]) & (nightlights <= intervals[2])] = 2
nightlights[(nightlights > intervals[2]) & (nightlights <= intervals[3])] = 3

# write out
[cols, rows] = nightlights.shape

output_raster = gdal.GetDriverByName('GTiff').Create('data/nightlights_africa_settlements_bin.tif', rows, cols, 1, gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(raster.GetGeoTransform())  # Specify its coordinates
output_raster.SetProjection(raster.GetProjection())
output_raster.GetRasterBand(1).SetNoDataValue(-99)

output_raster.GetRasterBand(1).WriteArray(nightlights)  # Writes my array to the raster
output_raster.FlushCache()  # saves to disk!!