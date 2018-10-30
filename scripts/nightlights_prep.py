"""
poverty product from NOAA is horrible (https://ngdc.noaa.gov/eog/dmsp/download_poverty.html)

Use this script in GEE to get nightlights:


var NLImgSet = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
            .select('avg_rad')
            .filterDate('2014-01-01', '2018-01-01')

var median = NLImgSet.reduce(ee.Reducer.median());
median = median.updateMask(median.gte(0.2))

Map.addLayer(median, {min: 0.01, max: 100, palette: ['blue', 'green', 'red']}, 'nightlights');

// EXPORT results
Export.image.toDrive({
  image: median,
  description: 'nightlights_africa',
  scale: 1000,
  region: polygon
});

this script bins them into classes for training.
"""

from osgeo import gdal
import numpy as np

raster = gdal.Open('data/nightlights_africa.tif')
nightlights = np.array(raster.GetRasterBand(1).ReadAsArray())

# nightlights = np.round(np.float64(nightlights), 2)
print('0: ', len(nightlights[(nightlights <= 0.3) | (np.isnan(nightlights))]) / len(nightlights.ravel()))
print('poor: ', len(nightlights[(nightlights > 0.3) & (nightlights <= 10)]) / len(nightlights.ravel()))
print('medium: ', len(nightlights[(nightlights > 10) & (nightlights <= 50)]) / len(nightlights.ravel()))
print('rich: ', len(nightlights[nightlights > 50]) / len(nightlights.ravel()))

# if need to aggregate georasters.aggregate(data,NDV,(10,10))

# Bin
nightlights[(nightlights <= 0.3) | (np.isnan(nightlights))] = np.nan
nightlights[(nightlights > 0.3) & (nightlights <= 10)] = 1
nightlights[(nightlights > 10) & (nightlights <= 50)] = 2
nightlights[nightlights > 50] = 3

# write out
[cols, rows] = nightlights.shape

output_raster = gdal.GetDriverByName('GTiff').Create('data/nightlights_africa_bin.tif', rows, cols, 1, gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(raster.GetGeoTransform())  # Specify its coordinates
output_raster.SetProjection(raster.GetProjection())
output_raster.GetRasterBand(1).SetNoDataValue(-99)

output_raster.GetRasterBand(1).WriteArray(nightlights)  # Writes my array to the raster
output_raster.FlushCache()  # saves to disk!!

# then for each country:
# gdalwarp -crop_to_cutline -cutline Downloads\UGA_adm_shp\UGA_adm0.shp nightlights_crop_bin.tif nightlights_bin_Uganda.tif